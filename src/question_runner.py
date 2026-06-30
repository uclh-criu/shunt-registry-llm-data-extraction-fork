"""
Generic LLM extraction loop + CLI entry point.

This version is fixed for UK National Shunt Registry extraction where each row is
a surgical episode/operation. It processes one operation row at a time, not one
MRN at a time, because the same MRN can have multiple shunt operations.

Run a single question:
    python question_runner.py q14

Run multiple questions:
    python question_runner.py q13 q14 q24

Run all registered questions:
    python question_runner.py all

Limit records:
    python question_runner.py q14 --max-records 20

Legacy alias:
    python question_runner.py q14 --max-mrns 20
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from llm_client import LLMClient, create_llm_client_from_config
from config import RESULTS_DATA_PATH

from utils import (
    append_results_to_csv,
    evaluate_predictions,
    extract_with_llm,
    print_evaluation_summary,
)


# ---------------------------------------------------------------------------
# Question specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuestionSpec:
    """Configuration for one registry question."""

    question_name: str
    """Display name, e.g. 'Q14 - ETV'."""

    gold_standard_col: str
    """Column in the gold/merged data for validation, e.g. 'ETV | Shunt Operation'."""

    prompt_file: str
    """Filename under prompts/, e.g. 'q14_prompt.txt'."""

    options: Optional[str]
    """Option list / registry text passed into the prompt template."""

    prediction_key: str
    """Prediction output column, e.g. 'Q14_ETV'."""

    note_sources: Tuple[str, ...] = ("Discharge Summary", "Op Note", "Clerking")
    """Wide-table columns to concatenate as input notes."""

    max_mrns: Optional[int] = None
    """Legacy limit. In this fixed version this is treated as max_records."""

    llm_kwargs: dict[str, Any] = field(default_factory=dict)
    """Provider-specific kwargs forwarded to generate_chat on every call.

    Examples:
        OpenAI: {"response_format": {"type": "json_object"}}
        Ollama: {"format": schema, "options": {"temperature": 0}}
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ID_COLUMNS = [
    "EpisodeID",
    "SurgicalCaseKey",
    "EncounterKey",
    "CSN",
    "MRN",
    "Identifier",
]


def _is_blank(value: Any) -> bool:
    """True for NaN/None/empty-ish values."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip() == ""


def _clean_cell(value: Any) -> str:
    """Return a safe string for notes/IDs."""
    if _is_blank(value):
        return ""
    return str(value).strip()


def _normalise_column_name(name: str) -> str:
    """Loose normalisation for matching headers with accidental spaces/commas."""
    name = str(name)
    name = name.replace("\ufeff", "")
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" ,\t\n\r")
    return name


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names without changing data."""
    df = df.copy()
    df.columns = [_normalise_column_name(c) for c in df.columns]
    return df


def find_column(df: pd.DataFrame, requested: str) -> Optional[str]:
    """
    Find a column exactly or approximately.

    This helps if CSV headers include leading/trailing spaces, stray commas,
    or inconsistent whitespace.
    """
    requested_norm = _normalise_column_name(requested)

    if requested_norm in df.columns:
        return requested_norm

    lower_map = {str(c).lower(): c for c in df.columns}
    if requested_norm.lower() in lower_map:
        return lower_map[requested_norm.lower()]

    compact_requested = re.sub(r"[\s,]+", "", requested_norm).lower()
    for col in df.columns:
        compact_col = re.sub(r"[\s,]+", "", str(col)).lower()
        if compact_col == compact_requested:
            return col

    return None


def get_row_identity(row: pd.Series) -> dict[str, str]:
    """Collect useful row identifiers for the output."""
    out: dict[str, str] = {}
    for col in ID_COLUMNS:
        out[col] = _clean_cell(row[col]) if col in row.index else ""
    return out


def make_record_id(row: pd.Series, row_index: int) -> str:
    """
    Stable operation-level identifier.

    Prefer surgical identifiers. Falls back to row index if no IDs exist.
    """
    parts = []
    for col in ["EpisodeID", "SurgicalCaseKey", "EncounterKey", "CSN", "MRN"]:
        if col in row.index:
            value = _clean_cell(row[col])
            if value:
                parts.append(f"{col}={value}")
    if parts:
        return " | ".join(parts)
    return f"row_index={row_index}"


def combine_medical_texts_for_row(row: pd.Series, note_sources: list[str] | tuple[str, ...]) -> str:
    """
    Combine note columns for a single surgical episode row.

    This replaces the old MRN-level combine step, which could accidentally mix
    notes from multiple operations belonging to the same patient.
    """
    chunks: list[str] = []

    for source in note_sources:
        col = source if source in row.index else None

        if col is None:
            # Defensive fallback for whitespace/comma variations.
            for candidate in row.index:
                if _normalise_column_name(candidate).lower() == _normalise_column_name(source).lower():
                    col = candidate
                    break

        if col is None:
            continue

        text = _clean_cell(row[col])
        if text:
            chunks.append(f"--- {source} ---\n{text}")

    return "\n\n".join(chunks).strip()


def get_gold_standard_for_row(row: pd.Series, gold_standard_col: str) -> Any:
    """
    Get gold standard value from this row.

    Returns None if the gold column is not present or the value is blank.
    """
    col = gold_standard_col if gold_standard_col in row.index else None

    if col is None:
        # Defensive fallback for whitespace/comma variations.
        requested_norm = _normalise_column_name(gold_standard_col).lower()
        for candidate in row.index:
            if _normalise_column_name(candidate).lower() == requested_norm:
                col = candidate
                break

    if col is None:
        return None

    value = row[col]
    if _is_blank(value):
        return None
    return value


def extract_answer_value(raw_prediction: Any, prediction_key: str) -> Any:
    """
    Convert LLM output into a scalar answer.

    Handles:
      - plain strings: "Yes"
      - JSON strings: {"answer": "Yes"}
      - JSON strings using prediction_key: {"Q14_ETV": "Yes"}
      - dictionaries returned by a client wrapper
    """
    if raw_prediction is None:
        return ""

    if isinstance(raw_prediction, dict):
        if prediction_key in raw_prediction:
            return raw_prediction[prediction_key]
        if "answer" in raw_prediction:
            return raw_prediction["answer"]
        if "response" in raw_prediction:
            return raw_prediction["response"]
        if len(raw_prediction) == 1:
            return next(iter(raw_prediction.values()))
        return json.dumps(raw_prediction, ensure_ascii=False)

    text = str(raw_prediction).strip()

    # Strip markdown code fences if a local model adds them.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Try JSON.
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                if prediction_key in obj:
                    return obj[prediction_key]
                if "answer" in obj:
                    return obj["answer"]
                if "response" in obj:
                    return obj["response"]
                if len(obj) == 1:
                    return next(iter(obj.values()))
        except Exception:
            pass

    return text


def normalise_for_metric(value: Any) -> str:
    """
    Normalise common registry value differences before evaluation.

    This does not over-map clinical categories; it mostly fixes obvious
    formatting/boolean/check-box differences.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    if text == "":
        return ""

    lowered = text.lower().strip()

    yes_values = {"true", "1", "1.0", "yes", "y", "checked", "tick", "t"}
    no_values = {"false", "0", "0.0", "no", "n", "unchecked", "f"}

    if lowered in yes_values:
        return "Yes"
    if lowered in no_values:
        return "No"

    # Common symbol normalisation.
    text = text.replace("≥", ">=")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Keep registry category case mostly intact but make whitespace robust.
    return text


def safe_evaluate_predictions(
    predictions: list[Any],
    gold_standards: list[Any],
    question_name: str,
) -> dict[str, Any]:
    """
    Evaluate predictions using project utils if possible.
    Falls back to simple exact-match metrics if utils.evaluate_predictions fails.
    """
    pred_norm = [normalise_for_metric(x) for x in predictions]
    gold_norm = [normalise_for_metric(x) for x in gold_standards]

    valid_pairs = [
        (p, g)
        for p, g in zip(pred_norm, gold_norm)
        if g not in {"", "Unavailable", "None", "nan"}
    ]

    if not valid_pairs:
        return {
            "question_name": question_name,
            "n": 0,
            "accuracy": None,
            "note": "No available gold standards for this question.",
        }

    try:
        return evaluate_predictions(
            [p for p, _ in valid_pairs],
            [g for _, g in valid_pairs],
            question_name,
        )
    except Exception:
        correct = sum(p == g for p, g in valid_pairs)
        n = len(valid_pairs)
        return {
            "question_name": question_name,
            "n": n,
            "correct": correct,
            "incorrect": n - correct,
            "accuracy": correct / n if n else None,
        }


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def run_question(
    data_merged: pd.DataFrame,
    llm: LLMClient,
    spec: QuestionSpec,
    merged_data_path: str | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Run one question for each surgical episode/operation row.

    Returns a DataFrame containing identifiers, prediction, gold standard, and
    matched/unmatched status.
    """
    if merged_data_path is None:
        from config import MERGED_DATA_PATH
        merged_data_path = MERGED_DATA_PATH

    data_merged = clean_dataframe_columns(data_merged)

    # Check expected note columns early but do not crash if one is missing.
    missing_note_cols = [c for c in spec.note_sources if find_column(data_merged, c) is None]
    if missing_note_cols:
        print(
            f"WARNING: These note source columns were not found for {spec.question_name}: "
            f"{missing_note_cols}",
            flush=True,
        )

    gold_col = find_column(data_merged, spec.gold_standard_col)
    if gold_col is None:
        print(
            f"WARNING: Gold-standard column not found for {spec.question_name}: "
            f"{spec.gold_standard_col!r}. Gold values will be marked unavailable.",
            flush=True,
        )
    else:
        # Use cleaned actual column in case of whitespace differences.
        spec = replace(spec, gold_standard_col=gold_col)

    df_to_process = data_merged.copy()

    # Legacy behaviour: spec.max_mrns now limits rows/records.
    if spec.max_mrns is not None:
        df_to_process = df_to_process.head(spec.max_mrns)

    results: list[dict[str, Any]] = []
    predictions: list[Any] = []
    gold_standards: list[Any] = []
    mrns_for_log: list[str] = []
    csns_for_log: list[str] = []
    episode_ids_for_log: list[str] = []
    surgical_case_keys_for_log: list[str] = []
    encounter_keys_for_log: list[str] = []
    identifiers_for_log: list[str] = []
    record_ids_for_log: list[str] = []

    iterator = tqdm(
        df_to_process.iterrows(),
        total=len(df_to_process),
        desc=spec.question_name,
    )

    for row_index, row in iterator:
        identity = get_row_identity(row)
        record_id = make_record_id(row, row_index)

        mrn = identity.get("MRN", "")
        csn = identity.get("CSN", "")

        gold_standard = get_gold_standard_for_row(row, spec.gold_standard_col)

        try:
            note_text = combine_medical_texts_for_row(row, spec.note_sources)

            if not note_text:
                raw_prediction = ""
                prediction = ""
                error = "No note text available"
            else:
                raw_prediction = extract_with_llm(
                    spec.prompt_file,
                    spec.options or "",
                    note_text,
                    llm,
                    **spec.llm_kwargs,
                )
                prediction = extract_answer_value(raw_prediction, spec.prediction_key)
                error = ""

            pred_norm = normalise_for_metric(prediction)
            gold_norm = normalise_for_metric(gold_standard)
            match = "" if gold_norm == "" else pred_norm == gold_norm

            row_result = {
                "Record_ID": record_id,
                **identity,
                "Question": spec.question_name,
                "Prediction_Key": spec.prediction_key,
                "Prediction": prediction,
                "Raw_Prediction": raw_prediction,
                spec.prediction_key: prediction,
                "Prediction_Normalised": pred_norm,
                "Gold_Standard": gold_standard if gold_standard is not None else "Unavailable",
                "Gold_Standard_Normalised": gold_norm,
                "Match": match,
                "Error": error,
            }
            results.append(row_result)

            predictions.append(prediction)
            gold_standards.append(gold_standard)
            mrns_for_log.append(mrn)
            csns_for_log.append(csn)
            episode_ids_for_log.append(identity.get("EpisodeID", ""))
            surgical_case_keys_for_log.append(identity.get("SurgicalCaseKey", ""))
            encounter_keys_for_log.append(identity.get("EncounterKey", ""))
            identifiers_for_log.append(identity.get("Identifier", ""))
            record_ids_for_log.append(record_id)

            print(f"{record_id} -> {prediction}")

        except Exception as e:
            err = f"ERROR: {str(e)}"
            row_result = {
                "Record_ID": record_id,
                **identity,
                "Question": spec.question_name,
                "Prediction_Key": spec.prediction_key,
                "Prediction": err,
                "Raw_Prediction": "",
                spec.prediction_key: err,
                "Prediction_Normalised": err,
                "Gold_Standard": gold_standard if gold_standard is not None else "Unavailable",
                "Gold_Standard_Normalised": normalise_for_metric(gold_standard),
                "Match": False if gold_standard is not None else "",
                "Error": err,
            }
            results.append(row_result)

            predictions.append(err)
            gold_standards.append(gold_standard)
            mrns_for_log.append(mrn)
            csns_for_log.append(csn)
            episode_ids_for_log.append(identity.get("EpisodeID", ""))
            surgical_case_keys_for_log.append(identity.get("SurgicalCaseKey", ""))
            encounter_keys_for_log.append(identity.get("EncounterKey", ""))
            identifiers_for_log.append(identity.get("Identifier", ""))
            record_ids_for_log.append(record_id)

            print(f"Error on {record_id}: {e}")

    df_results = pd.DataFrame(results)
    print(f"\nProcessed {len(df_results)} operation record(s) for {spec.question_name}")

    # Append to existing project-wide results CSV.
    # This preserves your old logging pipeline, but uses operation-level repeated
    # MRN/CSN values where available.
    try:
        try:
            rows_logged = append_results_to_csv(
                question_name=spec.question_name,
                predictions=predictions,
                gold_standards=gold_standards,
                mrns=mrns_for_log,
                csns=csns_for_log,
                llm=llm,
                merged_data_path=merged_data_path,
                episode_ids=episode_ids_for_log,
                surgical_case_keys=surgical_case_keys_for_log,
                encounter_keys=encounter_keys_for_log,
                identifiers=identifiers_for_log,
                record_ids=record_ids_for_log,
            )
        except TypeError:
            # Backwards compatibility with older utils.py versions.
            rows_logged = append_results_to_csv(
                question_name=spec.question_name,
                predictions=predictions,
                gold_standards=gold_standards,
                mrns=mrns_for_log,
                csns=csns_for_log,
                llm=llm,
                merged_data_path=merged_data_path,
            )
    except Exception as e:
        rows_logged = 0
        print(f"WARNING: Could not append to central results CSV: {e}", flush=True)

    metrics = safe_evaluate_predictions(predictions, gold_standards, spec.question_name)
    try:
        print_evaluation_summary(metrics, spec.question_name)
    except Exception:
        print(f"Evaluation summary for {spec.question_name}: {metrics}", flush=True)

    results_path = Path(RESULTS_DATA_PATH).resolve()
    if rows_logged:
        print(f"Results file: appended {rows_logged} row(s) to {results_path}", flush=True)
    else:
        print(f"Results file: nothing appended to central file. Path: {results_path}", flush=True)

    # Per-question output file.
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", spec.prediction_key).strip("_")
        out_csv = output_dir / f"{safe_key}_predictions.csv"
        df_results.to_csv(out_csv, index=False)
        print(f"Per-question output written to: {out_csv}", flush=True)

    return df_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LLM extraction for one or more UK Shunt Registry questions.",
        epilog="Questions available: see QUESTION_REGISTRY in questions.py",
    )
    parser.add_argument(
        "questions",
        nargs="+",
        metavar="QUESTION",
        help="Question key(s) to run, e.g. q13 q14 q24, or 'all'.",
    )
    parser.add_argument(
        "--input",
        "--data-path",
        dest="data_path",
        default=None,
        help=(
            "Path to the input/merged CSV. Defaults to MERGED_DATA_PATH from config. "
            "Can be InputData if you only want predictions, or a merged file if you want live metrics."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        metavar="N",
        help="Limit operation rows per question. 0 = all rows. Default = all rows.",
    )
    parser.add_argument(
        "--max-mrns",
        type=int,
        default=None,
        metavar="N",
        help="Legacy alias for --max-records. In this fixed version it limits rows, not unique MRNs.",
    )
    parser.add_argument(
        "--out-dir",
        default="QUESTION_RUNNER_OUTPUT",
        help="Folder for per-question prediction CSVs.",
    )
    return parser


def main() -> None:
    from config import MERGED_DATA_PATH
    from questions import QUESTION_REGISTRY

    args = _build_parser().parse_args()

    # Resolve question keys.
    if args.questions == ["all"]:
        keys = list(QUESTION_REGISTRY.keys())
    else:
        keys = args.questions
        unknown = [k for k in keys if k not in QUESTION_REGISTRY]
        if unknown:
            raise SystemExit(
                f"Unknown question(s): {unknown}\n"
                f"Available: {list(QUESTION_REGISTRY.keys())}"
            )

    data_path = args.data_path or MERGED_DATA_PATH
    data_path = str(data_path)

    if data_path.lower().endswith((".xlsx", ".xls")):
        data_merged = pd.read_excel(data_path)
    else:
        data_merged = pd.read_csv(data_path, dtype=object, keep_default_na=False)

    data_merged = clean_dataframe_columns(data_merged)

    llm = create_llm_client_from_config()

    # Prefer --max-records; fall back to legacy --max-mrns.
    max_records = args.max_records
    if max_records is None and args.max_mrns is not None:
        max_records = args.max_mrns

    if max_records == 0:
        max_records = None

    all_outputs: list[pd.DataFrame] = []

    for key in keys:
        spec = QUESTION_REGISTRY[key]

        if max_records is not None:
            spec = replace(spec, max_mrns=max_records)

        print(f"\n{'=' * 70}")
        print(f"Running: {spec.question_name}")
        print(f"{'=' * 70}")

        df_results = run_question(
            data_merged=data_merged,
            llm=llm,
            spec=spec,
            merged_data_path=data_path,
            output_dir=args.out_dir,
        )
        df_results.insert(0, "Question_Key", key)
        all_outputs.append(df_results)

    if all_outputs:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(all_outputs, ignore_index=True)
        combined_path = out_dir / "all_predictions_long.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined output written to: {combined_path}", flush=True)


if __name__ == "__main__":
    main()
