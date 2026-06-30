"""
Shared helpers for the UK National Shunt Registry extraction pipeline.

Fixed version:
  - keeps backwards-compatible MRN-level helpers
  - adds operation/episode row-level helpers
  - normalises registry values more robustly for validation
  - handles exact registry headings with "| Shunt Operation"
  - logs operation identifiers as well as MRN/CSN
  - supports structured JSON outputs from both OpenAI-style and Ollama-style clients

Sections:
  - Prompt + LLM helpers
  - Column/header helpers
  - Notes helpers
  - Gold standard + metrics helpers
  - Results logging helpers
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from config import RESULTS_DATA_PATH
from llm_client import LLMClient


# ---------------------------------------------------------------------------
# Prompt + LLM helpers
# ---------------------------------------------------------------------------

def load_prompt(
    prompt_file: str,
    options_text: str | None,
    note_text: str,
    max_length: int = 20000,
) -> str:
    """
    Load a prompt template from prompts/ and fill in {options} and {note_text}.

    max_length is deliberately higher than the old 4000 because operation notes
    plus implant notes can be long. If needed, override at call site.
    """
    prompt_path = Path("prompts") / prompt_file
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    options_text = options_text or ""
    note_text = note_text or ""

    truncated_note = note_text[:max_length] if len(note_text) > max_length else note_text

    try:
        return prompt_template.format(
            options=options_text,
            note_text=truncated_note,
        )
    except KeyError as e:
        raise KeyError(
            f"Prompt formatting failed for {prompt_file!r}. "
            f"This usually means the prompt contains literal JSON braces that "
            f"need escaping as '{{{{' and '}}}}'. Missing placeholder/key: {e}"
        ) from e


def extract_with_llm(
    prompt_file: str,
    options_text: str | None,
    note_text: str,
    llm: LLMClient,
    **kwargs,
) -> str:
    """
    Load prompt template and run chat completion via the given LLM client.

    Extra **kwargs are forwarded to llm.generate_chat.
    """
    prompt_content = load_prompt(prompt_file, options_text, note_text)
    messages = [{"role": "user", "content": prompt_content}]
    raw = llm.generate_chat(messages, **kwargs)
    return unwrap_structured_answer(raw)


def options_to_enum_schema(options_text: str | None) -> dict:
    """
    Parse '- Label' lines from an options block into a JSON Schema enum.

    Returns:
        {
          "type": "object",
          "properties": {"answer": {"type": "string", "enum": labels}},
          "required": ["answer"]
        }

    Safe if options_text is None/blank.
    """
    options_text = options_text or ""

    labels = [
        m.group(1).strip()
        for m in re.finditer(r"^- (.+)$", options_text, re.MULTILINE)
        if m.group(1).strip()
    ]

    # Preserve old behaviour, but avoid duplicate UNKNOWN.
    if "UNKNOWN" in options_text and "UNKNOWN" not in labels:
        labels.append("UNKNOWN")

    # Fallback for accidental comma-separated options.
    if not labels and "," in options_text:
        labels = [x.strip() for x in options_text.split(",") if x.strip()]

    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": labels},
        },
        "required": ["answer"],
    }


def free_text_answer_schema() -> dict:
    """JSON Schema for a single free-text answer."""
    return {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }


def unwrap_structured_answer(raw: Any) -> str:
    """
    Pull answer out of structured JSON; otherwise return stripped text.

    Handles:
      - {"answer": "..."}
      - {"response": "..."}
      - {"Q14_ETV": "..."} when only one key exists
      - ```json fenced blocks
    """
    if raw is None:
        return ""

    if isinstance(raw, dict):
        if "answer" in raw and raw["answer"] is not None:
            return str(raw["answer"]).strip()
        if "response" in raw and raw["response"] is not None:
            return str(raw["response"]).strip()
        if len(raw) == 1:
            value = next(iter(raw.values()))
            return "" if value is None else str(value).strip()
        return json.dumps(raw, ensure_ascii=False)

    s = str(raw).strip()
    if not s:
        return s

    m = re.match(
        r"^```(?:json)?\s*\r?\n?(.*?)\r?\n?```\s*$",
        s,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        s = m.group(1).strip()

    if not s.lstrip().startswith("{"):
        return s

    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return s

    if isinstance(data, dict):
        if "answer" in data and data["answer"] is not None:
            return str(data["answer"]).strip()
        if "response" in data and data["response"] is not None:
            return str(data["response"]).strip()
        if len(data) == 1:
            value = next(iter(data.values()))
            return "" if value is None else str(value).strip()

    return s


# ---------------------------------------------------------------------------
# Column/header helpers
# ---------------------------------------------------------------------------

def is_blank(value: Any) -> bool:
    """True for None/NaN/empty string."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip() == ""


def clean_cell(value: Any) -> str:
    """Return a safe stripped string."""
    if is_blank(value):
        return ""
    return str(value).strip()


def normalise_column_name(name: Any) -> str:
    """Clean accidental whitespace, commas, and BOM from headers."""
    name = str(name).replace("\ufeff", "")
    name = re.sub(r"\s+", " ", name)
    return name.strip(" ,\t\n\r")


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with cleaned column names."""
    df = df.copy()
    df.columns = [normalise_column_name(c) for c in df.columns]
    return df


def find_column(df: pd.DataFrame, requested: str) -> Optional[str]:
    """
    Find a column exactly or approximately.

    Useful for registry headings such as:
        Grade of primary surgeon | Shunt Operation
    """
    requested_norm = normalise_column_name(requested)

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


# ---------------------------------------------------------------------------
# Notes helpers
# ---------------------------------------------------------------------------

def combine_medical_texts_for_row(
    row: pd.Series,
    sources: Iterable[str] = ("Discharge Summary", "Op Note", "Clerking"),
) -> str:
    """
    Combine notes for ONE operation/episode row.

    Use this for the Shunt Registry pipeline. It prevents accidental mixing of
    notes from several operations belonging to the same MRN.
    """
    combined_parts: list[str] = []

    for source in sources:
        col = source if source in row.index else None

        if col is None:
            source_norm = normalise_column_name(source).lower()
            for candidate in row.index:
                if normalise_column_name(candidate).lower() == source_norm:
                    col = candidate
                    break

        if col is None:
            continue

        text = clean_cell(row[col])
        if text:
            combined_parts.append(f"--- {source} ---\n{text}")

    return "\n\n".join(combined_parts).strip()


def combine_medical_texts(
    data: pd.DataFrame,
    mrn: Any,
    sources: Iterable[str] = ("Discharge Summary", "Op Note", "Clerking"),
):
    """
    Legacy MRN-level helper.

    Kept for backwards compatibility only. For registry extraction, prefer
    combine_medical_texts_for_row(row, sources) because MRN-level extraction can
    mix separate shunt operations from the same patient.
    """
    if "MRN" not in data.columns:
        return ""

    mrn_data = data[data["MRN"] == mrn]

    if mrn_data.empty:
        return ""

    row = mrn_data.iloc[0]
    return combine_medical_texts_for_row(row, sources)


# ---------------------------------------------------------------------------
# Gold standard + evaluation helpers
# ---------------------------------------------------------------------------

def normalize_text(text: Any) -> str:
    """
    Normalise values for comparison.

    Handles:
      - True/False, 1/0 -> Yes/No
      - whitespace/case
      - ≥ vs >=
      - hyphen variants
    """
    if is_blank(text):
        return ""

    s = str(text).strip()
    low = s.lower().strip()

    yes_values = {"true", "1", "1.0", "yes", "y", "checked", "tick", "t"}
    no_values = {"false", "0", "0.0", "no", "n", "unchecked", "f"}

    if low in yes_values:
        return "yes"
    if low in no_values:
        return "no"

    s = s.replace("≥", ">=")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)

    return s.lower().strip()


def get_gold_standard_for_row(row: pd.Series, column_name: str) -> Any:
    """Get gold standard value for one operation/episode row."""
    col = column_name if column_name in row.index else None

    if col is None:
        target = normalise_column_name(column_name).lower()
        for candidate in row.index:
            if normalise_column_name(candidate).lower() == target:
                col = candidate
                break

    if col is None:
        return None

    value = row[col]
    return None if is_blank(value) else value


def get_gold_standard(data_merged: pd.DataFrame, mrn: Any, column_name: str):
    """
    Legacy MRN-level gold helper.

    Kept for backwards compatibility only. For registry extraction, prefer
    get_gold_standard_for_row(row, column_name).
    """
    if "MRN" not in data_merged.columns:
        return None

    col = find_column(data_merged, column_name)
    if col is None:
        return None

    mrn_data = data_merged[data_merged["MRN"] == mrn]
    if mrn_data.empty:
        return None

    value = mrn_data[col].iloc[0]
    return None if is_blank(value) else value


def _prf_for_class(preds_norm, golds_norm, cls):
    """Precision, recall, F1 for one class, one-vs-rest."""
    tp = sum(1 for p, g in zip(preds_norm, golds_norm) if p == cls and g == cls)
    fp = sum(1 for p, g in zip(preds_norm, golds_norm) if p == cls and g != cls)
    fn = sum(1 for p, g in zip(preds_norm, golds_norm) if p != cls and g == cls)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1_score


def evaluate_predictions(predictions, gold_standards, question_name):
    """
    Evaluate predictions against gold standards using exact normalised matching.

    Returns macro precision/recall/F1 across observed labels.
    """
    eval_data = [
        (p, g)
        for p, g in zip(predictions, gold_standards)
        if g is not None and normalize_text(g) != ""
    ]

    if not eval_data:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "total": len(predictions),
            "with_gold_standard": 0,
            "correct": [],
            "incorrect": [],
            "class_metrics": {},
        }

    preds, golds = zip(*eval_data)

    preds_normalized = [normalize_text(p) for p in preds]
    golds_normalized = [normalize_text(g) for g in golds]

    correct = sum(1 for p, g in zip(preds_normalized, golds_normalized) if p == g)
    total = len(preds_normalized)
    accuracy = correct / total if total > 0 else 0.0

    all_classes = sorted(set(preds_normalized + golds_normalized))

    class_metrics: dict[str, dict] = {}
    for cls in all_classes:
        prec, rec, f1_score = _prf_for_class(preds_normalized, golds_normalized, cls)
        support = sum(1 for g in golds_normalized if g == cls)
        predicted = sum(1 for p in preds_normalized if p == cls)
        class_metrics[cls] = {
            "precision": prec,
            "recall": rec,
            "f1": f1_score,
            "support": support,
            "predicted": predicted,
        }

    n_cls = len(class_metrics)
    precision = sum(m["precision"] for m in class_metrics.values()) / n_cls if n_cls else 0.0
    recall = sum(m["recall"] for m in class_metrics.values()) / n_cls if n_cls else 0.0
    f1 = sum(m["f1"] for m in class_metrics.values()) / n_cls if n_cls else 0.0

    correct_examples = []
    incorrect_examples = []

    for pred, gold, pred_norm, gold_norm in zip(
        preds, golds, preds_normalized, golds_normalized
    ):
        if pred_norm == gold_norm:
            correct_examples.append((pred, gold))
        else:
            incorrect_examples.append((pred, gold))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": len(predictions),
        "with_gold_standard": total,
        "correct": correct_examples[:5],
        "incorrect": incorrect_examples[:5],
        "class_metrics": class_metrics,
    }


def print_evaluation_summary(metrics, question_name):
    """Print a formatted evaluation summary."""
    print(f"\n{'=' * 60}")
    print(f"Evaluation Results for {question_name}")
    print(f"{'=' * 60}")

    if metrics["with_gold_standard"] == 0:
        print("⚠️  No gold standard available for evaluation")
        print(f"   Total records processed: {metrics['total']}")
        return

    print(f"Total records: {metrics['total']}")
    print(f"Records with gold standard: {metrics['with_gold_standard']}")
    print(f"Records without gold standard: {metrics['total'] - metrics['with_gold_standard']}")

    print("\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")

    if metrics.get("class_metrics"):
        print("\nPer-class:")
        for cls, vals in metrics["class_metrics"].items():
            print(
                f"  {cls}: "
                f"P={vals['precision']:.3f}, "
                f"R={vals['recall']:.3f}, "
                f"F1={vals['f1']:.3f}, "
                f"support={vals['support']}, "
                f"predicted={vals['predicted']}"
            )

    if metrics["correct"]:
        print("\n✓ Correct Examples, up to 5:")
        for i, (pred, gold) in enumerate(metrics["correct"], 1):
            print(f"  {i}. Predicted: {pred!r} | Gold: {gold!r}")

    if metrics["incorrect"]:
        print("\n✗ Incorrect Examples, up to 5:")
        for i, (pred, gold) in enumerate(metrics["incorrect"], 1):
            print(f"  {i}. Predicted: {pred!r} | Gold: {gold!r}")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Results logging helpers
# ---------------------------------------------------------------------------

def append_results_to_csv(
    question_name,
    predictions,
    gold_standards,
    mrns,
    csns,
    llm: LLMClient,
    merged_data_path: str,
    episode_ids: Optional[Iterable[Any]] = None,
    surgical_case_keys: Optional[Iterable[Any]] = None,
    encounter_keys: Optional[Iterable[Any]] = None,
    identifiers: Optional[Iterable[Any]] = None,
    record_ids: Optional[Iterable[Any]] = None,
) -> int:
    """
    Append extraction results for a question to the central CSV.

    Backwards compatible with the old signature, but can now store operation
    identifiers so multiple operations under the same MRN are distinguishable.
    """
    run_ts = datetime.now(timezone.utc).isoformat()

    n = len(predictions)

    def _list_or_blank(values):
        if values is None:
            return [""] * n
        values = list(values)
        if len(values) != n:
            return values[:n] + [""] * max(0, n - len(values))
        return values

    episode_ids = _list_or_blank(episode_ids)
    surgical_case_keys = _list_or_blank(surgical_case_keys)
    encounter_keys = _list_or_blank(encounter_keys)
    identifiers = _list_or_blank(identifiers)
    record_ids = _list_or_blank(record_ids)

    rows = []
    for i, (mrn, csn, pred, gold) in enumerate(
        zip(mrns, csns, predictions, gold_standards)
    ):
        has_gold = gold is not None and normalize_text(gold) != ""
        is_correct = has_gold and normalize_text(pred) == normalize_text(gold)

        rows.append(
            {
                "Run_Timestamp": run_ts,
                "Record_ID": clean_cell(record_ids[i]),
                "EpisodeID": clean_cell(episode_ids[i]),
                "SurgicalCaseKey": clean_cell(surgical_case_keys[i]),
                "EncounterKey": clean_cell(encounter_keys[i]),
                "MRN": clean_cell(mrn),
                "CSN": clean_cell(csn),
                "Identifier": clean_cell(identifiers[i]),
                "Question": question_name,
                "Provider": getattr(llm, "provider", ""),
                "Model": getattr(llm, "model_id", ""),
                "Prediction": pred,
                "Has_Gold": has_gold,
                "Gold_Standard": "" if gold is None else gold,
                "Correct": "Yes" if is_correct else "No",
                "Merged_Data_Path": merged_data_path,
            }
        )

    df_append = pd.DataFrame(rows)

    write_header = not os.path.exists(RESULTS_DATA_PATH)
    df_append.to_csv(RESULTS_DATA_PATH, mode="a", header=write_header, index=False)
    return len(rows)
