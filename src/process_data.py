"""
Pre-process InputData / combined registry CSV into the merged dataset used by the
LLM extraction pipeline.

Fixed for the UK National Shunt Registry episode-level pipeline.

Key fixes:
  - keeps exact registry headings, e.g. "ETV | Shunt Operation"
  - does NOT rename fields to short names such as "ETV"
  - preserves operation-level identifiers: EpisodeID, SurgicalCaseKey, EncounterKey, CSN
  - preserves all note columns used for prompts, including Implant_note
  - supports CSV and Excel input
  - strips "(GOLD)" suffixes if your combined file has gold-standard columns named that way
  - keeps implant fields 1-10 for validation/export
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from config import INPUT_DATA_PATH, MERGED_DATA_PATH


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def clean_col(name) -> str:
    """Normalise accidental spaces, commas, and BOMs in column names."""
    name = str(name).replace("\ufeff", "")
    name = re.sub(r"\s+", " ", name)
    return name.strip(" ,\t\n\r")


def strip_gold_suffix(name: str) -> str:
    """
    Convert:
        "ETV | Shunt Operation(GOLD)"
    to:
        "ETV | Shunt Operation"

    Leaves normal columns unchanged.
    """
    name = clean_col(name)
    name = re.sub(r"\s*\(GOLD\)\s*$", "", name, flags=re.IGNORECASE)
    return clean_col(name)


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with cleaned and registry-compatible column names."""
    df = df.copy()
    df.columns = [strip_gold_suffix(c) for c in df.columns]

    # If stripping (GOLD) creates duplicate columns, combine first non-null value.
    if df.columns.duplicated().any():
        new_df = pd.DataFrame(index=df.index)
        for col in dict.fromkeys(df.columns):
            same = df.loc[:, df.columns == col]
            if same.shape[1] == 1:
                new_df[col] = same.iloc[:, 0]
            else:
                new_df[col] = same.bfill(axis=1).iloc[:, 0]
        df = new_df

    return df


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or Excel."""
    path = Path(path)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    # Try utf-8 first, then latin-1 for exported NHS/registry CSVs.
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


# ---------------------------------------------------------------------------
# Columns to preserve
# ---------------------------------------------------------------------------

ID_COLS = [
    "EpisodeID",
    "SurgicalCaseKey",
    "EncounterKey",
    "CSN",
    "Surname",
    "Midname",
    "Forename",
    "Identifier",
    "MRN",
    "Gender",
    "DOB",
    "DOD",
    "Postcode",
    "GP Surgery",
]

NOTE_COLS = [
    "Clerking",
    "Op Note",
    "Discharge Summary",
    "Imaging Report",
    "MDT Outcome Pre Proc Date",
    "MDT Outcome Pre Proc",
    "MDT Outcome Post Proc Date",
    "MDT Outcome Post Proc",
    "ImplantName",
    "ManufacturerFull",
    "Implant_note",
]

SHUNT_OPERATION_COLS = [
    "Primary reason for shunting | Shunt Operation",
    "EVD insertion in the last 30 days | Shunt Operation",
    "Procedure type | Shunt Operation",
    "Primary reason for revision | Shunt Operation",
    "Replacement with EVD | Shunt Operation",
    "ETV | Shunt Operation",
    "Choroid plexectomy | Shunt Operation",
    "Subtemporal decompression | Shunt Operation",
    "Ventricular size prior to surgery | Shunt Operation",
    "Concurrent chemoradiotherapy for primary CNS tumour | Shunt Operation",
    "Co-existing CNS infection | Shunt Operation",
    "CNS infection in the last 6 months | Shunt Operation",
    "Knife to skin | Shunt Operation",
    "Final suture | Shunt Operation",
    "Primary Surgeon | Shunt Operation",
    "Grade of primary surgeon | Shunt Operation",
    "Consultant presence | Shunt Operation",
    "Responsible consultant | Shunt Operation",
    "Total number of surgeons | Shunt Operation",
    "Other surgeons | Shunt Operation",
    "Include note | Shunt Operation",
    "Operation title | Shunt Operation",
    "Anaesthetist | Shunt Operation",
    "Procedure | Shunt Operation",
    "Post-op plan | Shunt Operation",
    "New implants inserted | Shunt Operation",
    "Old implants removed | Shunt Operation",
    "Other implants removed | Shunt Operation",
    "Other implants removed (not listed above) | Shunt Operation",
    "Procedure1(Code:Description) | Shunt Operation",
]


def implant_cols(max_implants: int = 10) -> list[str]:
    """Build implant field names for Implant 1-N."""
    base_fields = [
        "Implant type",
        "Manufacturer",
        "Implant (Catalogue number - Name)",
        "Catheter type",
        "Drainage site",
        "Insertion site",
        "Image guided placement",
        "Reservoir type",
        "Serial number",
        "Programmable",
        "Initial setting",
        "UDI Code",
        "Other implant information",
        "Present",
    ]

    cols: list[str] = []
    for i in range(1, max_implants + 1):
        for field in base_fields:
            cols.append(f"{field} | Implant | {i}")
    return cols


EXPECTED_COLS = ID_COLS + SHUNT_OPERATION_COLS + implant_cols(10) + NOTE_COLS


# ---------------------------------------------------------------------------
# Build merged dataset
# ---------------------------------------------------------------------------

def build_merged_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Keep exact registry headings and note columns for the LLM pipeline.

    This function deliberately avoids renaming:
        "ETV | Shunt Operation" -> "ETV"

    because question_registry.py now expects the exact registry headings.
    """
    data = clean_dataframe_columns(data)

    # Harmonise the two possible Q36 variants.
    verbose_q36 = "Other implants removed (not listed above) | Shunt Operation"
    short_q36 = "Other implants removed | Shunt Operation"

    if verbose_q36 in data.columns and short_q36 not in data.columns:
        data = data.rename(columns={verbose_q36: short_q36})

    # Preserve only columns that exist.
    cols_to_keep = []
    seen = set()
    for col in EXPECTED_COLS:
        col = short_q36 if col == verbose_q36 else col
        if col in data.columns and col not in seen:
            cols_to_keep.append(col)
            seen.add(col)

    missing = []
    for col in EXPECTED_COLS:
        col = short_q36 if col == verbose_q36 else col
        if col not in data.columns and col not in missing:
            missing.append(col)

    data_merged = data[cols_to_keep].copy()

    print("Merged dataset created:")
    print(f"  - Total records: {len(data_merged)}")
    print(f"  - Columns retained: {len(data_merged.columns)}")

    id_present = [c for c in ID_COLS if c in data_merged.columns]
    note_present = [c for c in NOTE_COLS if c in data_merged.columns]
    shunt_present = [c for c in SHUNT_OPERATION_COLS if c in data_merged.columns]
    implant_present = [c for c in implant_cols(10) if c in data_merged.columns]

    print(f"  - Identifier columns retained: {len(id_present)}")
    print(f"  - Note columns retained: {len(note_present)}")
    print(f"  - Shunt operation columns retained: {len(shunt_present)}")
    print(f"  - Implant columns retained: {len(implant_present)}")

    important_missing = [
        c for c in [
            "EpisodeID",
            "SurgicalCaseKey",
            "EncounterKey",
            "CSN",
            "MRN",
            "Op Note",
            "ETV | Shunt Operation",
            "Grade of primary surgeon | Shunt Operation",
            "Replacement with EVD | Shunt Operation",
        ]
        if c not in data_merged.columns
    ]

    if important_missing:
        print("\nWARNING: important expected columns were not found:")
        for c in important_missing:
            print(f"  - {c}")

    if missing:
        print("\nOther expected columns not found, this may be fine depending on your file:")
        for c in missing[:60]:
            print(f"  - {c}")
        if len(missing) > 60:
            print(f"  ... and {len(missing) - 60} more")

    # Helpful counts for key gold/target fields.
    print("\nNon-empty counts for key registry fields:")
    for col in [
        "Primary reason for shunting | Shunt Operation",
        "EVD insertion in the last 30 days | Shunt Operation",
        "Procedure type | Shunt Operation",
        "Primary reason for revision | Shunt Operation",
        "Replacement with EVD | Shunt Operation",
        "ETV | Shunt Operation",
        "Choroid plexectomy | Shunt Operation",
        "Subtemporal decompression | Shunt Operation",
        "Ventricular size prior to surgery | Shunt Operation",
        "Concurrent chemoradiotherapy for primary CNS tumour | Shunt Operation",
        "Co-existing CNS infection | Shunt Operation",
        "CNS infection in the last 6 months | Shunt Operation",
        "Grade of primary surgeon | Shunt Operation",
        "Consultant presence | Shunt Operation",
        "New implants inserted | Shunt Operation",
        "Old implants removed | Shunt Operation",
    ]:
        if col in data_merged.columns:
            print(f"  - {col}: {data_merged[col].notna().sum()}")

    return data_merged


def load_data(path: str | Path = INPUT_DATA_PATH) -> pd.DataFrame:
    """Load InputData or combined notes/gold file."""
    return read_table(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-process UK Shunt Registry InputData into merged dataset for LLM extraction."
    )
    parser.add_argument(
        "--input",
        default=INPUT_DATA_PATH,
        help="InputData/combined CSV or Excel path. Defaults to INPUT_DATA_PATH from config.",
    )
    parser.add_argument(
        "--output",
        default=MERGED_DATA_PATH,
        help="Output merged CSV path. Defaults to MERGED_DATA_PATH from config.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    data = load_data(args.input)
    data_merged = build_merged_dataset(data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_merged.to_csv(output_path, index=False)

    print(f"\nSaved merged dataset to: {output_path}")


if __name__ == "__main__":
    main()
