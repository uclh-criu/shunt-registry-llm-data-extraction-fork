"""
Central registry of all QuestionSpec definitions for the UK National Shunt Registry.

This file uses the TRUE registry field numbers.

To add a new question:
  1. Add its options string to registry_options.py.
  2. Add a prompt file to prompts/.
  3. Add a QuestionSpec entry to QUESTION_REGISTRY below.

Not registered here:
- Fields 1-7: mapped demographic fields.
- Field 8: GP Surgery is an NLP/option-set field but is not currently prompt-based here.
- Fields 21-23: mapped theatre/surgeon fields.
- Fields 26-28: mapped consultant/surgeon-count fields.
- Field 29 "Include note": in analysis.ipynb this is a simple check that Op Note text exists (Yes/No), not prompt-based extraction.
- Fields 30-33: ignored note text fields.
"""

from question_runner import QuestionSpec
from registry_options import (
    q9_options,
    q10_options,
    q11_options,
    q12_options,
    q13_options,
    q14_options,
    q15_options,
    q16_options,
    q17_options,
    q18_options,
    q19_options,
    q20_options,
    q24_options,
    q25_options,
    new_implants_inserted_options,
    old_implants_removed_options,
    implant_type_options,
    manufacturer_options,
    catalogue_number_name_options,
    catheter_type_options,
    drainage_site_options,
    insertion_site_options,
    image_guided_placement_options,
    reservoir_type_options,
    programmable_options,
    present_status_options,
)
from utils import options_to_enum_schema


_q9_schema = options_to_enum_schema(q9_options)
_q10_schema = options_to_enum_schema(q10_options)
_q11_schema = options_to_enum_schema(q11_options)
_q12_schema = options_to_enum_schema(q12_options)
_q13_schema = options_to_enum_schema(q13_options)
_q14_schema = options_to_enum_schema(q14_options)
_q15_schema = options_to_enum_schema(q15_options)
_q16_schema = options_to_enum_schema(q16_options)
_q17_schema = options_to_enum_schema(q17_options)
_q18_schema = options_to_enum_schema(q18_options)
_q19_schema = options_to_enum_schema(q19_options)
_q20_schema = options_to_enum_schema(q20_options)
_q24_schema = options_to_enum_schema(q24_options)
_q25_schema = options_to_enum_schema(q25_options)


def _labels(options_text: str) -> list[str]:
    """Extract '- option' labels from registry_options.py blocks."""
    return [
        line.strip()[2:].strip()
        for line in options_text.splitlines()
        if line.strip().startswith("- ")
    ]


_q34_q50_implants_schema = {
    "type": "object",
    "properties": {
        "new_implants_inserted": {
            "type": "string",
            "enum": _labels(new_implants_inserted_options),
        },
        "old_implants_removed": {
            "type": "string",
            "enum": _labels(old_implants_removed_options),
        },
        "other_implants_removed": {"type": "string"},
        "implants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "implant_type": {
                        "type": "string",
                        "enum": _labels(implant_type_options) + [""],
                    },
                    "manufacturer": {
                        "type": "string",
                        "enum": _labels(manufacturer_options) + [""],
                    },
                    # Do not enum-constrain this because the catalogue list is very large
                    # and matching is better handled in post-processing/validation.
                    "catalogue_number_name": {"type": "string"},
                    "catheter_type": {
                        "type": "string",
                        "enum": _labels(catheter_type_options) + [""],
                    },
                    "drainage_site": {
                        "type": "string",
                        "enum": _labels(drainage_site_options) + [""],
                    },
                    "insertion_site": {
                        "type": "string",
                        "enum": _labels(insertion_site_options) + [""],
                    },
                    "image_guided_placement": {
                        "type": "string",
                        "enum": _labels(image_guided_placement_options) + [""],
                    },
                    "reservoir_type": {
                        "type": "string",
                        "enum": _labels(reservoir_type_options) + [""],
                    },
                    "serial_number": {"type": "string"},
                    "programmable": {
                        "type": "string",
                        "enum": _labels(programmable_options) + [""],
                    },
                    "initial_setting": {"type": "string"},
                    "udi_code": {"type": "string"},
                    "other_implant_information": {"type": "string"},
                    "present_status": {
                        "type": "string",
                        "enum": _labels(present_status_options),
                    },
                },
                "required": [
                    "implant_type",
                    "manufacturer",
                    "catalogue_number_name",
                    "catheter_type",
                    "drainage_site",
                    "insertion_site",
                    "image_guided_placement",
                    "reservoir_type",
                    "serial_number",
                    "programmable",
                    "initial_setting",
                    "udi_code",
                    "other_implant_information",
                    "present_status",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "new_implants_inserted",
        "old_implants_removed",
        "other_implants_removed",
        "implants",
    ],
    "additionalProperties": False,
}



ALL_NOTES_SOURCES = (
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
)

OP_NOTE_ONLY = ("Op Note",)

OP_NOTE_AND_IMPLANT_NOTE = ("Op Note", "Implant_note")


# ---------------------------------------------------------------------------
# Question specs — one entry per prompt-based registry field
# ---------------------------------------------------------------------------

QUESTION_REGISTRY: dict[str, QuestionSpec] = {
    # -----------------------------------------------------------------------
    # Shunt Operation NLP fields
    # -----------------------------------------------------------------------
    "q9": QuestionSpec(
        question_name="Q9 - Primary reason for shunting",
        gold_standard_col="Primary reason for shunting | Shunt Operation",
        prompt_file="q9_prompt.txt",
        options=q9_options,
        prediction_key="Q9_Primary_Reason_Shunting",
        note_sources=ALL_NOTES_SOURCES,
        llm_kwargs={
            "format": _q9_schema,
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),
    "q10": QuestionSpec(
        question_name="Q10 - EVD insertion in the last 30 days",
        gold_standard_col="EVD insertion in the last 30 days | Shunt Operation",
        prompt_file="q10_prompt.txt",
        options=q10_options,
        prediction_key="Q10_EVD_Insertion_Last_30_Days",
        note_sources=ALL_NOTES_SOURCES,
        llm_kwargs={
            "format": _q10_schema,
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),
    "q11": QuestionSpec(
        question_name="Q11 - Procedure type",
        gold_standard_col="Procedure type | Shunt Operation",
        prompt_file="q11_prompt.txt",
        options=q11_options,
        prediction_key="Q11_Procedure_Type",
        note_sources=ALL_NOTES_SOURCES,
        llm_kwargs={
            "format": _q11_schema,
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),
    "q12": QuestionSpec(
        question_name="Q12 - Primary reason for revision",
        gold_standard_col="Primary reason for revision | Shunt Operation",
        prompt_file="q12_prompt.txt",
        options=q12_options,
        prediction_key="Q12_Primary_Reason_Revision",
        note_sources=ALL_NOTES_SOURCES,
        llm_kwargs={
            "format": _q12_schema,
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),
    "q13": QuestionSpec(
        question_name="Q13 - Replacement with EVD",
        gold_standard_col="Replacement with EVD | Shunt Operation",
        prompt_file="q13_prompt.txt",
        options=q13_options,
        prediction_key="Q13_Replacement_With_EVD",
        note_sources=OP_NOTE_ONLY,
        llm_kwargs={
            "format": _q13_schema,
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),
    "q14": QuestionSpec(
        question_name="Q14 - ETV",
        gold_standard_col="ETV | Shunt Operation",
        prompt_file="q14_prompt.txt",
        options=q14_options,
        prediction_key="Q14_ETV",
        note_sources=OP_NOTE_ONLY,
        llm_kwargs={
            "format": _q14_schema,
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),
    "q15": QuestionSpec(
        question_name="Q15 - Choroid plexectomy",
        gold_standard_col="Choroid plexectomy | Shunt Operation",
        prompt_file="q15_prompt.txt",
        options=q15_options,
        prediction_key="Q15_Choroid_Plexectomy",
        note_sources=OP_NOTE_ONLY,
        llm_kwargs={
            "format": _q15_schema,
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),
    "q16": QuestionSpec(
        question_name="Q16 - Subtemporal decompression",
        gold_standard_col="Subtemporal decompression | Shunt Operation",
        prompt_file="q16_prompt.txt",
        options=q16_options,
        prediction_key="Q16_Subtemporal_Decompression",
        note_sources=OP_NOTE_ONLY,
        llm_kwargs={
            "format": _q16_schema,
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),
    "q17": QuestionSpec(
        question_name="Q17 - Ventricular size prior to surgery",
        gold_standard_col="Ventricular size prior to surgery | Shunt Operation",
        prompt_file="q17_prompt.txt",
        options=q17_options,
        prediction_key="Q17_Ventricular_Size_Prior_Surgery",
        note_sources=ALL_NOTES_SOURCES,
        llm_kwargs={
            "format": _q17_schema,
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),
    "q18": QuestionSpec(
        question_name="Q18 - Concurrent chemoradiotherapy for primary CNS tumour",
        gold_standard_col="Concurrent chemoradiotherapy for primary CNS tumour | Shunt Operation",
        prompt_file="q18_prompt.txt",
        options=q18_options,
        prediction_key="Q18_Concurrent_Chemoradiotherapy",
        note_sources=ALL_NOTES_SOURCES,
        llm_kwargs={
            "format": _q18_schema,
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),
    "q19": QuestionSpec(
        question_name="Q19 - Co-existing CNS infection",
        gold_standard_col="Co-existing CNS infection | Shunt Operation",
        prompt_file="q19_prompt.txt",
        options=q19_options,
        prediction_key="Q19_Coexisting_CNS_Infection",
        note_sources=ALL_NOTES_SOURCES,
        llm_kwargs={
            "format": _q19_schema,
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),
    "q20": QuestionSpec(
        question_name="Q20 - CNS infection in the last 6 months",
        gold_standard_col="CNS infection in the last 6 months | Shunt Operation",
        prompt_file="q20_prompt.txt",
        options=q20_options,
        prediction_key="Q20_CNS_Infection_Last_6_Months",
        note_sources=ALL_NOTES_SOURCES,
        llm_kwargs={
            "format": _q20_schema,
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),
    "q24": QuestionSpec(
        question_name="Q24 - Grade of primary surgeon",
        gold_standard_col="Grade of primary surgeon | Shunt Operation",
        prompt_file="q24_prompt.txt",
        options=q24_options,
        prediction_key="Q24_Grade_Primary_Surgeon",
        note_sources=OP_NOTE_ONLY,
        llm_kwargs={
            "format": _q24_schema,
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),
    "q25": QuestionSpec(
        question_name="Q25 - Consultant presence",
        gold_standard_col="Consultant presence | Shunt Operation",
        prompt_file="q25_prompt.txt",
        options=q25_options,
        prediction_key="Q25_Consultant_Presence",
        note_sources=OP_NOTE_ONLY,
        llm_kwargs={
            "format": _q25_schema,
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),

    # -----------------------------------------------------------------------
    # Implant fields Q34-Q50.
    #
    # This block extracts:
    # - Q34 New implants inserted
    # - Q35 Old implants removed
    # - Q36 Other implants removed
    # - Q37-Q50 repeated implant component rows
    # -----------------------------------------------------------------------
    "q34_q50_implants": QuestionSpec(
        question_name="Q34-Q50 - Implant extraction",
        gold_standard_col="Implants",
        prompt_file="q34_q50_implants_prompt.txt",
        options=None,
        prediction_key="Q34_Q50_Implants",
        note_sources=OP_NOTE_AND_IMPLANT_NOTE,
        llm_kwargs={
            "format": {
                "type": "object",
                "properties": {
                    "new_implants_inserted": {"type": "string"},
                    "old_implants_removed": {"type": "string"},
                    "other_implants_removed": {"type": "string"},
                    "implants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "implant_type": {"type": "string"},
                                "manufacturer": {"type": "string"},
                                "catalogue_number_name": {"type": "string"},
                                "catheter_type": {"type": "string"},
                                "drainage_site": {"type": "string"},
                                "insertion_site": {"type": "string"},
                                "image_guided_placement": {"type": "string"},
                                "reservoir_type": {"type": "string"},
                                "serial_number": {"type": "string"},
                                "programmable": {"type": "string"},
                                "initial_setting": {"type": "string"},
                                "udi_code": {"type": "string"},
                                "other_implant_information": {"type": "string"},
                                "present_status": {"type": "string"},
                            },
                            "required": [
                                "implant_type",
                                "manufacturer",
                                "catalogue_number_name",
                                "catheter_type",
                                "drainage_site",
                                "insertion_site",
                                "image_guided_placement",
                                "reservoir_type",
                                "serial_number",
                                "programmable",
                                "initial_setting",
                                "udi_code",
                                "other_implant_information",
                                "present_status",
                            ],
                        },
                    },
                },
                "required": [
                    "new_implants_inserted",
                    "old_implants_removed",
                    "other_implants_removed",
                    "implants",
                ],
            },
            "response_format": {"type": "json_object"},
            "options": {"temperature": 0},
        },
    ),
}

# WARNING: Could not automatically replace q34_q50_implants block.
