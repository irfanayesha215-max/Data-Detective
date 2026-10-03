import os
import json
from typing import Literal

from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator

# load the GROQ_API_KEY from the .env file into the environment
load_dotenv()

_API_KEY = os.environ.get("GROQ_API_KEY")
# client is created lazily (see get_client()) so a missing key doesn't crash
# the whole app at import time - it only fails when a diagnosis is actually
# requested, with a message that says exactly what to fix.
_client = None


def get_client() -> Groq:
    global _client
    if not _API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a .env file in backend/ with "
            "GROQ_API_KEY=your_key_here (get a free key at console.groq.com)."
        )
    if _client is None:
            _client = Groq(
            api_key=_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
    return _client


# Groq retires/renames models fairly often. If you get a "model_not_found"
# error, list current models with: Groq().models.list() and update this line.
MODEL_NAME = "openai/gpt-oss-120b"
LLM_TIMEOUT_SECONDS = 30.0
LLM_MAX_RETRIES = 3        # the SDK retries 429s, 5xx errors and dropped connections with backoff
MAX_LIST_ITEMS = 10        # cap on how many example values per list we send to the LLM

SYSTEM_PROMPT = """You are a data cleaning assistant. You will be given a JSON
report describing issues found in a dataset (missing values, duplicates,
inconsistent categories, type mismatches, outliers, formatting issues).

For EACH issue in the report, respond with:
- a short plain-English explanation of the problem
- a specific suggested fix
- a confidence score from 0-100, meaning: "how safe is it to apply this fix
  automatically without a human checking it first?"
  - 90-100: unambiguous, obviously safe (e.g. trimming whitespace)
  - 60-89: probably right, but context-dependent
  - below 60: genuinely unsure, a human should decide

Respond ONLY with valid JSON in this exact structure, and nothing else -
no explanation, no markdown formatting, no code fences:

{
  "diagnoses": [
    {
      "column": "column_name",
      "issue_type": "missing_values | duplicates | inconsistent_categories | type_mismatch | outliers | formatting",
      "explanation": "plain English explanation",
      "suggested_fix": "specific fix description",
      "confidence": 85
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# Validation: the LLM is the one part of the pipeline we don't control, so
# nothing it returns reaches the fixer until it passes this schema.
# ---------------------------------------------------------------------------

IssueType = Literal[
    "missing_values",
    "duplicates",
    "inconsistent_categories",
    "type_mismatch",
    "outliers",
    "formatting",
]
ISSUE_TYPE_ALIASES = {
    "type_mismatches": "type_mismatch",
    "formatting_issues": "formatting",
    "duplicate_rows": "duplicates",
    "duplicate": "duplicates",
    "missing_value": "missing_values",
    "outlier": "outliers",
    "inconsistent_category": "inconsistent_categories",
}

class Diagnosis(BaseModel):
    column: str
    issue_type: IssueType
    explanation: str
    suggested_fix: str
    confidence: float = 0

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0   # same "safest" fallback confidence.py uses
        return max(0.0, min(100.0, v))

    @field_validator("issue_type", mode="before")
    @classmethod
    def normalize_issue_type(cls, v):
        if isinstance(v, str):
            key = v.strip().lower().replace(" ", "_").replace("-", "_")
            return ISSUE_TYPE_ALIASES.get(key, key)
        return v
    
def validate_diagnoses(parsed) -> dict:
    """Keep only well-formed diagnoses; never let a bad one reach the fixer."""
    items = parsed.get("diagnoses") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        raise ValueError("The AI response did not contain a 'diagnoses' list.")

    valid, dropped = [], []
    for item in items:
        try:
            valid.append(Diagnosis.model_validate(item).model_dump())
        except ValidationError:
            dropped.append(item)

    if dropped:
        print(f"Warning: dropped {len(dropped)} malformed diagnosis item(s): {dropped}")
    return {"diagnoses": valid}
def _trim_lists(value, limit=MAX_LIST_ITEMS):
    """Return a copy with every list cut to `limit` items (original is untouched)."""
    if isinstance(value, dict):
        return {k: _trim_lists(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_trim_lists(v, limit) for v in value[:limit]]
    return value


def prepare_report_for_llm(report: dict) -> dict:
    """
    The LLM only needs counts plus a few examples to explain an issue.
    Sending every outlier value or duplicate index from a 100k-row file
    would blow the token limit, so big lists are trimmed here. The full
    report is still what the fixer works from.
    """
    trimmed = _trim_lists(report)
    # inconsistent_groups has no count field, so add one before it's cut down
    for col, info in report.get("inconsistent_categories", {}).items():
        trimmed["inconsistent_categories"][col]["total_groups"] = len(
            info.get("inconsistent_groups", [])
        )
    return trimmed

# Without the AI there's no second opinion, so only the two fixes that can't
# change meaning get a score high enough to auto-apply (threshold is 85).
# Everything else is routed to human review.
FALLBACK_CONFIDENCE = {
    "formatting": 95,
    "duplicates": 90,
    "inconsistent_categories": 75,
    "type_mismatch": 50,
    "missing_values": 40,
    "outliers": 30,
}


def fallback_diagnosis(report: dict) -> dict:
    """
    Build diagnoses straight from the profiler's findings, with generic
    wording, for when the AI service can't be reached. Same shape as
    diagnose_report() returns, so the rest of the pipeline is unchanged.
    """
    diagnoses = []

    def add(column, issue_type, explanation, fix):
        diagnoses.append({
            "column": column,
            "issue_type": issue_type,
            "explanation": explanation,
            "suggested_fix": fix,
            "confidence": FALLBACK_CONFIDENCE[issue_type],
        })

    for col, info in report.get("missing_values", {}).items():
        add(col, "missing_values",
            f"{info['missing_count']} value(s) ({info['missing_percent']}%) are missing.",
            "Fill with the median (numbers) or most common value (text), or leave blank.")

    dup = report.get("duplicate_rows")
    if dup:
        add("All columns", "duplicates",
            f"{dup['duplicate_count']} row(s) are exact copies of another row.",
            "Remove the duplicate rows, keeping the first copy.")

    for col, info in report.get("inconsistent_categories", {}).items():
        n = len(info.get("inconsistent_groups", []))
        add(col, "inconsistent_categories",
            f"{n} group(s) of values look like the same thing spelled differently.",
            "Standardize each group to its most common spelling.")

    for col, info in report.get("type_mismatches", {}).items():
        examples = info.get("bad_values", [])[:5]
        add(col, "type_mismatch",
            f"This is mostly a number column, but {info.get('bad_value_count', '?')} value(s) "
            f"aren't numbers, e.g. {examples}.",
            "Set the non-numeric values to blank, or correct them by hand.")

    for col, info in report.get("outliers", {}).items():
        low, high = info.get("normal_range", ["?", "?"])
        add(col, "outliers",
            f"{info['outlier_count']} value(s) fall outside the usual range ({low:.1f} to {high:.1f}).",
            "Check whether they're real. If not, set them to blank.")

    for col, info in report.get("formatting_issues", {}).items():
        add(col, "formatting",
            f"{info['whitespace_issue_count']} value(s) have extra spaces at the start or end.",
            "Trim the extra whitespace.")

    # run through the same validator as the AI output, so the shape is guaranteed
    return validate_diagnoses({"diagnoses": diagnoses})

def diagnose_report(profiling_report: dict) -> dict:
    """
    Send the raw profiling report to the LLM and get back structured
    diagnoses with plain-English explanations and confidence scores.
    """
    response = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prepare_report_for_llm(profiling_report))}
        ],
        temperature=0.2,   # low temperature = more consistent, less "creative"
    )

    raw_text = response.choices[0].message.content or ""

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # the model occasionally wraps JSON in markdown code fences despite
        # instructions - this strips them as a fallback before giving up
        cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)

    return validate_diagnoses(parsed)


if __name__ == "__main__":
    # a small fake profiling report to test with, shaped like what
    # profiler.py actually produces
    sample_report = {
        "missing_values": {"Age": {"missing_count": 1, "missing_percent": 12.5}},
        "duplicate_rows": {"duplicate_count": 1, "duplicate_row_indexes": [4]},
        "inconsistent_categories": {"Country": {"inconsistent_groups": [["USA", "usa"]]}},
        "type_mismatches": {"Salary": {"expected_type": "numeric", "bad_values": ["unknown"]}},
        "outliers": {"Age": {"outlier_count": 1, "outlier_values": [250.0]}},
        "formatting_issues": {"Name": {"example_values": [" Grace "]}},
    }

    result = diagnose_report(sample_report)
    print(json.dumps(result, indent=2))