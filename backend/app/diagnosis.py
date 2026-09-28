import os
import json
from groq import Groq
from dotenv import load_dotenv

# load the GROQ_API_KEY from the .env file into the environment
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# Groq retires/renames models fairly often. If you get a "model_not_found"
# error, list current models with: Groq().models.list() and update this line.
MODEL_NAME = "openai/gpt-oss-120b"

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


def diagnose_report(profiling_report: dict) -> dict:
    """
    Send the raw profiling report to the LLM and get back structured
    diagnoses with plain-English explanations and confidence scores.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(profiling_report)}
        ],
        temperature=0.2,   # low temperature = more consistent, less "creative"
    )

    raw_text = response.choices[0].message.content

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # the model occasionally wraps JSON in markdown code fences despite
        # instructions - this strips them as a fallback before giving up
        cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)


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