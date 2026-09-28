"""
Confidence cap layer.

The AI suggests a confidence score for each fix, but AI models tend to sound
more sure than they should be. So we also set our own maximum score for each
type of issue, based on how risky that kind of fix is. The final score is
whichever number is LOWER.
"""

# Maximum confidence allowed per issue type.
# Fixes that only change formatting are safe; fixes that guess or
# delete real data are risky.
CONFIDENCE_CAPS = {
    "duplicates": 99,               # removing exact copies is almost always safe
    "formatting": 98,               # trimming spaces doesn't change meaning
    "inconsistent_categories": 90,  # "usa" -> "USA" is safe, but merging can be wrong
    "type_mismatch": 60,            # replacing bad values changes the data
    "missing_values": 40,           # filling blanks means inventing data
    "outliers": 35,                 # an "outlier" might be a real value
}

DEFAULT_CAP = 50           # for any issue type we don't recognize: be cautious
AUTO_APPLY_THRESHOLD = 85  # final score must reach this to be applied automatically


def apply_confidence_caps(diagnosis_result: dict, threshold: int = AUTO_APPLY_THRESHOLD) -> list:
    """
    Take the AI's diagnosis output and return a list of diagnoses, each with:
      - ai_confidence:    what the AI said
      - cap:              our maximum for this issue type
      - final_confidence: the lower of the two
      - auto_apply:       True only if final_confidence >= threshold
    """
    results = []

    for item in diagnosis_result.get("diagnoses", []):
        # if the AI's score is missing or not a number, treat it as 0 (safest)
        try:
            ai_confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            ai_confidence = 0.0

        cap = CONFIDENCE_CAPS.get(item.get("issue_type"), DEFAULT_CAP)
        final_confidence = min(ai_confidence, cap)

        results.append({
            **item,
            "ai_confidence": ai_confidence,
            "cap": cap,
            "final_confidence": final_confidence,
            "auto_apply": final_confidence >= threshold,
        })

    return results


if __name__ == "__main__":
    # Test using a shortened copy of the real AI output from diagnosis.py
    ai_output = {
        "diagnoses": [
            {"column": "Age", "issue_type": "missing_values", "confidence": 80},
            {"column": "All", "issue_type": "duplicates", "confidence": 95},
            {"column": "Country", "issue_type": "inconsistent_categories", "confidence": 90},
            {"column": "Salary", "issue_type": "type_mismatch", "confidence": 85},
            {"column": "Age", "issue_type": "outliers", "confidence": 70},
            {"column": "Name", "issue_type": "formatting", "confidence": 95},
        ]
    }

    print(f"{'Column':<10}{'Issue':<26}{'AI':>5}{'Cap':>6}{'Final':>7}  Auto-apply?")
    print("-" * 68)
    for r in apply_confidence_caps(ai_output):
        print(
            f"{r['column']:<10}{r['issue_type']:<26}"
            f"{r['ai_confidence']:>5.0f}{r['cap']:>6}{r['final_confidence']:>7.0f}  "
            f"{'YES' if r['auto_apply'] else 'no - needs review'}"
        )