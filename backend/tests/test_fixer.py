"""
Tests for fixer.py - the code that actually changes data.

These matter more than tests for the profiler or the AI layer, because a
bug here means real data gets changed incorrectly. Run with:
    cd backend
    pytest tests/test_fixer.py -v
"""
import pandas as pd

from app.fixer import (
    fix_formatting,
    fix_inconsistent_categories,
    fix_duplicates,
    fix_missing_values,
    fix_type_mismatch,
    fix_outliers,
    apply_fixes,
)


# ---------------------------------------------------------------------------
# Individual fixers - each one should do exactly one thing, and do nothing
# when there's nothing to fix.
# ---------------------------------------------------------------------------

def test_fix_formatting_trims_whitespace():
    df = pd.DataFrame({"Name": ["Alice", " Bob "]})
    df, change = fix_formatting(df, "Name")
    assert df["Name"].tolist() == ["Alice", "Bob"]
    assert change["column"] == "Name"


def test_fix_formatting_does_nothing_when_clean():
    df = pd.DataFrame({"Name": ["Alice", "Bob"]})
    df, change = fix_formatting(df, "Name")
    assert change is None


def test_fix_inconsistent_categories_uses_most_common_spelling():
    # "USA" appears 3 times, "usa" once - "USA" should win
    df = pd.DataFrame({"Country": ["USA", "usa", "USA", "USA"]})
    df, change = fix_inconsistent_categories(df, "Country")
    assert (df["Country"] == "USA").all()
    assert change is not None


def test_fix_duplicates_keeps_first_occurrence():
    df = pd.DataFrame({"Name": ["Alice", "Bob", "Alice"], "Age": [25, 30, 25]})
    df, change = fix_duplicates(df)
    assert len(df) == 2
    assert change["column"] == "(all columns)"


def test_fix_duplicates_does_nothing_when_no_duplicates():
    df = pd.DataFrame({"Name": ["Alice", "Bob"]})
    df, change = fix_duplicates(df)
    assert change is None


def test_fix_missing_values_numeric_uses_median():
    df = pd.DataFrame({"Age": [20.0, 30.0, 40.0, None]})
    df, change = fix_missing_values(df, "Age")
    assert df["Age"].tolist() == [20.0, 30.0, 40.0, 30.0]   # median of 20/30/40
    assert change is not None


def test_fix_missing_values_text_uses_most_common():
    df = pd.DataFrame({"City": ["NY", "NY", "LA", None]})
    df, change = fix_missing_values(df, "City")
    assert df["City"].tolist() == ["NY", "NY", "LA", "NY"]


def test_fix_missing_values_does_nothing_when_no_gaps():
    df = pd.DataFrame({"Age": [20, 30]})
    df, change = fix_missing_values(df, "Age")
    assert change is None


def test_fix_type_mismatch_converts_bad_values_to_blank():
    df = pd.DataFrame({"Salary": [50000, "unknown", 60000]})
    df, change = fix_type_mismatch(df, "Salary")
    assert df["Salary"].isna().sum() == 1
    assert pd.api.types.is_numeric_dtype(df["Salary"])
    assert change is not None


def test_fix_outliers_blanks_values_outside_iqr_range():
    df = pd.DataFrame({"Age": [25, 28, 30, 32, 250]})
    df, change = fix_outliers(df, "Age")
    assert df["Age"].isna().sum() == 1
    assert change is not None


def test_fix_outliers_does_nothing_with_too_few_values():
    df = pd.DataFrame({"Age": [25, 250]})  # only 2 values, IQR needs 4+
    df, change = fix_outliers(df, "Age")
    assert change is None


# ---------------------------------------------------------------------------
# apply_fixes - the orchestration function that decides what runs and tags
# each result with a real status.
# ---------------------------------------------------------------------------

def _result(id, column, issue_type, auto_apply, final_confidence=90):
    """Shorthand for building a fake diagnosis result in tests."""
    return {
        "id": id, "column": column, "issue_type": issue_type,
        "explanation": "test", "suggested_fix": "test",
        "final_confidence": final_confidence, "auto_apply": auto_apply,
    }


def test_apply_fixes_only_runs_auto_apply_results():
    df = pd.DataFrame({"Name": [" Bob "], "Age": [None]})
    results = [
        _result(0, "Name", "formatting", auto_apply=True),
        _result(1, "Age", "missing_values", auto_apply=False),
    ]
    cleaned, tagged = apply_fixes(df, results)

    assert cleaned["Name"].iloc[0] == "Bob"          # applied
    assert pd.isna(cleaned["Age"].iloc[0])            # left alone
    assert tagged[0]["status"] == "applied"
    assert tagged[1]["status"] == "needs_review"


def test_apply_fixes_does_not_mutate_the_original_dataframe():
    df = pd.DataFrame({"Name": [" Bob "]})
    original_value = df["Name"].iloc[0]
    results = [_result(0, "Name", "formatting", auto_apply=True)]

    apply_fixes(df, results)

    # the ORIGINAL df passed in must be untouched
    assert df["Name"].iloc[0] == original_value


def test_apply_fixes_missing_values_runs_before_outliers():
    """
    Regression test for a real bug: outliers used to get blanked first,
    then missing_values would fill that new blank too, over-counting the
    "filled" total. Missing values must run first.
    """
    df = pd.DataFrame({"Age": [25.0, 28.0, 30.0, 32.0, None, 250.0]})
    results = [
        _result(0, "Age", "missing_values", auto_apply=True),
        _result(1, "Age", "outliers", auto_apply=True),
    ]
    cleaned, tagged = apply_fixes(df, results)

    missing_result = next(r for r in tagged if r["issue_type"] == "missing_values")
    outlier_result = next(r for r in tagged if r["issue_type"] == "outliers")

    # exactly 1 value should have been filled (the true gap) -
    # the outlier's blank must NOT be counted here
    assert "1 value(s) filled" in missing_result["status_detail"]
    assert "250.0" in outlier_result["status_detail"]
    assert cleaned["Age"].isna().sum() == 1   # the outlier, now blank


def test_apply_fixes_marks_unfixable_type_as_skipped():
    df = pd.DataFrame({"Name": ["Alice"]})
    results = [_result(0, "Name", "some_future_issue_type", auto_apply=True)]
    cleaned, tagged = apply_fixes(df, results)
    assert tagged[0]["status"] == "skipped"


if __name__ == "__main__":
    # allows a quick manual run without pytest installed, e.g.:
    #     python3 tests/test_fixer.py
    import sys
    failures = 0
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {test.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)