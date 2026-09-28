"""
The full Data Detective pipeline:

    CSV file -> profiler (finds problems) -> AI (explains + suggests fixes)
             -> confidence caps (decides what's safe to auto-apply)

Run from the backend/ folder:
    python3 -m app.pipeline                  (uses the sample file)
    python3 -m app.pipeline path/to/your.csv
"""
import sys
from pathlib import Path

import pandas as pd

from app.profiler import (
    check_missing_values,
    check_duplicate_rows,
    check_inconsistent_categories,
    check_type_mismatches,
    check_outliers,
    check_formatting_issues,
)
from app.diagnosis import diagnose_report
from app.confidence import apply_confidence_caps
from app.fixer import clean_and_save

# project_root/sample_data/messy_sample.csv
DEFAULT_FILE = Path(__file__).resolve().parents[2] / "sample_data" / "messy_sample.csv"


def build_report(df: pd.DataFrame) -> dict:
    """
    Run all six profiling checks and keep ONLY the real problems.
    (Otherwise the AI would waste time "diagnosing" columns with 0 issues.)
    """
    report = {}

    # missing values: keep only columns that actually have some missing
    missing = {
        column: info
        for column, info in check_missing_values(df).items()
        if info["missing_count"] > 0
    }
    if missing:
        report["missing_values"] = missing

    # duplicates: keep only if there are any
    duplicates = check_duplicate_rows(df)
    if duplicates["duplicate_count"] > 0:
        report["duplicate_rows"] = duplicates

    # the other four checks already return an empty dict when all is well
    other_checks = [
        ("inconsistent_categories", check_inconsistent_categories),
        ("type_mismatches", check_type_mismatches),
        ("outliers", check_outliers),
        ("formatting_issues", check_formatting_issues),
    ]
    for name, check in other_checks:
        result = check(df)
        if result:
            report[name] = result

    return report


def run_pipeline(file_path) -> tuple:
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {Path(file_path).name}")

    report = build_report(df)
    if not report:
        print("No issues found - this dataset looks clean!")
        return df, []

    print(f"Profiler found problems in {len(report)} categories. Asking the AI to diagnose...")
    diagnosis = diagnose_report(report)
    return df, apply_confidence_caps(diagnosis)

def print_results(results: list) -> None:
    if not results:
        return

    auto = [r for r in results if r["auto_apply"]]
    review = [r for r in results if not r["auto_apply"]]
    print(f"\n{len(auto)} fixes are safe to auto-apply, {len(review)} need your review\n")

    for label, group in (("AUTO-APPLY", auto), ("NEEDS REVIEW", review)):
        for r in group:
            print(f"[{label}] {r['column']} - {r['issue_type']} (confidence {r['final_confidence']:.0f})")
            print(f"    Problem: {r['explanation']}")
            print(f"    Fix:     {r['suggested_fix']}\n")


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    df, results = run_pipeline(file_path)
    print_results(results)
    if results:
        clean_and_save(df, results, file_path)