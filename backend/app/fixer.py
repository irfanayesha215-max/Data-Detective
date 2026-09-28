"""
The fixer: applies the fixes the confidence layer marked as safe,
saves a cleaned copy of the data, and writes a report of every change.

Important design choice: the AI only EXPLAINS problems. The actual fixing is
done by the small, tested functions below, so the AI can never make the tool
do something unpredictable to your data.
"""
from pathlib import Path

import pandas as pd

from app.profiler import check_inconsistent_categories

# project_root/output/  (cleaned files and reports are saved here)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


# ---------------------------------------------------------------------------
# Individual fixes. Each one takes (df, column) and returns (df, change)
# where `change` describes what happened, or None if there was nothing to do.
# ---------------------------------------------------------------------------

def fix_formatting(df: pd.DataFrame, column: str):
    """Trim leading/trailing spaces from a text column."""
    if column not in df.columns or not pd.api.types.is_string_dtype(df[column]):
        return df, None

    original = df[column]
    stripped = original.str.strip()
    changed_mask = original.notna() & (original != stripped)
    changed = int(changed_mask.sum())
    if changed == 0:
        return df, None

    example_before = original[changed_mask].iloc[0]
    example_after = stripped[changed_mask].iloc[0]
    df[column] = stripped
    return df, {
        "column": column,
        "action": "Trimmed extra whitespace",
        "detail": f"{changed} value(s) changed, e.g. {example_before!r} -> {example_after!r}",
    }


def fix_inconsistent_categories(df: pd.DataFrame, column: str):
    """Make variants of the same value (USA / usa) use the most common spelling."""
    if column not in df.columns:
        return df, None

    groups = check_inconsistent_categories(df).get(column, {}).get("inconsistent_groups", [])
    if not groups:
        return df, None

    counts = df[column].value_counts()
    mapping = {}
    for variants in groups:
        # the spelling used most often in the data wins
        canonical = max(variants, key=lambda v: counts.get(v, 0))
        for variant in variants:
            if variant != canonical:
                mapping[variant] = canonical

    if not mapping:
        return df, None

    affected = int(df[column].isin(list(mapping)).sum())
    df[column] = df[column].replace(mapping)
    changes = ", ".join(f"{old!r} -> {new!r}" for old, new in mapping.items())
    return df, {
        "column": column,
        "action": "Standardized inconsistent values",
        "detail": f"{affected} value(s) changed: {changes}",
    }


def fix_duplicates(df: pd.DataFrame, column: str = None):
    """Remove exact duplicate rows (the first copy is kept)."""
    duplicate_mask = df.duplicated()
    count = int(duplicate_mask.sum())
    if count == 0:
        return df, None

    removed_rows = df[duplicate_mask].index.tolist()
    df = df.drop_duplicates()
    return df, {
        "column": "(all columns)",
        "action": "Removed duplicate rows",
        "detail": f"{count} row(s) removed (original row numbers: {removed_rows})",
    }


def fix_missing_values(df: pd.DataFrame, column: str):
    """
    Fill missing values: the median for number columns, the most common
    value for text columns. This is only run when a person has explicitly
    approved it - guessing at real data is never done automatically.
    """
    if column not in df.columns:
        return df, None

    col = df[column]
    missing_mask = col.isna()
    count = int(missing_mask.sum())
    if count == 0:
        return df, None

    if pd.api.types.is_numeric_dtype(col):
        fill_value = float(col.median())
        method = "the median"
    else:
        mode = col.mode(dropna=True)
        if mode.empty:
            return df, None
        fill_value = mode.iloc[0]
        method = "the most common value"

    df[column] = col.fillna(fill_value)
    return df, {
        "column": column,
        "action": f"Filled missing values with {method}",
        "detail": f"{count} value(s) filled with {fill_value!r}",
    }


def fix_type_mismatch(df: pd.DataFrame, column: str):
    """
    Turn values that don't fit the column's type (e.g. "unknown" in a
    number column) into blanks, so the column becomes properly numeric.
    """
    if column not in df.columns:
        return df, None

    col = df[column]
    numeric = pd.to_numeric(col, errors="coerce")
    bad_mask = col.notna() & numeric.isna()
    count = int(bad_mask.sum())
    if count == 0:
        return df, None

    bad_values = col[bad_mask].unique().tolist()
    df[column] = numeric
    return df, {
        "column": column,
        "action": "Converted non-numeric values to blank",
        "detail": f"{count} value(s) {bad_values} could not be read as numbers, so they were set to blank",
    }


def fix_outliers(df: pd.DataFrame, column: str):
    """
    Set values far outside the normal range (using the same IQR rule as
    the profiler) to blank, so they don't skew any analysis using this data.
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return df, None

    values = df[column].dropna()
    if len(values) < 4:
        return df, None

    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - (1.5 * iqr), q3 + (1.5 * iqr)

    outlier_mask = df[column].notna() & ((df[column] < lower) | (df[column] > upper))
    count = int(outlier_mask.sum())
    if count == 0:
        return df, None

    outlier_values = df.loc[outlier_mask, column].tolist()
    df.loc[outlier_mask, column] = pd.NA
    return df, {
        "column": column,
        "action": "Set outlier values to blank",
        "detail": f"{count} value(s) {outlier_values} were outside the normal range ({lower:.1f}-{upper:.1f}) and set to blank",
    }


# which fixer handles which issue type - and the order they must run in:
# trim spaces and standardize first, then fix values, and remove
# duplicates LAST (because cleaning can turn near-copies into exact copies)
FIXERS = {
    "formatting": fix_formatting,
    "inconsistent_categories": fix_inconsistent_categories,
    "type_mismatch": fix_type_mismatch,
    "outliers": fix_outliers,
    "missing_values": fix_missing_values,
    "duplicates": fix_duplicates,
}
FIX_ORDER = ["formatting", "inconsistent_categories", "type_mismatch", "outliers", "missing_values", "duplicates"]


def apply_fixes(df: pd.DataFrame, results: list):
    """
    Apply every fix marked auto_apply. Return (cleaned_df, applied, skipped).
    Nothing is ever applied silently: every fix ends up in one of the two lists.
    """
    df = df.copy()   # never modify the original data
    applied, skipped = [], []
    to_apply = []

    for r in results:
        if not r["auto_apply"]:
            reason = f"Needs your review (confidence {r['final_confidence']:.0f})"
            skipped.append({**r, "reason": reason})
        elif r["issue_type"] not in FIXERS:
            skipped.append({**r, "reason": "No automatic fixer available yet"})
        else:
            to_apply.append(r)

    to_apply.sort(key=lambda r: FIX_ORDER.index(r["issue_type"]))

    for r in to_apply:
        df, change = FIXERS[r["issue_type"]](df, r["column"])
        if change:
            applied.append(change)
        else:
            skipped.append({**r, "reason": "Nothing left to change"})

    return df, applied, skipped


def write_report(path: Path, source_name: str, rows_before: int, rows_after: int,
                 applied: list, skipped: list) -> None:
    review = [s for s in skipped if s["reason"].startswith("Needs your review")]

    lines = [
        "# Cleaning Report",
        "",
        f"- **Source file:** {source_name}",
        f"- **Rows:** {rows_before} -> {rows_after}",
        f"- **Fixes applied automatically:** {len(applied)}",
        f"- **Issues left for your review:** {len(review)}",
        "",
        "## Applied automatically",
        "",
    ]
    if applied:
        for i, change in enumerate(applied, start=1):
            lines.append(f"{i}. **{change['column']}**: {change['action']}. {change['detail']}")
    else:
        lines.append("_No fixes were applied._")

    lines += ["", "## Not applied - needs your review", ""]
    if review:
        for s in review:
            lines.append(f"- **{s['column']}** ({s['issue_type']}, confidence {s['final_confidence']:.0f})")
            lines.append(f"  - Problem: {s['explanation']}")
            lines.append(f"  - Suggested fix: {s['suggested_fix']}")
    else:
        lines.append("_Nothing needs review._")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_and_save(df: pd.DataFrame, results: list, source_path, output_dir=None) -> None:
    """Apply safe fixes, save the cleaned CSV, and save the cleaning report."""
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source_path).stem

    cleaned_df, applied, skipped = apply_fixes(df, results)

    cleaned_path = output_dir / f"{stem}_cleaned.csv"
    report_path = output_dir / f"{stem}_cleaning_report.md"
    # a column with blanks becomes decimals in pandas (25 -> 25.0), even though
    # we never touched it. convert_dtypes() turns whole-number columns back
    # into integers, so the file only changes where we said it did.
    cleaned_df.convert_dtypes().to_csv(cleaned_path, index=False)
    write_report(report_path, Path(source_path).name, len(df), len(cleaned_df), applied, skipped)

    print(f"Applied {len(applied)} fixes. Rows: {len(df)} -> {len(cleaned_df)}")
    print(f"Cleaned file:    {cleaned_path}")
    print(f"Cleaning report: {report_path}")