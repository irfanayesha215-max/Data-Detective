import pandas as pd


def check_missing_values(df: pd.DataFrame) -> dict:
    """
    Look at every column and report how many values are missing,
    and what percent of the column that represents.
    """
    total_rows = len(df)
    report = {}

    for column in df.columns:
        missing_count = df[column].isna().sum()   # counts NaN / None / missing values
        missing_percent = float(round((missing_count / total_rows) * 100, 1))

        report[column] = {
            "missing_count": int(missing_count),
            "missing_percent": missing_percent
        }

    return report


def check_duplicate_rows(df: pd.DataFrame) -> dict:
    """
    Check whether any rows in the dataset are exact copies of each other.
    """
    total_rows = len(df)

    # duplicated() returns True for every row that is an exact repeat
    # of a row seen earlier in the dataset (the first occurrence stays False)
    duplicate_mask = df.duplicated()
    duplicate_count = int(duplicate_mask.sum())
    duplicate_percent = float(round((duplicate_count / total_rows) * 100, 1))

    return {
        "duplicate_count": duplicate_count,
        "duplicate_percent": duplicate_percent,
        "duplicate_row_indexes": df[duplicate_mask].index.tolist()
    }


def check_inconsistent_categories(df: pd.DataFrame) -> dict:
    """
    For text columns, find values that are probably the same thing
    written differently - e.g. "USA", "usa", " USA " are likely one category.
    """
    report = {}

    for column in df.columns:
        # only check text columns - skip numbers
        # (use pandas's own helper instead of comparing dtype names directly,
        # since "text" can show up as "object" or "str" depending on pandas version)
        if not pd.api.types.is_string_dtype(df[column]):
            continue

        # drop missing values before comparing - they're handled separately
        values = df[column].dropna()

        # group original values by their "cleaned" version (lowercase, trimmed)
        groups = {}
        for original_value in values.unique():
            cleaned = str(original_value).strip().lower()
            groups.setdefault(cleaned, []).append(original_value)

        # keep only the groups where more than one original spelling exists
        inconsistent_groups = [
            variants for variants in groups.values() if len(variants) > 1
        ]

        if inconsistent_groups:
            report[column] = {
                "inconsistent_groups": inconsistent_groups
            }

    return report


def check_type_mismatches(df: pd.DataFrame) -> dict:
    """
    Look for columns that are *mostly* one type (like numbers) but have
    a small number of values that don't fit - e.g. a numeric column
    with a few "unknown" or "N/A" text values mixed in.
    """
    report = {}

    for column in df.columns:
        values = df[column].dropna()
        if len(values) == 0:
            continue

        # try converting every value to a number
        # errors="coerce" turns anything that fails into NaN instead of crashing
        numeric_attempt = pd.to_numeric(values, errors="coerce")

        # values that were real numbers before, but became NaN after conversion,
        # are the ones that don't actually fit as numbers
        failed_to_convert = values[numeric_attempt.isna()]
        succeeded = values[~numeric_attempt.isna()]

        # only flag this column if MOST values are numeric but a few aren't -
        # if it's mostly text, it's just a text column, not a "mismatch"
        if len(succeeded) > 0 and len(failed_to_convert) > 0:
            mostly_numeric = len(succeeded) > len(failed_to_convert)
            if mostly_numeric:
                report[column] = {
                    "expected_type": "numeric",
                    "bad_values": failed_to_convert.unique().tolist(),
                    "bad_value_count": int(len(failed_to_convert))
                }

    return report


def check_outliers(df: pd.DataFrame) -> dict:
    """
    For numeric columns, flag values that are unusually far from
    the rest of the data using the IQR (interquartile range) method -
    a standard, simple way to spot outliers.
    """
    report = {}

    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        values = df[column].dropna()
        if len(values) < 4:
            # too few values to reasonably judge what's "normal"
            continue

        # Q1 = value below which 25% of the data falls
        # Q3 = value below which 75% of the data falls
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        # anything beyond 1.5x the IQR past Q1/Q3 is considered unusual -
        # this 1.5 multiplier is a widely used statistical convention
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outliers = values[(values < lower_bound) | (values > upper_bound)]

        if len(outliers) > 0:
            report[column] = {
                "outlier_count": int(len(outliers)),
                "outlier_values": outliers.tolist(),
                "normal_range": [float(lower_bound), float(upper_bound)]
            }

    return report


def check_formatting_issues(df: pd.DataFrame) -> dict:
    """
    For text columns, flag values that have formatting problems on their own -
    like extra leading/trailing whitespace - regardless of whether another
    value matches them once cleaned.
    """
    report = {}

    for column in df.columns:
        if not pd.api.types.is_string_dtype(df[column]):
            continue

        values = df[column].dropna()

        # a value "has whitespace issues" if it doesn't match its own
        # stripped version - meaning it has extra spaces somewhere
        has_extra_whitespace = values[values != values.str.strip()]

        if len(has_extra_whitespace) > 0:
            report[column] = {
                "whitespace_issue_count": int(len(has_extra_whitespace)),
                "example_values": has_extra_whitespace.unique().tolist()[:5]
            }

    return report


if __name__ == "__main__":
    # Quick manual test with a tiny fake messy dataset
    # (row 4 is an exact duplicate of row 1, on purpose)
    sample_data = {
        "Name": ["Alice", "Bob", None, "David", "Alice", "Eve", "Frank", " Grace "],
        "Age": [25, None, 30, 40, 25, 28, 32, 250],   # 250 is an obvious outlier
        "Country": ["USA", "usa", "USA", None, "USA", "USA", "USA", "USA"],
        "Salary": [50000, 60000, "unknown", 55000, 50000, 58000, 62000, 59000],
    }
    df = pd.DataFrame(sample_data)

    print("Missing values:")
    print(check_missing_values(df))

    print("\nDuplicate rows:")
    print(check_duplicate_rows(df))

    print("\nInconsistent categories:")
    print(check_inconsistent_categories(df))

    print("\nType mismatches:")
    print(check_type_mismatches(df))

    print("\nOutliers:")
    print(check_outliers(df))

    print("\nFormatting issues:")
    print(check_formatting_issues(df))