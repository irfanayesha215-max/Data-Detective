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


if __name__ == "__main__":
    # Quick manual test with a tiny fake messy dataset
    # (row 4 is an exact duplicate of row 1, on purpose)
    sample_data = {
        "Name": ["Alice", "Bob", None, "David", "Alice"],
        "Age": [25, None, 30, 40, 25],
        "Country": ["USA", "usa", "USA", None, "USA"],
    }
    df = pd.DataFrame(sample_data)

    print("Missing values:")
    print(check_missing_values(df))

    print("\nDuplicate rows:")
    print(check_duplicate_rows(df))

    print("\nInconsistent categories:")
    print(check_inconsistent_categories(df))