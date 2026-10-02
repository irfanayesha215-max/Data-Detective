from pathlib import Path
import pandas as pd

MAX_ROWS = 100_000
SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


class LoadError(ValueError):
    """A problem with the uploaded file that we can explain to the user."""


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    head = path.read_bytes()[:2]
    if head in (b"\xff\xfe", b"\xfe\xff"):          # UTF-16 (Excel "Unicode Text")
        encodings = ["utf-16"]
    else:                                            # utf-8-sig also handles a BOM
        encodings = ["utf-8-sig", "cp1252", "latin-1"]

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
        # One giant column whose header contains ; or tab = wrong delimiter
        if df.shape[1] == 1 and any(d in str(df.columns[0]) for d in (";", "\t")):
            try:
                alt = pd.read_csv(path, encoding=enc, sep=None, engine="python", **kwargs)
                if alt.shape[1] > 1:
                    df = alt
            except Exception:
                pass
        return df

    raise LoadError("Couldn't work out this file's text encoding. "
                    "Try re-saving it as 'CSV UTF-8' from Excel.")


def _read(path: Path, ext: str, **kwargs) -> pd.DataFrame:
    if ext == ".xlsx":
        return pd.read_excel(path, engine="openpyxl", **kwargs)  # first sheet only
    return _read_csv(path, **kwargs)


def _looks_headerless(df: pd.DataFrame) -> bool:
    def is_number(x):
        try:
            float(str(x))
            return True
        except ValueError:
            return False
    return len(df.columns) > 0 and all(is_number(c) for c in df.columns)


def load_file(path: Path, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise LoadError("Please upload a .csv or .xlsx file.")

    try:
        df = _read(path, ext)
        if _looks_headerless(df):
            df = _read(path, ext, header=None)
            df.columns = [f"column_{i + 1}" for i in range(df.shape[1])]
    except LoadError:
        raise
    except pd.errors.EmptyDataError:
        raise LoadError("That file is empty.")
    except Exception as e:
        raise LoadError(f"Could not read that file: {e}")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)          # fully blank rows

    # Trailing commas in a CSV create empty "Unnamed: N" columns - drop those
    junk = [c for c in df.columns if c.startswith("Unnamed:") and df[c].isna().all()]
    df = df.drop(columns=junk)

    if df.shape[1] == 0 or len(df) == 0:
        raise LoadError("That file has no data rows to clean.")
    if len(df) > MAX_ROWS:
        raise LoadError(f"That file has {len(df):,} rows. The limit is {MAX_ROWS:,}.")
    return df