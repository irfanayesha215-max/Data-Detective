import pandas as pd
import pytest
from app import loader
from app.loader import load_file, LoadError


def test_loads_utf8_csv(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("name,age\nAlice,25\nBob,30\n", encoding="utf-8")
    df = load_file(p, "a.csv")
    assert list(df.columns) == ["name", "age"]
    assert len(df) == 2


def test_falls_back_to_cp1252(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes("name,city\nAmélie,Café\n".encode("cp1252"))
    df = load_file(p, "a.csv")
    assert df.loc[0, "city"] == "Café"


def test_semicolon_delimiter(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("name;age\nAlice;25\nBob;30\n", encoding="utf-8")
    assert load_file(p, "a.csv").shape[1] == 2


def test_empty_file_gives_friendly_error(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("")
    with pytest.raises(LoadError):
        load_file(p, "a.csv")


def test_header_only_file_rejected(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("name,age\n")
    with pytest.raises(LoadError):
        load_file(p, "a.csv")


def test_headerless_file_gets_generated_names(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("1,2,3\n4,5,6\n7,8,9\n")
    df = load_file(p, "a.csv")
    assert list(df.columns) == ["column_1", "column_2", "column_3"]
    assert len(df) == 3


def test_reads_xlsx(tmp_path):
    p = tmp_path / "a.xlsx"
    pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}).to_excel(p, index=False)
    assert len(load_file(p, "a.xlsx")) == 2


def test_drops_empty_unnamed_columns(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("a,b,\n1,2,\n3,4,\n")
    assert list(load_file(p, "a.csv").columns) == ["a", "b"]


def test_row_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "MAX_ROWS", 3)
    p = tmp_path / "a.csv"
    p.write_text("a\n1\n2\n3\n4\n")
    with pytest.raises(LoadError):
        load_file(p, "a.csv")


def test_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("a\n1\n")
    with pytest.raises(LoadError):
        load_file(p, "a.txt")
