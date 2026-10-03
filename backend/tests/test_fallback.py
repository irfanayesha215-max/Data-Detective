import pandas as pd
from fastapi.testclient import TestClient

from app import main
from app.confidence import apply_confidence_caps
from app.diagnosis import fallback_diagnosis
from app.pipeline import build_report


def _messy_df():
    return pd.DataFrame({
        "name": [" Alice ", "Bob", "Carol", "Dave", "Eve", "Bob"],
        "age": [25, None, 28, 31, 999, 30],
        "country": ["USA", "usa", "USA", "UK", "USA", "usa"],
    })


def test_fallback_covers_every_issue_the_profiler_found():
    report = build_report(_messy_df())
    result = fallback_diagnosis(report)
    types = {d["issue_type"] for d in result["diagnoses"]}
    assert {"formatting", "missing_values", "inconsistent_categories"} <= types
    assert all(d["explanation"] and d["suggested_fix"] for d in result["diagnoses"])


def test_fallback_only_auto_applies_the_safest_fixes():
    results = apply_confidence_caps(fallback_diagnosis(build_report(_messy_df())))
    for r in results:
        if r["auto_apply"]:
            assert r["issue_type"] in {"formatting", "duplicates"}
    assert any(r["issue_type"] == "formatting" and r["auto_apply"] for r in results)


CSV = b"name,age\n Alice ,25\nBob,30\nCarol,28\nDave,31\nEve,29\n"


def test_endpoint_falls_back_when_ai_is_down(monkeypatch):
    def boom(report):
        raise RuntimeError("simulated outage")
    monkeypatch.setattr(main, "diagnose_report", boom)

    client = TestClient(main.app)
    r = client.post("/api/clean", files={"file": ("a.csv", CSV, "text/csv")}, data={"mode": "review"})
    assert r.status_code == 200
    assert "notice" in r.json()


def test_full_auto_refuses_when_ai_is_down(monkeypatch):
    def boom(report):
        raise RuntimeError("simulated outage")
    monkeypatch.setattr(main, "diagnose_report", boom)

    client = TestClient(main.app)
    r = client.post("/api/clean", files={"file": ("a.csv", CSV, "text/csv")}, data={"mode": "full_auto"})
    assert r.status_code == 503
