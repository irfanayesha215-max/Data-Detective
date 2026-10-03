import json
from types import SimpleNamespace

from app import diagnosis
from app.diagnosis import prepare_report_for_llm


def test_big_lists_trimmed_but_counts_kept_and_original_untouched():
    report = {
        "outliers": {"Age": {"outlier_count": 5000, "outlier_values": list(range(5000))}},
        "duplicate_rows": {"duplicate_count": 3000, "duplicate_row_indexes": list(range(3000))},
    }
    out = prepare_report_for_llm(report)
    assert len(out["outliers"]["Age"]["outlier_values"]) == 10
    assert out["outliers"]["Age"]["outlier_count"] == 5000
    assert len(out["duplicate_rows"]["duplicate_row_indexes"]) == 10
    assert len(report["outliers"]["Age"]["outlier_values"]) == 5000


def test_inconsistent_groups_get_a_total_count():
    groups = [[f"a{i}", f"A{i}"] for i in range(40)]
    report = {"inconsistent_categories": {"Country": {"inconsistent_groups": groups}}}
    out = prepare_report_for_llm(report)["inconsistent_categories"]["Country"]
    assert len(out["inconsistent_groups"]) == 10
    assert out["total_groups"] == 40


def test_diagnose_report_sends_the_trimmed_report(monkeypatch):
    sent = {}

    class FakeCompletions:
        def create(self, **kwargs):
            sent["payload"] = kwargs["messages"][1]["content"]
            msg = SimpleNamespace(content='{"diagnoses": []}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(diagnosis, "get_client", lambda: fake)

    report = {"outliers": {"Age": {"outlier_count": 900, "outlier_values": list(range(900))}}}
    result = diagnosis.diagnose_report(report)

    assert result == {"diagnoses": []}
    assert len(json.loads(sent["payload"])["outliers"]["Age"]["outlier_values"]) == 10