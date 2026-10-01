"""
Tests for the validation layer in diagnosis.py. No API call is made.
Run with:
    cd backend
    pytest tests/test_diagnosis.py -v
"""
import pytest

from app.diagnosis import validate_diagnoses

GOOD = {
    "column": "Age", "issue_type": "outliers",
    "explanation": "x", "suggested_fix": "y", "confidence": 70,
}


def test_valid_item_passes():
    result = validate_diagnoses({"diagnoses": [GOOD]})["diagnoses"]
    assert len(result) == 1
    assert result[0]["column"] == "Age"


def test_missing_key_is_dropped_not_crashed():
    bad = {k: v for k, v in GOOD.items() if k != "explanation"}
    result = validate_diagnoses({"diagnoses": [GOOD, bad]})["diagnoses"]
    assert len(result) == 1


def test_unknown_issue_type_is_dropped():
    result = validate_diagnoses({"diagnoses": [{**GOOD, "issue_type": "typos"}]})["diagnoses"]
    assert result == []


def test_confidence_is_clamped_and_defaulted():
    hi = validate_diagnoses({"diagnoses": [{**GOOD, "confidence": 250}]})["diagnoses"][0]
    junk = validate_diagnoses({"diagnoses": [{**GOOD, "confidence": "very sure"}]})["diagnoses"][0]
    assert hi["confidence"] == 100.0
    assert junk["confidence"] == 0.0


def test_wrong_top_level_shape_raises():
    with pytest.raises(ValueError):
        validate_diagnoses({"results": []})
    with pytest.raises(ValueError):
        validate_diagnoses([GOOD])