import pytest

from json_extract_eval.core.comparators.oneof import compare_oneof


def test_match() -> None:
    result = compare_oneof("PVD", "Sputtering", {"values": ["PVD", "Sputtering"]})
    assert result.score == 1.0
    assert result.comparator == "oneof"


def test_no_match() -> None:
    result = compare_oneof("PVD", "CVD", {"values": ["PVD", "Sputtering"]})
    assert result.score == 0.0
    assert result.reason == "no match in values"


def test_empty_values() -> None:
    result = compare_oneof("PVD", "PVD", {"values": []})
    assert result.score == 0.0


def test_extracted_in_values_gold_not_in_values() -> None:
    result = compare_oneof("PVD", "CVD", {"values": ["PVD", "Sputtering"]})
    assert result.score == 0.0
    assert result.reason == "no match in values"
    assert result.score == 0.0


def test_empty_string() -> None:
    result = compare_oneof("PVD", "", {"values": []})
    assert result.score == 0.0


def test_empty_string_pass() -> None:
    result = compare_oneof("PVD", "", {"values": [""]})
    assert result.score == 1.0


def test_missing_values_key() -> None:
    with pytest.raises(TypeError, match="requires 'values' to be a list"):
        compare_oneof("PVD", "PVD", {})


def test_none_values() -> None:
    with pytest.raises(TypeError, match="requires 'values' to be a list"):
        compare_oneof("PVD", "PVD", {"values": None})


def test_exact_gold_in_values() -> None:
    result = compare_oneof("PVD", "PVD", {"values": ["PVD", "Sputtering"]})
    assert result.score == 1.0


def test_case_sensitive() -> None:
    result = compare_oneof("PVD", "pvd", {"values": ["PVD", "Sputtering"]})
    assert result.score == 0.0
