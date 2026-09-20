import pytest

from json_extract_eval.core.comparators.comparator import ComparatorResult


def test_defaults() -> None:
    result = ComparatorResult(score=1.0, comparator="exact")
    assert result.score == 1.0
    assert result.comparator == "exact"
    assert result.reason is None


def test_all_fields() -> None:
    result = ComparatorResult(
        score=0.0,
        comparator="numeric",
        reason="relative error exceeds 0.01",
    )
    assert result.score == 0.0
    assert result.comparator == "numeric"
    assert result.reason == "relative error exceeds 0.01"


def test_frozen() -> None:
    result = ComparatorResult(score=1.0, comparator="exact")
    with pytest.raises(AttributeError):
        result.score = 0.5  # type: ignore[misc]


def test_score_boundary_values() -> None:
    assert ComparatorResult(score=0.0, comparator="numeric").score == 0.0
    assert ComparatorResult(score=1.0, comparator="numeric").score == 1.0
    assert ComparatorResult(score=0.5, comparator="numeric").score == 0.5
