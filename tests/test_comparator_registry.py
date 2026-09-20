from typing import Any

import pytest

from json_extract_eval.core.comparators.comparator import ComparatorResult
from json_extract_eval.core.comparators.registry import (
    ComparatorNotFoundError,
    _clear_registry,
    get_comparator,
    register,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Reset custom registry before each test."""
    _clear_registry()


def test_builtins_registered() -> None:
    # "semantic" is no longer a builtin -- it's a BatchComparator that the
    # user must register explicitly before calling evaluate().
    for name in ("exact", "numeric", "oneof"):
        fn = get_comparator(name)
        assert callable(fn)


def test_get_unknown_raises() -> None:
    with pytest.raises(ComparatorNotFoundError, match="Unknown comparator"):
        get_comparator("nonexistent")


def test_register_custom() -> None:
    def my_comparator(gold: object, extracted: object, params: dict[str, Any]) -> ComparatorResult:
        return ComparatorResult(score=1.0, comparator="custom")

    register("custom", my_comparator)
    assert get_comparator("custom") is my_comparator


def test_duplicate_registration_raises() -> None:
    def dummy(gold: object, extracted: object, params: dict[str, Any]) -> ComparatorResult:
        return ComparatorResult(score=1.0, comparator="dummy")

    register("dummy", dummy)
    with pytest.raises(ValueError, match="already registered"):
        register("dummy", dummy)


def test_overwrite_replaces_existing() -> None:
    def v1(gold: object, extracted: object, params: dict[str, Any]) -> ComparatorResult:
        return ComparatorResult(score=1.0, comparator="v1")

    def v2(gold: object, extracted: object, params: dict[str, Any]) -> ComparatorResult:
        return ComparatorResult(score=0.0, comparator="v2")

    register("my_comp", v1)
    assert get_comparator("my_comp") is v1
    register("my_comp", v2, overwrite=True)
    assert get_comparator("my_comp") is v2


def test_overwrite_false_raises_on_duplicate() -> None:
    def dummy(gold: object, extracted: object, params: dict[str, Any]) -> ComparatorResult:
        return ComparatorResult(score=1.0, comparator="dummy")

    register("dup", dummy)
    with pytest.raises(ValueError, match="already registered"):
        register("dup", dummy, overwrite=False)


def test_cannot_overwrite_builtin() -> None:
    def fake(gold: object, extracted: object, params: dict[str, Any]) -> ComparatorResult:
        return ComparatorResult(score=1.0, comparator="fake")

    with pytest.raises(ValueError, match="Cannot overwrite built-in"):
        register("exact", fake)
    with pytest.raises(ValueError, match="Cannot overwrite built-in"):
        register("exact", fake, overwrite=True)


def test_builtin_exact() -> None:
    fn = get_comparator("exact")
    result = fn("hello", "hello", {})
    assert result.score == 1.0


def test_builtin_numeric() -> None:
    fn = get_comparator("numeric")
    result = fn(42.0, 42.0, {})
    assert result.score == 1.0


def test_builtin_oneof() -> None:
    fn = get_comparator("oneof")
    result = fn("PVD", "PVD", {"values": ["PVD", "Sputtering"]})
    assert result.score == 1.0
