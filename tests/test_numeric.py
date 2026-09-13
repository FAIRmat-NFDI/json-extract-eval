import math

from struct_extract_eval.core.comparators.numeric import compare_numeric


def test_exact_match() -> None:
    result = compare_numeric(42.0, 42.0, {})
    assert result.score == 1.0
    assert result.comparator == "numeric"


def test_exact_mismatch() -> None:
    result = compare_numeric(42.0, 42.1, {})
    assert result.score == 0.0
    assert result.reason == "values differ"


def test_relative_tolerance_pass() -> None:
    result = compare_numeric(100.0, 100.5, {"tolerance": {"rel": 0.01}})
    assert result.score == 1.0


def test_relative_tolerance_fail() -> None:
    result = compare_numeric(100.0, 102.0, {"tolerance": {"rel": 0.01}})
    assert result.score == 0.0
    assert "relative error" in (result.reason or "")


def test_absolute_tolerance_pass() -> None:
    result = compare_numeric(10.0, 10.3, {"tolerance": {"abs": 0.5}})
    assert result.score == 1.0


def test_absolute_tolerance_fail() -> None:
    result = compare_numeric(10.0, 11.0, {"tolerance": {"abs": 0.5}})
    assert result.score == 0.0
    assert "absolute error" in (result.reason or "")


def test_type_error_string() -> None:
    result = compare_numeric(42.0, "not a number", {})
    assert result.score == 0.0
    assert result.reason == "type_error"


def test_type_error_none() -> None:
    result = compare_numeric(None, 42.0, {})
    assert result.score == 0.0
    assert result.reason == "type_error"


def test_nan_both() -> None:
    """Both NaN should match."""
    result = compare_numeric(math.nan, math.nan, {})
    assert result.score == 1.0


def test_nan_one_side() -> None:
    """One NaN, one number should be type_error."""
    result = compare_numeric(math.nan, 42.0, {})
    assert result.score == 0.0
    assert result.reason == "type_error"

    result = compare_numeric(42.0, math.nan, {})
    assert result.score == 0.0
    assert result.reason == "type_error"


def test_none_both() -> None:
    """Both None should match -- identical values, mirrors the NaN policy."""
    result = compare_numeric(None, None, {})
    assert result.score == 1.0


def test_none_one_side() -> None:
    """One None, one number should be type_error."""
    result = compare_numeric(None, 42.0, {})
    assert result.score == 0.0
    assert result.reason == "type_error"

    result = compare_numeric(42.0, None, {})
    assert result.score == 0.0
    assert result.reason == "type_error"


def test_string_numbers() -> None:
    """Numeric strings should be castable."""
    result = compare_numeric("42.5", "42.5", {})
    assert result.score == 1.0


def test_integer_inputs() -> None:
    result = compare_numeric(100, 101, {"tolerance": {"abs": 2}})
    assert result.score == 1.0


def test_zero_gold_relative() -> None:
    """When gold is 0, relative tolerance uses denom=1.0."""
    result = compare_numeric(0.0, 0.005, {"tolerance": {"rel": 0.01}})
    assert result.score == 1.0


def test_both_tolerances_both_pass() -> None:
    """Both rel and abs specified, both satisfied."""
    result = compare_numeric(100.0, 100.5, {"tolerance": {"rel": 0.01, "abs": 1.0}})
    assert result.score == 1.0


def test_both_tolerances_rel_fails() -> None:
    """Both specified, rel fails."""
    result = compare_numeric(100.0, 102.0, {"tolerance": {"rel": 0.01, "abs": 5.0}})
    assert result.score == 0.0
    assert "relative error" in (result.reason or "")


def test_both_tolerances_abs_fails() -> None:
    """Both specified, abs fails."""
    result = compare_numeric(100.0, 100.5, {"tolerance": {"rel": 0.01, "abs": 0.1}})
    assert result.score == 0.0
    assert "absolute error" in (result.reason or "")
