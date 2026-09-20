import math
from typing import Any

import pytest

from json_extract_eval.core.transforms.builtins import (
    transform_lowercase,
    transform_normalize_whitespace,
    transform_round_digits,
    transform_sort_tokens,
    transform_strip,
    transform_type_convert,
)
from json_extract_eval.core.transforms.transform import Transform

# --- lowercase ---


def test_lowercase_basic() -> None:
    assert transform_lowercase("HELLO", {}) == "hello"


def test_lowercase_mixed() -> None:
    assert transform_lowercase("Hello World", {}) == "hello world"


def test_lowercase_already_lower() -> None:
    assert transform_lowercase("hello", {}) == "hello"


def test_lowercase_empty() -> None:
    assert transform_lowercase("", {}) == ""


def test_lowercase_non_string_raises() -> None:
    with pytest.raises(TypeError, match="requires a string"):
        transform_lowercase(42, {})


# --- strip ---


def test_strip_basic() -> None:
    assert transform_strip("  hello  ", {}) == "hello"


def test_strip_tabs_newlines() -> None:
    assert transform_strip("\t\nhello\n\t", {}) == "hello"


def test_strip_no_whitespace() -> None:
    assert transform_strip("hello", {}) == "hello"


def test_strip_empty() -> None:
    assert transform_strip("", {}) == ""


def test_strip_non_string_raises() -> None:
    with pytest.raises(TypeError, match="requires a string"):
        transform_strip(42, {})


# --- normalize_whitespace ---


def test_normalize_whitespace_multiple_spaces() -> None:
    assert transform_normalize_whitespace("hello   world", {}) == "hello world"


def test_normalize_whitespace_newlines() -> None:
    assert transform_normalize_whitespace("hello\n\nworld", {}) == "hello world"


def test_normalize_whitespace_tabs_and_spaces() -> None:
    assert transform_normalize_whitespace("hello\t  \nworld", {}) == "hello world"


def test_normalize_whitespace_leading_trailing() -> None:
    assert transform_normalize_whitespace("  hello  world  ", {}) == "hello world"


def test_normalize_whitespace_empty() -> None:
    assert transform_normalize_whitespace("", {}) == ""


def test_normalize_whitespace_non_string_raises() -> None:
    with pytest.raises(TypeError, match="requires a string"):
        transform_normalize_whitespace(42, {})


# --- sort_tokens ---


def test_sort_tokens_basic() -> None:
    assert transform_sort_tokens("banana apple cherry", {}) == "apple banana cherry"


def test_sort_tokens_already_sorted() -> None:
    assert transform_sort_tokens("a b c", {}) == "a b c"


def test_sort_tokens_single() -> None:
    assert transform_sort_tokens("hello", {}) == "hello"


def test_sort_tokens_empty() -> None:
    assert transform_sort_tokens("", {}) == ""


def test_sort_tokens_non_string_raises() -> None:
    with pytest.raises(TypeError, match="requires a string"):
        transform_sort_tokens(42, {})


# --- round_digits ---


def test_round_digits_basic() -> None:
    assert transform_round_digits(3.14159, {"digits": 2}) == 3.14


def test_round_digits_digits_param() -> None:
    assert transform_round_digits(3.14159, {"digits": 3}) == 3.142


def test_round_digits_zero() -> None:
    assert transform_round_digits(3.7, {"digits": 0}) == 4.0


def test_round_digits_integer_input() -> None:
    assert transform_round_digits(42, {"digits": 2}) == 42


def test_round_digits_negative() -> None:
    assert transform_round_digits(1234.5, {"digits": -2}) == 1200.0


def test_round_digits_non_number_raises() -> None:
    with pytest.raises(TypeError, match="requires a number"):
        transform_round_digits("3.14", {"digits": 2})


def test_round_digits_missing_param_raises() -> None:
    with pytest.raises(TypeError, match="requires 'digits'"):
        transform_round_digits(3.14, {})


def test_round_digits_nan() -> None:
    result = transform_round_digits(float("nan"), {"digits": 2})
    assert math.isnan(result)


# --- type_convert ---


def test_type_convert_str_to_float() -> None:
    assert transform_type_convert("3.14", {"to": "float"}) == 3.14


def test_type_convert_str_to_int() -> None:
    assert transform_type_convert("42", {"to": "int"}) == 42


def test_type_convert_int_to_str() -> None:
    assert transform_type_convert(42, {"to": "str"}) == "42"


def test_type_convert_float_to_int() -> None:
    assert transform_type_convert(3.0, {"to": "int"}) == 3


def test_type_convert_str_to_bool_truthy() -> None:
    assert transform_type_convert("1", {"to": "bool"}) is True
    assert transform_type_convert("true", {"to": "bool"}) is True
    assert transform_type_convert("True", {"to": "bool"}) is True
    assert transform_type_convert("TRUE", {"to": "bool"}) is True


def test_type_convert_str_to_bool_falsy() -> None:
    assert transform_type_convert("0", {"to": "bool"}) is False
    assert transform_type_convert("false", {"to": "bool"}) is False
    assert transform_type_convert("False", {"to": "bool"}) is False
    assert transform_type_convert("FALSE", {"to": "bool"}) is False


def test_type_convert_int_to_bool() -> None:
    assert transform_type_convert(1, {"to": "bool"}) is True
    assert transform_type_convert(0, {"to": "bool"}) is False


def test_type_convert_bool_to_bool() -> None:
    assert transform_type_convert(True, {"to": "bool"}) is True
    assert transform_type_convert(False, {"to": "bool"}) is False


def test_type_convert_invalid_bool_raises() -> None:
    with pytest.raises(TypeError, match="Cannot convert"):
        transform_type_convert("not_a_bool", {"to": "bool"})
    with pytest.raises(TypeError, match="Cannot convert"):
        transform_type_convert("yes", {"to": "bool"})
    with pytest.raises(TypeError, match="Cannot convert"):
        transform_type_convert("2", {"to": "bool"})
    with pytest.raises(TypeError, match="Cannot convert"):
        transform_type_convert(42, {"to": "bool"})


def test_type_convert_int_to_float() -> None:
    assert transform_type_convert(300, {"to": "float"}) == 300.0


def test_type_convert_missing_param_raises() -> None:
    with pytest.raises(TypeError, match="requires 'to'"):
        transform_type_convert("42", {})


def test_type_convert_invalid_target_raises() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        transform_type_convert("42", {"to": "complex"})


def test_type_convert_unconvertible_raises() -> None:
    with pytest.raises(TypeError, match="Cannot convert"):
        transform_type_convert("not_a_number", {"to": "float"})


# --- None handling ---
# Built-in transforms are None-safe: they no-op on None (return None) instead
# of raising, so a sometimes-null field with a string/number transform does not
# crash the scoring chain. A non-None wrong type still raises.


@pytest.mark.parametrize(
    "transform, params",
    [
        (transform_lowercase, {}),
        (transform_strip, {}),
        (transform_normalize_whitespace, {}),
        (transform_sort_tokens, {}),
        (transform_round_digits, {"digits": 2}),
        (transform_type_convert, {"to": "float"}),
    ],
)
def test_builtin_transform_none_is_noop(transform: Transform, params: dict[str, Any]) -> None:
    assert transform(None, params) is None


def test_apply_transforms_passes_none_to_custom_transform() -> None:
    """_apply_transforms must hand None to transforms so they can rewrite it."""
    from json_extract_eval.core.scoring import _apply_transforms
    from json_extract_eval.core.transforms.registry import _clear_registry, register
    from json_extract_eval.core.transforms.transform import TransformSpec

    def none_to_empty(value: object, params: dict[str, Any]) -> object:
        return "" if value is None else value

    _clear_registry()  # hermetic: ensure no leftover registration collides
    try:
        register("none_to_empty", none_to_empty)
        result = _apply_transforms(None, [TransformSpec(name="none_to_empty", params={})])
        assert result == ""
    finally:
        _clear_registry()
