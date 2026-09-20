from typing import Any

import pytest

from json_extract_eval.core.transforms.registry import (
    TransformNotFoundError,
    _clear_registry,
    get_transform,
    register,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Reset custom registry before each test."""
    _clear_registry()


def test_builtins_registered() -> None:
    for name in ("lowercase", "strip", "normalize_whitespace", "sort_tokens", "round_digits"):
        fn = get_transform(name)
        assert callable(fn)


def test_get_unknown_raises() -> None:
    with pytest.raises(TransformNotFoundError, match="Unknown transform"):
        get_transform("nonexistent")


def test_register_custom() -> None:
    def my_transform(value: Any, params: dict[str, Any]) -> Any:
        return value

    register("custom", my_transform)
    assert get_transform("custom") is my_transform


def test_duplicate_registration_raises() -> None:
    def dummy(value: Any, params: dict[str, Any]) -> Any:
        return value

    register("dummy", dummy)
    with pytest.raises(ValueError, match="already registered"):
        register("dummy", dummy)


def test_builtin_override_raises() -> None:
    def fake(value: Any, params: dict[str, Any]) -> Any:
        return value

    with pytest.raises(ValueError, match="already registered"):
        register("lowercase", fake)


def test_register_non_callable_raises() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        register("bad", "not a function")  # type: ignore[arg-type]


def test_builtin_lowercase() -> None:
    fn = get_transform("lowercase")
    assert fn("HELLO", {}) == "hello"


def test_builtin_strip() -> None:
    fn = get_transform("strip")
    assert fn("  hello  ", {}) == "hello"


def test_builtin_round_digits() -> None:
    fn = get_transform("round_digits")
    assert fn(3.14159, {"digits": 2}) == 3.14
