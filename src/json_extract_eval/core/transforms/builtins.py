import re
from typing import Any


def transform_lowercase(value: Any, params: dict[str, Any]) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"lowercase transform requires a string, got {type(value).__name__}")
    return value.lower()


def transform_strip(value: Any, params: dict[str, Any]) -> Any:
    """Strip leading/trailing whitespace."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"strip transform requires a string, got {type(value).__name__}")
    return value.strip()


def transform_normalize_whitespace(value: Any, params: dict[str, Any]) -> Any:
    """Collapse multiple spaces/newlines to a single space, strip leading/trailing whitespace."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"normalize_whitespace transform requires a string, got {type(value).__name__}"
        )
    return re.sub(r"\s+", " ", value).strip()


def transform_sort_tokens(value: Any, params: dict[str, Any]) -> Any:
    """Alphabetize whitespace-separated tokens."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"sort_tokens transform requires a string, got {type(value).__name__}"
        )
    return " ".join(sorted(value.split()))


def transform_round_digits(value: Any, params: dict[str, Any]) -> Any:
    """Round numeric value to N decimal places.

    Params: {"digits": int}
    """
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"round_digits transform requires a number, got {type(value).__name__}"
        )
    if "digits" not in params:
        raise TypeError("round_digits transform requires 'digits' parameter")
    digits = int(params["digits"])
    return round(value, digits)


_TRUTHY = frozenset({"1", "true"})
_FALSY = frozenset({"0", "false"})


def _convert_bool(value: Any) -> bool:
    """Parse a value to bool with strict validation.

    Accepts: bool, 0/1 (int), and case-insensitive strings
    "1"/"true" or "0"/"false".
    Raises TypeError for anything else.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in _TRUTHY:
            return True
        if lower in _FALSY:
            return False
    raise TypeError(
        f"Cannot convert {type(value).__name__} {value!r} to bool"
    )


_TYPE_CONVERTERS: dict[str, type] = {
    "float": float,
    "int": int,
    "str": str,
}

_VALID_TARGETS = frozenset({"float", "int", "str", "bool"})


def transform_type_convert(value: Any, params: dict[str, Any]) -> Any:
    """Convert value to the specified type.

    Params: {"to": "float" | "int" | "str" | "bool"}
    """
    if value is None:
        return None
    if "to" not in params:
        raise TypeError("type_convert transform requires 'to' parameter")
    target = params["to"]
    if target not in _VALID_TARGETS:
        raise ValueError(
            f"type_convert 'to' must be one of {sorted(_VALID_TARGETS)}, got {target!r}"
        )
    if target == "bool":
        return _convert_bool(value)
    converter = _TYPE_CONVERTERS[target]
    try:
        return converter(value)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"Cannot convert {type(value).__name__} {value!r} to {target}"
        ) from exc
