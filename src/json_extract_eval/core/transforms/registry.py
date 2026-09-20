from json_extract_eval.core.transforms.builtins import (
    transform_lowercase,
    transform_normalize_whitespace,
    transform_round_digits,
    transform_sort_tokens,
    transform_strip,
    transform_type_convert,
)
from json_extract_eval.core.transforms.transform import Transform


class TransformNotFoundError(KeyError):
    """Raised when a transform name is not found in the registry."""


_BUILTIN_TRANSFORMS: dict[str, Transform] = {
    "lowercase": transform_lowercase,
    "normalize_whitespace": transform_normalize_whitespace,
    "round_digits": transform_round_digits,
    "sort_tokens": transform_sort_tokens,
    "strip": transform_strip,
    "type_convert": transform_type_convert,
}

_registry: dict[str, Transform] = {}


def register(name: str, fn: Transform) -> None:
    """Register a custom transform function under the given name.

    Raises ValueError if a transform with this name is already registered.
    Raises TypeError if fn is not callable.
    """
    if not callable(fn):
        raise TypeError(f"Transform must be callable, got {type(fn).__name__}")
    if name in _registry or name in _BUILTIN_TRANSFORMS:
        raise ValueError(f"Transform '{name}' is already registered")
    _registry[name] = fn


def get_transform(name: str) -> Transform:
    if name in _registry:
        return _registry[name]
    if name in _BUILTIN_TRANSFORMS:
        return _BUILTIN_TRANSFORMS[name]
    raise TransformNotFoundError(f"Unknown transform: '{name}'")


def _clear_registry() -> None:
    """Clear the custom registry. For testing only."""
    _registry.clear()
