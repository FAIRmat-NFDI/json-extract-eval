from json_extract_eval.core.comparators.comparator import (
    BatchComparator,
    Comparator,
)
from json_extract_eval.core.comparators.exact import compare_exact
from json_extract_eval.core.comparators.numeric import compare_numeric
from json_extract_eval.core.comparators.oneof import compare_oneof


class ComparatorNotFoundError(KeyError):
    """Raised when a comparator name is not found in the registry."""


# Both Comparator and BatchComparator share the same registry. The dispatcher
# uses ``isinstance(fn, BatchComparator)`` to decide which call protocol to use.
_BUILTIN_COMPARATORS: dict[str, Comparator | BatchComparator] = {
    "exact": compare_exact,
    "numeric": compare_numeric,
    "oneof": compare_oneof,
}

# _registry is for registering custom comparators.
_registry: dict[str, Comparator | BatchComparator] = {}


def register(
    name: str, fn: Comparator | BatchComparator, *, overwrite: bool = False,
) -> None:
    """Register a custom comparator (per-field or batch) under the given name.

    Args:
        name: Name to register under (referenced by ``x-eval-compare``).
        fn: Callable comparator (per-field or batch).
        overwrite: If True, silently replace an existing custom registration.
            If False (default), raises ValueError on duplicate.
            Built-in comparators (exact, numeric, oneof) cannot be overwritten.

    Raises ValueError if a comparator with this name is already registered
    and overwrite is False, or if the name collides with a built-in.
    Raises TypeError if fn is not callable.
    """
    if not callable(fn):
        raise TypeError(f"Comparator must be callable, got {type(fn).__name__}")
    if name in _BUILTIN_COMPARATORS:
        raise ValueError(f"Cannot overwrite built-in comparator '{name}'")
    if not overwrite and name in _registry:
        raise ValueError(f"Comparator '{name}' is already registered")
    _registry[name] = fn


def get_comparator(name: str) -> Comparator | BatchComparator:
    if name in _registry:
        return _registry[name]
    if name in _BUILTIN_COMPARATORS:
        return _BUILTIN_COMPARATORS[name]
    raise ComparatorNotFoundError(f"Unknown comparator: '{name}'")


def _clear_registry() -> None:
    """Clear the custom registry. For testing only."""
    _registry.clear()
