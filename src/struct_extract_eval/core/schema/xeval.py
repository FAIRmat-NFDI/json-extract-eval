"""x-eval-* utilities.

``annotate_xeval`` annotates a resolved schema in-place with ``x-eval-*``
defaults. Leaf fields without an existing ``x-eval-compare`` or
``x-eval-skip`` get a comparator from the internal type-defaults mapping.

Use ``set_type_default`` to change the default comparator for a JSON type.

``parse_xeval_entry`` is the shared parser for the two-shape rule used
by both ``x-eval-transform`` and ``x-eval-compare``.
"""

from struct_extract_eval.core.schema.utils import get_children, is_leaf, resolve_type

_BUILTIN_TYPE_DEFAULTS: dict[str, str | dict[str, object]] = {
    "string": "exact",
    "number": "numeric",
    "integer": "numeric",
    "boolean": "exact",
}


def set_type_default(
    json_type: str, comparator: str | dict[str, object],
) -> None:
    """Set the default comparator for a JSON type.

    Affects all subsequent ``annotate_xeval()`` calls. Persistent for
    the lifetime of the process.

    Args:
        json_type: JSON Schema type (e.g. ``"string"``, ``"number"``).
        comparator: Comparator name (string) or comparator with params
            (single-key dict). Uses the same two-shape rule as
            ``x-eval-compare``. Must be registered in the comparator
            registry before ``evaluate()`` is called.

    Examples::

        set_type_default("string", "semantic")
        set_type_default("number", {"numeric": {"tolerance": {"rel": 0.01}}})
        annotate_xeval(schema)
    """
    if not isinstance(json_type, str) or not json_type:
        raise ValueError(
            f"json_type must be a non-empty string, got {json_type!r}"
        )
    # Validate using the same two-shape rule as x-eval-compare
    if isinstance(comparator, str):
        if not comparator:
            raise ValueError("comparator must be a non-empty string")
    elif isinstance(comparator, dict):
        parse_xeval_entry(comparator)  # validates structure
    else:
        raise TypeError(
            f"comparator must be a string or single-key dict, "
            f"got {type(comparator).__name__}"
        )
    _BUILTIN_TYPE_DEFAULTS[json_type] = comparator


def reset_type_defaults() -> None:
    """Reset the type-defaults mapping to the built-in defaults.

    Undoes all ``set_type_default()`` calls. Useful in tests or when
    switching between configurations in the same process.
    """
    _BUILTIN_TYPE_DEFAULTS.clear()
    _BUILTIN_TYPE_DEFAULTS.update({
        "string": "exact",
        "number": "numeric",
        "integer": "numeric",
        "boolean": "exact",
    })



def parse_xeval_entry(entry: str | dict[str, object]) -> tuple[str, dict[str, object]]:
    """Parse the two-shape config rule into ``(function name, function params)``.

    String form: ``"exact"`` -> ``("exact", {})``.
    Object form: ``{"numeric": {"tolerance": ...}}`` -> ``("numeric", {"tolerance": ...})``.

    Raises ``TypeError`` for invalid types, ``ValueError`` for bad structure.
    """
    if isinstance(entry, str):
        if not entry:
            raise ValueError("Config name must be a non-empty string")
        return entry, {}
    if isinstance(entry, dict):
        if len(entry) != 1:
            raise ValueError(
                f"Config object must have exactly one key, got {len(entry)}: {list(entry)}"
            )
        (name, params), = entry.items()
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"Config key must be a non-empty string, got {type(name).__name__}: {name!r}"
            )
        if not isinstance(params, dict):
            raise ValueError(
                f"Params for '{name}' must be a dict, got {type(params).__name__}"
            )
        return name, params
    raise TypeError(
        f"Config must be a string or single-key dict, got {type(entry).__name__}"
    )


def annotate_xeval(schema: dict[str, object]) -> dict[str, object]:
    """Annotate a resolved schema in-place with ``x-eval-*`` defaults.

    Walks the schema tree. For each leaf node without an explicit
    ``x-eval-compare`` or ``x-eval-skip``, assigns a comparator from
    the internal type-defaults mapping. Use ``set_type_default()`` to
    customize the mapping before calling.

    Other keys in the schema (e.g. ``required``, ``type``, ``properties``)
    are left untouched.

    Returns:
        The schema for convenience (same object, mutated in-place).
    """
    _annotate_node(schema)
    return schema


def _annotate_node(schema: dict[str, object]) -> None:
    """Recursively annotate a single schema node."""
    if is_leaf(schema):
        if "x-eval-compare" not in schema and "x-eval-skip" not in schema:
            json_type = resolve_type(schema)
            comparator = _BUILTIN_TYPE_DEFAULTS.get(json_type or "", "exact")
            schema["x-eval-compare"] = comparator
        return

    # A container with its own x-eval-compare is scored as one unit. Its
    # children are never compared, so do not give them defaults.
    if "x-eval-compare" in schema:
        return

    # Container node (object or array): recurse into children.
    for _field_name, child_schema, _child_path in get_children(schema):
        _annotate_node(child_schema)
