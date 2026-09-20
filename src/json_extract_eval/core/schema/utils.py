"""
JSON Schema traversal utilities.

Assumes the schema is already resolved -- no $ref, allOf, anyOf, oneOf,
if/then/else, or other composition keywords. Only type, properties, items,
required, and x-eval-* keys are expected. Schemas should be pre-resolved
before using these utilities.

No x-eval-* knowledge -- these functions operate on structure only.
"""

import json
from collections.abc import Callable, Iterator
from pathlib import Path


def load_schema(path: str | Path) -> dict[str, object]:
    """Load a resolved JSON Schema from a file.

    Raises ``ValueError`` if the file does not contain a JSON object.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Schema must be a JSON object, got {type(raw).__name__}")
    return raw


MULTI_TYPE = "multi"


def non_null_types(type_value: object) -> list[str]:
    """Non-null type strings from a list-valued `type` (drops "null")."""
    if not isinstance(type_value, list):
        return []
    return [t for t in type_value if isinstance(t, str) and t != "null"]


def resolve_type(schema: dict[str, object]) -> str | None:
    """Return the effective JSON Schema type, or None if absent.

    JSON Schema allows `type` to be a list. We reduce it:
    - drop "null" (nullable is handled by value presence, not type)
    - exactly one type left -> that type (e.g. ["string", "null"] -> "string")
    - two or more left -> ``MULTI_TYPE`` (a comparator-owned multi-type node;
      see ``get_children`` and ``SchemaNode.allowed_types``)
    """
    t = schema.get("type")
    if isinstance(t, str):
        return t
    if isinstance(t, list):
        non_null = non_null_types(t)
        if len(non_null) == 1:
            return non_null[0]
        if len(non_null) >= 2:
            return MULTI_TYPE
    return None


def is_leaf(schema: dict[str, object]) -> bool:
    """True if the schema node has no children to recurse into.
    """
    return not get_children(schema)


def get_children(
    schema: dict[str, object],
    path: str = "",
) -> list[tuple[str, dict[str, object], str]]:
    """Return immediate children of a resolved schema's node given a path.

    get ``schema``'s children. ``path`` is the path of ``schema`` itself.
    Returns a list of ``(field_name, child_schema, child_path)`` tuples.

    - For objects: one entry per property. ``field_name`` is the property key.
    - For arrays: a single entry for ``items``. ``field_name`` is ``"[]"``.
    - For leaves: empty list.
    - For a multi-type node (``type`` is a list of >= 2 non-null types): empty
      list -- it is scored as one unit by its comparator, not structurally,
      even if it also declares ``properties``/``items``.
    """
    if len(non_null_types(schema.get("type"))) >= 2:
        return []

    children: list[tuple[str, dict[str, object], str]] = []

    props = schema.get("properties")
    if isinstance(props, dict):
        for name, prop_schema in props.items():
            if isinstance(prop_schema, dict):
                child_path = f"{path}.{name}" if path else name
                children.append((name, prop_schema, child_path))

    items = schema.get("items")
    if isinstance(items, dict) and not children:
        child_path = f"{path}[]"
        # "[]" is a special field name for array items, to distinguish
        # from object properties named "items"
        # do not add . between path and [] like child_path = f"{path}.[]" if path else "[]" to distinguish field name called "[]"
        children.append(("[]", items, child_path))

    return children


def iter_schema(
    schema: dict[str, object],
    path: str = "",
) -> Iterator[tuple[dict[str, object], str]]:
    """Yield ``(node_schema, path)`` tuples in pre-order depth-first order."""
    yield schema, path
    for _name, child_schema, child_path in get_children(schema, path):
        yield from iter_schema(child_schema, child_path)


def get_leaf_paths(schema: dict[str, object]) -> list[str]:
    """Return paths of all terminal (leaf) fields in the schema."""
    return [path for node, path in iter_schema(schema) if is_leaf(node)]


def get_node_at_path(
    schema: dict[str, object],
    path: str,
) -> dict[str, object] | None:
    """Return the schema node at a dot-notation path, e.g. ``'steps[].name'``.

    Empty string returns the root node. Returns None if the path doesn't exist.
    """
    if not path:
        return schema
    parts = [p for p in path.replace("[]", ".[]").split(".") if p]
    node: dict[str, object] | None = schema
    for part in parts:
        if node is None:
            return None
        if part == "[]": # if field name is "[]", the path will be "[].", which means the part will not be "[]", items will not be confused with a field name called "[]"
            items = node.get("items")
            node = items if isinstance(items, dict) else None
        else:
            props = node.get("properties")
            node = props.get(part) if isinstance(props, dict) else None  # type: ignore[union-attr]
    return node
