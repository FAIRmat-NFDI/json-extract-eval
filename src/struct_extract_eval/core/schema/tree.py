"""SchemaNode tree and parse_eval_schema.

Parses an eval schema (resolved schema + x-eval-* extensions) into a
SchemaNode tree. All downstream scoring code works with SchemaNode.

Call annotate_xeval() to get eval schema -- parse_eval_schema
does not assign comparator defaults for leaf nodes, it validates
and parses. Container nodes (objects/arrays) get a placeholder
comparator since they are scored via their children, not directly.
"""

import logging
from dataclasses import dataclass, field

from struct_extract_eval.core.comparators.comparator import ComparatorSpec
from struct_extract_eval.core.comparators.registry import (
    ComparatorNotFoundError,
    get_comparator,
)
from struct_extract_eval.core.schema.utils import (
    get_children,
    is_leaf,
    non_null_types,
    resolve_type,
)
from struct_extract_eval.core.schema.xeval import parse_xeval_entry
from struct_extract_eval.core.transforms.registry import (
    TransformNotFoundError,
    get_transform,
)
from struct_extract_eval.core.transforms.transform import TransformSpec

logger = logging.getLogger(__name__)

_KNOWN_XEVAL_KEYS = frozenset(
    {
        "x-eval-compare",
        "x-eval-transform",
        "x-eval-skip",
        "x-eval-align",
    }
)


class SchemaError(Exception):
    """Raised when an eval schema is invalid or contains bad x-eval-* config."""

    def __init__(self, message: str, path: str = "") -> None:
        self.path = path
        full = f"{path}: {message}" if path else message
        super().__init__(full)


@dataclass
class SchemaNode:
    """A single node in the parsed evaluation schema tree."""

    path: str  # field path
    json_type: str
    # Key of this node in its parent: property name for object children,
    # "[]" for array items, "" for the root.
    name: str = ""
    comparator: ComparatorSpec = field(default_factory=ComparatorSpec)
    children: list["SchemaNode"] = field(default_factory=list)
    skip: bool = False
    transforms: list[TransformSpec] = field(default_factory=list)
    # Array-only. None for leaf and object nodes.
    align: dict[str, object] | None = None
    # Set when `type` was a list of >= 2 non-null types (json_type is MULTI_TYPE).
    # The non-null types the value may take. None otherwise.
    allowed_types: list[str] | None = None


def _validate_xeval(schema: dict[str, object], path: str) -> None:
    """Validate x-eval-* keys at parse time. Raises SchemaError on invalid config."""
    for key in schema:
        if not isinstance(key, str):
            raise SchemaError(
                f"Schema keys must be strings, got {type(key).__name__!r}",
                path,
            )
        if key.startswith("x-eval-") and key not in _KNOWN_XEVAL_KEYS:
            logger.warning("Unknown x-eval key '%s' at path '%s'", key, path)

    if "x-eval-skip" in schema and not isinstance(schema["x-eval-skip"], bool):
        raise SchemaError("x-eval-skip must be a boolean", path)

    if "x-eval-align" in schema:
        if resolve_type(schema) != "array":
            raise SchemaError(
                "x-eval-align is only valid on array nodes, "
                f"but this node has type '{resolve_type(schema)}'",
                path,
            )
        raw_align = schema["x-eval-align"]
        if not isinstance(raw_align, dict):
            raise SchemaError("x-eval-align must be a dict", path)
        if "ordered" not in raw_align and "match_by" not in raw_align:
            raise SchemaError(
                "x-eval-align must have 'ordered' or 'match_by' key", path
            )
        if "ordered" in raw_align and not isinstance(raw_align["ordered"], bool):
            raise SchemaError(
                "x-eval-align 'ordered' must be a boolean", path
            )
        if raw_align.get("ordered") is False and "match_by" not in raw_align:
            raise SchemaError(
                "x-eval-align with ordered=false must specify 'match_by'",
                path,
            )
        match_by = raw_align.get("match_by")
        if match_by is not None:
            if not isinstance(match_by, str) or not match_by:
                raise SchemaError(
                    "x-eval-align 'match_by' must be a non-empty string",
                    path,
                )
            _VALID_MATCH_BY = {"key_field", "hungarian"}
            if match_by not in _VALID_MATCH_BY:
                raise SchemaError(
                    f"x-eval-align 'match_by' must be one of "
                    f"{sorted(_VALID_MATCH_BY)}, got {match_by!r}",
                    path,
                )
        if match_by == "key_field":
            if "key" not in raw_align:
                raise SchemaError(
                    "x-eval-align with match_by='key_field' "
                    "requires a 'key'",
                    path,
                )
            key_name = raw_align["key"]
            if not isinstance(key_name, str) or not key_name:
                raise SchemaError(
                    "x-eval-align 'key' must be a non-empty string",
                    path,
                )
            # Validate key field against items schema.
            # Errors for structurally impossible cases.
            # Warns when matching might work on data but schema is incomplete,
            # for example, array have object type, without items property,
            # match_by key comparator cannot check if the key exists.
            items_schema = schema.get("items")
            if not isinstance(items_schema, dict):
                raise SchemaError(
                    "x-eval-align with match_by='key_field' requires "
                    "'items' to be an object schema",
                    path,
                )
            items_type = resolve_type(items_schema)
            if items_type is not None and items_type != "object":
                raise SchemaError(
                    f"x-eval-align with match_by='key_field' requires "
                    f"items type 'object', got '{items_type}'",
                    path,
                )
            items_props = items_schema.get("properties")
            if not isinstance(items_props, dict):
                logger.warning(
                    "Path '%s': x-eval-align match_by='key_field' but items "
                    "has no 'properties'. Key-field matching will attempt to "
                    "match on data, but no per-field scoring is possible.",
                    path,
                )
            elif key_name not in items_props:
                logger.warning(
                    "Path '%s': x-eval-align key '%s' not found in items "
                    "properties %s. Key-field matching will attempt to match "
                    "on data anyway.",
                    path, key_name, sorted(items_props),
                )

    if "x-eval-compare" in schema:
        raw_compare = schema["x-eval-compare"]
        if isinstance(raw_compare, str):
            comparator_name = raw_compare
        elif isinstance(raw_compare, dict):
            if len(raw_compare) != 1:
                raise SchemaError(
                    "x-eval-compare object must have exactly one key", path
                )
            comparator_name, params = next(iter(raw_compare.items()))
            if not isinstance(params, dict):
                raise SchemaError(
                    f"params for '{comparator_name}' must be a dict, "
                    f"got {type(params).__name__}",
                    path,
                )
        else:
            raise SchemaError(
                "x-eval-compare must be a string or object", path
            )
        try:
            get_comparator(comparator_name)
        except ComparatorNotFoundError as err:
            raise SchemaError(f"Unknown comparator: '{comparator_name}'", path) from err

        # Warn about type-comparator mismatches for built-in comparators.
        # Custom comparators are not checked -- the user knows their intent.
        json_type = resolve_type(schema)
        if comparator_name == "numeric" and json_type in ("string", "boolean"):
            logger.warning(
                "Path '%s': 'numeric' comparator on '%s' field. "
                "numeric expects a number -- did you mean 'exact'?",
                path, json_type,
            )
        if comparator_name == "exact" and json_type in ("number", "integer"):
            logger.warning(
                "Path '%s': 'exact' comparator on '%s' field. "
                "exact requires identical type+value (e.g. 42 != 42.0). "
                "Consider 'numeric' for tolerance-based comparison.",
                path, json_type,
            )
    if "x-eval-transform" in schema:
        raw = schema["x-eval-transform"]
        if not isinstance(raw, list):
            raise SchemaError("x-eval-transform must be a list", path)
        for i, item in enumerate(raw):
            if not isinstance(item, (str, dict)):
                raise SchemaError(
                    f"x-eval-transform[{i}] must be a string or object",
                    path,
                )
            try:
                transform_name, _ = parse_xeval_entry(item)
            except (SchemaError, StopIteration, ValueError, TypeError) as err:
                raise SchemaError(
                    f"x-eval-transform[{i}]: {err}", path
                ) from err
            try:
                get_transform(transform_name)
            except TransformNotFoundError as err:
                raise SchemaError(f"Unknown transform: '{transform_name}'", path) from err


def _resolve_comparator_spec(
    schema: dict[str, object], path: str, required: bool = True
) -> ComparatorSpec:
    """Build a ComparatorSpec from x-eval-compare.

    Raises SchemaError if x-eval-compare is missing on a non-skip leaf node,
    unless ``required`` is False: a leaf below a node that has its own
    comparator is never compared, so it needs no comparator of its own.
    Non-leaf nodes without x-eval-compare get an empty placeholder
    since container nodes are scored via their children, not directly.
    Skip nodes without x-eval-compare get an empty placeholder since
    the comparator is never called.
    """
    if "x-eval-compare" not in schema:
        if required and is_leaf(schema) and not schema.get("x-eval-skip"):
            raise SchemaError(
                "missing x-eval-compare -- run annotate_xeval first", path
            )
        return ComparatorSpec()

    raw = schema["x-eval-compare"]
    try:
        name, params = parse_xeval_entry(raw)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"invalid x-eval-compare: {exc}", path) from exc
    return ComparatorSpec(name=name, params=params)


def _resolve_transform_specs(
    schema: dict[str, object], path: str
) -> list[TransformSpec]:
    """Build a list of TransformSpec from x-eval-transform.

    Returns an empty list if x-eval-transform is absent. Raises SchemaError
    if it is present but not a list. Entry shapes and name resolution have
    already been verified by _validate_xeval.
    """
    if "x-eval-transform" not in schema:
        return []
    raw = schema["x-eval-transform"]
    if not isinstance(raw, list):
        raise SchemaError("x-eval-transform must be a list", path)
    specs: list[TransformSpec] = []
    for entry in raw:
        name, params = parse_xeval_entry(entry)
        specs.append(TransformSpec(name=name, params=params))
    return specs


def _build_node(
    schema: dict[str, object], path: str, name: str = "", scored_by_ancestor: bool = False
) -> SchemaNode:
    """Recursively build a SchemaNode tree from a resolved eval schema.

    ``scored_by_ancestor`` is True below a node that has its own
    ``x-eval-compare``. Such a node is scored as one unit, so nothing under it
    is ever compared and its leaves are not required to carry a comparator.
    """
    _validate_xeval(schema, path)

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for property_name in properties:
            if isinstance(property_name, str) and "." in property_name:
                raise SchemaError(
                    f"property name {property_name!r} contains '.', which is reserved "
                    "as the path separator. Rename the key.",
                    path,
                )

    json_type = resolve_type(schema)
    if json_type is None:
        raise SchemaError("Missing or invalid 'type'", path)

    non_null = non_null_types(schema.get("type"))
    allowed_types: list[str] | None = non_null if len(non_null) >= 2 else None

    comparator = _resolve_comparator_spec(schema, path, required=not scored_by_ancestor)
    transforms = _resolve_transform_specs(schema, path)

    below_comparator = scored_by_ancestor or "x-eval-compare" in schema
    children = [
        _build_node(child_schema, child_path, child_name, below_comparator)
        for child_name, child_schema, child_path in get_children(schema, path)
    ]

    node = SchemaNode(
        path=path,
        json_type=json_type,
        name=name,
        comparator=comparator,
        children=children,
        transforms=transforms,
        allowed_types=allowed_types,
    )
    if schema.get("x-eval-skip"):
        node.skip = True
    if "x-eval-align" in schema:
        # Validation already confirmed it's a dict
        raw_align = schema["x-eval-align"]
        assert isinstance(raw_align, dict)  # for mypy
        node.align = raw_align
    return node


def parse_eval_schema(raw_schema: dict[str, object]) -> SchemaNode:
    """Parse an eval schema into a SchemaNode tree.

    Expects:
    - $ref and allOf resolved by the caller
    - x-eval-* defaults filled in by annotate_xeval

    Validates all x-eval-* keys at parse time: unknown comparators,
    unknown transforms, invalid types, bad x-eval-* config all raise
    SchemaError immediately. Does not assign defaults.
    """
    if not isinstance(raw_schema, dict):
        raise SchemaError("Eval schema must be an object")

    return _build_node(raw_schema, "")
