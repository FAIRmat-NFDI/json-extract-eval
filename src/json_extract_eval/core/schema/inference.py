import json
import logging
from copy import deepcopy
from typing import Any

import jsonref

logger = logging.getLogger(__name__)


def _json_type(value: object) -> str:
    """JSON Schema type name for a value.

    ``bool`` is checked before ``int`` because ``bool`` is an ``int`` subclass
    in Python. Any value whose type isn't a JSON type falls back to
    ``"string"`` (a safe default for the inferred schema).
    """
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"  # non-JSON type: safe default for the inferred schema


def _type_family(value: object) -> str:
    """Coarse type family for polymorphism detection. Json schem's number
     can be any numeric value, including decimals

    Like :func:`_json_type` but collapses ``integer`` and ``number`` into one
    family: the numeric comparator already treats ``5`` and ``5.0`` as equal,
    so a field holding both is not considered polymorphic.
    """
    json_type = _json_type(value)
    return "number" if json_type in ("integer", "number") else json_type


def infer_schema(values: list[object], path: str = "") -> dict[str, object]:
    """Infer a resolved schema from a list of values observed at one position.

    ``values`` are the values seen at the same place in the data. This function
    is recursive, so what "one position" means depends on depth:

    - top level: ``values`` is the whole record list (each value is a dict)
    - inside an object: ``values`` is one field's value across every record
      (e.g. field ``temp`` -> ``[1.5, 3.2, None]``)
    - inside an array: ``values`` is every element pooled across all the arrays
      seen at that position

    At every level each value may be any JSON type (str, int, float, bool,
    list, dict, None). Structure is merged across values, so objects capture
    the union of all keys seen.

    All-null positions default to ``{"type": "string"}``.

    When a position is polymorphic -- its values span more than one JSON type
    family (e.g. sometimes a string, sometimes an object) -- the result is a
    list-valued type such as ``{"type": ["object", "string"]}``, and a warning
    is logged. This is the same form ``resolve_schema_references`` produces
    when it collapses an ``anyOf`` of typed branches. A multi-type node is
    scored as one unit by its comparator, never structurally, so the inner
    ``properties`` / ``items`` of the object or array shapes are deliberately
    not included. Assign an explicit ``x-eval-compare`` that understands all
    the shapes (the default is ``exact``).

    Otherwise the inferred type comes from the values' single type family.
    """
    if not values:
        raise ValueError("infer_schema requires at least one value")

    present_values = [value for value in values if value is not None]
    if not present_values:
        return {"type": "string"}

    families = sorted({_type_family(value) for value in present_values})
    if len(families) > 1:
        logger.warning(
            "Polymorphic field at '%s': observed multiple JSON types %s across "
            "records. Emitting a multi-type schema; it is scored as one unit by "
            "its comparator (default exact). Assign an x-eval-compare that "
            "handles all the shapes.",
            path or "<root>", families,
        )
        # Sorted, so the result does not depend on record order. No
        # properties/items: a multi-type node is never scored structurally
        # (get_children returns nothing for it), and this matches what
        # resolve_schema_references emits when it collapses an anyOf.
        return {"type": list(families)}

    # The first non-null value decides the inferred type for this position.
    first_type = _json_type(present_values[0])
    if first_type == "array":
        # Array position: pool every element from all the arrays seen here into
        # one flat list, then infer a single items schema from the combined
        # elements.
        # e.g. record 1 has tags=["a","b"], record 2 has tags=["c"]
        #   -> pooled_elements = ["a", "b", "c"]
        #   -> items_schema = {"type": "string"}
        pooled_elements: list[object] = [
            element
            for array in present_values
            if isinstance(array, list)
            for element in array
        ]
        items_schema = (
            infer_schema(pooled_elements, f"{path}[]")
            if pooled_elements
            else {"type": "string"}
        )
        return {"type": "array", "items": items_schema}
    if first_type == "object":
        #   present_values = [{"name": "A", "value": 1.5},
        #                     {"name": "B", "value": 3.2, "unit": "nm"}]

        # Step 1: Keep only dicts (filter out non-dict values if mixed types).
        #   -> object_values = [{"name": "A", "value": 1.5},
        #                       {"name": "B", "value": 3.2, "unit": "nm"}]
        object_values = [value for value in present_values if isinstance(value, dict)]

        # Step 2: Union of all keys across all records.
        #   record 0 keys: {"name", "value"}
        #   record 1 keys: {"name", "value", "unit"}
        #   -> all_keys = {"name", "value", "unit"}
        all_keys: set[str] = set()
        for record in object_values:
            all_keys.update(record.keys())

        # Step 3: For each key, collect that field's value from every record and
        # recurse. record.get(key) returns None when the key is absent.
        #
        #   key="name":  ["A", "B"]      -> infer_schema(...) -> {"type": "string"}
        #   key="unit":  [None, "nm"]    -> infer_schema(...) -> {"type": "string"}
        #   key="value": [1.5, 3.2]      -> infer_schema(...) -> {"type": "number"}
        properties: dict[str, object] = {}
        for key in sorted(all_keys):
            field_values: list[object] = [record.get(key) for record in object_values]
            child_path = f"{path}.{key}" if path else key
            properties[key] = infer_schema(field_values, child_path)

        return {"type": "object", "properties": properties}
    # Scalar position (boolean, integer, number, string -- and "string" for any
    # non-JSON type, per _json_type's fallback).
    return {"type": first_type}


def merge_all_of(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merges 'allOf' lists into a single dictionary.
    """
    if not isinstance(schema, dict):
        return schema

    if "allOf" in schema:
        all_of_list = schema.pop("allOf")
        for subschema in all_of_list:
            # Recursively merge the subschema first
            merged_sub = merge_all_of(subschema)
            # Update the base schema with subschema properties
            for key, value in merged_sub.items():
                if key == "properties":
                    schema_properties = schema.get("properties", {})
                    for ik, iv in value.items():
                        # skip for overriden values
                        if ik not in schema_properties:
                            schema_properties.update({ik: iv})
                    schema["properties"] = schema_properties
                elif key not in schema:
                    schema[key] = value
    if "properties" in schema:
        for k, v in schema["properties"].items():
            schema["properties"][k] = merge_all_of(v)

    if "items" in schema:
        schema["items"] = merge_all_of(schema["items"])

    return schema


def remove_null_anyof(schema: dict[str, Any] | list[Any]) -> Any:
    """Recursively removes {'type': 'null'} from anyOf lists"""
    if isinstance(schema, dict):
        if "anyOf" in schema:
            anyOf = remove_null_anyof(
                [
                    i
                    for i in schema.pop("anyOf", [])
                    if i != {"type": "null"} and i != {"type": None}
                ]
            )
            if len(anyOf) == 1:
                schema.update(anyOf[0])
            else:
                schema["anyOf"] = anyOf
        return {k: remove_null_anyof(v) for k, v in schema.items()}
    elif isinstance(schema, list):
        return [remove_null_anyof(i) for i in schema]
    return schema


# Branch keys that are safe to lose when an anyOf collapses to a list-valued
# type -- they carry no structure or constraints.
_COLLAPSE_IGNORED_KEYS = frozenset({"type", "description", "title"})


def _collapse_to_type_list(branches: list[Any], path: str) -> list[str] | None:
    """Reduce anyOf branches to a flat list of JSON type names, if possible.

    Returns None when a branch has no usable ``type`` (e.g. an unresolved
    ``$ref`` or a bare ``enum``) or when fewer than two distinct non-null
    types remain (e.g. two object shapes) -- the caller keeps the anyOf then.
    """
    types: list[str] = []
    for branch in branches:
        if not isinstance(branch, dict):
            return None
        branch_type = branch.get("type")
        if isinstance(branch_type, str):
            branch_types = [branch_type]
        elif isinstance(branch_type, list) and all(
            isinstance(t, str) for t in branch_type
        ):
            branch_types = branch_type
        else:
            return None
        types.extend(t for t in branch_types if t != "null")

    deduped = list(dict.fromkeys(types))
    if len(deduped) < 2:
        return None

    dropped_keys = sorted(
        {
            key
            for branch in branches
            for key in branch
            if key not in _COLLAPSE_IGNORED_KEYS
        }
    )
    if dropped_keys:
        logger.warning(
            "anyOf at '%s' collapsed to multi-type %s; branch keys %s were "
            "dropped. The field is scored as one unit by its comparator "
            "(assign x-eval-compare), not structurally.",
            path or "<root>", deduped, dropped_keys,
        )
    return deduped


def collapse_multi_type_anyof(
    schema: dict[str, Any] | list[Any], path: str = "",
) -> Any:
    """Recursively collapse anyOf lists of typed branches to a list-valued type.

    ``anyOf: [{"type": "string"}, {"type": "number"}]`` becomes
    ``type: ["string", "number"]`` -- a comparator-owned multi-type node
    (see issue #82). Branch keys beyond ``type`` (``properties``, ``enum``,
    ...) are dropped with a warning: a multi-type node is scored as one unit
    by its comparator, not structurally.

    The anyOf is kept as-is when a branch has no usable ``type`` (e.g. an
    unresolved ``$ref``) or when fewer than two distinct non-null types
    remain (e.g. two object shapes) -- so run this only after ``$ref`` and
    ``allOf`` resolution, when branches carry their real types.

    ``path`` labels the node in warnings.
    """
    if isinstance(schema, dict):
        anyOf = schema.get("anyOf")
        if isinstance(anyOf, list):
            collapsed = _collapse_to_type_list(anyOf, path)
            if collapsed is not None:
                schema.pop("anyOf")
                schema["type"] = collapsed
        return {
            k: collapse_multi_type_anyof(v, f"{path}.{k}" if path else k)
            for k, v in schema.items()
        }
    elif isinstance(schema, list):
        return [
            collapse_multi_type_anyof(i, f"{path}[{index}]")
            for index, i in enumerate(schema)
        ]
    return schema


def resolve_schema_references(schema: dict[str, Any]) -> Any:
    """Resolve a JSON schema into a simplified form for evaluation.

    Replaces ``$ref`` references, merges ``allOf`` lists, simplifies
    ``anyOf`` lists (null branches removed; multiple non-null typed branches
    collapse to a list-valued ``type``), and drops ``$defs``.

    Warns about JSON Schema keywords that are not handled:

    - ``oneOf`` -- requires choosing one branch (not the same as anyOf)
    - ``if``/``then``/``else`` -- conditional subschemas
    - Constraint keywords (``enum``, ``const``, ``default``, ``minLength``,
      ``maxLength``, ``minimum``, ``maximum``, ``exclusiveMinimum``,
      ``exclusiveMaximum``, ``pattern``, ``format``) -- left in the
      schema but ignored by the evaluator
    """
    logger.warning(_RESOLVE_WARNING)
    schema = deepcopy(schema)
    schema = remove_null_anyof(schema)
    schema = dict(jsonref.replace_refs(schema, jsonschema=True, proxies=False))
    schema = merge_all_of(schema)
    # After ref/allOf resolution: every anyOf branch that can carry a `type`
    # now does (a branch that was `{"$ref": ...}` had none before), so the
    # remaining anyOf lists can collapse to a list-valued `type`.
    schema = collapse_multi_type_anyof(schema)
    schema.pop("$defs", None)
    return json.loads(json.dumps(schema))


_RESOLVE_WARNING = """\
resolve_schema_references handles $ref, allOf, anyOf[type, null], and anyOf
with multiple non-null types (collapsed to a list-valued `type` -- a
comparator-owned multi-type field; branch properties/items are dropped).
The following are NOT handled:
  - oneOf: type info lost, field has no 'type' key -> SchemaError at parse time
  - anyOf whose branches share one type (e.g. two object shapes) or lack a
    'type' key: anyOf is kept -> SchemaError at parse time
  - if/then/else: conditional properties lost -- may cause SchemaError or silently miss fields
For schemas with these keywords, use infer_schema(instances) instead."""
