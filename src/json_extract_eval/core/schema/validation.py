"""Gold data validation.

``validate_gold(gold, schema, ...)``
    Validates gold data against an eval schema. Raises ``GoldValidationError``
    on:
    - Extra fields in gold that are not in the schema (all gold fields must
      be defined in the eval schema)

    Warns about (does not raise):
    - Container type mismatches (a dict/list expected by the schema's
      ``json_type`` but gold has another type). ``json_type`` is only a hint --
      a field may be polymorphic -- so this is not treated as an error; the
      scorer compares the value as-is. See ``_score_node``.
    - Fields in the schema that are missing from a gold record (``warn_missing``)

Schema validation is handled by ``parse_eval_schema()`` in ``core/schema.py``.
``validate_gold`` calls it internally, so schema errors are caught automatically.

Run before evaluation so issues surface early.
"""

import logging

from json_extract_eval.core.schema.tree import SchemaNode, parse_eval_schema

logger = logging.getLogger(__name__)


def _value_type(value: object) -> str:
    """JSON Schema type name of a Python value (bool before int)."""
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
    return "null"


def _type_matches(actual: str, declared: str) -> bool:
    """Whether a value's type satisfies a declared JSON Schema type.

    ``integer`` and ``number`` are treated as interchangeable so a valid numeric
    value (e.g. ``5`` for a ``number`` field) is not flagged.
    """
    if actual == declared:
        return True
    return actual in ("integer", "number") and declared in ("integer", "number")


class GoldValidationError(Exception):
    """Raised when gold data has a structural issue."""

    def __init__(
        self, message: str, record_id: str | int = "", path: str = ""
    ) -> None:
        self.record_id = record_id
        self.path = path
        super().__init__(message)


def validate_gold(
    gold: list[dict[str, object]],
    schema: dict[str, object],
    id_field: str | None = None,
    warn_missing: bool = True,
    strict_types: bool = False,
    allow_extra_fields: bool = False,
) -> None:
    """Validate gold data against an eval schema.

    Raises ``GoldValidationError`` on:
    - Extra fields: field present in gold but not defined in schema. All gold
      fields must be in the eval schema. If a field shouldn't be scored, add
      it to the schema with ``x-eval-skip: true``.

    Warns about (does not raise):
    - Container type mismatch: gold isn't the dict/list the schema's
      ``json_type`` declares. ``json_type`` is a hint, not a constraint -- the
      field may be polymorphic -- so the scorer compares the value as-is
      (a type difference becomes a mismatch unless an ``x-eval-compare``
      comparator handles it). Fix the gold if it is actually malformed.
    - ``warn_missing``: field defined in schema but absent from a gold record.
      These fields will be ignored during scoring for that record (not an
      omission -- omission only happens when gold has a field and extracted
      doesn't).

    With large datasets or many optional fields, missing-field warnings can
    be verbose. Turn them off if needed::

        validate_gold(gold, schema, warn_missing=False)

    Args:
        gold: Gold (ground truth) instances.
        schema: Eval schema (resolved schema with x-eval-* annotations).
        id_field: Field name to use as record ID. Defaults to integer index.
        warn_missing: Warn when a schema field is absent from a gold record.
        strict_types: Off by default (type is a hint -- mismatches only warn).
            When True, every node's gold value must match the schema-declared
            type, or be one of the declared types when ``type`` is a list, or be
            ``null``; otherwise a ``GoldValidationError`` is raised. Use with
            hand-written schemas where the declared types are a real constraint;
            leave off for inferred schemas (where the type is only a hint) and
            for fields whose comparator deliberately coerces types.
        allow_extra_fields: When True, gold fields absent from the schema emit
            a warning instead of raising. Scoring reports them as skipped.

    Raises:
        GoldValidationError: if gold contains fields not defined in the schema
            and ``allow_extra_fields`` is False, or (when ``strict_types``) a
            gold value is not a declared type.
        SchemaError: if the schema itself is invalid (checked first).
    """
    tree = parse_eval_schema(schema)
    # id_field is a record identifier, not a scored field -- exclude it
    # from the extra-key check so it doesn't need to be in the schema.
    ignore_keys = {id_field} if id_field else set()
    for i, g in enumerate(gold):
        if id_field:
            if id_field not in g:
                raise GoldValidationError(
                    f"Record {i!r}: id field '{id_field}' is missing",
                    record_id=i,
                    path=id_field,
                )
            raw_id = g[id_field]
            if isinstance(raw_id, bool) or not isinstance(
                raw_id, (str, int)
            ):
                raise GoldValidationError(
                    f"Record {i!r}: id field '{id_field}' must be a "
                    f"string or integer, got {type(raw_id).__name__}",
                    record_id=i,
                    path=id_field,
                )
            record_id: str | int = raw_id
        else:
            record_id = i
        _validate_node(
            tree,
            g,
            record_id,
            warn_missing,
            ignore_keys,
            strict_types,
            allow_extra_fields,
        )


def _validate_node(
    node: SchemaNode,
    gold_value: object,
    record_id: str | int,
    warn_missing: bool,
    ignore_keys: set[str] | None = None,
    strict: bool = False,
    allow_extra_fields: bool = False,
) -> None:
    """Recursively validate a gold value against a schema node."""
    if gold_value is None:
        return
    # Strict mode: the gold value's type must match the schema-declared type
    # (or be one of them for a multi-type node). null is exempt (handled above).
    # Raises before the lenient warnings below, and applies to every node.
    if strict:
        actual = _value_type(gold_value)
        declared = (
            node.allowed_types if node.allowed_types is not None else [node.json_type]
        )
        if not any(_type_matches(actual, d) for d in declared):
            raise GoldValidationError(
                f"Record {record_id!r}: gold at '{node.path}' is {actual}, not "
                f"the schema-declared type {declared} (strict_types=True; only "
                f"the declared type(s) and null are allowed).",
                record_id=record_id,
                path=node.path,
            )
    # Multi-type node (`type` was a list of >= 2 non-null types): gold may be
    # any of the declared types. Warn only if it is none of them; the field's
    # comparator scores it as-is regardless. Checked before the comparator
    # short-circuit so the union check still runs (multi-type nodes carry a
    # default `exact` comparator).
    if node.allowed_types is not None:
        actual = _value_type(gold_value)
        if not any(_type_matches(actual, d) for d in node.allowed_types):
            logger.warning(
                "Record %r: gold at '%s' is %s, not one of the declared types "
                "%s. The field's comparator will compare it as-is.",
                record_id, node.path, actual, node.allowed_types,
            )
        return
    # A node with an explicit comparator owns its value's type: the comparator
    # handles any shape (see _score_node's escape hatch), so a polymorphic field
    # configured with x-eval-compare is intentional and must not be flagged
    # here. Leaves always carry a comparator and were never type-checked anyway.
    if node.comparator.name:
        return
    if node.json_type == "object" and node.children:
        if not isinstance(gold_value, dict):
            # json_type is a hint, not a constraint: the field may be
            # polymorphic, so this is a warning, not an error. The scorer
            # compares the value as-is. Can't validate children of a non-dict.
            logger.warning(
                "Record %r: gold at '%s' is %s, not the schema's 'object' "
                "type. Treated as a hint (the field may be polymorphic); the "
                "scorer compares it as-is. Add an custom x-eval-compare comparator on "
                "'%s', or fix the gold if it is malformed.",
                record_id, node.path, type(gold_value).__name__, node.path,
            )
            return

        schema_fields: set[str] = set()
        for child in node.children:
            field_name = child.name
            schema_fields.add(field_name)
            if field_name in gold_value:
                _validate_node(
                    child, gold_value[field_name], record_id,
                    warn_missing, strict=strict,
                    allow_extra_fields=allow_extra_fields,
                )
            elif warn_missing:
                logger.warning(
                    "Record %r: field '%s' is in schema but missing from "
                    "gold. It will not be scored for this record.",
                    record_id, child.path,
                )

        skip = ignore_keys or set()
        for key in gold_value:
            if isinstance(key, str) and "." in key:
                path = f"{node.path}.{key}" if node.path else key
                location = node.path or "<root>"
                raise GoldValidationError(
                    f"Record {record_id!r}: {location}: property name {key!r} "
                    "contains '.', which is reserved as the path separator. "
                    "Rename the key.",
                    record_id=record_id,
                    path=path,
                )
            if key not in schema_fields and key not in skip:
                path = f"{node.path}.{key}" if node.path else key
                if allow_extra_fields:
                    logger.warning(
                        "Record %r: gold field '%s' is not in the schema and "
                        "will be skipped during scoring.",
                        record_id,
                        path,
                    )
                    continue
                raise GoldValidationError(
                    f"Record {record_id!r}: field '{path}' is in gold but "
                    f"not in schema. All gold fields must be defined in the "
                    f"eval schema. If this field should not be scored, add it "
                    f"to the schema with x-eval-skip: true.",
                    record_id=record_id,
                    path=path,
                )

    elif node.json_type == "array" and node.children:
        if not isinstance(gold_value, list):
            # json_type is a hint, not a constraint (see the object branch).
            logger.warning(
                "Record %r: gold at '%s' is %s, not the schema's 'array' "
                "type. Treated as a hint (the field may be polymorphic); the "
                "scorer compares it as-is. Add an x-eval-compare comparator on "
                "'%s' to silence this, or fix the gold if it is malformed.",
                record_id, node.path, type(gold_value).__name__, node.path,
            )
            return
        items_node = node.children[0]
        for item in gold_value:
            _validate_node(
                items_node, item, record_id, warn_missing, strict=strict,
                allow_extra_fields=allow_extra_fields,
            )
