"""Batch comparator dispatch.

After ``score_record`` finishes, some FieldResults have ``status="pending"``
(provisional placeholders for fields that use a BatchComparator).
``process_batches`` finds these, groups them by ``comparator`` name, looks up
the registered handler, and calls it once per group with the full list of items.

Comparator params (from ``x-eval-compare`` in the schema) are read from the
schema tree -- NOT stored on FieldResult. This keeps FieldResult a pure result
object with no config fields. The tree is passed as a parameter.

Each handler returns a positional list (one entry per input item, in order):
- ``ComparatorResult``: the handler decided this item -- score and status are
  applied to the corresponding FieldResult
- ``None``: the handler couldn't decide for this item -- the FieldResult gets
  ``status=batch_error``

Failure handling:
- Handler raises (whole batch) -> all items in that group get status=batch_error
- Handler returns non-list -> all items get batch_error
- Handler returns short list -> trailing items get batch_error
- Handler returns long list -> trimmed with a warning (trailing entries discarded)
- Per-item None -> only THAT item gets batch_error; others are unaffected
- Per-item non-ComparatorResult value -> THAT item gets batch_error

Batch errors are EXCLUDED from precision/recall/F1, like skipped fields.
"""

import logging
import re

from json_extract_eval.core.comparators.comparator import (
    BatchComparator,
    BatchItem,
    ComparatorResult,
)
from json_extract_eval.core.comparators.registry import get_comparator
from json_extract_eval.core.schema import SchemaNode
from json_extract_eval.core.field_result import FieldResult

logger = logging.getLogger(__name__)


def _to_schema_path(path: str) -> str:
    """Normalize an instance path back to a schema path for path_map lookup.

    ``"students[0].surname"`` -> ``"students[].surname"``
    ``"layers[2].steps[-1].name"`` -> ``"layers[].steps[].name"``
    """
    return re.sub(r"\[-?\d+\]", "[]", path)


def _build_path_map(tree: SchemaNode) -> dict[str, SchemaNode]:
    """Build a flat path -> SchemaNode lookup from the tree.

    Walks the tree recursively. Used by process_batches to look up
    comparator params without traversing the tree per field.
    """
    result: dict[str, SchemaNode] = {}

    def _walk(node: SchemaNode) -> None:
        if node.path:
            result[node.path] = node
        for child in node.children:
            _walk(child)

    _walk(tree)
    return result


def process_batches(
    field_results: list[FieldResult],
    tree: SchemaNode,
) -> list[FieldResult]:
    """Find pending batch fields, group by comparator name, dispatch to handlers.

    Mutates ``field_results`` in place AND returns it (for chaining). After this
    runs, no FieldResult should still have ``status="pending"``.

    The ``tree`` parameter provides access to comparator params (from
    ``x-eval-compare`` in the schema) so they can be passed through to
    ``BatchItem.params`` without storing config on FieldResult.
    """
    # Group pending fields by comparator name, preserving order within each group
    groups: dict[str, list[FieldResult]] = {}
    for r in field_results:
        if r.status == "pending":
            groups.setdefault(r.comparator, []).append(r)

    if not groups:
        return field_results

    # Build path -> node map once for all groups
    path_map = _build_path_map(tree)

    for name, results in groups.items():
        try:
            fn = get_comparator(name)
        except KeyError:
            logger.error(
                "No comparator registered for pending batch '%s' "
                "(at paths %s). Marking %d items as batch_error.",
                name, [r.path for r in results], len(results),
            )
            _mark_all_error(results)
            continue

        if not isinstance(fn, BatchComparator):
            logger.error(
                "Comparator '%s' does not inherit BatchComparator "
                "but %d fields were dispatched to it. Marking as batch_error.",
                name, len(results),
            )
            _mark_all_error(results)
            continue

        items = []
        for r in results:
            schema_path = _to_schema_path(r.path)
            node = path_map.get(schema_path)
            items.append(BatchItem(
                path=r.path,
                gold_raw=r.gold_value,
                extracted_raw=r.extracted_value,
                gold_compared=r.gold_compared,
                extracted_compared=r.extracted_compared,
                params=node.comparator.params if node else {},
            ))

        try:
            outputs = fn(items)
        except Exception as exc:
            logger.error(
                "Batch comparator '%s' raised for %d items at paths %s: %s. "
                "Marking all as batch_error.",
                name, len(items), [it.path for it in items], exc,
            )
            _mark_all_error(results)
            continue

        if not isinstance(outputs, list):
            logger.error(
                "Batch comparator '%s' returned %s instead of a list "
                "(expected %d results). Marking all as batch_error.",
                name, type(outputs).__name__, len(items),
            )
            _mark_all_error(results)
            continue

        if len(outputs) > len(items):
            logger.warning(
                "Batch comparator '%s' returned %d results for %d items; "
                "trimming extras.",
                name, len(outputs), len(items),
            )
            outputs = outputs[: len(items)]
        elif len(outputs) < len(items):
            logger.error(
                "Batch comparator '%s' returned %d results for %d items; "
                "missing trailing items marked batch_error.",
                name, len(outputs), len(items),
            )

        # Apply outputs back to FieldResults in order. Each output entry can be:
        # - ComparatorResult: success, apply score+status
        # - None: per-item failure, mark this field as batch_error (not the rest)
        # - anything else: invalid handler return type, mark as batch_error
        # field_result is updated in place, FieldResult must be mutable, (not frozen=True)
        for i, r in enumerate(results):
            if i >= len(outputs):
                # Short list: trailing items get batch_error
                r.status = "batch_error"
                continue
            out = outputs[i]
            if out is None:
                # Per-item failure -- explicit None signals "couldn't decide"
                r.status = "batch_error"
                continue
            if not isinstance(out, ComparatorResult):
                logger.error(
                    "Batch comparator '%s' returned %s at index %d (path '%s'); "
                    "expected ComparatorResult or None. Marking as batch_error.",
                    name, type(out).__name__, i, r.path,
                )
                r.status = "batch_error"
                continue
            r.reason = out.reason
            if out.skip:
                r.score = 0.0
                r.status = "skipped"
            else:
                r.score = out.score
                r.status = "match" if out.score >= 1.0 else "mismatch"

    return field_results


def _mark_all_error(results: list[FieldResult]) -> None:
    """Mark every FieldResult as batch_error."""
    for r in results:
        r.status = "batch_error"
