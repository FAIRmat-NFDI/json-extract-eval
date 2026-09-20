"""Batch error propagation: if any item in a batch failed, taint the whole batch.

When a batch comparator (e.g. the LLM judge) produces a ``batch_error`` for
one or more items, the other items in the same batch may also be unreliable:
the judge might have miscounted, misaligned responses, or partially failed.
This post-processor marks sibling items from a tainted batch as
``batch_error`` so they don't pollute metrics.

Items are grouped by ``comparator`` name (the batch label). If ANY item in a
group has ``status="batch_error"``, all other items in that group are also
marked ``batch_error`` -- except omission and hallucination, which represent
structural extraction failures detected before the batch comparator runs
and should remain visible in metrics.

This post-processor runs after ``process_batches``, so ``pending`` statuses
should not exist. The statuses actually converted are: ``match``,
``mismatch``, ``skipped``, and ``batch_error`` (already that status).
Items from unaffected batch comparators (or per-field comparators) are left
untouched.

Usage::

    from json_extract_eval.postprocess import propagate_batch_errors
    from json_extract_eval import evaluate

    result = evaluate(
        gold, extracted, schema,
        post_process=propagate_batch_errors,
    )

Or combine with null handling::

    from json_extract_eval.postprocess import (
        NullHandling, reclassify_nulls, propagate_batch_errors,
    )

    def my_post_process(frs):
        propagate_batch_errors(frs)
        reclassify_nulls(frs, NullHandling(absent_values=[None, ""]))
        return frs

    result = evaluate(gold, extracted, schema, post_process=my_post_process)
"""

from json_extract_eval.core.field_result import FieldResult


def propagate_batch_errors(
    field_results: list[FieldResult],
) -> list[FieldResult]:
    """If any item in a batch has batch_error, taint all items in that batch.

    Groups FieldResults by ``comparator`` name. For each group where at
    least one result has ``status="batch_error"``, marks sibling results
    as ``status="batch_error"`` with a reason explaining why.

    Omission and hallucination results are never affected -- they are
    structural issues detected before the comparator runs.

    Per-field comparators (exact, numeric, oneof, etc.) are never affected
    because they don't produce batch_error.

    Mutates ``field_results`` in place AND returns it (for chaining).
    """
    # Find which comparator names have at least one batch_error
    tainted: set[str] = set()
    for fr in field_results:
        if fr.status == "batch_error" and fr.comparator:
            tainted.add(fr.comparator)

    if not tainted:
        return field_results

    # Mark batch-path items from tainted comparators as batch_error.
    # Leave omission/hallucination untouched -- those are structural
    # issues that exist regardless of whether the batch comparator worked.
    for fr in field_results:
        # "omission", "hallucination" are labeled before sending to batch processor
        # due to one field missing
        if fr.comparator in tainted and fr.status not in ("omission", "hallucination"):
            fr.status = "batch_error"
            fr.score = 0.0
            fr.reason = (
                f"batch tainted: comparator '{fr.comparator}' had errors "
                f"in this record — all its results are excluded"
            )

    return field_results
