"""Semantic batch comparator -- LLM-judge wrapper that fits the BatchComparator protocol.

Wraps a ``Judge`` (e.g. ``GroqJudge``) into a class that the scoring layer
can dispatch to like any other comparator. The user registers an instance under
the name ``"semantic"`` (or any name they like) and schemas reference it via
``x-eval-compare``:

    from json_extract_eval.core.comparators.registry import register
    from json_extract_eval.batch import GroqJudge, SemanticBatchComparator

    register("semantic", SemanticBatchComparator(GroqJudge()))

The exact-match short-circuit is implemented here so the LLM is never called
for trivially equal pairs -- this is the highest-leverage optimization.
"""

import logging

from json_extract_eval.batch.llm_judge import Judge, JudgeItem
from json_extract_eval.core.comparators.comparator import (
    BatchComparator,
    BatchItem,
    ComparatorResult,
)

logger = logging.getLogger(__name__)


class SemanticBatchComparator(BatchComparator):
    """BatchComparator that defers non-exact-match cases to an LLM judge.

    For each input item:

    - If gold and extracted compare equal (same type, same value) -> score 1.0
      with reason "exact" -- no LLM call.
    - Otherwise -> the item goes into a batch sent to the underlying ``Judge``.

    Returns a positional list of ``ComparatorResult | None`` matching the
    input length. ``None`` at index i means "this item couldn't be judged"
    (e.g. the LLM returned an invalid value, or the call raised). The caller
    (``process_batches``) marks those positions as ``batch_error``.

    Note: positions are preserved even when the judge fails. A judge failure
    on item #2 results in ``results[2] = None`` -- it does NOT shift
    subsequent items, so unrelated fields stay correctly scored.
    """

    def __init__(self, judge: Judge, name: str = "semantic"):
        self.judge = judge
        self.name = name

    def __call__(self, items: list[BatchItem]) -> list[ComparatorResult | None]:
        if not items:
            return []

        # Positional scratch list. Each slot starts as None and is filled
        # either by the exact-match short-circuit or by the judge result.
        # Slots that remain None at the end mean "item not judged" -- the
        # caller marks these as batch_error.
        results: list[ComparatorResult | None] = [None] * len(items)

        pending: list[JudgeItem] = []
        pending_indices: list[int] = []

        for i, item in enumerate(items):
            g = item.gold_compared
            e = item.extracted_compared
            if type(g) is type(e) and g == e:
                results[i] = ComparatorResult(
                    score=1.0, comparator=self.name, reason="exact"
                )
            else:
                pending.append(JudgeItem(path=item.path, gold=g, extracted=e))
                pending_indices.append(i)

        if pending:
            try:
                scores = self.judge.judge_batch(pending)
            except Exception as exc:
                logger.error(
                    "Semantic judge raised for %d items at paths %s: %s. "
                    "Marking all pending as batch_error.",
                    len(pending),
                    [p.path for p in pending],
                    exc,
                )
                scores = []

            if not isinstance(scores, list):
                logger.error(
                    "Semantic judge returned %s instead of a list for %d items. "
                    "Marking all pending as batch_error.",
                    type(scores).__name__,
                    len(pending),
                )
                scores = []

            if scores and len(scores) < len(pending):
                logger.warning(
                    "Semantic judge returned %d scores for %d items "
                    "(at paths %s). Missing items will be marked as "
                    "batch_error. The judge may have made mistakes — "
                    "all returned scores should be treated with caution.",
                    len(scores),
                    len(pending),
                    [p.path for p in pending],
                )
            elif len(scores) > len(pending):
                logger.warning(
                    "Semantic judge returned %d scores for %d items "
                    "(at paths %s). Extra scores are discarded. The "
                    "judge may have made mistakes — all returned "
                    "scores should be treated with caution.",
                    len(scores),
                    len(pending),
                    [p.path for p in pending],
                )

            for j, idx in enumerate(pending_indices):
                if j >= len(scores):
                    # Judge returned a short list -- this position stays None
                    # -> process_batches marks batch_error.
                    continue
                score = scores[j]
                if score is None:
                    # Per-item failure (e.g. LLM returned 0.5). Already None.
                    continue
                results[idx] = ComparatorResult(
                    score=score,
                    comparator=self.name,
                    reason="judge match" if score >= 1.0 else "judge mismatch",
                )

        # Return the FULL positional list, including None entries.
        # process_batches will mark None positions as batch_error.
        return results
