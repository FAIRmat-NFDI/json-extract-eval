import math
from typing import Any

from struct_extract_eval.core.comparators.comparator import ComparatorResult


def compare_numeric(gold: Any, extracted: Any, params: dict[str, Any]) -> ComparatorResult:
    """Compare two numeric values within a tolerance.

    Params should contain tolerance as {"rel": float} or/and {"abs": float}.
    If no tolerance is specified, defaults to exact float equality.
    if both rel and abs are specified, both conditions must be satisfied for score=1.0
    """
    # allow both None as correct, otherwise type error. Mirrors the NaN
    # policy below: identical "no value" on both sides is a match, but a
    # value-vs-None disagreement is not. Checked before float() since
    # float(None) raises.
    if gold is None or extracted is None:
        if gold is None and extracted is None:
            return ComparatorResult(score=1.0, comparator="numeric")
        return ComparatorResult(score=0.0, comparator="numeric", reason="type_error")

    try:
        gold_f = float(gold)
        extracted_f = float(extracted)

    except (TypeError, ValueError):
        return ComparatorResult(score=0.0, comparator="numeric", reason="type_error")

    # allow both NaN as correct, otherwise type error
    if math.isnan(gold_f) or math.isnan(extracted_f):
        if math.isnan(gold_f) and math.isnan(extracted_f):
            return ComparatorResult(score=1.0, comparator="numeric")
        return ComparatorResult(score=0.0, comparator="numeric", reason="type_error")

    tolerance = params.get("tolerance", {}) or {}
    rel = tolerance.get("rel")
    abs_ = tolerance.get("abs")

    # no tolerance: exact match
    if rel is None and abs_ is None:
        if gold_f == extracted_f:
            return ComparatorResult(score=1.0, comparator="numeric")
        return ComparatorResult(
            score=0.0,
            comparator="numeric",
            reason="values differ",
        )

    diff = abs(gold_f - extracted_f)

    # both tolerances: require both to pass
    if rel is not None and abs_ is not None:
        denom = abs(gold_f) if gold_f != 0.0 else 1.0
        rel_err = diff / denom
        if rel_err > rel:
            return ComparatorResult(
                score=0.0,
                comparator="numeric",
                reason=f"relative error exceeds {rel}",
            )
        if diff > abs_:
            return ComparatorResult(
                score=0.0,
                comparator="numeric",
                reason=f"absolute error exceeds {abs_}",
            )
        return ComparatorResult(score=1.0, comparator="numeric")

    if rel is not None:
        denom = abs(gold_f) if gold_f != 0.0 else 1.0
        if diff / denom > rel:
            return ComparatorResult(
                score=0.0,
                comparator="numeric",
                reason=f"relative error exceeds {rel}",
            )
        return ComparatorResult(score=1.0, comparator="numeric")

    if abs_ is not None:
        if diff > abs_:
            return ComparatorResult(
                score=0.0,
                comparator="numeric",
                reason=f"absolute error exceeds {abs_}",
            )
        return ComparatorResult(score=1.0, comparator="numeric")
