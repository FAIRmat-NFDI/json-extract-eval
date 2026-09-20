from typing import Any

from json_extract_eval.core.comparators.comparator import ComparatorResult


def compare_exact(gold: Any, extracted: Any, params: dict[str, Any]) -> ComparatorResult:
    """
    compare_exact returns 1.0 if gold and extracted are exactly equal (including type), and 0.0 otherwise.
    """
    if type(gold) is type(extracted) and gold == extracted:
        return ComparatorResult(score=1.0, comparator="exact")
    return ComparatorResult(score=0.0, comparator="exact", reason="mismatch")
