from typing import Any

from json_extract_eval.core.comparators.comparator import ComparatorResult


def compare_oneof(gold: Any, extracted: Any, params: dict[str, Any]) -> ComparatorResult:
    """Compare extracted value against a list of acceptable values"""
    values = params.get("values")
    if not isinstance(values, list):
        raise TypeError(
            f"oneof comparator requires 'values' to be a list, got {type(values).__name__}"
        )
    if extracted in values:
        return ComparatorResult(score=1.0, comparator="oneof")
    return ComparatorResult(score=0.0, comparator="oneof", reason="no match in values")
