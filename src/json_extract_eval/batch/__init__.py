"""Batch comparator implementations: LLM judge clients and built-ins.

The generic dispatcher lives in ``core/comparators/batch.py`` -- it has no
I/O and core scoring depends on it. This package holds only the I/O-bound
parts.
"""

from json_extract_eval.batch.llm_judge import (
    DEFAULT_PROMPT_TEMPLATE,
    FakeJudge,
    GroqJudge,
    Judge,
    JudgeItem,
)
from json_extract_eval.batch.semantic_comparator import SemanticBatchComparator

__all__ = [
    "DEFAULT_PROMPT_TEMPLATE",
    "FakeJudge",
    "GroqJudge",
    "Judge",
    "JudgeItem",
    "SemanticBatchComparator",
]
