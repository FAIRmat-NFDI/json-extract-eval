"""Domain-agnostic benchmark for evaluating LLM JSON extraction."""

from struct_extract_eval.core.comparators.comparator import (
    BatchItem,
    ComparatorResult,
    ComparatorSpec,
    CompoundComparator,
)
from struct_extract_eval.core.comparators.registry import register
from struct_extract_eval.core.field_result import FieldResult
from struct_extract_eval.core.record import (
    FieldAggregation,
    RecordResult,
    RunResult,
)
from struct_extract_eval.core.schema import (
    GoldValidationError,
    SchemaNode,
    annotate_xeval,
    infer_schema,
    parse_eval_schema,
    parse_xeval_entry,
    reset_type_defaults,
    resolve_schema_references,
    set_type_default,
    validate_gold,
)
from struct_extract_eval.core.scoring import score_record
from struct_extract_eval.core.transforms.transform import TransformSpec
from struct_extract_eval.evaluator import evaluate
from struct_extract_eval.postprocess import NullHandling, reclassify_nulls

__all__ = [
    "BatchItem",
    "ComparatorResult",
    "ComparatorSpec",
    "CompoundComparator",
    "FieldAggregation",
    "FieldResult",
    "GoldValidationError",
    "NullHandling",
    "RecordResult",
    "RunResult",
    "SchemaNode",
    "TransformSpec",
    "annotate_xeval",
    "evaluate",
    "infer_schema",
    "parse_eval_schema",
    "parse_xeval_entry",
    "reclassify_nulls",
    "register",
    "reset_type_defaults",
    "resolve_schema_references",
    "score_record",
    "set_type_default",
    "validate_gold",
]
