"""Per-record and per-run result aggregation.

RecordResult: precision, recall, F1 from a list of FieldResult.
RunResult: aggregate metrics across all records, per-field breakdown.
"""

from dataclasses import dataclass
from statistics import mean

from json_extract_eval.core.field_result import FieldResult


@dataclass(frozen=True)
class RecordResult:
    """Evaluation result for a single gold/extracted pair."""

    record_id: str | int
    field_results: list[FieldResult]
    gold: dict[str, object]
    extracted: dict[str, object]
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class FieldAggregation:
    """Per-field-path statistics across all records."""

    mean_score: float
    matches: int
    mismatches: int
    omissions: int
    hallucinations: int


@dataclass(frozen=True)
class RunResult:
    """Aggregate evaluation result across all records."""

    records: list[RecordResult]
    mean_precision: float
    mean_recall: float
    mean_f1: float
    total_records: int
    total_fields: int
    total_omissions: int
    total_hallucinations: int
    total_batch_errors: int
    per_field: dict[str, FieldAggregation]


def build_record_result(
    record_id: str | int,
    field_results: list[FieldResult],
    gold: dict[str, object],
    extracted: dict[str, object],
) -> RecordResult:
    """Compute precision, recall, F1 from field results.

    Counting logic (skipped, pending, and batch_error fields are present in
    results for visibility but excluded from all metric calculations):
    - match/mismatch: contributes to both precision and recall denominators
    - omission (FN): contributes to recall denominator only
    - hallucination (FP): contributes to precision denominator only
    - skipped: excluded from all counts
    - pending: excluded from all counts (awaiting batch handler)
    - batch_error: excluded from all counts (batch handler failed)
    """
    precision_num = 0.0
    precision_den = 0.0
    recall_num = 0.0
    recall_den = 0.0

    for fr in field_results:
        if fr.status in ("skipped", "pending", "batch_error"):
            continue
        if fr.status == "omission":
            recall_den += 1.0
        elif fr.status == "hallucination":
            precision_den += 1.0
        else:
            # match or mismatch: both sides present
            precision_num += fr.score
            precision_den += 1.0
            recall_num += fr.score
            recall_den += 1.0

    precision = precision_num / precision_den if precision_den > 0 else 1.0
    recall = recall_num / recall_den if recall_den > 0 else 1.0
    if precision_den == 0 and recall_den == 0:
        f1 = 1.0  # no scorable fields
    elif (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return RecordResult(
        record_id=record_id,
        field_results=field_results,
        gold=gold,
        extracted=extracted,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def build_run_result(records: list[RecordResult]) -> RunResult:
    """Aggregate RecordResults into a RunResult with per-field breakdown.

    No records will result in precision, recall, and F1 of 1.0 since there
    are no mismatches or omissions by definition.
    """
    if not records:
        return RunResult(
            records=[],
            mean_precision=1.0,
            mean_recall=1.0,
            mean_f1=1.0,
            total_records=0,
            total_fields=0,
            total_omissions=0,
            total_hallucinations=0,
            total_batch_errors=0,
            per_field={},
        )

    # Per-field accumulation
    field_scores: dict[str, list[float]] = {}
    field_statuses: dict[str, dict[str, int]] = {}

    total_fields = 0
    total_omissions = 0
    total_hallucinations = 0
    total_batch_errors = 0

    for record in records:
        for fr in record.field_results:
            if fr.status in ("skipped", "pending"):
                continue
            if fr.status == "batch_error":
                total_batch_errors += 1
                continue
            total_fields += 1
            if fr.status == "omission":
                total_omissions += 1
            elif fr.status == "hallucination":
                total_hallucinations += 1

            if fr.path not in field_scores:
                field_scores[fr.path] = []
                field_statuses[fr.path] = {
                    "match": 0,
                    "mismatch": 0,
                    "omission": 0,
                    "hallucination": 0,
                }

            field_scores[fr.path].append(fr.score)
            field_statuses[fr.path][fr.status] += 1

    # Build per-field aggregation
    per_field: dict[str, FieldAggregation] = {}
    for path, scores in field_scores.items():
        counts = field_statuses[path]
        per_field[path] = FieldAggregation(
            mean_score=mean(scores),
            matches=counts["match"],
            mismatches=counts["mismatch"],
            omissions=counts["omission"],
            hallucinations=counts["hallucination"],
        )

    return RunResult(
        records=records,
        mean_precision=mean(r.precision for r in records),
        mean_recall=mean(r.recall for r in records),
        mean_f1=mean(r.f1 for r in records),
        total_records=len(records),
        total_fields=total_fields,
        total_omissions=total_omissions,
        total_hallucinations=total_hallucinations,
        total_batch_errors=total_batch_errors,
        per_field=per_field,
    )
