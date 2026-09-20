"""Tests for batch error propagation post-processor."""

from json_extract_eval.core.field_result import FieldResult
from json_extract_eval.postprocess import propagate_batch_errors


class TestPropagateBatchErrors:
    def test_no_errors_no_changes(self) -> None:
        results = [
            FieldResult(
                path="a", score=1.0, comparator="semantic",
                gold_value="x", extracted_value="x", status="match",
            ),
            FieldResult(
                path="b", score=0.0, comparator="semantic",
                gold_value="x", extracted_value="y", status="mismatch",
            ),
        ]
        propagate_batch_errors(results)
        assert results[0].status == "match"
        assert results[1].status == "mismatch"

    def test_one_error_taints_all_in_same_comparator(self) -> None:
        results = [
            FieldResult(
                path="a", score=1.0, comparator="semantic",
                gold_value="x", extracted_value="x", status="match",
            ),
            FieldResult(
                path="b", score=0.0, comparator="semantic",
                gold_value="x", extracted_value="y", status="batch_error",
            ),
            FieldResult(
                path="c", score=1.0, comparator="semantic",
                gold_value="z", extracted_value="z", status="match",
            ),
        ]
        propagate_batch_errors(results)
        # All semantic fields are now batch_error
        assert all(r.status == "batch_error" for r in results)
        assert all("tainted" in (r.reason or "") for r in results)

    def test_different_comparator_unaffected(self) -> None:
        results = [
            FieldResult(
                path="a", score=1.0, comparator="exact",
                gold_value="x", extracted_value="x", status="match",
            ),
            FieldResult(
                path="b", score=0.0, comparator="semantic",
                gold_value="x", extracted_value="y", status="batch_error",
            ),
            FieldResult(
                path="c", score=1.0, comparator="semantic",
                gold_value="z", extracted_value="z", status="match",
            ),
            FieldResult(
                path="d", score=1.0, comparator="numeric",
                gold_value=42, extracted_value=42, status="match",
            ),
        ]
        propagate_batch_errors(results)
        # exact and numeric untouched
        assert results[0].status == "match"
        assert results[3].status == "match"
        # semantic tainted
        assert results[1].status == "batch_error"
        assert results[2].status == "batch_error"

    def test_multiple_batch_comparators_independent(self) -> None:
        """bc1 has error, bc2 does not — only bc1 is tainted."""
        results = [
            FieldResult(
                path="a", score=1.0, comparator="bc1",
                gold_value="x", extracted_value="x", status="match",
            ),
            FieldResult(
                path="b", score=0.0, comparator="bc1",
                gold_value="x", extracted_value="y", status="batch_error",
            ),
            FieldResult(
                path="c", score=1.0, comparator="bc2",
                gold_value="z", extracted_value="z", status="match",
            ),
            FieldResult(
                path="d", score=0.0, comparator="bc2",
                gold_value="w", extracted_value="v", status="mismatch",
            ),
        ]
        propagate_batch_errors(results)
        # bc1: all batch_error
        assert results[0].status == "batch_error"
        assert results[1].status == "batch_error"
        # bc2: untouched
        assert results[2].status == "match"
        assert results[3].status == "mismatch"

    def test_skipped_sibling_also_tainted(self) -> None:
        """A batch-handler skip (ComparatorResult(skip=True)) is also tainted."""
        results = [
            FieldResult(
                path="a", score=0.0, comparator="semantic",
                gold_value="x", extracted_value="y", status="skipped",
                reason="batch handler skipped",
            ),
            FieldResult(
                path="b", score=0.0, comparator="semantic",
                gold_value="x", extracted_value="y", status="batch_error",
            ),
        ]
        propagate_batch_errors(results)
        # Both become batch_error when the batch is tainted
        assert results[0].status == "batch_error"
        assert results[1].status == "batch_error"

    def test_omission_not_converted(self) -> None:
        """Omission results are structural -- not affected by batch tainting."""
        results = [
            FieldResult(
                path="a", score=0.0, comparator="semantic",
                gold_value="x", extracted_value=None, status="omission",
            ),
            FieldResult(
                path="b", score=0.0, comparator="semantic",
                gold_value="y", extracted_value="z", status="batch_error",
            ),
        ]
        propagate_batch_errors(results)
        assert results[0].status == "omission"
        assert results[1].status == "batch_error"

    def test_hallucination_not_converted(self) -> None:
        """Hallucination results are structural -- not affected by batch tainting."""
        results = [
            FieldResult(
                path="a", score=0.0, comparator="semantic",
                gold_value=None, extracted_value="x", status="hallucination",
            ),
            FieldResult(
                path="b", score=0.0, comparator="semantic",
                gold_value="y", extracted_value="z", status="batch_error",
            ),
        ]
        propagate_batch_errors(results)
        assert results[0].status == "hallucination"
        assert results[1].status == "batch_error"
