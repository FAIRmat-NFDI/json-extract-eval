"""Tests for the BatchComparator subsystem.

Covers:
- The semantic batch comparator (using FakeJudge -- no network)
- process_batches dispatch by label, multiple groups, error paths
- End-to-end via explicit register(...) + evaluate()
- A unit-aware QuantityBatchComparator example -- demonstrates non-LLM usage
"""

from typing import ClassVar

import pytest

from json_extract_eval.batch import (
    FakeJudge,
    SemanticBatchComparator,
)
from json_extract_eval.batch.llm_judge import (
    JudgeItem,
    _coerce_binary_score,
    _parse_judge_response,
)
from json_extract_eval.core.comparators.batch import process_batches
from json_extract_eval.core.comparators.comparator import (
    BatchComparator,
    BatchItem,
    ComparatorResult,
    CompoundComparator,
)
from json_extract_eval.core.comparators.registry import (
    _clear_registry,
    register,
)
from json_extract_eval.core.schema import SchemaNode
from json_extract_eval.core.field_result import FieldResult
from json_extract_eval.evaluator import evaluate

# Minimal empty tree for process_batches tests that don't need real schema params.
_EMPTY_TREE = SchemaNode(path="", json_type="object", comparator="")


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Reset custom registry and remove any leftover registration."""
    _clear_registry()
    yield
    _clear_registry()


# --- SemanticBatchComparator ---


class TestSemanticBatchComparator:
    def test_exact_match_short_circuits(self) -> None:
        # Both pairs are equal -> no LLM call
        judge = FakeJudge()
        comp = SemanticBatchComparator(judge)
        items = [
            BatchItem("a", "PVD", "PVD", "PVD", "PVD"),
            BatchItem("b", 42, 42, 42, 42),
        ]
        results = comp(items)
        assert len(results) == 2
        assert results[0].score == 1.0
        assert results[0].reason == "exact"
        assert results[1].score == 1.0
        # No judge call was needed for either
        assert judge.calls == []

    def test_non_exact_defers_to_judge(self) -> None:
        judge = FakeJudge(responses={("PVD", "physical vapor deposition"): 1.0})
        comp = SemanticBatchComparator(judge)
        items = [
            BatchItem("method", "PVD", "physical vapor deposition",
                      "PVD", "physical vapor deposition"),
        ]
        results = comp(items)
        assert results[0].score == 1.0
        assert results[0].reason == "judge match"
        assert len(judge.calls) == 1

    def test_mixed_exact_and_judged(self) -> None:
        judge = FakeJudge(responses={("PVD", "sputtering"): 0.0})
        comp = SemanticBatchComparator(judge)
        items = [
            BatchItem("a", "Alice", "Alice", "Alice", "Alice"),  # exact
            BatchItem("b", "PVD", "sputtering", "PVD", "sputtering"),  # judged
        ]
        results = comp(items)
        assert results[0].score == 1.0
        assert results[0].reason == "exact"
        assert results[1].score == 0.0
        assert results[1].reason == "judge mismatch"
        # Judge called once with only the non-exact item
        assert len(judge.calls) == 1
        assert len(judge.calls[0]) == 1
        assert judge.calls[0][0].path == "b"

    def test_uses_post_transform_values(self) -> None:
        # gold_compared / extracted_compared are what the judge sees, not the raw values
        judge = FakeJudge(responses={("trimmed", "trimmed"): 1.0})
        comp = SemanticBatchComparator(judge)
        items = [
            BatchItem(
                path="x",
                gold_raw="  trimmed  ",
                extracted_raw="trimmed",
                gold_compared="trimmed",  # post-transform
                extracted_compared="trimmed",
            ),
        ]
        results = comp(items)
        assert results[0].score == 1.0
        assert results[0].reason == "exact"  # short-circuits because compared values match


# --- process_batches dispatch ---


class TestProcessBatches:
    def test_no_pending_returns_unchanged(self) -> None:
        results = [
            FieldResult("a", 1.0, "exact", "x", "x", "match"),
            FieldResult("b", 0.0, "exact", "y", "z", "mismatch"),
        ]
        process_batches(results, _EMPTY_TREE)
        assert results[0].status == "match"
        assert results[1].status == "mismatch"

    def test_dispatches_to_registered_handler(self) -> None:
        # Register a fake batch comparator that returns 1 for all items
        class AlwaysMatch(BatchComparator):
            def __call__(self, items: list[BatchItem]) -> list[ComparatorResult | None]:
                return [ComparatorResult(score=1.0, comparator="always") for _ in items]
        register("always", AlwaysMatch())

        results = [
            FieldResult(
                path="x", score=0.0, comparator="always",
                gold_value="g", extracted_value="e",
                status="pending",
                gold_compared="g", extracted_compared="e",
            ),
        ]
        process_batches(results, _EMPTY_TREE)
        assert results[0].status == "match"
        assert results[0].score == 1.0

    def test_unregistered_handler_marks_batch_error(self) -> None:
        results = [
            FieldResult(
                path="x", score=0.0, comparator="missing",
                gold_value="g", extracted_value="e",
                status="pending",
                gold_compared="g", extracted_compared="e",
            ),
        ]
        process_batches(results, _EMPTY_TREE)
        assert results[0].status == "batch_error"

    def test_handler_raises_marks_all_batch_error(self) -> None:
        class Raising(BatchComparator):
            def __call__(self, items: list[BatchItem]) -> list[ComparatorResult | None]:
                raise RuntimeError("nope")
        register("raising", Raising())

        results = [
            FieldResult("a", 0.0, "raising", "g1", "e1", "pending",
                        gold_compared="g1", extracted_compared="e1"),
            FieldResult("b", 0.0, "raising", "g2", "e2", "pending",
                        gold_compared="g2", extracted_compared="e2"),
        ]
        process_batches(results, _EMPTY_TREE)
        assert all(r.status == "batch_error" for r in results)

    def test_short_response_marks_trailing_batch_error(self) -> None:
        class ShortResponse(BatchComparator):
            def __call__(self, items: list[BatchItem]) -> list[ComparatorResult | None]:
                return [ComparatorResult(score=1.0, comparator="short")]  # only 1
        register("short", ShortResponse())

        results = [
            FieldResult("a", 0.0, "short", "g1", "e1", "pending",
                        gold_compared="g1", extracted_compared="e1"),
            FieldResult("b", 0.0, "short", "g2", "e2", "pending",
                        gold_compared="g2", extracted_compared="e2"),
        ]
        process_batches(results, _EMPTY_TREE)
        assert results[0].status == "match"
        assert results[1].status == "batch_error"

    def test_extra_results_trimmed(self) -> None:
        class ExtraResponse(BatchComparator):
            def __call__(self, items: list[BatchItem]) -> list[ComparatorResult | None]:
                return [
                    ComparatorResult(score=1.0, comparator="extra"),
                    ComparatorResult(score=0.0, comparator="extra"),
                    ComparatorResult(score=1.0, comparator="extra"),  # extra
                ]
        register("extra", ExtraResponse())

        results = [
            FieldResult("a", 0.0, "extra", "g1", "e1", "pending",
                        gold_compared="g1", extracted_compared="e1"),
            FieldResult("b", 0.0, "extra", "g2", "e2", "pending",
                        gold_compared="g2", extracted_compared="e2"),
        ]
        process_batches(results, _EMPTY_TREE)
        assert results[0].status == "match"
        assert results[1].status == "mismatch"

    def test_groups_by_label(self) -> None:
        # Two different labels in one record -> two separate handler calls
        calls: list[str] = []

        class HandlerA(BatchComparator):
            def __call__(self, items: list[BatchItem]) -> list[ComparatorResult | None]:
                calls.append("A")
                return [ComparatorResult(score=1.0, comparator="A") for _ in items]

        class HandlerB(BatchComparator):
            def __call__(self, items: list[BatchItem]) -> list[ComparatorResult | None]:
                calls.append("B")
                return [ComparatorResult(score=0.0, comparator="B") for _ in items]

        register("A", HandlerA())
        register("B", HandlerB())

        results = [
            FieldResult("p1", 0.0, "A", "g", "e", "pending",
                        gold_compared="g", extracted_compared="e"),
            FieldResult("p2", 0.0, "B", "g", "e", "pending",
                        gold_compared="g", extracted_compared="e"),
            FieldResult("p3", 0.0, "A", "g", "e", "pending",
                        gold_compared="g", extracted_compared="e"),
        ]
        process_batches(results, _EMPTY_TREE)
        assert calls == ["A", "B"] or calls == ["B", "A"]
        assert len(calls) == 2  # one call per label, not per item
        assert results[0].score == 1.0  # from A
        assert results[1].score == 0.0  # from B
        assert results[2].score == 1.0  # from A


    def test_skip_flag_sets_status_skipped(self) -> None:
        """ComparatorResult(skip=True) -> process_batches sets status='skipped'."""
        class SkipHandler(BatchComparator):
            def __call__(self, items: list[BatchItem]) -> list[ComparatorResult | None]:
                return [
                    ComparatorResult(score=1.0, comparator="skipper"),
                    ComparatorResult(score=0.0, comparator="skipper", reason="supporting", skip=True),
                ]
        register("skipper", SkipHandler())

        results = [
            FieldResult("a", 0.0, "skipper", "g1", "e1", "pending",
                        gold_compared="g1", extracted_compared="e1"),
            FieldResult("b", 0.0, "skipper", "g2", "e2", "pending",
                        gold_compared="g2", extracted_compared="e2"),
        ]
        process_batches(results, _EMPTY_TREE)
        assert results[0].status == "match"
        assert results[0].score == 1.0
        assert results[1].status == "skipped"
        assert results[1].score == 0.0
        assert results[1].reason == "supporting"


# --- end-to-end via manual semantic registration ---


class TestEvaluateWithSemanticRegistration:
    def test_registered_semantic_used_for_scoring(self) -> None:
        register(
            "semantic",
            SemanticBatchComparator(
                FakeJudge(responses={("PVD", "physical vapor deposition"): 1.0})
            ),
        )

        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "method": {"type": "string", "x-eval-compare": "semantic"},
            },
        }
        gold = [{"method": "PVD"}]
        extracted = [{"method": "physical vapor deposition"}]

        result = evaluate(gold, extracted, schema)
        assert result.records[0].field_results[0].status == "match"
        assert result.records[0].field_results[0].score == 1.0
        assert result.mean_f1 == 1.0

    def test_unregistered_semantic_raises_at_parse(self) -> None:
        # If "semantic" is referenced but no handler is registered, parse_eval_schema
        # raises with the missing comparator name in the message.
        from json_extract_eval.core.schema import SchemaError
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "method": {"type": "string", "x-eval-compare": "semantic"},
            },
        }
        with pytest.raises(SchemaError, match="semantic"):
            evaluate([{"method": "x"}], [{"method": "x"}], schema)

    def test_unregistered_custom_name_raises_with_that_name(self) -> None:
        # The error uses the actual missing comparator name, not a hardcoded "semantic".
        from json_extract_eval.core.schema import SchemaError
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "method": {"type": "string", "x-eval-compare": "semantic_fast"},
            },
        }
        with pytest.raises(SchemaError, match="semantic_fast"):
            evaluate([{"method": "x"}], [{"method": "x"}], schema)

    def test_two_judges_under_different_names(self) -> None:
        # Demonstrates the multi-judge use case: register two semantic comparators
        # under different names and let the schema pick which to use per field.
        register(
            "semantic_strict",
            SemanticBatchComparator(FakeJudge(default_score=0.0)),  # always disagrees
        )
        register(
            "semantic",
            SemanticBatchComparator(FakeJudge(default_score=1.0)),  # always agrees
        )

        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "strict_field": {"type": "string", "x-eval-compare": "semantic_strict"},
                "semantic_field": {"type": "string", "x-eval-compare": "semantic"},
            },
        }
        gold = [{"strict_field": "A", "semantic_field": "A"}]
        extracted = [{"strict_field": "B", "semantic_field": "B"}]

        result = evaluate(gold, extracted, schema)
        by_path = {r.path: r for r in result.records[0].field_results}
        assert by_path["strict_field"].status == "mismatch"
        assert by_path["semantic_field"].status == "match"

    def test_batch_error_excluded_from_metrics(self) -> None:
        class RaisingJudge:
            def judge_batch(self, items: list[JudgeItem]) -> list[float | None]:
                raise RuntimeError("api down")

        register("semantic", SemanticBatchComparator(RaisingJudge()))

        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "x-eval-compare": "exact"},
                "method": {"type": "string", "x-eval-compare": "semantic"},
            },
        }
        gold = [{"name": "Alice", "method": "PVD"}]
        extracted = [{"name": "Alice", "method": "physical vapor deposition"}]

        result = evaluate(gold, extracted, schema)
        record = result.records[0]

        statuses = [r.status for r in record.field_results]
        assert "match" in statuses  # the name field
        assert "batch_error" in statuses  # the semantic field

        # Only the matching field counts toward metrics
        assert record.precision == 1.0
        assert record.recall == 1.0
        assert record.f1 == 1.0
        assert result.total_batch_errors == 1
        assert result.total_fields == 1


# --- non-LLM example: unit-aware comparison ---


class QuantityBatchComparator(BatchComparator):
    """Example multi-field BatchComparator: pairs sibling 'value' and 'unit' fields."""

    UNITS_TO_METERS: ClassVar[dict[str, float]] = {
        "m": 1.0, "meter": 1.0, "meters": 1.0,
        "km": 1000.0, "cm": 0.01, "mm": 0.001,
    }

    def __call__(self, items: list[BatchItem]) -> list[ComparatorResult | None]:
        # Group by parent path, tracking original indices for O(n) lookup.
        by_parent: dict[str, list[tuple[int, BatchItem]]] = {}
        for i, item in enumerate(items):
            parent = item.path.rsplit(".", 1)[0]
            by_parent.setdefault(parent, []).append((i, item))

        # Build results in original order
        result_by_index: dict[int, ComparatorResult] = {}
        for parent, group in by_parent.items():
            length = next(((i, it) for i, it in group if it.path.endswith(".length")), None)
            unit = next(((i, it) for i, it in group if it.path.endswith(".unit")), None)

            if length is None or unit is None:
                for idx, _item in group:
                    result_by_index[idx] = ComparatorResult(
                        score=0.0, comparator="quantity",
                        reason=f"incomplete pair under {parent}",
                    )
                continue

            score = self._compare(length[1], unit[1])
            for idx, _item in group:
                result_by_index[idx] = ComparatorResult(
                    score=score, comparator="quantity",
                    reason=f"{length[1].gold_compared}{unit[1].gold_compared} vs "
                           f"{length[1].extracted_compared}{unit[1].extracted_compared}",
                )

        return [result_by_index[i] for i in range(len(items))]

    def _compare(self, length: BatchItem, unit: BatchItem) -> float:
        g_unit = (unit.gold_compared or "").lower()
        e_unit = (unit.extracted_compared or "").lower()
        if g_unit not in self.UNITS_TO_METERS or e_unit not in self.UNITS_TO_METERS:
            return 0.0
        try:
            g_meters = float(length.gold_compared) * self.UNITS_TO_METERS[g_unit]
            e_meters = float(length.extracted_compared) * self.UNITS_TO_METERS[e_unit]
        except (TypeError, ValueError):
            return 0.0
        return 1.0 if abs(g_meters - e_meters) < 1e-9 else 0.0


class TestQuantityBatchComparatorExample:
    def test_value_and_unit_compared_jointly(self) -> None:
        register("quantity", QuantityBatchComparator())

        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "measurement": {
                    "type": "object",
                    "properties": {
                        "length": {"type": "number", "x-eval-compare": "quantity"},
                        "unit": {"type": "string", "x-eval-compare": "quantity"},
                    },
                },
            },
        }
        gold = [{"measurement": {"length": 10, "unit": "m"}}]
        extracted = [{"measurement": {"length": 10, "unit": "meter"}}]

        result = evaluate(gold, extracted, schema)
        record = result.records[0]

        # Both length and unit fields end up as matches via the quantity handler
        assert len(record.field_results) == 2
        assert all(r.status == "match" for r in record.field_results)
        assert all(r.score == 1.0 for r in record.field_results)
        assert record.f1 == 1.0

    def test_unit_conversion_works(self) -> None:
        register("quantity", QuantityBatchComparator())

        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "measurement": {
                    "type": "object",
                    "properties": {
                        "length": {"type": "number", "x-eval-compare": "quantity"},
                        "unit": {"type": "string", "x-eval-compare": "quantity"},
                    },
                },
            },
        }
        # 1000mm == 1m
        gold = [{"measurement": {"length": 1000, "unit": "mm"}}]
        extracted = [{"measurement": {"length": 1, "unit": "m"}}]

        result = evaluate(gold, extracted, schema)
        assert result.mean_f1 == 1.0

    def test_different_quantities_mismatch(self) -> None:
        register("quantity", QuantityBatchComparator())

        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "measurement": {
                    "type": "object",
                    "properties": {
                        "length": {"type": "number", "x-eval-compare": "quantity"},
                        "unit": {"type": "string", "x-eval-compare": "quantity"},
                    },
                },
            },
        }
        gold = [{"measurement": {"length": 10, "unit": "m"}}]
        extracted = [{"measurement": {"length": 5, "unit": "m"}}]  # different value

        result = evaluate(gold, extracted, schema)
        record = result.records[0]
        assert all(r.status == "mismatch" for r in record.field_results)
        assert record.f1 == 0.0


# --- compound comparator using CompoundComparator base class ---


class NameCompoundComparator(CompoundComparator):
    """Example: scores family_name + given_name as one full name.

    Uses the CompoundComparator base class -- only the compare() method
    needs to be written. Grouping, primary/skip, incomplete handling are
    all handled by the base class.
    """

    def __init__(self) -> None:
        super().__init__(
            fields=["family_name", "given_name"],
            primary="family_name",
            name="name_compound",
        )

    def compare(
        self, gold: dict[str, object], extracted: dict[str, object]
    ) -> float:
        def normalize(name: str) -> str:
            return " ".join(str(name).lower().split())

        gold_full = normalize(f"{gold['given_name']} {gold['family_name']}")
        ext_full = normalize(f"{extracted['given_name']} {extracted['family_name']}")
        if gold_full == ext_full:
            return 1.0
        # Try swapped order
        ext_swapped = normalize(f"{extracted['family_name']} {extracted['given_name']}")
        if gold_full == ext_swapped:
            return 1.0
        return 0.0


class TestNameCompoundComparator:
    """Compound comparator: family_name + given_name scored as one logical name."""

    def _schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "x-eval-compare": "exact"},
                "family_name": {"type": "string", "x-eval-compare": "name_compound"},
                "given_name": {"type": "string", "x-eval-compare": "name_compound"},
                "gender": {"type": "string", "x-eval-compare": "exact"},
            },
        }

    def test_both_correct(self) -> None:
        register("name_compound", NameCompoundComparator())
        gold = [{"id": "S1", "family_name": "Smith", "given_name": "John", "gender": "M"}]
        ext = [{"id": "S1", "family_name": "Smith", "given_name": "John", "gender": "M"}]

        result = evaluate(gold, ext, schema=self._schema())
        record = result.records[0]

        # family_name: compound match, given_name: skipped
        by_path = {r.path: r for r in record.field_results}
        assert by_path["family_name"].status == "match"
        assert by_path["family_name"].score == 1.0
        assert by_path["given_name"].status == "skipped"
        assert by_path["id"].status == "match"
        assert by_path["gender"].status == "match"

        # Only 3 fields counted in metrics (id, family_name, gender), not 4
        assert record.f1 == 1.0

    def test_names_swapped(self) -> None:
        """family/given swapped -> compound still matches."""
        register("name_compound", NameCompoundComparator())
        gold = [{"id": "S1", "family_name": "Smith", "given_name": "John", "gender": "M"}]
        ext = [{"id": "S1", "family_name": "John", "given_name": "Smith", "gender": "M"}]

        result = evaluate(gold, ext, schema=self._schema())
        record = result.records[0]

        by_path = {r.path: r for r in record.field_results}
        assert by_path["family_name"].status == "match"
        assert by_path["family_name"].score == 1.0
        assert by_path["given_name"].status == "skipped"
        # Without compound, both would mismatch -> F1 = 0.5
        # With compound, the name matches -> F1 = 1.0
        assert record.f1 == 1.0

    def test_different_person(self) -> None:
        register("name_compound", NameCompoundComparator())
        gold = [{"id": "S1", "family_name": "Smith", "given_name": "John", "gender": "M"}]
        ext = [{"id": "S1", "family_name": "Doe", "given_name": "Jane", "gender": "F"}]

        result = evaluate(gold, ext, schema=self._schema())
        record = result.records[0]

        by_path = {r.path: r for r in record.field_results}
        assert by_path["family_name"].status == "mismatch"
        assert by_path["family_name"].score == 0.0
        assert by_path["given_name"].status == "skipped"
        assert by_path["gender"].status == "mismatch"
        # 3 fields scored: id(match), family_name(mismatch), gender(mismatch)
        # P = 1/3, R = 1/3
        assert record.precision < 0.4
        assert record.recall < 0.4

    def test_supporting_field_excluded_from_metrics(self) -> None:
        """given_name is skipped -- it must not affect P/R/F1."""
        register("name_compound", NameCompoundComparator())
        # Only name fields, no id/gender -- isolates the compound behavior
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "family_name": {"type": "string", "x-eval-compare": "name_compound"},
                "given_name": {"type": "string", "x-eval-compare": "name_compound"},
            },
        }
        gold = [{"family_name": "Smith", "given_name": "John"}]
        ext = [{"family_name": "Smith", "given_name": "John"}]

        result = evaluate(gold, ext, schema=schema)
        record = result.records[0]

        # Only 1 field counted (family_name), given_name is skipped
        scored = [r for r in record.field_results if r.status != "skipped"]
        assert len(scored) == 1
        assert scored[0].path == "family_name"
        assert record.f1 == 1.0


    def test_compound_inside_array(self) -> None:
        """Compound comparator works inside arrays via instance paths.

        Each array element gets its own group (students[0] vs students[1])
        so the handler pairs each element's fields correctly.
        """
        register("name_compound", NameCompoundComparator())
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "students": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "family_name": {"type": "string", "x-eval-compare": "name_compound"},
                            "given_name": {"type": "string", "x-eval-compare": "name_compound"},
                        },
                    },
                },
            },
        }
        gold = [{"students": [
            {"family_name": "Smith", "given_name": "John"},
            {"family_name": "Kim", "given_name": "Soo"},
        ]}]
        ext = [{"students": [
            {"family_name": "Smith", "given_name": "Jane"},  # wrong name
            {"family_name": "Kim", "given_name": "Soo"},     # correct
        ]}]

        result = evaluate(gold, ext, schema=schema)
        record = result.records[0]
        by_path = {r.path: r for r in record.field_results}

        # Element 0: wrong name -> mismatch on primary, skip on supporting
        assert by_path["students[0].family_name"].status == "mismatch"
        assert by_path["students[0].given_name"].status == "skipped"

        # Element 1: correct -> match on primary, skip on supporting
        assert by_path["students[1].family_name"].status == "match"
        assert by_path["students[1].given_name"].status == "skipped"

        # 2 fields scored (one per element's primary), not 4
        scored = [r for r in record.field_results if r.status != "skipped"]
        assert len(scored) == 2


# --- CompoundComparator base class ---


class TestCompoundComparatorBaseClass:
    """Tests for the CompoundComparator base class itself."""

    def test_primary_must_be_in_fields(self) -> None:
        with pytest.raises(ValueError, match="primary field"):
            CompoundComparator(fields=["a", "b"], primary="c")

    def test_minimal_subclass(self) -> None:
        """A simple quantity comparator using the base class."""

        class QuantityCC(CompoundComparator):
            def __init__(self) -> None:
                super().__init__(fields=["value", "unit"], primary="value", name="qty")

            def compare(self, gold: dict[str, object], extracted: dict[str, object]) -> float:
                # Trivial: just check both fields match exactly
                return 1.0 if gold == extracted else 0.0

        register("qty", QuantityCC())
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "measurement": {"type": "object", "properties": {
                    "value": {"type": "number", "x-eval-compare": "qty"},
                    "unit": {"type": "string", "x-eval-compare": "qty"},
                }},
            },
        }
        gold = [{"measurement": {"value": 10, "unit": "m"}}]
        ext = [{"measurement": {"value": 10, "unit": "m"}}]
        result = evaluate(gold, ext, schema=schema)
        record = result.records[0]

        by_path = {r.path: r for r in record.field_results}
        assert by_path["measurement.value"].status == "match"
        assert by_path["measurement.value"].score == 1.0
        assert by_path["measurement.unit"].status == "skipped"
        assert record.f1 == 1.0

    def test_mismatch_scores_zero(self) -> None:
        class AlwaysMismatch(CompoundComparator):
            def __init__(self) -> None:
                super().__init__(fields=["a", "b"], primary="a", name="test")

            def compare(self, gold: dict[str, object], extracted: dict[str, object]) -> float:
                return 0.0

        register("test", AlwaysMismatch())
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "a": {"type": "string", "x-eval-compare": "test"},
                "b": {"type": "string", "x-eval-compare": "test"},
            },
        }
        result = evaluate([{"a": "x", "b": "y"}], [{"a": "x", "b": "y"}], schema=schema)
        record = result.records[0]
        by_path = {r.path: r for r in record.field_results}
        assert by_path["a"].status == "mismatch"
        assert by_path["a"].score == 0.0
        assert by_path["b"].status == "skipped"

    def test_incomplete_group_scores_zero(self) -> None:
        """If one sibling field is omitted, the remaining field scores 0."""

        class SimpleCC(CompoundComparator):
            def __init__(self) -> None:
                super().__init__(fields=["a", "b"], primary="a", name="simple")

            def compare(self, gold: dict[str, object], extracted: dict[str, object]) -> float:
                return 1.0

        register("simple", SimpleCC())
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "a": {"type": "string", "x-eval-compare": "simple"},
                "b": {"type": "string", "x-eval-compare": "simple"},
            },
        }
        # "b" missing from extracted -> omission for "b", incomplete compound for "a"
        result = evaluate([{"a": "x", "b": "y"}], [{"a": "x"}], schema=schema)
        record = result.records[0]
        by_path = {r.path: r for r in record.field_results}
        # "b" is an omission (handled by _score_object, never reaches batch)
        assert by_path["b"].status == "omission"
        # "a" arrives alone in the batch -> incomplete compound
        assert by_path["a"].score == 0.0
        assert "incomplete" in (by_path["a"].reason or "")


# --- Value validation in the LLM judge response parser ---


class TestCoerceBinaryScore:
    """Strict coercion: only unambiguous 0/1 inputs become 0.0/1.0; rest is None."""

    def test_bools(self) -> None:
        assert _coerce_binary_score(True) == 1.0
        assert _coerce_binary_score(False) == 0.0

    def test_ints_zero_one(self) -> None:
        assert _coerce_binary_score(0) == 0.0
        assert _coerce_binary_score(1) == 1.0

    def test_floats_exact_zero_one(self) -> None:
        assert _coerce_binary_score(0.0) == 0.0
        assert _coerce_binary_score(1.0) == 1.0

    def test_other_numbers_are_none(self) -> None:
        # 0.5 silently becoming 0 was the bug; now it's None.
        assert _coerce_binary_score(0.5) is None
        assert _coerce_binary_score(2) is None
        assert _coerce_binary_score(-1) is None
        assert _coerce_binary_score(1.5) is None
        assert _coerce_binary_score(0.999) is None

    def test_string_zero_one(self) -> None:
        assert _coerce_binary_score("0") == 0.0
        assert _coerce_binary_score("1") == 1.0

    def test_string_true_false_case_insensitive(self) -> None:
        assert _coerce_binary_score("true") == 1.0
        assert _coerce_binary_score("True") == 1.0
        assert _coerce_binary_score("TRUE") == 1.0
        assert _coerce_binary_score("false") == 0.0
        assert _coerce_binary_score("False") == 0.0
        assert _coerce_binary_score("FALSE") == 0.0

    def test_string_with_whitespace(self) -> None:
        assert _coerce_binary_score("  1  ") == 1.0
        assert _coerce_binary_score("\ttrue\n") == 1.0

    def test_invalid_strings_are_none(self) -> None:
        assert _coerce_binary_score("maybe") is None
        assert _coerce_binary_score("yes") is None
        assert _coerce_binary_score("") is None
        assert _coerce_binary_score("0.5") is None

    def test_other_types_are_none(self) -> None:
        assert _coerce_binary_score(None) is None
        assert _coerce_binary_score([]) is None
        assert _coerce_binary_score({}) is None
        assert _coerce_binary_score([1]) is None


class TestParseJudgeResponse:
    """The full response parser. Per-item invalid values become None."""

    def test_clean_binary_response(self) -> None:
        scores = _parse_judge_response('{"results": [1, 0, 1]}', expected_count=3)
        assert scores == [1.0, 0.0, 1.0]

    def test_per_item_invalid_becomes_none(self) -> None:
        # Mix of valid and invalid -- valid items keep their scores, invalid -> None
        scores = _parse_judge_response('{"results": [1, 0.5, 0, "maybe", 1]}', expected_count=5)
        assert scores == [1.0, None, 0.0, None, 1.0]

    def test_invalid_json_returns_empty(self) -> None:
        scores = _parse_judge_response("not json at all", expected_count=3)
        assert scores == []

    def test_missing_results_key_returns_empty(self) -> None:
        scores = _parse_judge_response('{"foo": [1, 1]}', expected_count=2)
        assert scores == []

    def test_results_not_a_list_returns_empty(self) -> None:
        scores = _parse_judge_response('{"results": "1"}', expected_count=2)
        assert scores == []

    def test_code_fence_stripped(self) -> None:
        scores = _parse_judge_response(
            '```json\n{"results": [1, 1]}\n```', expected_count=2
        )
        assert scores == [1.0, 1.0]

    def test_bool_values(self) -> None:
        scores = _parse_judge_response('{"results": [true, false]}', expected_count=2)
        assert scores == [1.0, 0.0]


# --- Position bug regression ---


class TestPositionPreservation:
    """Regression tests for the position bug: a judge failure on item N must
    NOT corrupt other items in the same batch."""

    def test_judge_short_response_does_not_shift_others(self) -> None:
        # Items: [exact, pending, pending, exact]
        # Judge returns only 1 score for the 2 pending items.
        # Bug behavior: the second pending item would get the exact[3]'s score,
        # and exact[3] would become batch_error.
        # Correct behavior: pending #2 becomes batch_error; exact[3] stays match.

        # Force a short response: this judge returns only 1 entry for 2 inputs
        class ShortJudge:
            def __init__(self) -> None:
                self.calls: list[list[JudgeItem]] = []
            def judge_batch(self, items: list[JudgeItem]) -> list[float | None]:
                self.calls.append(list(items))
                # Return one score regardless of how many items came in
                return [1.0]

        comp = SemanticBatchComparator(ShortJudge())
        items = [
            BatchItem("p0", "x", "x", "x", "x"),       # exact match
            BatchItem("p1", "a", "ax", "a", "ax"),     # pending, judge returns 1.0
            BatchItem("p2", "b", "bx", "b", "bx"),     # pending, judge has no answer
            BatchItem("p3", "y", "y", "y", "y"),       # exact match
        ]
        results = comp(items)
        assert len(results) == 4  # positional, no filtering

        assert results[0] is not None and results[0].reason == "exact"
        assert results[1] is not None and results[1].score == 1.0
        # results[2] is None -- judge didn't return a score for it
        assert results[2] is None
        # results[3] is the exact match, NOT corrupted
        assert results[3] is not None and results[3].reason == "exact"

    def test_judge_per_item_invalid_does_not_shift_others(self) -> None:
        # Judge returns [1.0, None, 0.0] for 3 pending items.
        # The middle one is invalid (e.g. LLM said 0.5).
        class InvalidMiddleJudge:
            def judge_batch(self, items: list[JudgeItem]) -> list[float | None]:
                return [1.0, None, 0.0]

        comp = SemanticBatchComparator(InvalidMiddleJudge())
        items = [
            BatchItem("a", "g1", "e1", "g1", "e1"),
            BatchItem("b", "g2", "e2", "g2", "e2"),
            BatchItem("c", "g3", "e3", "g3", "e3"),
        ]
        results = comp(items)
        assert len(results) == 3
        assert results[0] is not None and results[0].score == 1.0
        assert results[1] is None  # invalid value -> None at correct position
        assert results[2] is not None and results[2].score == 0.0

    def test_end_to_end_per_item_failure_only_affects_one_field(self) -> None:
        # Three semantic fields in one record. Judge fails on the middle one
        # (returns None at index 1). The other two should be fine; only the
        # middle field becomes batch_error.

        class PartialJudge:
            def judge_batch(self, items: list[JudgeItem]) -> list[float | None]:
                return [1.0, None, 1.0]

        register("semantic", SemanticBatchComparator(PartialJudge()))

        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "a": {"type": "string", "x-eval-compare": "semantic"},
                "b": {"type": "string", "x-eval-compare": "semantic"},
                "c": {"type": "string", "x-eval-compare": "semantic"},
            },
        }
        gold = [{"a": "x1", "b": "x2", "c": "x3"}]
        extracted = [{"a": "y1", "b": "y2", "c": "y3"}]  # all non-exact

        result = evaluate(gold, extracted, schema)
        record = result.records[0]
        by_path = {r.path: r for r in record.field_results}
        assert by_path["a"].status == "match"
        assert by_path["b"].status == "batch_error"
        assert by_path["c"].status == "match"
        # b is excluded from metrics; a and c both match -> P=R=F1=1.0
        assert record.f1 == 1.0
        assert result.total_batch_errors == 1


# --- All-or-nothing failure (whole batch) ---


class TestWholeBatchFailure:
    def test_judge_raises_marks_all_pending_as_error(self) -> None:
        class RaisingJudge:
            def judge_batch(self, items: list[JudgeItem]) -> list[float | None]:
                raise RuntimeError("api down")

        register("semantic", SemanticBatchComparator(RaisingJudge()))

        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "a": {"type": "string", "x-eval-compare": "semantic"},
                "b": {"type": "string", "x-eval-compare": "semantic"},
            },
        }
        gold = [{"a": "x", "b": "x"}]
        extracted = [{"a": "y", "b": "y"}]  # both non-exact

        result = evaluate(gold, extracted, schema)
        record = result.records[0]
        assert all(r.status == "batch_error" for r in record.field_results)
        assert result.total_batch_errors == 2
        assert result.total_fields == 0

    def test_exact_matches_survive_whole_batch_failure(self) -> None:
        # Exact matches short-circuit before the judge call.
        # If the judge later raises, the exact matches should NOT be affected.
        class RaisingJudge:
            def judge_batch(self, items: list[JudgeItem]) -> list[float | None]:
                raise RuntimeError("nope")

        register("semantic", SemanticBatchComparator(RaisingJudge()))

        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "a": {"type": "string", "x-eval-compare": "semantic"},
                "b": {"type": "string", "x-eval-compare": "semantic"},
            },
        }
        gold = [{"a": "match_me", "b": "x"}]
        extracted = [{"a": "match_me", "b": "y"}]  # a is exact, b needs judge

        result = evaluate(gold, extracted, schema)
        by_path = {r.path: r for r in result.records[0].field_results}
        assert by_path["a"].status == "match"           # short-circuit survived
        assert by_path["b"].status == "batch_error"     # judge failed
        assert result.total_batch_errors == 1
