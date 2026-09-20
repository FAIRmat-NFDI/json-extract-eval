"""Tests for null handling (Approach C): reclassify absent values."""

from json_extract_eval.postprocess import NullHandling, reclassify_nulls
from json_extract_eval.core.field_result import FieldResult
from json_extract_eval.core.schema import annotate_xeval
from json_extract_eval.evaluator import evaluate


class TestReclassifyNulls:
    def test_gold_null_extracted_value_becomes_hallucination(self) -> None:
        results = [
            FieldResult(
                path="a", score=0.0, comparator="exact",
                gold_value=None, extracted_value="hello", status="mismatch",
                gold_compared=None, extracted_compared="hello",
            ),
        ]
        reclassify_nulls(results, NullHandling())
        assert results[0].status == "hallucination"
        assert results[0].reason == "gold is absent"

    def test_gold_value_extracted_null_becomes_omission(self) -> None:
        results = [
            FieldResult(
                path="a", score=0.0, comparator="exact",
                gold_value="hello", extracted_value=None, status="mismatch",
                gold_compared="hello", extracted_compared=None,
            ),
        ]
        reclassify_nulls(results, NullHandling())
        assert results[0].status == "omission"
        assert results[0].reason == "extracted is absent"

    def test_both_null_skip(self) -> None:
        results = [
            FieldResult(
                path="a", score=1.0, comparator="exact",
                gold_value=None, extracted_value=None, status="match",
                gold_compared=None, extracted_compared=None,
            ),
        ]
        reclassify_nulls(results, NullHandling(both_absent_skip=True))
        assert results[0].status == "skipped"
        assert results[0].reason == "both absent"

    def test_both_null_match(self) -> None:
        results = [
            FieldResult(
                path="a", score=1.0, comparator="exact",
                gold_value=None, extracted_value=None, status="match",
                gold_compared=None, extracted_compared=None,
            ),
        ]
        reclassify_nulls(results, NullHandling(both_absent_skip=False))
        assert results[0].status == "match"
        assert results[0].score == 1.0

    def test_empty_string_as_absent(self) -> None:
        config = NullHandling(absent_values=[None, ""], both_absent_skip=True)
        results = [
            FieldResult(
                path="a", score=0.0, comparator="exact",
                gold_value="PVD", extracted_value="", status="mismatch",
                gold_compared="PVD", extracted_compared="",
            ),
            FieldResult(
                path="b", score=0.0, comparator="exact",
                gold_value="", extracted_value="hello", status="mismatch",
                gold_compared="", extracted_compared="hello",
            ),
            FieldResult(
                path="c", score=1.0, comparator="exact",
                gold_value="", extracted_value="", status="match",
                gold_compared="", extracted_compared="",
            ),
        ]
        reclassify_nulls(results, config)
        assert results[0].status == "omission"
        assert results[1].status == "hallucination"
        assert results[2].status == "skipped"

    def test_mixed_absent_values(self) -> None:
        config = NullHandling(absent_values=[None, ""], both_absent_skip=True)
        results = [
            FieldResult(
                path="a", score=1.0, comparator="exact",
                gold_value=None, extracted_value=None, status="match",
                gold_compared=None, extracted_compared=None,
            ),
            FieldResult(
                path="b", score=0.0, comparator="exact",
                gold_value=None, extracted_value="", status="mismatch",
                gold_compared=None, extracted_compared="",
            ),
        ]
        reclassify_nulls(results, config)
        assert results[0].status == "skipped"  # null vs null
        assert results[1].status == "skipped"  # null vs "" (both absent)

    def test_normal_values_unchanged(self) -> None:
        results = [
            FieldResult(
                path="a", score=1.0, comparator="exact",
                gold_value="PVD", extracted_value="PVD", status="match",
                gold_compared="PVD", extracted_compared="PVD",
            ),
            FieldResult(
                path="b", score=0.0, comparator="exact",
                gold_value="PVD", extracted_value="CVD", status="mismatch",
                gold_compared="PVD", extracted_compared="CVD",
            ),
        ]
        reclassify_nulls(results, NullHandling())
        assert results[0].status == "match"
        assert results[1].status == "mismatch"

    def test_skipped_fields_not_touched(self) -> None:
        results = [
            FieldResult(
                path="a", score=0.0, comparator="",
                gold_value=None, extracted_value="hello", status="skipped",
            ),
        ]
        reclassify_nulls(results, NullHandling())
        assert results[0].status == "skipped"

    def test_bool_not_confused_with_int(self) -> None:
        """absent_values=[0] should NOT match False (bool != int)."""
        config = NullHandling(absent_values=[0])
        results = [
            FieldResult(
                path="a", score=0.0, comparator="exact",
                gold_value=False, extracted_value="hello", status="mismatch",
                gold_compared=False, extracted_compared="hello",
            ),
        ]
        reclassify_nulls(results, config)
        assert results[0].status == "mismatch"  # NOT hallucination

    def test_uses_compared_values_when_available(self) -> None:
        """Reclassification should use post-transform values."""
        config = NullHandling(absent_values=[None, ""])
        results = [
            FieldResult(
                path="a", score=0.0, comparator="exact",
                gold_value="  ", extracted_value="hello", status="mismatch",
                gold_compared="", extracted_compared="hello",  # strip made it ""
            ),
        ]
        reclassify_nulls(results, config)
        # gold_compared="" is absent -> hallucination
        assert results[0].status == "hallucination"


class TestEvaluateWithNullHandling:
    def test_null_means_absent_via_evaluate(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "temp": {"type": "number"},
                "notes": {"type": "string"},
            },
        }
        annotate_xeval(schema)

        gold = [{"method": "PVD", "temp": 300, "notes": None}]
        extracted = [{"method": "PVD", "temp": None, "notes": None}]

        config = NullHandling(absent_values=[None], both_absent_skip=True)
        result = evaluate(
            gold, extracted, schema,
            post_process=lambda frs: reclassify_nulls(frs, config),
        )

        by_path = {
            r.path: r for r in result.records[0].field_results
        }
        assert by_path["method"].status == "match"
        assert by_path["temp"].status == "omission"
        assert by_path["notes"].status == "skipped"

        record = result.records[0]
        assert record.precision == 1.0
        assert record.recall == 0.5

    def test_default_no_null_handling(self) -> None:
        """Default: null is a value, compared normally."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "temp": {"type": "number"},
            },
        }
        annotate_xeval(schema)

        gold = [{"method": "PVD", "temp": 300}]
        extracted = [{"method": "PVD", "temp": None}]

        result = evaluate(gold, extracted, schema)
        by_path = {
            r.path: r for r in result.records[0].field_results
        }
        assert by_path["temp"].status == "mismatch"

    def test_both_absent_skip_false(self) -> None:
        """both_absent_skip=False counts null-null as match."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "notes": {"type": "string"},
            },
        }
        annotate_xeval(schema)

        config = NullHandling(both_absent_skip=False)
        result = evaluate(
            [{"notes": None}],
            [{"notes": None}],
            schema,
            post_process=lambda frs: reclassify_nulls(frs, config),
        )
        assert result.records[0].field_results[0].status == "match"

    def test_both_absent_skip_false_mixed_markers(self) -> None:
        """both_absent_skip=False with None vs '' -> match (both absent, equivalent)."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "notes": {"type": "string"},
            },
        }
        annotate_xeval(schema)

        config = NullHandling(absent_values=[None, ""], both_absent_skip=False)
        result = evaluate(
            [{"notes": None}],
            [{"notes": ""}],
            schema,
            post_process=lambda frs: reclassify_nulls(frs, config),
        )
        fr = result.records[0].field_results[0]
        assert fr.status == "match"
        assert fr.score == 1.0

    def test_empty_string_absent(self) -> None:
        """Empty string treated as absent when configured."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
            },
        }
        annotate_xeval(schema)

        config = NullHandling(absent_values=[None, ""])
        result = evaluate(
            [{"method": "PVD"}],
            [{"method": ""}],
            schema,
            post_process=lambda frs: reclassify_nulls(frs, config),
        )
        assert result.records[0].field_results[0].status == "omission"

    def test_nested_object_null_extracted(self) -> None:
        """Null extracted for an object field -> child leaves are omissions."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
            },
        }
        annotate_xeval(schema)

        config = NullHandling(absent_values=[None])
        result = evaluate(
            [{"address": {"street": "Main St", "city": "NYC"}}],
            [{"address": None}],
            schema,
            post_process=lambda frs: reclassify_nulls(frs, config),
        )
        by_path = {r.path: r for r in result.records[0].field_results}
        assert by_path["address.street"].status == "omission"
        assert by_path["address.city"].status == "omission"

    def test_transform_normalizes_to_absent(self) -> None:
        """Transform that strips whitespace, making value absent ('')."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "x-eval-transform": ["strip"],
                },
            },
        }
        annotate_xeval(schema)

        config = NullHandling(absent_values=[None, ""])
        result = evaluate(
            [{"notes": "real note"}],
            [{"notes": "   "}],  # strip -> "" -> absent
            schema,
            post_process=lambda frs: reclassify_nulls(frs, config),
        )
        fr = result.records[0].field_results[0]
        assert fr.status == "omission"
        assert fr.extracted_compared == ""
