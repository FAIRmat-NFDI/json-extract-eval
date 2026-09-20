from json_extract_eval.core.field_result import FieldResult
from json_extract_eval.core.record import (
    RecordResult,
    build_record_result,
    build_run_result,
)


# --- build_record_result ---


class TestBuildRecordResult:
    def test_all_match(self) -> None:
        fields = [
            FieldResult("name", 1.0, "exact", "Alice", "Alice", "match"),
            FieldResult("age", 1.0, "numeric", 30, 30, "match"),
        ]
        r = build_record_result(0, fields, {"name": "Alice", "age": 30}, {"name": "Alice", "age": 30})
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1 == 1.0

    def test_all_mismatch(self) -> None:
        fields = [
            FieldResult("name", 0.0, "exact", "Alice", "Bob", "mismatch"),
            FieldResult("age", 0.0, "numeric", 30, 99, "mismatch"),
        ]
        r = build_record_result(0, fields, {}, {})
        assert r.precision == 0.0
        assert r.recall == 0.0
        assert r.f1 == 0.0

    def test_one_omission(self) -> None:
        fields = [
            FieldResult("name", 1.0, "exact", "Alice", "Alice", "match"),
            FieldResult("age", 0.0, "numeric", 30, None, "omission"),
        ]
        r = build_record_result(0, fields, {}, {})
        # recall: 1 / 2 = 0.5, precision: 1 / 1 = 1.0
        assert r.precision == 1.0
        assert r.recall == 0.5
        assert r.f1 > 0.0

    def test_one_hallucination(self) -> None:
        fields = [
            FieldResult("name", 1.0, "exact", "Alice", "Alice", "match"),
            FieldResult("extra", 0.0, "exact", None, "ghost", "hallucination"),
        ]
        r = build_record_result(0, fields, {}, {})
        # precision: 1 / 2 = 0.5, recall: 1 / 1 = 1.0
        assert r.precision == 0.5
        assert r.recall == 1.0
        assert r.f1 > 0.0

    def test_skipped_excluded_from_metrics(self) -> None:
        # Skipped fields are present in results but excluded from all counts
        fields = [
            FieldResult("name", 1.0, "exact", "Alice", "Alice", "match"),
            FieldResult("comment", 0.0, "", "blah", "blah", "skipped"),
        ]
        r = build_record_result(0, fields, {}, {})
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1 == 1.0

    def test_empty_field_results(self) -> None:
        r = build_record_result(0, [], {}, {})
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1 == 1.0

    def test_stores_gold_and_extracted(self) -> None:
        gold = {"name": "Alice"}
        extracted = {"name": "Bob"}
        fields = [FieldResult("name", 0.0, "exact", "Alice", "Bob", "mismatch")]
        r = build_record_result("doc-1", fields, gold, extracted)
        assert r.gold is gold
        assert r.extracted is extracted
        assert r.record_id == "doc-1"

    def test_mixed_statuses(self) -> None:
        fields = [
            FieldResult("a", 1.0, "exact", "x", "x", "match"),
            FieldResult("b", 0.0, "exact", "y", "z", "mismatch"),
            FieldResult("c", 0.0, "exact", "w", None, "omission"),
            FieldResult("d", 0.0, "exact", None, "ghost", "hallucination"),
            FieldResult("e", 0.0, "", "free", "text", "skipped"),
        ]
        r = build_record_result(0, fields, {}, {})
        # skipped field (e) excluded from all calculations
        # precision denominator: match(a) + mismatch(b) + hallucination(d) = 3
        # precision numerator: score(a)=1 + score(b)=0 + score(d)=0 = 1
        # precision = 1/3
        assert abs(r.precision - 1 / 3) < 1e-9
        # recall denominator: match(a) + mismatch(b) + omission(c) = 3
        # recall numerator: score(a)=1 + score(b)=0 + score(c)=0 = 1
        # recall = 1/3
        assert abs(r.recall - 1 / 3) < 1e-9


# --- build_run_result ---


class TestBuildRunResult:
    def _make_record(
        self,
        record_id: int,
        fields: list[FieldResult],
    ) -> RecordResult:
        return build_record_result(record_id, fields, {}, {})

    def test_single_record(self) -> None:
        fields = [
            FieldResult("name", 1.0, "exact", "Alice", "Alice", "match"),
        ]
        run = build_run_result([self._make_record(0, fields)])
        assert run.mean_f1 == 1.0
        assert run.total_records == 1
        assert run.total_fields == 1

    def test_aggregate_metrics(self) -> None:
        r1 = self._make_record(0, [
            FieldResult("name", 1.0, "exact", "A", "A", "match"),
        ])
        r2 = self._make_record(1, [
            FieldResult("name", 0.0, "exact", "A", "B", "mismatch"),
        ])
        run = build_run_result([r1, r2])
        assert run.mean_precision == 0.5
        assert run.mean_recall == 0.5
        assert run.total_records == 2

    def test_per_field_aggregation(self) -> None:
        r1 = self._make_record(0, [
            FieldResult("name", 1.0, "exact", "A", "A", "match"),
            FieldResult("age", 0.0, "numeric", 30, 99, "mismatch"),
        ])
        r2 = self._make_record(1, [
            FieldResult("name", 1.0, "exact", "B", "B", "match"),
            FieldResult("age", 1.0, "numeric", 50, 50, "match"),
        ])
        run = build_run_result([r1, r2])
        assert run.per_field["name"].mean_score == 1.0
        assert run.per_field["name"].matches == 2
        assert run.per_field["age"].mean_score == 0.5
        assert run.per_field["age"].mismatches == 1

    def test_per_field_omission_hallucination_counts(self) -> None:
        r1 = self._make_record(0, [
            FieldResult("name", 0.0, "exact", "A", None, "omission"),
            FieldResult("extra", 0.0, "exact", None, "ghost", "hallucination"),
        ])
        run = build_run_result([r1])
        assert run.per_field["name"].omissions == 1
        assert run.per_field["extra"].hallucinations == 1
        assert run.total_omissions == 1
        assert run.total_hallucinations == 1

    def test_empty_run(self) -> None:
        run = build_run_result([])
        assert run.mean_f1 == 1.0
        assert run.total_records == 0
        assert run.per_field == {}

    def test_skipped_excluded_from_run_counts(self) -> None:
        # Skipped fields are in results but excluded from total_fields and per_field
        r1 = self._make_record(0, [
            FieldResult("comment", 0.0, "", "a", "b", "skipped"),
        ])
        run = build_run_result([r1])
        assert "comment" not in run.per_field
        assert run.total_fields == 0
