import logging

import pytest

from struct_extract_eval.core.comparators.comparator import ComparatorResult
from struct_extract_eval.core.comparators.registry import _clear_registry, register
from struct_extract_eval.core.schema import annotate_xeval, parse_eval_schema
from struct_extract_eval.core.scoring import _score_object, score_record


def _make_schema(raw: dict[str, object]) -> "SchemaNode":  # noqa: F821
    """Helper: add defaults + parse into SchemaNode."""
    annotate_xeval(raw)
    return parse_eval_schema(raw)


# --- Flat objects ---


class TestFlatObject:
    def test_all_fields_match(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        })
        results = score_record(schema, {"name": "Alice", "age": 30}, {"name": "Alice", "age": 30})
        assert len(results) == 2
        assert all(r.score == 1.0 for r in results)
        assert all(r.status == "match" for r in results)

    def test_one_field_mismatches(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        })
        results = score_record(schema, {"name": "Alice", "age": 30}, {"name": "Bob", "age": 30})
        by_path = {r.path: r for r in results}
        assert by_path["name"].score == 0.0
        assert by_path["name"].status == "mismatch"
        assert by_path["age"].score == 1.0

    def test_boolean_exact(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {"active": {"type": "boolean"}},
        })
        results = score_record(schema, {"active": True}, {"active": True})
        assert results[0].score == 1.0

    def test_boolean_mismatch(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {"active": {"type": "boolean"}},
        })
        results = score_record(schema, {"active": True}, {"active": False})
        assert results[0].score == 0.0


# --- Missing fields ---


class TestMissingFields:
    def test_required_field_missing_is_omission(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        })
        results = score_record(schema, {"name": "Alice"}, {})
        assert len(results) == 1
        assert results[0].status == "omission"
        assert results[0].score == 0.0

    def test_optional_field_in_gold_missing_in_extracted_is_omission(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
        })
        results = score_record(schema, {"name": "Alice", "email": "a@b.com"}, {"name": "Alice"})
        # "email" is in gold, so extractor is expected to produce it -- omission
        assert len(results) == 2
        by_path = {r.path: r for r in results}
        assert by_path["name"].score == 1.0
        assert by_path["email"].status == "omission"
        assert by_path["email"].score == 0.0

    def test_optional_field_missing_in_both_is_not_counted(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
        })
        results = score_record(schema, {"name": "Alice"}, {"name": "Alice"})
        # "email" absent in both -- not counted
        assert len(results) == 1
        assert results[0].path == "name"

    def test_field_missing_in_gold_present_in_extracted_is_hallucination(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
        })
        results = score_record(schema, {"name": "Alice"}, {"name": "Alice", "email": "a@b.com"})
        # "email" not in gold but in extracted -- hallucination
        assert len(results) == 2
        by_path = {r.path: r for r in results}
        assert by_path["name"].score == 1.0
        assert by_path["email"].status == "hallucination"
        assert by_path["email"].score == 0.0

    def test_extra_field_not_in_schema_is_hallucination(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        })
        # "extra" is not in the schema -- detected as hallucination
        results = score_record(schema, {"name": "Alice"}, {"name": "Alice", "extra": "ignored"})
        by_path = {r.path: r for r in results}
        assert len(results) == 2
        assert by_path["name"].score == 1.0
        assert by_path["extra"].status == "hallucination"
        assert by_path["extra"].extracted_value == "ignored"
        assert by_path["extra"].gold_value is None

    def test_gold_field_not_in_schema_is_skipped_when_extracted_is_missing(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })

        results = score_record(
            schema,
            {"name": "Alice", "provenance": "annotator-1"},
            {"name": "Alice"},
        )
        by_path = {result.path: result for result in results}

        assert by_path["provenance"].status == "skipped"
        assert by_path["provenance"].extracted_value is None
        assert by_path["provenance"].reason == (
            "gold field not in schema, no comparator to apply"
        )

    def test_matching_unknown_gold_field_is_consumed_as_skip(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })

        results = score_record(
            schema,
            {"name": "Alice", "provenance": "annotator-1"},
            {"name": "Alice", "provenance": "annotator-1"},
        )
        by_path = {result.path: result for result in results}

        assert by_path["provenance"].status == "skipped"
        assert by_path["provenance"].extracted_value == "annotator-1"
        assert by_path["provenance"].reason == (
            "gold field not in schema, no comparator to apply; "
            "extracted also has this field"
        )
        assert not any(result.status == "hallucination" for result in results)

    def test_multiple_extra_fields_sorted_order(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        })
        # Multiple extra keys -- hallucinations emitted in sorted order
        results = score_record(
            schema,
            {"name": "Alice"},
            {"name": "Alice", "zebra": 1, "alpha": 2, "mid": 3},
        )
        hallucinations = [r for r in results if r.status == "hallucination"]
        assert len(hallucinations) == 3
        assert [h.path for h in hallucinations] == ["alpha", "mid", "zebra"]

    def test_extra_field_in_nested_object(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "details": {
                    "type": "object",
                    "properties": {
                        "color": {"type": "string"},
                    },
                },
            },
        })
        results = score_record(
            schema,
            {"details": {"color": "red"}},
            {"details": {"color": "red", "weight": 5}},
        )
        by_path = {r.path: r for r in results}
        assert by_path["details.color"].status == "match"
        assert by_path["details.weight"].status == "hallucination"
        assert by_path["details.weight"].extracted_value == 5


# --- Null handling ---


class TestNullHandling:
    def test_null_vs_null_is_match(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {"val": {"type": "string"}},
        })
        results = score_record(schema, {"val": None}, {"val": None})
        assert results[0].score == 1.0
        assert results[0].status == "match"

    def test_null_vs_value_is_mismatch(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {"val": {"type": "string"}},
        })
        results = score_record(schema, {"val": None}, {"val": "hello"})
        assert results[0].score == 0.0

    def test_value_vs_null_is_mismatch(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {"val": {"type": "string"}},
        })
        results = score_record(schema, {"val": "hello"}, {"val": None})
        assert results[0].score == 0.0


# --- Nested objects ---


class TestNestedObject:
    def test_inner_fields_compared(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "experiment": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "temp": {"type": "number"},
                    },
                },
            },
        })
        gold = {"experiment": {"name": "XRD", "temp": 300.0}}
        extracted = {"experiment": {"name": "XRD", "temp": 301.0}}
        results = score_record(schema, gold, extracted)
        by_path = {r.path: r for r in results}
        assert by_path["experiment.name"].score == 1.0
        # numeric comparator uses exact equality here (no tolerance configured) -- 300 vs 301
        assert by_path["experiment.temp"].score == 0.0

    def test_missing_nested_object_omits_children(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "experiment": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "temp": {"type": "number"},
                    },
                },
            },
        })
        results = score_record(schema, {"experiment": {"name": "XRD", "temp": 300}}, {})
        assert len(results) == 2
        assert all(r.status == "omission" for r in results)

    def test_empty_objects_match(self) -> None:
        # {} vs {} is one match for the object node itself, mirroring empty
        # arrays ([] vs [] -> match). Both sides agree there are no fields.
        schema = _make_schema({
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "properties": {"author": {"type": "string"}},
                },
            },
        })
        results = score_record(schema, {"meta": {}}, {"meta": {}})
        assert len(results) == 1
        assert results[0].path == "meta"
        assert results[0].status == "match"
        assert results[0].score == 1.0


# --- Ordered arrays ---


class TestOrderedArray:
    def test_same_length_match(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(schema, {"tags": ["a", "b"]}, {"tags": ["a", "b"]})
        assert len(results) == 2
        assert all(r.score == 1.0 for r in results)

    def test_same_length_mismatch(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(schema, {"tags": ["a", "b"]}, {"tags": ["a", "c"]})
        assert results[0].score == 1.0  # "a" == "a"
        assert results[1].score == 0.0  # "b" != "c"

    def test_gold_longer_produces_omissions(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(schema, {"tags": ["a", "b", "c"]}, {"tags": ["a"]})
        assert len(results) == 3
        assert results[0].score == 1.0
        assert results[1].status == "omission"
        assert results[2].status == "omission"

    def test_extracted_longer_produces_hallucinations(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(schema, {"tags": ["a"]}, {"tags": ["a", "b", "c"]})
        assert len(results) == 3
        assert results[0].score == 1.0
        assert results[1].status == "hallucination"
        assert results[2].status == "hallucination"

    def test_array_of_objects(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "temp": {"type": "number"},
                        },
                    },
                },
            },
        })
        gold = {"steps": [{"name": "deposit", "temp": 300}, {"name": "anneal", "temp": 500}]}
        extracted = {"steps": [{"name": "deposit", "temp": 300}, {"name": "anneal", "temp": 501}]}
        results = score_record(schema, gold, extracted)
        # First element: both fields match
        # Second element: name matches, temp mismatches
        assert len(results) == 4
        name_results = [r for r in results if "name" in r.path]
        assert all(r.score == 1.0 for r in name_results)

    def test_empty_arrays_match(self) -> None:
        # [] vs [] is one match for the array node itself.
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(schema, {"tags": []}, {"tags": []})
        assert len(results) == 1
        assert results[0].path == "tags"
        assert results[0].status == "match"
        assert results[0].score == 1.0

    def test_missing_array_in_extracted_produces_omissions(self) -> None:
        # gold has an array, extracted is missing the field entirely.
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(schema, {"tags": ["a", "b", "c"]}, {})
        assert len(results) == 3
        assert all(r.status == "omission" for r in results)
        assert all(r.score == 0 for r in results)
        # One omission per missing element, indexed by gold position -- the same
        # instance paths a present array would use (not the bare schema path tags[]).
        assert [r.path for r in results] == ["tags[0]", "tags[1]", "tags[2]"]


    def test_missing_array_in_gold_produces_hallucinations(self) -> None:
        # extracted has an array, gold is missing the field entirely.
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(schema, {}, {"tags": ["a", "b", "c"]})
        assert len(results) == 3
        assert all(r.status == "hallucination" for r in results)
        # Distinct negative indices (no gold counterpart), matching a present array.
        assert [r.path for r in results] == ["tags[-1]", "tags[-2]", "tags[-3]"]

    def test_missing_array_of_objects_indexes_each_element(self) -> None:
        # A fully-missing array of objects indexes each element distinctly
        # (steps[0].name, steps[1].name) instead of merging them at steps[].
        schema = _make_schema({
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "temp": {"type": "number"},
                        },
                    },
                },
            },
        })
        gold = {"steps": [{"name": "a", "temp": 1}, {"name": "b", "temp": 2}]}
        results = score_record(schema, gold, {})
        assert sorted(r.path for r in results) == [
            "steps[0].name", "steps[0].temp", "steps[1].name", "steps[1].temp",
        ]
        assert all(r.status == "omission" for r in results)

    def test_empty_gold_array_missing_extracted_is_omission(self) -> None:
        # gold has [], extracted is missing the field: 1 omission for the array node.
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(schema, {"tags": []}, {})
        assert len(results) == 1
        assert results[0].path == "tags"
        assert results[0].status == "omission"

    def test_empty_extracted_array_missing_gold_is_hallucination(self) -> None:
        # extracted has [], gold is missing the field: 1 hallucination for the array node.
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(schema, {}, {"tags": []})
        assert len(results) == 1
        assert results[0].path == "tags"
        assert results[0].status == "hallucination"

    def test_array_inside_missing_object(self) -> None:
        # Issue #31 nested case: object containing an array is missing entirely.
        schema = _make_schema({
            "type": "object",
            "properties": {
                "experiment": {
                    "type": "object",
                    "properties": {
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        })
        gold = {"experiment": {"tags": ["a", "b"]}}
        results = score_record(schema, gold, {})
        assert len(results) == 2
        assert all(r.status == "omission" for r in results)


class TestArrayTypeErrors:
    """Wrong container type at array paths: a value is present but not a list.

    Decided behavior (issues #56 / #82): a present-but-wrong container type
    counts as a single mismatch (penalizes precision and recall). A missing
    value (absent key, or null) stays omission/hallucination -- a missing
    list of N elements is N omissions. See array.md "Polymorphic fields".
    """

    def _schema(self) -> "SchemaNode":  # noqa: F821
        return _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })

    def test_gold_list_extracted_not_list(self) -> None:
        # gold=[a,b], extracted="bad" -> mismatch (extracted wrong type)
        schema = self._schema()
        results = score_record(schema, {"tags": ["a", "b"]}, {"tags": "bad"})
        assert len(results) == 1
        assert results[0].path == "tags"
        assert results[0].status == "mismatch"
        assert "type mismatch" in results[0].reason

    def test_gold_not_list_extracted_list(self) -> None:
        # gold="bad", extracted=[a,b] -> mismatch (symmetric, gold wrong type)
        schema = self._schema()
        results = score_record(schema, {"tags": "bad"}, {"tags": ["a", "b"]})
        assert len(results) == 1
        assert results[0].path == "tags"
        assert results[0].status == "mismatch"
        assert "type mismatch" in results[0].reason

    def test_gold_empty_extracted_not_list(self) -> None:
        # gold=[], extracted="bad" -> mismatch (extracted wrong type)
        schema = self._schema()
        results = score_record(schema, {"tags": []}, {"tags": "bad"})
        assert len(results) == 1
        assert results[0].path == "tags"
        assert results[0].status == "mismatch"

    def test_gold_not_list_extracted_empty(self) -> None:
        # gold="bad", extracted=[] -> mismatch (gold wrong type)
        schema = self._schema()
        results = score_record(schema, {"tags": "bad"}, {"tags": []})
        assert len(results) == 1
        assert results[0].path == "tags"
        assert results[0].status == "mismatch"

    def test_both_not_list(self) -> None:
        # gold="bad", extracted="also_bad" -> mismatch (both wrong type, differ)
        schema = self._schema()
        results = score_record(schema, {"tags": "bad"}, {"tags": "also_bad"})
        assert len(results) == 1
        assert results[0].path == "tags"
        assert results[0].status == "mismatch"

    def test_both_not_list_but_equal_is_match(self) -> None:
        # gold="bad", extracted="bad" -> match: the extractor reproduced gold
        # exactly, even though both are off-shape (no comparator assigned).
        schema = self._schema()
        results = score_record(schema, {"tags": "bad"}, {"tags": "bad"})
        assert len(results) == 1
        assert results[0].path == "tags"
        assert results[0].status == "match"
        assert results[0].score == 1.0

    def test_gold_not_list_extracted_missing(self) -> None:
        # gold="bad", extracted missing -> 1 omission for the array node
        schema = self._schema()
        results = score_record(schema, {"tags": "bad"}, {})
        assert len(results) == 1
        assert results[0].path == "tags"
        assert results[0].status == "omission"

    def test_gold_missing_extracted_not_list(self) -> None:
        # gold missing, extracted="bad" -> 1 hallucination for the array node
        schema = self._schema()
        results = score_record(schema, {}, {"tags": "bad"})
        assert len(results) == 1
        assert results[0].path == "tags"
        assert results[0].status == "hallucination"

    def test_gold_list_extracted_null(self) -> None:
        # gold=[a,b], extracted=null -> omissions (null is "missing", scaled
        # to data lost: a 2-element list missing -> 2 omissions)
        schema = self._schema()
        results = score_record(schema, {"tags": ["a", "b"]}, {"tags": None})
        assert len(results) == 2
        assert all(r.status == "omission" for r in results)

    def test_gold_null_extracted_list(self) -> None:
        # gold=null, extracted=[a,b] -> hallucinations
        schema = self._schema()
        results = score_record(schema, {"tags": None}, {"tags": ["a", "b"]})
        assert len(results) == 2
        assert all(r.status == "hallucination" for r in results)

    def test_gold_not_list_extracted_null(self) -> None:
        # gold="bad", extracted=null -> 1 omission (extracted absent)
        schema = self._schema()
        results = score_record(schema, {"tags": "bad"}, {"tags": None})
        assert len(results) == 1
        assert results[0].path == "tags"
        assert results[0].status == "omission"

    def test_gold_null_extracted_not_list(self) -> None:
        # gold=null, extracted="bad" -> 1 hallucination (gold absent)
        schema = self._schema()
        results = score_record(schema, {"tags": None}, {"tags": "bad"})
        assert len(results) == 1
        assert results[0].path == "tags"
        assert results[0].status == "hallucination"

    def test_both_null(self) -> None:
        # gold=null, extracted=null -> nothing scorable
        schema = self._schema()
        results = score_record(schema, {"tags": None}, {"tags": None})
        assert results == []


class TestObjectTypeErrors:
    """Wrong container type at object paths.

    Object nodes now share the same wrong-type / missing policy as arrays
    (issues #56 / #82): present-but-wrong type -> mismatch (or match if the raw
    values are equal); null -> omission/hallucination. Previously a non-dict at
    an object path was silently coerced to {} (all omissions/hallucinations).
    """

    def _schema(self) -> "SchemaNode":  # noqa: F821
        return _make_schema({
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                },
            },
        })

    def test_gold_dict_extracted_not_dict(self) -> None:
        # gold={a:1}, extracted="bad" -> single mismatch (not all-omissions)
        schema = self._schema()
        results = score_record(schema, {"meta": {"a": "1"}}, {"meta": "bad"})
        assert len(results) == 1
        assert results[0].path == "meta"
        assert results[0].status == "mismatch"

    def test_gold_not_dict_extracted_dict(self) -> None:
        # gold="bad", extracted={a:1} -> single mismatch (not all-hallucinations)
        schema = self._schema()
        results = score_record(schema, {"meta": "bad"}, {"meta": {"a": "1"}})
        assert len(results) == 1
        assert results[0].path == "meta"
        assert results[0].status == "mismatch"

    def test_both_not_dict_but_equal_is_match(self) -> None:
        # gold="bad", extracted="bad" -> match (reproduced exactly, off-shape)
        schema = self._schema()
        results = score_record(schema, {"meta": "bad"}, {"meta": "bad"})
        assert len(results) == 1
        assert results[0].path == "meta"
        assert results[0].status == "match"

    def test_gold_dict_extracted_null(self) -> None:
        # gold={a:1}, extracted=null -> omission for the present gold field
        schema = self._schema()
        results = score_record(schema, {"meta": {"a": "1"}}, {"meta": None})
        assert len(results) == 1
        assert results[0].path == "meta.a"
        assert results[0].status == "omission"

    def test_gold_null_extracted_dict(self) -> None:
        # gold=null, extracted={a:1} -> hallucination for the extracted field
        schema = self._schema()
        results = score_record(schema, {"meta": None}, {"meta": {"a": "1"}})
        assert len(results) == 1
        assert results[0].path == "meta.a"
        assert results[0].status == "hallucination"


class TestPolymorphicObjectOrArray:
    """A field that is sometimes an object, sometimes an array, handled by a
    custom comparator via the escape hatch (issue #82). Element-level scoring
    of both shapes is tracked in issue #83."""

    @pytest.fixture(autouse=True)
    def _clean(self) -> None:
        _clear_registry()

    @staticmethod
    def _shape_aware(
        gold: object, extracted: object, params: dict[str, object]
    ) -> ComparatorResult:
        # Treat equal-by-value as a match regardless of object-vs-array shape.
        return ComparatorResult(
            score=1.0 if gold == extracted else 0.0, comparator="shape_aware"
        )

    def test_comparator_handles_object_and_array_forms(self) -> None:
        register("shape_aware", self._shape_aware)
        schema = _make_schema({
            "type": "object",
            "properties": {
                "body": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "x-eval-compare": "shape_aware",
                },
            },
        })
        # gold is an object, extracted is an array -> comparator owns both
        mismatch = score_record(schema, {"body": {"a": "1"}}, {"body": ["x"]})
        assert mismatch[0].comparator == "shape_aware"
        assert mismatch[0].status == "mismatch"
        # both arrays and equal -> match (no structural recursion)
        match = score_record(schema, {"body": ["x"]}, {"body": ["x"]})
        assert match[0].comparator == "shape_aware"
        assert match[0].status == "match"


# --- Transforms ---


class TestTransforms:
    def test_lowercase_before_compare(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-eval-transform": ["lowercase"],
                },
            },
        })
        results = score_record(schema, {"name": "ALICE"}, {"name": "alice"})
        assert results[0].score == 1.0

    def test_transform_skipped_for_null(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-eval-transform": ["lowercase"],
                },
            },
        })
        results = score_record(schema, {"name": None}, {"name": None})
        assert results[0].score == 1.0


# --- Deeply nested ---


class TestDeeplyNested:
    def test_object_array_object_leaf(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "experiment": {
                    "type": "object",
                    "properties": {
                        "samples": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "value": {"type": "number"},
                                },
                            },
                        },
                    },
                },
            },
        })
        gold = {"experiment": {"samples": [
            {"id": "S1", "value": 1.5},
            {"id": "S2", "value": 3.2},
        ]}}
        extracted = {"experiment": {"samples": [
            {"id": "S1", "value": 1.5},
        ]}}
        results = score_record(schema, gold, extracted)
        # 2 fields from matched element + 2 omission fields from missing element
        matched = [r for r in results if r.status == "match"]
        omissions = [r for r in results if r.status == "omission"]
        assert len(matched) == 2  # S1's id and value
        assert len(omissions) == 2  # S2's id and value


# --- Comparator escape hatch (issue #82) ---


class TestComparatorEscapeHatch:
    """Explicit x-eval-compare on a container node bypasses structural scoring.

    The comparator receives the whole raw value, so a polymorphic field whose
    type varies (scalar OR list OR object) is handled by the comparator rather
    than the object/array machinery.
    """

    @pytest.fixture(autouse=True)
    def _clean(self) -> None:
        _clear_registry()

    @staticmethod
    def _poly(gold: object, extracted: object, params: dict[str, object]) -> ComparatorResult:
        # Equal regardless of container type -- the comparator owns the decision.
        return ComparatorResult(
            score=1.0 if gold == extracted else 0.0, comparator="poly"
        )

    def test_array_node_comparator_sees_raw_scalar(self) -> None:
        register("poly", self._poly)
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "x-eval-compare": "poly",
                },
            },
        })
        # gold is a list, extracted is a scalar: no structural skip/mismatch --
        # the comparator decides on the raw values.
        results = score_record(schema, {"tags": ["a"]}, {"tags": "a"})
        assert len(results) == 1
        assert results[0].path == "tags"
        assert results[0].comparator == "poly"
        assert results[0].status == "mismatch"  # raw list != raw scalar

    def test_array_node_comparator_polymorphic_match(self) -> None:
        register("poly", self._poly)
        schema = _make_schema({
            "type": "object",
            "properties": {
                "val": {
                    "type": "array",
                    "items": {"type": "string"},
                    "x-eval-compare": "poly",
                },
            },
        })
        # both scalars, equal -> the comparator returns match (no array recursion)
        results = score_record(schema, {"val": "x"}, {"val": "x"})
        assert len(results) == 1
        assert results[0].comparator == "poly"
        assert results[0].status == "match"

    def test_object_node_comparator_scored_as_whole(self) -> None:
        register("poly", self._poly)
        schema = _make_schema({
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "x-eval-compare": "poly",
                },
            },
        })
        # object node handled by the comparator as a whole, not per-child
        results = score_record(schema, {"meta": {"a": "1"}}, {"meta": {"a": "1"}})
        assert len(results) == 1
        assert results[0].path == "meta"
        assert results[0].comparator == "poly"
        assert results[0].status == "match"


# --- Array Instance paths ---


class TestArrayInstancePaths:
    """FieldResult.path should carry element indices for array elements."""

    def test_flat_array_elements_have_indices(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        })
        results = score_record(schema, {"tags": ["a", "b"]}, {"tags": ["a", "b"]})
        assert results[0].path == "tags[0]"
        assert results[1].path == "tags[1]"

    def test_array_of_objects_elements_have_indices(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": {"type": "object", "properties": {
                    "name": {"type": "string"},
                    "temp": {"type": "number"},
                }}},
            },
        })
        gold = {"steps": [{"name": "deposit", "temp": 300}, {"name": "anneal", "temp": 500}]}
        ext = {"steps": [{"name": "deposit", "temp": 300}, {"name": "anneal", "temp": 999}]}
        results = score_record(schema, gold, ext)
        paths = [r.path for r in results]
        assert "steps[0].name" in paths
        assert "steps[0].temp" in paths
        assert "steps[1].name" in paths
        assert "steps[1].temp" in paths

    def test_omissions_have_indices(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        })
        results = score_record(schema, {"tags": ["a", "b", "c"]}, {"tags": ["a"]})
        assert results[0].path == "tags[0]"  # match
        assert results[1].path == "tags[1]"  # omission
        assert results[2].path == "tags[2]"  # omission

    def test_hallucinations_have_unique_negative_indices(self) -> None:
        # Hallucinated elements (no gold counterpart) get distinct negative
        # indices counting down (-1, -2, ...), mirroring how omitted/matched
        # elements use gold's positive indices. This keeps each hallucinated
        # element distinct in per-field aggregation and parent-path grouping.
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        })
        results = score_record(schema, {"tags": ["a"]}, {"tags": ["a", "b", "c"]})
        assert results[0].path == "tags[0]"  # match
        assert results[1].path == "tags[-1]"  # 1st hallucination
        assert results[2].path == "tags[-2]"  # 2nd hallucination

    def test_hallucinated_objects_do_not_collide(self) -> None:
        # Two hallucinated object elements must not share a parent path (which
        # would merge their fields in aggregation / compound grouping).
        schema = _make_schema({
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "temp": {"type": "number"},
                        },
                    },
                },
            },
        })
        results = score_record(
            schema,
            {"steps": []},
            {"steps": [{"name": "a", "temp": 1}, {"name": "b", "temp": 2}]},
        )
        paths = sorted(r.path for r in results)
        assert paths == ["steps[-1].name", "steps[-1].temp", "steps[-2].name", "steps[-2].temp"]
        assert all(r.status == "hallucination" for r in results)

    def test_nested_arrays_both_levels_have_indices(self) -> None:
        """layers[0].steps[1].temp -- each array level gets its own index."""
        schema = _make_schema({
            "type": "object",
            "properties": {
                "layers": {"type": "array", "items": {"type": "object", "properties": {
                    "name": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "object", "properties": {
                        "action": {"type": "string"},
                        "temp": {"type": "number"},
                    }}},
                }}},
            },
        })
        gold = {"layers": [
            {"name": "L1", "steps": [
                {"action": "deposit", "temp": 300},
                {"action": "anneal", "temp": 500},
            ]},
            {"name": "L2", "steps": [
                {"action": "etch", "temp": 100},
            ]},
        ]}
        ext = {"layers": [
            {"name": "L1", "steps": [
                {"action": "deposit", "temp": 300},
                {"action": "anneal", "temp": 999},
            ]},
            {"name": "L2", "steps": [
                {"action": "etch", "temp": 100},
            ]},
        ]}
        results = score_record(schema, gold, ext)
        paths = [r.path for r in results]
        # Outer array: layers[0], layers[1]
        assert "layers[0].name" in paths
        assert "layers[1].name" in paths
        # Inner array: steps[0], steps[1] under each layer
        assert "layers[0].steps[0].action" in paths
        assert "layers[0].steps[0].temp" in paths
        assert "layers[0].steps[1].action" in paths
        assert "layers[0].steps[1].temp" in paths
        assert "layers[1].steps[0].action" in paths
        assert "layers[1].steps[0].temp" in paths
        # Verify the mismatch is at the right path
        mismatch = [r for r in results if r.status == "mismatch"]
        assert len(mismatch) == 1
        assert mismatch[0].path == "layers[0].steps[1].temp"

    def test_empty_array_match_uses_array_path_not_element(self) -> None:
        """[] vs [] is a match on the array node itself, no element index."""
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        })
        results = score_record(schema, {"tags": []}, {"tags": []})
        assert len(results) == 1
        assert results[0].path == "tags"  # array-level, no index


# --- Skip fields ---


class TestSkipFields:
    def test_skip_field_included_in_results_as_skipped(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string", "x-eval-skip": True},
            },
        })
        results = score_record(
            schema,
            {"name": "Alice", "description": "some text"},
            {"name": "Alice", "description": "other text"},
        )
        # description is skip -- present in results for visibility, with status "skipped"
        assert len(results) == 2
        by_path = {r.path: r for r in results}
        assert by_path["name"].status == "match"
        assert by_path["description"].status == "skipped"
        assert by_path["description"].reason == "skipped by x-eval-skip"
        assert by_path["description"].gold_value == "some text"
        assert by_path["description"].extracted_value == "other text"

    def test_skip_field_missing_in_extracted_no_omission(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string", "x-eval-skip": True},
            },
        })
        results = score_record(
            schema,
            {"name": "Alice", "description": "some text"},
            {"name": "Alice"},
        )
        # description is skip -- still "skipped", not "omission"
        assert len(results) == 2
        by_path = {r.path: r for r in results}
        assert by_path["name"].status == "match"
        assert by_path["description"].status == "skipped"

    def test_skip_field_missing_in_gold_no_hallucination(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string", "x-eval-skip": True},
            },
        })
        results = score_record(
            schema,
            {"name": "Alice"},
            {"name": "Alice", "description": "extra text"},
        )
        # description is skip -- still "skipped", not "hallucination"
        assert len(results) == 2
        by_path = {r.path: r for r in results}
        assert by_path["name"].status == "match"
        assert by_path["description"].status == "skipped"

    def test_skip_excluded_from_metrics(self) -> None:
        # x-eval-skip: true
        # skip fields appear in results but don't affect precision/recall/F1
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string", "x-eval-skip": True},
            },
        })
        results = score_record(
            schema,
            {"name": "Alice", "description": "text"},
            {"name": "Alice"},
        )
        assert len(results) == 2
        # Only "name" should contribute to metrics
        scored = [r for r in results if r.status != "skipped"]
        assert len(scored) == 1
        assert scored[0].path == "name"



class TestScoreObjectEdgeCases:
    def test_same_field_name_at_different_depths(self) -> None:
        # "name" appears at experiment.name and at experiment.sample.name.
        # Lookup is per node, so the two must not interfere with each other.
        schema = _make_schema({
            "type": "object",
            "properties": {
                "experiment": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "sample": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        },
                    },
                },
            },
        })
        gold = {"experiment": {"name": "XRD run", "sample": {"name": "Si wafer"}}}
        extracted = {"experiment": {"name": "XRD run", "sample": {"name": "WRONG"}}}
        results = score_record(schema, gold, extracted)
        by_path = {r.path: r for r in results}
        assert len(results) == 2
        assert by_path["experiment.name"].status == "match"
        assert by_path["experiment.sample.name"].status == "mismatch"
        assert by_path["experiment.sample.name"].gold_value == "Si wafer"
        assert by_path["experiment.sample.name"].extracted_value == "WRONG"

    def test_skip_on_object_node_does_not_traverse_children(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "x-eval-skip": True,
                    "properties": {"a": {"type": "string"}},
                },
            },
        })
        results = score_record(schema, {"meta": {"a": "1"}}, {"meta": {"a": "2"}})
        assert len(results) == 1
        assert results[0].path == "meta"
        assert results[0].status == "skipped"
        assert results[0].gold_value == {"a": "1"}
        assert results[0].extracted_value == {"a": "2"}

    def test_extra_key_with_container_value_is_single_hallucination(self) -> None:
        # An unknown key is reported once with its whole value, not expanded
        # per inner field -- there is no schema to expand it against.
        schema = _make_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        blob = {"p": 1, "q": [1, 2]}
        results = score_record(schema, {"name": "x"}, {"name": "x", "blob": blob})
        by_path = {r.path: r for r in results}
        assert len(results) == 2
        assert by_path["name"].status == "match"
        assert by_path["blob"].status == "hallucination"
        assert by_path["blob"].gold_value is None
        assert by_path["blob"].extracted_value == blob

    def test_nested_empty_gold_vs_nonempty_extracted(self) -> None:
        # The both-empty short-circuit ({} vs {} -> match) must not fire when
        # only one side is empty; children are scored normally.
        schema = _make_schema({
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                },
            },
        })
        results = score_record(schema, {"meta": {}}, {"meta": {"a": "1"}})
        assert len(results) == 1
        assert results[0].path == "meta.a"
        assert results[0].status == "hallucination"

    def test_nested_nonempty_gold_vs_empty_extracted(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                },
            },
        })
        results = score_record(schema, {"meta": {"a": "1"}}, {"meta": {}})
        assert len(results) == 1
        assert results[0].path == "meta.a"
        assert results[0].status == "omission"

    def test_result_order_is_schema_order_then_sorted_extras(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "zeta": {"type": "string"},
                "alpha": {"type": "string"},
            },
        })
        gold = {"zeta": "1", "alpha": "2"}
        extracted = {"alpha": "2", "zeta": "1", "b_extra": "x", "a_extra": "y"}
        results = score_record(schema, gold, extracted)
        assert [r.path for r in results] == ["zeta", "alpha", "a_extra", "b_extra"]

    def test_skip_field_absent_on_both_sides_still_emitted(self) -> None:
        # Skip results are emitted unconditionally for visibility, even when
        # neither side has the field.
        schema = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string", "x-eval-skip": True},
            },
        })
        results = score_record(schema, {"name": "Alice"}, {"name": "Alice"})
        by_path = {r.path: r for r in results}
        assert len(results) == 2
        assert by_path["description"].status == "skipped"
        assert by_path["description"].gold_value is None
        assert by_path["description"].extracted_value is None


# --- _score_object called directly (issue #115) ---


class TestScoreObjectDirect:

    def test_requires_dicts_on_both_sides(self) -> None:
        # _score_container is responsible for routing non-dicts elsewhere.
        node = _make_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        with pytest.raises(AssertionError):
            _score_object(node, {"name": "a"}, "not a dict")
        with pytest.raises(AssertionError):
            _score_object(node, None, {"name": "a"})

    def test_both_empty_is_single_match_at_node_path(self) -> None:
        node = _make_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        results = _score_object(node, {}, {})
        assert len(results) == 1
        assert results[0].path == ""
        assert results[0].status == "match"
        assert results[0].score == 1.0
        assert results[0].gold_value == {}
        assert results[0].extracted_value == {}

    def test_child_present_on_both_sides_is_compared(self) -> None:
        node = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        })
        results = _score_object(node, {"name": "a", "age": 1}, {"name": "a", "age": 2})
        by_path = {r.path: r for r in results}
        assert len(results) == 2
        assert by_path["name"].status == "match"
        assert by_path["age"].status == "mismatch"

    def test_child_in_gold_only_is_omission(self) -> None:
        node = _make_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        results = _score_object(node, {"name": "a"}, {})
        assert len(results) == 1
        assert results[0].path == "name"
        assert results[0].status == "omission"
        assert results[0].gold_value == "a"
        assert results[0].extracted_value is None

    def test_child_in_extracted_only_is_hallucination(self) -> None:
        node = _make_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        results = _score_object(node, {}, {"name": "a"})
        assert len(results) == 1
        assert results[0].path == "name"
        assert results[0].status == "hallucination"
        assert results[0].gold_value is None
        assert results[0].extracted_value == "a"

    def test_child_absent_on_both_sides_produces_nothing(self) -> None:
        node = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        })
        results = _score_object(node, {"name": "a"}, {"name": "a"})
        assert [r.path for r in results] == ["name"]

    def test_skip_child_is_always_emitted_and_never_compared(self) -> None:
        node = _make_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "notes": {"type": "string", "x-eval-skip": True},
            },
        })
        # Present on both sides with different values: still "skipped".
        results = _score_object(node, {"name": "a", "notes": "x"}, {"name": "a", "notes": "y"})
        by_path = {r.path: r for r in results}
        assert by_path["notes"].status == "skipped"
        assert by_path["notes"].score == 0.0
        assert by_path["notes"].gold_value == "x"
        assert by_path["notes"].extracted_value == "y"
        # Absent on both sides: still emitted.
        results = _score_object(node, {"name": "a"}, {"name": "a"})
        by_path = {r.path: r for r in results}
        assert by_path["notes"].status == "skipped"
        assert by_path["notes"].gold_value is None
        assert by_path["notes"].extracted_value is None

    def test_unknown_extracted_key_is_hallucination_with_whole_value(self) -> None:
        node = _make_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        blob = {"p": 1, "q": [1, 2]}
        results = _score_object(node, {"name": "a"}, {"name": "a", "blob": blob})
        by_path = {r.path: r for r in results}
        assert len(results) == 2
        assert by_path["blob"].status == "hallucination"
        assert by_path["blob"].score == 0.0
        assert by_path["blob"].comparator == ""
        assert by_path["blob"].gold_value is None
        assert by_path["blob"].extracted_value == blob

    def test_output_order_is_schema_order_then_sorted_unknown_keys(self) -> None:
        node = _make_schema({
            "type": "object",
            "properties": {
                "zeta": {"type": "string"},
                "alpha": {"type": "string"},
            },
        })
        gold = {"alpha": "1", "zeta": "2"}
        extracted = {"b_extra": "x", "zeta": "2", "a_extra": "y", "alpha": "1"}
        results = _score_object(node, gold, extracted)
        assert [r.path for r in results] == ["zeta", "alpha", "a_extra", "b_extra"]

    def test_nested_node_prefixes_paths_with_node_path(self) -> None:
        # Call _score_object on a child node, not the root, and check every
        # emitted path (schema child, unknown key) carries the node's prefix.
        root = _make_schema({
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                },
            },
        })
        meta_node = root.children[0]
        assert meta_node.path == "meta"
        results = _score_object(meta_node, {"a": "1"}, {"a": "1", "extra": "x"})
        assert [r.path for r in results] == ["meta.a", "meta.extra"]
        assert results[0].status == "match"
        assert results[1].status == "hallucination"

    def test_nested_node_both_empty_uses_node_path(self) -> None:
        root = _make_schema({
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                },
            },
        })
        meta_node = root.children[0]
        results = _score_object(meta_node, {}, {})
        assert len(results) == 1
        assert results[0].path == "meta"
        assert results[0].status == "match"

    def test_same_field_name_at_different_depths(self) -> None:
        root = _make_schema({
            "type": "object",
            "properties": {
                "experiment": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "sample": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        },
                    },
                },
            },
        })
        experiment_node = root.children[0]
        results = _score_object(
            experiment_node,
            {"name": "XRD run", "sample": {"name": "Si wafer"}},
            {"name": "XRD run", "sample": {"name": "WRONG"}},
        )
        by_path = {r.path: r for r in results}
        assert len(results) == 2
        assert by_path["experiment.name"].status == "match"
        assert by_path["experiment.sample.name"].status == "mismatch"


class TestListValuedType:
    def test_multi_type_default_exact_match(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {"q": {"type": ["string", "object"],
                                  "properties": {"value": {"type": "number"}}}},
        })
        # identical objects -> exact match
        results = score_record(schema, {"q": {"value": 1}}, {"q": {"value": 1}})
        assert len(results) == 1
        assert results[0].path == "q"
        assert results[0].status == "match"

    def test_multi_type_default_exact_cross_shape_mismatch(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {"q": {"type": ["string", "object"],
                                  "properties": {"value": {"type": "number"}}}},
        })
        # string vs object -> exact mismatch (default coarse behavior)
        results = score_record(schema, {"q": "x"}, {"q": {"value": 1}})
        assert len(results) == 1
        assert results[0].status == "mismatch"

    def test_multi_type_custom_comparator(self) -> None:
        def cmp_loose(gold: object, extracted: object, params: dict[str, object]) -> ComparatorResult:
            return ComparatorResult(score=1.0, comparator="loose")

        register("loose", cmp_loose)
        try:
            schema = _make_schema({
                "type": "object",
                "properties": {"q": {"type": ["string", "object"],
                                     "x-eval-compare": "loose"}},
            })
            results = score_record(schema, {"q": "x"}, {"q": {"value": 1}})
            assert results[0].status == "match"
        finally:
            _clear_registry()


# --- One-sided walk (omission / hallucination of a whole subtree) ---


def _meta_schema() -> "SchemaNode":  # noqa: F821
    # Fresh dict each call: _make_schema mutates its input.
    return _make_schema({
        "type": "object",
        "properties": {
            "meta": {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "note": {"type": "string", "x-eval-skip": True},
                },
            },
        },
    })


class TestOneSidedWalk:
    """Branches of the one-sided walk that no other test reaches."""

    def test_omitted_subtree_emits_nothing_for_skip_child(self) -> None:
        results = score_record(_meta_schema(), {"meta": {"a": "1", "note": "x"}}, {})
        assert [(r.path, r.status) for r in results] == [("meta.a", "omission")]

    def test_hallucinated_subtree_emits_nothing_for_skip_child(self) -> None:
        results = score_record(_meta_schema(), {}, {"meta": {"a": "1", "note": "x"}})
        assert [(r.path, r.status) for r in results] == [("meta.a", "hallucination")]

    def test_omitted_object_with_wrong_typed_gold_warns_and_emits_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            results = score_record(_meta_schema(), {"meta": "oops"}, {})
        assert results == []
        assert "Expected dict at 'meta', got str in gold" in caplog.text

    def test_hallucinated_object_with_wrong_typed_extracted_warns_and_emits_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            results = score_record(_meta_schema(), {}, {"meta": "oops"})
        assert results == []
        assert "Expected dict at 'meta', got str in extracted" in caplog.text
