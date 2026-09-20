"""Tests for array alignment strategies (x-eval-align).

Covers:
- Key-field alignment: match by a unique identifier field
- Hungarian alignment: optimal bipartite matching by F1
- Default behavior: no x-eval-align = ordered
- Edge cases: empty arrays, missing key fields, duplicate keys
"""

import pytest

from json_extract_eval.core.comparators.comparator import BatchItem, ComparatorResult
from json_extract_eval.core.comparators.registry import _clear_registry, register
from json_extract_eval.core.schema import (
    SchemaError,
    annotate_xeval,
    parse_eval_schema,
)
from json_extract_eval.core.scoring import score_record
from json_extract_eval.evaluator import evaluate


def _make_schema(raw: dict[str, object]) -> "SchemaNode":
    annotate_xeval(raw)
    return parse_eval_schema(raw)


# --- Schema validation ---


class TestAlignValidation:
    def test_align_must_be_dict(self) -> None:
        with pytest.raises(SchemaError, match="must be a dict"):
            _make_schema({
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "x-eval-align": "ordered",
                    },
                },
            })

    def test_align_must_have_ordered_or_match_by(self) -> None:
        with pytest.raises(SchemaError, match="'ordered' or 'match_by'"):
            _make_schema({
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "x-eval-align": {"foo": "bar"},
                    },
                },
            })

    def test_key_field_requires_key(self) -> None:
        with pytest.raises(SchemaError, match="requires a 'key'"):
            _make_schema({
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                        },
                        "x-eval-align": {"match_by": "key_field"},
                    },
                },
            })

    def test_valid_key_field_align_parses(self) -> None:
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
                    "x-eval-align": {
                        "match_by": "key_field",
                        "key": "name",
                    },
                },
            },
        })
        steps_node = schema.children[0]
        assert steps_node.align is not None
        assert steps_node.align["match_by"] == "key_field"
        assert steps_node.align["key"] == "name"

    def test_ordered_true_parses(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "x-eval-align": {"ordered": True},
                },
            },
        })
        tags_node = schema.children[0]
        assert tags_node.align is not None
        assert tags_node.align.get("ordered") is True


# --- Key-field alignment ---


def _steps_schema(
    align: dict[str, object] | None = None,
) -> dict[str, object]:
    """Helper: schema with a steps array of {name, temp} objects."""
    schema: dict[str, object] = {
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
    }
    if align is not None:
        schema["properties"]["steps"]["x-eval-align"] = align  # type: ignore[index]
    return schema


class TestKeyFieldAlignment:
    def test_reordered_elements_match(self) -> None:
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {
            "steps": [
                {"name": "anneal", "temp": 500},
                {"name": "deposit", "temp": 300},
            ]
        }
        extracted = {
            "steps": [
                {"name": "deposit", "temp": 300},
                {"name": "anneal", "temp": 500},
            ]
        }
        results = score_record(schema, gold, extracted)
        assert all(r.status == "match" for r in results)
        assert len(results) == 4  # two names, two temps

    def test_ordered_mismatch_same_data(self) -> None:
        # Same data as above, but with ordered matching: different order = mismatch
        schema = _make_schema(_steps_schema())  # no align = ordered
        gold = {
            "steps": [
                {"name": "anneal", "temp": 500},
                {"name": "deposit", "temp": 300},
            ]
        }
        extracted = {
            "steps": [
                {"name": "deposit", "temp": 300},
                {"name": "anneal", "temp": 500},
            ]
        }
        results = score_record(schema, gold, extracted)
        # Positional: anneal vs deposit = mismatch, deposit vs anneal = mismatch
        name_results = [r for r in results if r.path.endswith("name")]
        assert all(r.status == "mismatch" for r in name_results)

    def test_missing_extracted_element_is_omission(self) -> None:
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {
            "steps": [
                {"name": "anneal", "temp": 500},
                {"name": "deposit", "temp": 300},
            ]
        }
        extracted = {
            "steps": [
                {"name": "anneal", "temp": 500},
            ]
        }
        results = score_record(schema, gold, extracted)
        matches = [r for r in results if r.status == "match"]
        omissions = [r for r in results if r.status == "omission"]
        assert len(matches) == 2  # anneal.name + anneal.temp
        assert len(omissions) == 2  # deposit.name + deposit.temp

    def test_extra_extracted_element_is_hallucination(self) -> None:
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {
            "steps": [
                {"name": "anneal", "temp": 500},
            ]
        }
        extracted = {
            "steps": [
                {"name": "anneal", "temp": 500},
                {"name": "etch", "temp": 200},
            ]
        }
        results = score_record(schema, gold, extracted)
        matches = [r for r in results if r.status == "match"]
        hallucinations = [r for r in results if r.status == "hallucination"]
        assert len(matches) == 2
        assert len(hallucinations) == 2  # etch.name + etch.temp

    def test_value_mismatch_on_matched_key(self) -> None:
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {"steps": [{"name": "anneal", "temp": 500}]}
        extracted = {"steps": [{"name": "anneal", "temp": 999}]}
        results = score_record(schema, gold, extracted)
        by_path = {r.path: r for r in results}
        # Instance paths: steps[0].name, steps[0].temp
        assert by_path["steps[0].name"].status == "match"
        assert by_path["steps[0].temp"].status == "mismatch"

    def test_empty_arrays_match(self) -> None:
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        results = score_record(schema, {"steps": []}, {"steps": []})
        assert len(results) == 1
        assert results[0].status == "match"
        assert results[0].path == "steps"

    def test_gold_empty_extracted_has_elements(self) -> None:
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {"steps": []}
        extracted = {"steps": [{"name": "anneal", "temp": 500}]}
        results = score_record(schema, gold, extracted)
        hallucinations = [r for r in results if r.status == "hallucination"]
        assert len(hallucinations) == 2

    def test_extracted_element_missing_key_hallucination(self) -> None:
        # Extracted has an element without the key field — can't match.
        # Only actually-present fields in the extra element count as
        # hallucinations. The missing "name" is not hallucinated because
        # the extractor didn't produce it.
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {"steps": [{"name": "anneal", "temp": 500}]}
        extracted = {"steps": [
            {"name": "anneal", "temp": 500},
            {"temp": 999},  # no "name" key — unmatched extra element
        ]}
        results = score_record(schema, gold, extracted)
        matches = [r for r in results if r.status == "match"]
        hallucinations = [r for r in results if r.status == "hallucination"]
        assert len(matches) == 2  # anneal.name + anneal.temp
        assert len(hallucinations) == 1  # only temp (name not produced)

    # TODO: test this case later with Hungarian match
    def test_duplicate_keys_in_extracted(self) -> None:
        # Two extracted elements share the same key. First wins the match;
        # the duplicate is treated as an unmatched hallucination.
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {"steps": [{"name": "anneal", "temp": 500}]}
        extracted = {"steps": [
            {"name": "anneal", "temp": 500},
            {"name": "anneal", "temp": 999},  # duplicate key,
        ]}
        results = score_record(schema, gold, extracted)
        matches = [r for r in results if r.status == "match"]
        hallucinations = [r for r in results if r.status == "hallucination"]
        assert len(matches) == 2  # first anneal matched
        assert len(hallucinations) == 2  # duplicate anneal: name + temp

    def test_duplicate_keys_in_extracted_order_matters(self) -> None:
        # Same as test_duplicate_keys_in_extracted but with extracted order
        # swapped. The FIRST extracted element with key "anneal" wins the
        # lookup slot. Here the first has temp=999 which mismatches gold's
        # temp=500, while the second (temp=500) would have matched perfectly
        # — but it's treated as a duplicate and becomes a hallucination.
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {"steps": [{"name": "anneal", "temp": 500}]}
        extracted = {"steps": [
            {"name": "anneal", "temp": 999},  # first: wins lookup, but temp mismatches
            {"name": "anneal", "temp": 500},  # second: would match, but duplicate
        ]}
        results = score_record(schema, gold, extracted)
        matches = [r for r in results if r.status == "match"]
        mismatches = [r for r in results if r.status == "mismatch"]
        hallucinations = [r for r in results if r.status == "hallucination"]
        assert len(matches) == 1      # name matched
        assert len(mismatches) == 1   # temp: 999 vs 500
        assert len(hallucinations) == 2  # duplicate element: name + temp

    def test_duplicate_keys_in_gold(self) -> None:
        # Two gold elements share the same key. First matches; the second
        # can't match anyone (extracted already consumed) -> omission.
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {"steps": [
            {"name": "anneal", "temp": 500},
            {"name": "anneal", "temp": 600},  # duplicate key
        ]}
        extracted = {"steps": [{"name": "anneal", "temp": 500}]}
        results = score_record(schema, gold, extracted)
        matches = [r for r in results if r.status == "match"]
        omissions = [r for r in results if r.status == "omission"]
        assert len(matches) == 2  # first anneal matched
        assert len(omissions) == 2  # second anneal: name + temp

    def test_duplicate_keys_in_gold_order_matters(self) -> None:
        # Same as test_duplicate_keys_in_gold but with gold order swapped.
        # The FIRST gold element with key "anneal" wins the match.
        # Here the first has temp=600 which mismatches extracted's temp=500,
        # while the second (temp=500) would have matched perfectly — but it
        # loses because the first element already consumed the key.
        # This demonstrates that gold order affects results with duplicate keys.
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {"steps": [
            {"name": "anneal", "temp": 600},  # first: wins match, but temp mismatches
            {"name": "anneal", "temp": 500},  # second: would match, but key consumed
        ]}
        extracted = {"steps": [{"name": "anneal", "temp": 500}]}
        results = score_record(schema, gold, extracted)
        matches = [r for r in results if r.status == "match"]
        mismatches = [r for r in results if r.status == "mismatch"]
        omissions = [r for r in results if r.status == "omission"]
        assert len(matches) == 1    # name matched
        assert len(mismatches) == 1  # temp: 600 vs 500
        assert len(omissions) == 2  # second anneal: name + temp

    # todo: strange case, to be discussed!
    def test_gold_element_missing_field_omission_only_present(self) -> None:
        # Gold element has only "temp" (no "name"). The element is unmatched
        # (missing key → omission). Only the actually-present "temp" should
        # be an omission — the absent "name" can't be omitted.
        schema = _make_schema(
            _steps_schema({"match_by": "key_field", "key": "name"})
        )
        gold = {"steps": [{"temp": 500}]}  # no "name" key
        extracted = {"steps": [{"name": "anneal", "temp": 500}]}
        results = score_record(schema, gold, extracted)
        omissions = [r for r in results if r.status == "omission"]
        hallucinations = [r for r in results if r.status == "hallucination"]
        assert len(omissions) == 1  # only temp (name not in gold)
        assert len(hallucinations) == 2  # anneal unmatched: name + temp

    def test_end_to_end_via_evaluate(self) -> None:
        raw_schema = _steps_schema(
            {"match_by": "key_field", "key": "name"}
        )
        annotate_xeval(raw_schema)
        gold = [
            {
                "steps": [
                    {"name": "deposit", "temp": 300},
                    {"name": "anneal", "temp": 500},
                ]
            }
        ]
        extracted = [
            {
                "steps": [
                    {"name": "anneal", "temp": 500},
                    {"name": "deposit", "temp": 300},
                ]
            }
        ]
        result = evaluate(gold, extracted, raw_schema)
        assert result.mean_f1 == 1.0
        assert result.total_fields == 4


# --- Explicit ordered ---


class TestExplicitOrdered:
    def test_explicit_ordered_same_as_default(self) -> None:
        schema_default = _make_schema(_steps_schema())
        schema_explicit = _make_schema(
            _steps_schema({"ordered": True})
        )
        gold = {
            "steps": [
                {"name": "anneal", "temp": 500},
                {"name": "deposit", "temp": 300},
            ]
        }
        extracted = {
            "steps": [
                {"name": "deposit", "temp": 300},
                {"name": "anneal", "temp": 500},
            ]
        }
        results_default = score_record(schema_default, gold, extracted)
        results_explicit = score_record(schema_explicit, gold, extracted)
        # Both should produce the same statuses (positional mismatch)
        assert (
            [r.status for r in results_default]
            == [r.status for r in results_explicit]
        )


# --- Hungarian alignment ---


class TestHungarianAlignment:
    def test_reordered_primitives_match(self) -> None:
        """Hungarian finds the optimal pairing regardless of order."""
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "x-eval-align": {"match_by": "hungarian"},
                },
            },
        })
        results = score_record(
            schema,
            {"tags": ["a", "b"]},
            {"tags": ["b", "a"]},
        )
        assert len(results) == 2
        assert all(r.status == "match" for r in results)
        # Instance paths use gold indices
        paths = {r.path for r in results}
        assert "tags[0]" in paths
        assert "tags[1]" in paths

    def test_ordered_would_mismatch_same_data(self) -> None:
        """Same data with ordered matching produces mismatches."""
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        })
        results = score_record(
            schema,
            {"tags": ["a", "b"]},
            {"tags": ["b", "a"]},
        )
        assert all(r.status == "mismatch" for r in results)

    def test_partial_overlap(self) -> None:
        """Some elements match, some don't."""
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "x-eval-align": {"match_by": "hungarian"},
                },
            },
        })
        results = score_record(
            schema,
            {"tags": ["a", "b", "c"]},
            {"tags": ["c", "d"]},
        )
        matches = [r for r in results if r.status == "match"]
        omissions = [r for r in results if r.status == "omission"]
        hallucinations = [r for r in results if r.status == "hallucination"]
        assert len(matches) == 1      # "c" matched
        assert len(omissions) == 2    # "a", "b" unmatched in gold
        assert len(hallucinations) == 1  # "d" extra in extracted

    def test_empty_arrays_match(self) -> None:
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "x-eval-align": {"match_by": "hungarian"},
                },
            },
        })
        results = score_record(
            schema, {"tags": []}, {"tags": []},
        )
        assert len(results) == 1
        assert results[0].status == "match"

    def test_objects_reordered(self) -> None:
        """Hungarian matches objects by best F1, not by position."""
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
                    "x-eval-align": {"match_by": "hungarian"},
                },
            },
        })
        gold = {"steps": [
            {"name": "anneal", "temp": 500},
            {"name": "deposit", "temp": 300},
        ]}
        extracted = {"steps": [
            {"name": "deposit", "temp": 300},
            {"name": "anneal", "temp": 500},
        ]}
        results = score_record(schema, gold, extracted)
        assert all(r.status == "match" for r in results)
        assert len(results) == 4  # 2 elements x 2 fields


    def test_objects_mixed_results(self) -> None:
        """Reordered objects with a value mismatch on one element.

        gold:      anneal/500, deposit/300, etch/100
        extracted: deposit/300, anneal/500, etch/200  (etch temp wrong)

        Hungarian should pair by best F1:
          anneal <-> anneal (match)
          deposit <-> deposit (match)
          etch <-> etch (name match, temp mismatch)
        """
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
                    "x-eval-align": {"match_by": "hungarian"},
                },
            },
        })
        gold = {"steps": [
            {"name": "anneal", "temp": 500},
            {"name": "deposit", "temp": 300},
            {"name": "etch", "temp": 100},
        ]}
        extracted = {"steps": [
            {"name": "deposit", "temp": 300},
            {"name": "anneal", "temp": 500},
            {"name": "etch", "temp": 200},
        ]}
        results = score_record(schema, gold, extracted)

        # 3 elements x 2 fields = 6 results
        assert len(results) == 6

        matches = [r for r in results if r.status == "match"]
        mismatches = [r for r in results if r.status == "mismatch"]

        # anneal: name match + temp match = 2 matches
        # deposit: name match + temp match = 2 matches
        # etch: name match + temp mismatch (100 vs 200) = 1 match + 1 mismatch
        assert len(matches) == 5
        assert len(mismatches) == 1
        assert mismatches[0].gold_value == 100
        assert mismatches[0].extracted_value == 200

    def test_unequal_lengths_with_best_pairing(self) -> None:
        """More extracted than gold; best pairing chosen, rest hallucinated."""
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "x-eval-align": {"match_by": "hungarian"},
                },
            },
        })
        results = score_record(
            schema,
            {"tags": ["x"]},
            {"tags": ["a", "x", "b"]},
        )
        matches = [r for r in results if r.status == "match"]
        hallucinations = [r for r in results if r.status == "hallucination"]
        assert len(matches) == 1      # "x" matched
        assert len(hallucinations) == 2  # "a", "b"

    def test_no_matches_all_omissions_and_hallucinations(self) -> None:
        """No overlap at all."""
        schema = _make_schema({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "x-eval-align": {"match_by": "hungarian"},
                },
            },
        })
        results = score_record(
            schema,
            {"tags": ["a", "b"]},
            {"tags": ["x", "y"]},
        )
        matches = [r for r in results if r.status == "match"]
        omission = [r for r in results if r.status == "omission"]
        hallucinations = [r for r in results if r.status == "hallucination"]
        assert len(matches) == 0
        assert len(omission) == 2
        assert len(hallucinations) == 2

    def test_end_to_end_via_evaluate(self) -> None:
        raw_schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "x-eval-align": {"match_by": "hungarian"},
                },
            },
        }
        annotate_xeval(raw_schema)
        gold = [{"tags": ["a", "b", "c"]}]
        extracted = [{"tags": ["c", "a", "b"]}]
        result = evaluate(gold, extracted, raw_schema)
        assert result.mean_f1 == 1.0


class TestHungarianWithBatchComparator:
    """Hungarian matching resolves batch comparators before building cost matrix."""

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def test_batch_comparator_resolved_for_optimal_matching(self) -> None:
        """Batch comparator scores inform Hungarian matching.

        Without resolving batch fields, all pairs would have F1=1.0 (no
        scorable fields) and matching would be arbitrary. With resolution,
        Hungarian uses real scores to find the optimal pairing.

        Gold: ["alpha desc", "beta desc"]
        Extracted: ["beta desc", "alpha desc"]  (swapped order)

        The fake judge scores 1.0 for exact matches, 0.0 otherwise.
        Hungarian should match gold[0]->ext[1] and gold[1]->ext[0].
        """
        class FakeBatchComparator:
            is_batch = True
            name = "fake_semantic"
            def __call__(self, items: list[BatchItem]) -> list[ComparatorResult | None]:
                results: list[ComparatorResult | None] = []
                for item in items:
                    score = 1.0 if item.gold_compared == item.extracted_compared else 0.0
                    results.append(ComparatorResult(
                        score=score, comparator=self.name,
                        reason="match" if score == 1.0 else "mismatch",
                    ))
                return results

        register("fake_semantic", FakeBatchComparator())

        schema = _make_schema({
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "x-eval-align": {"match_by": "hungarian"},
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "x-eval-compare": "fake_semantic",
                            },
                        },
                    },
                },
            },
        })
        results = score_record(
            schema,
            {"steps": [
                {"description": "alpha desc"},
                {"description": "beta desc"},
            ]},
            {"steps": [
                {"description": "beta desc"},
                {"description": "alpha desc"},
            ]},
        )
        matches = [r for r in results if r.status == "match"]
        mismatches = [r for r in results if r.status == "mismatch"]
        # Optimal matching: both pairs match perfectly
        assert len(matches) == 2
        assert len(mismatches) == 0


class TestWrongTypeAcrossAlignmentStrategies:
    """The wrong-type / missing policy (issues #56 / #82) applies to every
    alignment strategy, not just the default ordered scorer."""

    @pytest.mark.parametrize(
        "align",
        [
            {"match_by": "key_field", "key": "name"},
            {"match_by": "hungarian"},
        ],
    )
    def test_extracted_wrong_type_is_mismatch(
        self, align: dict[str, object]
    ) -> None:
        schema = _make_schema(_steps_schema(align))
        results = score_record(
            schema, {"steps": [{"name": "anneal", "temp": 500}]}, {"steps": "bad"}
        )
        assert len(results) == 1
        assert results[0].path == "steps"
        assert results[0].status == "mismatch"

    @pytest.mark.parametrize(
        "align",
        [
            {"match_by": "key_field", "key": "name"},
            {"match_by": "hungarian"},
        ],
    )
    def test_extracted_null_is_omission(self, align: dict[str, object]) -> None:
        schema = _make_schema(_steps_schema(align))
        results = score_record(
            schema,
            {"steps": [{"name": "anneal", "temp": 500}]},
            {"steps": None},
        )
        # one gold element with two scored fields -> two omissions
        assert len(results) == 2
        assert all(r.status == "omission" for r in results)


def test_key_field_hallucinations_get_unique_negative_indices() -> None:
    # Two unmatched extracted elements (no gold counterpart) must get distinct
    # negative indices, even across key-field's two hallucination sources.
    schema = _make_schema(_steps_schema({"match_by": "key_field", "key": "name"}))
    gold = {"steps": [{"name": "anneal", "temp": 500}]}
    extracted = {
        "steps": [
            {"name": "anneal", "temp": 500},   # matched
            {"name": "etch", "temp": 200},      # unmatched by key -> halluc
            {"temp": 999},                       # missing key -> halluc (other source)
        ]
    }
    results = score_record(schema, gold, extracted)
    halluc_paths = sorted(r.path for r in results if r.status == "hallucination")
    # etch -> name+temp, the keyless element -> temp; distinct negative indices
    assert "steps[-1].name" in halluc_paths
    assert "steps[-2].temp" in halluc_paths
    # no two hallucinated elements share an index
    indices = {r.path.split(".")[0] for r in results if r.status == "hallucination"}
    assert indices == {"steps[-1]", "steps[-2]"}
