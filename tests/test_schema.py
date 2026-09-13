import logging

import pytest

from struct_extract_eval.core.comparators.comparator import ComparatorSpec
from struct_extract_eval.core.schema import (
    SchemaError,
    SchemaNode,
    annotate_xeval,
    parse_eval_schema,
)
from struct_extract_eval.core.schema.tree import _validate_xeval
from struct_extract_eval.core.transforms.transform import TransformSpec

# --- SchemaError ---


class TestSchemaError:
    def test_message_without_path(self) -> None:
        err = SchemaError("bad schema")
        assert str(err) == "bad schema"
        assert err.path == ""

    def test_message_with_path(self) -> None:
        err = SchemaError("bad type", path="steps[].duration")
        assert str(err) == "steps[].duration: bad type"
        assert err.path == "steps[].duration"

    def test_dotted_property_name_is_rejected_at_root(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "x.y": {"type": "string", "x-eval-compare": "exact"},
            },
        }

        with pytest.raises(SchemaError, match=r"property name 'x\.y' contains '\.'"):
            parse_eval_schema(schema)

    def test_dotted_property_name_reports_parent_path(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "parent": {
                    "type": "object",
                    "properties": {
                        "x.y": {"type": "string", "x-eval-compare": "exact"},
                    },
                },
            },
        }

        with pytest.raises(SchemaError) as exc_info:
            parse_eval_schema(schema)

        assert exc_info.value.path == "parent"
        assert str(exc_info.value).startswith("parent: property name 'x.y'")


# --- SchemaNode ---


class TestSchemaNode:
    def test_defaults(self) -> None:
        node = SchemaNode(path="", json_type="object", comparator=ComparatorSpec("exact"))
        assert node.children == []
        assert node.transforms == []
        assert node.comparator.params == {}


# --- _default_comparator ---


# --- _validate_xeval ---


class TestValidateXeval:
    def test_valid_config(self) -> None:
        schema: dict[str, object] = {
            "x-eval-compare": "exact",
            "x-eval-transform": ["lowercase", "strip"],
        }
        _validate_xeval(schema, "test")  # should not raise

    def test_unknown_comparator(self) -> None:
        with pytest.raises(SchemaError, match="Unknown comparator"):
            _validate_xeval({"x-eval-compare": "nonexistent"}, "test")

    def test_transform_not_list(self) -> None:
        with pytest.raises(SchemaError, match="x-eval-transform must be a list"):
            _validate_xeval({"x-eval-transform": "lowercase"}, "test")

    def test_transform_invalid_item(self) -> None:
        with pytest.raises(
            SchemaError, match="x-eval-transform\\[0\\] must be a string or object"
        ):
            _validate_xeval({"x-eval-transform": [123]}, "test")

    def test_transform_with_params(self) -> None:
        _validate_xeval(
            {"x-eval-transform": ["lowercase", {"round_digits": {"digits": 2}}]},
            "test",
        )

    def test_transform_scalar_params_raises(self) -> None:
        with pytest.raises(SchemaError, match="must be a dict"):
            _validate_xeval(
                {"x-eval-transform": [{"round_digits": 2}]},
                "test",
            )

    def test_transform_empty_dict_raises(self) -> None:
        with pytest.raises(SchemaError, match="x-eval-transform"):
            _validate_xeval({"x-eval-transform": [{}]}, "test")

    def test_transform_multi_key_dict_raises(self) -> None:
        with pytest.raises(SchemaError, match="x-eval-transform"):
            _validate_xeval(
                {"x-eval-transform": [{"lowercase": {}, "strip": {}}]},
                "test",
            )

    def test_transform_non_string_key_raises(self) -> None:
        with pytest.raises(SchemaError, match="x-eval-transform"):
            _validate_xeval({"x-eval-transform": [{1: {}}]}, "test")

    def test_unknown_transform_name_raises(self) -> None:
        with pytest.raises(SchemaError, match="Unknown transform"):
            _validate_xeval({"x-eval-transform": ["nonexistent"]}, "test")

    def test_unknown_transform_name_in_object_raises(self) -> None:
        with pytest.raises(SchemaError, match="Unknown transform"):
            _validate_xeval({"x-eval-transform": [{"bogus": {"x": 1}}]}, "test")

    def test_old_oneof_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """x-eval-oneof is no longer a known key -- should warn."""
        with caplog.at_level(logging.WARNING):
            _validate_xeval({"x-eval-oneof": ["a", "b"]}, "test")
        assert "Unknown x-eval key" in caplog.text

    def test_compare_as_object(self) -> None:
        _validate_xeval(
            {"x-eval-compare": {"numeric": {"tolerance": {"rel": 0.01}}}},
            "test",
        )

    def test_compare_object_unknown_comparator(self) -> None:
        with pytest.raises(SchemaError, match="Unknown comparator"):
            _validate_xeval({"x-eval-compare": {"bogus": {}}}, "test")

    def test_compare_object_multiple_keys(self) -> None:
        with pytest.raises(SchemaError, match="exactly one key"):
            _validate_xeval({"x-eval-compare": {"exact": {}, "numeric": {}}}, "test")

    def test_compare_scalar_params_raises(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "val": {"type": "number", "x-eval-compare": {"numeric": 2}},
            },
        }
        with pytest.raises(SchemaError, match="params for 'numeric' must be a dict"):
            parse_eval_schema(schema)

    def test_unknown_xeval_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _validate_xeval({"x-eval-custom-thing": True}, "test")
        assert "Unknown x-eval key" in caplog.text

    def test_old_tolerance_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """x-eval-tolerance is no longer a known key -- should warn."""
        with caplog.at_level(logging.WARNING):
            _validate_xeval({"x-eval-tolerance": {"rel": 0.01}}, "test")
        assert "Unknown x-eval key" in caplog.text

    def test_old_params_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """x-eval-params is no longer a known key -- should warn."""
        with caplog.at_level(logging.WARNING):
            _validate_xeval({"x-eval-params": {"ordered": True}}, "test")
        assert "Unknown x-eval key" in caplog.text


# --- parse_eval_schema ---


class TestParseSchema:
    def test_simple_string_field(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "name": {"type": "string", "x-eval-compare": "exact"},
            },
        }
        root = parse_eval_schema(schema)
        assert root.json_type == "object"
        assert root.path == ""
        assert len(root.children) == 1
        child = root.children[0]
        assert child.path == "name"
        assert child.json_type == "string"
        assert child.comparator.name == "exact"
        assert child.transforms == []

    def test_comparator_string_form(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "desc": {"type": "string", "x-eval-compare": "exact"},
            },
        }
        assert _root_child(schema, "desc").comparator.name == "exact"

    def test_xeval_transform(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "unit": {
                    "type": "string",
                    "x-eval-compare": "exact",
                    "x-eval-transform": ["lowercase", "strip"],
                },
            },
        }
        node = _root_child(schema, "unit")
        assert node.transforms == [TransformSpec("lowercase"), TransformSpec("strip")]

    def test_xeval_transform_with_params(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "val": {
                    "type": "number",
                    "x-eval-compare": "numeric",
                    "x-eval-transform": ["strip", {"round_digits": {"digits": 2}}],
                },
            },
        }
        node = _root_child(schema, "val")
        assert node.transforms == [
            TransformSpec("strip"),
            TransformSpec("round_digits", {"digits": 2}),
        ]

    def test_xeval_compare_oneof(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "unit": {
                    "type": "string",
                    "x-eval-compare": {"oneof": {"values": ["eV", "electronvolt"]}},
                },
            },
        }
        node = _root_child(schema, "unit")
        assert node.comparator.name == "oneof"
        assert node.comparator.params == {"values": ["eV", "electronvolt"]}

    def test_xeval_compare_with_params(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "val": {
                    "type": "number",
                    "x-eval-compare": {"numeric": {"tolerance": {"rel": 0.05}}},
                },
            },
        }
        node = _root_child(schema, "val")
        assert node.comparator.name == "numeric"
        assert node.comparator.params == {"tolerance": {"rel": 0.05}}

    def test_missing_xeval_compare_raises(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "name": {"type": "string"},
            },
        }
        with pytest.raises(SchemaError, match="missing x-eval-compare"):
            parse_eval_schema(schema)

    def test_nested_objects(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "outer": {
                    "type": "object",
                    "x-eval-compare": "exact",
                    "properties": {
                        "inner": {"type": "string", "x-eval-compare": "exact"},
                    },
                },
            },
        }
        root = parse_eval_schema(schema)
        outer = root.children[0]
        assert outer.path == "outer"
        inner = outer.children[0]
        assert inner.path == "outer.inner"

    def test_array_with_items(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "tags": {
                    "type": "array",
                    "x-eval-compare": "exact",
                    "items": {"type": "string", "x-eval-compare": "exact"},
                },
            },
        }
        tags = _root_child(schema, "tags")
        assert tags.json_type == "array"
        assert len(tags.children) == 1
        assert tags.children[0].path == "tags[]"
        assert tags.children[0].json_type == "string"

    def test_array_of_objects(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "ions": {
                    "type": "array",
                    "x-eval-compare": "exact",
                    "items": {
                        "type": "object",
                        "x-eval-compare": "exact",
                        "properties": {
                            "name": {"type": "string", "x-eval-compare": "exact"},
                            "coefficient": {"type": "number", "x-eval-compare": "numeric"},
                        },
                    },
                },
            },
        }
        ions = _root_child(schema, "ions")
        assert ions.json_type == "array"
        item = ions.children[0]
        assert item.path == "ions[]"
        assert item.json_type == "object"
        assert len(item.children) == 2

    def test_missing_type_raises(self) -> None:
        with pytest.raises(SchemaError, match="Missing or invalid 'type'"):
            parse_eval_schema(
                {
                    "x-eval-compare": "exact",
                    "properties": {"x": {"type": "string", "x-eval-compare": "exact"}},
                }
            )

    def test_invalid_comparator_raises(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "x": {"type": "string", "x-eval-compare": "bogus"},
            },
        }
        with pytest.raises(SchemaError, match="Unknown comparator"):
            parse_eval_schema(schema)

    def test_not_a_dict_raises(self) -> None:
        with pytest.raises(SchemaError, match="Eval schema must be an object"):
            parse_eval_schema("not a dict")  # type: ignore[arg-type]

    def test_unresolved_ref_raises(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "bg": {"$ref": "#/$defs/Bandgap"},
            },
        }
        with pytest.raises(SchemaError, match="Missing or invalid 'type'"):
            parse_eval_schema(schema)

    def test_deeply_nested(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "x-eval-compare": "exact",
            "properties": {
                "layers": {
                    "type": "array",
                    "x-eval-compare": "exact",
                    "items": {
                        "type": "object",
                        "x-eval-compare": "exact",
                        "properties": {
                            "name": {"type": "string", "x-eval-compare": "exact"},
                            "steps": {
                                "type": "array",
                                "x-eval-compare": "exact",
                                "items": {
                                    "type": "object",
                                    "x-eval-compare": "exact",
                                    "properties": {
                                        "duration": {
                                            "type": "number",
                                            "x-eval-compare": "numeric",
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        root = parse_eval_schema(schema)
        layers = root.children[0]
        layer_item = layers.children[0]
        steps = next(c for c in layer_item.children if c.path == "layers[].steps")
        step_item = steps.children[0]
        duration = step_item.children[0]
        assert duration.path == "layers[].steps[].duration"
        assert duration.comparator.name == "numeric"


# --- x-eval-align key_field validation ---


class TestKeyFieldValidation:
    def test_key_field_not_in_items_properties_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Key not in properties -> warns (schema incomplete, matching may still work on data)."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "x-eval-align": {"match_by": "key_field", "key": "name"},
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "x-eval-compare": "exact"},
                        },
                    },
                },
            },
        }
        with caplog.at_level(logging.WARNING):
            parse_eval_schema(schema)
        assert "key 'name' not found" in caplog.text

    def test_key_field_items_missing_raises(self) -> None:
        """Items missing entirely -> error (can't score array elements at all)."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "x-eval-align": {"match_by": "key_field", "key": "name"},
                },
            },
        }
        with pytest.raises(SchemaError, match="requires 'items'"):
            parse_eval_schema(schema)

    def test_key_field_items_primitive_type_raises(self) -> None:
        """Items is a primitive type (string) -> error (key_field needs objects)."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "x-eval-align": {"match_by": "key_field", "key": "name"},
                    "items": {"type": "string", "x-eval-compare": "exact"},
                },
            },
        }
        with pytest.raises(SchemaError, match="requires items type 'object'"):
            parse_eval_schema(schema)

    def test_key_field_items_no_properties_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Opaque object (no properties) -> warns (matching may work, no per-field scoring)."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "x-eval-align": {"match_by": "key_field", "key": "name"},
                    "items": {"type": "object", "x-eval-compare": "exact"},
                },
            },
        }
        with caplog.at_level(logging.WARNING):
            parse_eval_schema(schema)
        assert "no 'properties'" in caplog.text

    def test_key_field_valid_passes(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "x-eval-align": {"match_by": "key_field", "key": "name"},
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "x-eval-compare": "exact"},
                            "temp": {"type": "number", "x-eval-compare": "numeric"},
                        },
                    },
                },
            },
        }
        root = parse_eval_schema(schema)
        assert root.children[0].align == {"match_by": "key_field", "key": "name"}


# --- Test helpers ---


def _root_child(schema: dict[str, object], name: str) -> SchemaNode:
    """Parse schema and return the named child of the root node."""
    root = parse_eval_schema(schema)
    for child in root.children:
        if child.path == name:
            return child
    raise AssertionError(f"Child '{name}' not found in {[c.path for c in root.children]}")


class TestListValuedType:
    """JSON Schema allows `type` to be a list of types."""

    @staticmethod
    def _tree(raw: dict[str, object]) -> SchemaNode:
        annotate_xeval(raw)
        return parse_eval_schema(raw)

    def test_nullable_list_resolves_to_single_type(self) -> None:
        # ["string", "null"] is just a nullable string -- collapses to "string".
        tree = self._tree({
            "type": "object",
            "properties": {"name": {"type": ["string", "null"]}},
        })
        node = tree.children[0]
        assert node.json_type == "string"
        assert node.allowed_types is None
        assert node.comparator.name == "exact"

    def test_multi_type_defaults_to_exact_and_is_a_leaf(self) -> None:
        # >= 2 non-null types -> comparator-owned leaf, default comparator exact.
        tree = self._tree({
            "type": "object",
            "properties": {
                "q": {
                    "type": ["string", "object"],
                    "properties": {"value": {"type": "number"}},
                },
            },
        })
        node = tree.children[0]
        assert node.allowed_types == ["string", "object"]
        assert node.children == []            # not scored structurally
        assert node.comparator.name == "exact"   # default, regardless of first type

    def test_multi_type_default_exact_even_when_first_type_numeric(self) -> None:
        tree = self._tree({
            "type": "object",
            "properties": {"x": {"type": ["number", "string"]}},
        })
        assert tree.children[0].comparator.name == "exact"

    def test_multi_type_explicit_comparator_wins(self) -> None:
        tree = self._tree({
            "type": "object",
            "properties": {"x": {"type": ["string", "object"], "x-eval-compare": "exact"}},
        })
        assert tree.children[0].comparator.name == "exact"
