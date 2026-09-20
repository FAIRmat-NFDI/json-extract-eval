import pytest

from json_extract_eval.core.schema.xeval import (
    _BUILTIN_TYPE_DEFAULTS,
    annotate_xeval,
    parse_xeval_entry,
    set_type_default,
)


class TestParseXevalEntry:
    def test_string_form(self) -> None:
        assert parse_xeval_entry("exact") == ("exact", {})

    def test_object_form(self) -> None:
        assert parse_xeval_entry({"numeric": {"tolerance": {"rel": 0.01}}}) == (
            "numeric",
            {"tolerance": {"rel": 0.01}},
        )

    def test_object_empty_params(self) -> None:
        assert parse_xeval_entry({"exact": {}}) == ("exact", {})

    def test_object_scalar_params_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a dict"):
            parse_xeval_entry({"round_digits": 2})

    def test_object_multiple_keys_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one key"):
            parse_xeval_entry({"a": {}, "b": {}})

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            parse_xeval_entry("")

    def test_empty_dict_key_raises(self) -> None:
        with pytest.raises(TypeError, match="non-empty string"):
            parse_xeval_entry({"": {"param": 1}})

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a string or single-key dict"):
            parse_xeval_entry(123)  # type: ignore[arg-type]


class TestAnnotateXeval:
    def test_string_field_gets_exact(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        annotate_xeval(schema)
        assert schema["properties"]["name"]["x-eval-compare"] == "exact"  # type: ignore[index]

    def test_number_field_gets_numeric(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "temp": {"type": "number"},
            },
        }
        annotate_xeval(schema)
        assert schema["properties"]["temp"]["x-eval-compare"] == "numeric"  # type: ignore[index]

    def test_integer_field_gets_numeric(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        }
        annotate_xeval(schema)
        assert schema["properties"]["count"]["x-eval-compare"] == "numeric"  # type: ignore[index]

    def test_boolean_field_gets_exact(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "active": {"type": "boolean"},
            },
        }
        annotate_xeval(schema)
        assert schema["properties"]["active"]["x-eval-compare"] == "exact"  # type: ignore[index]

    def test_long_string_gets_exact(self) -> None:
        """All strings default to exact, regardless of maxLength."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "description": {"type": "string", "maxLength": 200},
            },
        }
        annotate_xeval(schema)
        assert schema["properties"]["description"]["x-eval-compare"] == "exact"  # type: ignore[index]

    def test_object_no_properties_gets_exact(self) -> None:
        # Opaque objects (no properties) are leaves and get "exact" by default
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "metadata": {"type": "object"},
            },
        }
        annotate_xeval(schema)
        assert schema["properties"]["metadata"]["x-eval-compare"] == "exact"  # type: ignore[index]

    def test_explicit_compare_not_overridden(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "x-eval-compare": "semantic"},
            },
        }
        annotate_xeval(schema)
        assert schema["properties"]["name"]["x-eval-compare"] == "semantic"  # type: ignore[index]

    def test_type_defaults_override(self) -> None:
        """Users can call set_type_default() to change the mapping."""
        original = dict(_BUILTIN_TYPE_DEFAULTS)
        try:
            set_type_default("string", "my_custom_str")
            set_type_default("number", "my_custom_num")
            schema: dict[str, object] = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "temp": {"type": "number"},
                    "active": {"type": "boolean"},
                },
            }
            annotate_xeval(schema)
            props = schema["properties"]
            assert props["name"]["x-eval-compare"] == "my_custom_str"  # type: ignore[index]
            assert props["temp"]["x-eval-compare"] == "my_custom_num"  # type: ignore[index]
            assert props["active"]["x-eval-compare"] == "exact"  # type: ignore[index]
        finally:
            _BUILTIN_TYPE_DEFAULTS.clear()
            _BUILTIN_TYPE_DEFAULTS.update(original)

    def test_type_defaults_with_params(self) -> None:
        """set_type_default() accepts dict form with params."""
        original = dict(_BUILTIN_TYPE_DEFAULTS)
        try:
            set_type_default("number", {"numeric": {"tolerance": {"rel": 0.01}}})
            schema: dict[str, object] = {
                "type": "object",
                "properties": {
                    "temp": {"type": "number"},
                    "name": {"type": "string"},
                },
            }
            annotate_xeval(schema)
            props = schema["properties"]
            expected = {"numeric": {"tolerance": {"rel": 0.01}}}
            assert props["temp"]["x-eval-compare"] == expected  # type: ignore[index]
            assert props["name"]["x-eval-compare"] == "exact"  # type: ignore[index]
        finally:
            _BUILTIN_TYPE_DEFAULTS.clear()
            _BUILTIN_TYPE_DEFAULTS.update(original)

    def test_type_defaults_do_not_override_explicit(self) -> None:
        """Explicit x-eval-compare on a field takes precedence over set_type_default."""
        original = dict(_BUILTIN_TYPE_DEFAULTS)
        try:
            set_type_default("string", "my_custom")
            schema: dict[str, object] = {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "x-eval-compare": "oneof"},
                },
            }
            annotate_xeval(schema)
            assert schema["properties"]["name"]["x-eval-compare"] == "oneof"  # type: ignore[index]
        finally:
            _BUILTIN_TYPE_DEFAULTS.clear()
            _BUILTIN_TYPE_DEFAULTS.update(original)

    def test_required_array_preserved(self) -> None:
        """annotate_xeval does not remove or modify the required array."""
        schema: dict[str, object] = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "optional_field": {"type": "string"},
            },
        }
        annotate_xeval(schema)
        assert schema["required"] == ["name"]

    def test_nested_object(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "sample": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                        "label": {"type": "string"},
                    },
                },
            },
        }
        annotate_xeval(schema)
        sample = schema["properties"]["sample"]  # type: ignore[index]
        assert sample["properties"]["name"]["x-eval-compare"] == "exact"

    def test_array_items(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        annotate_xeval(schema)
        items = schema["properties"]["tags"]["items"]  # type: ignore[index]
        assert items["x-eval-compare"] == "exact"

    def test_array_of_objects(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string"},
                            "duration": {"type": "number"},
                            "comment": {"type": "string"},
                        },
                    },
                },
            },
        }
        annotate_xeval(schema)
        items = schema["properties"]["steps"]["items"]  # type: ignore[index]
        item_props = items["properties"]
        assert item_props["name"]["x-eval-compare"] == "exact"
        assert item_props["duration"]["x-eval-compare"] == "numeric"

    def test_deeply_nested(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "layers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "duration": {"type": "number"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        annotate_xeval(schema)
        duration = (
            schema["properties"]["layers"]["items"]  # type: ignore[index]
            ["properties"]["steps"]["items"]["properties"]["duration"]
        )
        assert duration["x-eval-compare"] == "numeric"

    def test_mutates_in_place(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        original_id = id(schema)
        result = annotate_xeval(schema)
        assert result is schema
        assert id(result) == original_id

    def test_root_object_not_annotated_with_compare(self) -> None:
        """Root object and intermediate objects with properties don't need x-eval-compare."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        annotate_xeval(schema)
        # Root should not get x-eval-compare since it has children
        assert "x-eval-compare" not in schema

    def test_intermediate_object_not_annotated(self) -> None:
        """Objects that have properties are containers, not leaf fields."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "sample": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                },
            },
        }
        annotate_xeval(schema)
        assert "x-eval-compare" not in schema["properties"]["sample"]  # type: ignore[index]

    def test_array_not_annotated_with_compare(self) -> None:
        """Arrays are containers, not leaf fields."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        annotate_xeval(schema)
        assert "x-eval-compare" not in schema["properties"]["tags"]  # type: ignore[index]


def test_annotate_does_not_descend_under_node_with_comparator() -> None:
    raw: dict[str, object] = {
        "type": "object",
        "properties": {
            "person": {
                "type": "object",
                "x-eval-compare": "exact",
                "properties": {"surname": {"type": "string"}},
            },
            "plain": {"type": "string"},
        },
    }
    annotate_xeval(raw)
    person = raw["properties"]["person"]  # type: ignore[index]
    assert "x-eval-compare" not in person["properties"]["surname"]  # type: ignore[index]
    assert raw["properties"]["plain"]["x-eval-compare"] == "exact"  # type: ignore[index]
