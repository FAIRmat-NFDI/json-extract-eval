from json_extract_eval.core.schema.utils import (
    get_children,
    get_node_at_path,
    is_leaf,
    resolve_type,
)

# --- Shared fixtures ---

FLAT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
    },
}

NESTED_SCHEMA: dict[str, object] = {
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
}

ARRAY_OF_PRIMITIVES: dict[str, object] = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

ARRAY_OF_OBJECTS: dict[str, object] = {
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
}

DEEPLY_NESTED: dict[str, object] = {
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
                            "measurements": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "property": {"type": "string"},
                                        "value": {"type": "number"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


# --- resolve_type ---


class TestResolveType:
    def test_string_type(self) -> None:
        assert resolve_type({"type": "string"}) == "string"

    def test_object_type(self) -> None:
        assert resolve_type({"type": "object"}) == "object"

    def test_missing_type(self) -> None:
        assert resolve_type({}) is None

    def test_non_string_type(self) -> None:
        assert resolve_type({"type": 42}) is None  # type: ignore[dict-item]


# --- is_leaf ---


class TestIsLeaf:
    def test_string_field(self) -> None:
        assert is_leaf({"type": "string"}) is True

    def test_number_field(self) -> None:
        assert is_leaf({"type": "number"}) is True

    def test_object_with_properties(self) -> None:
        assert is_leaf({"type": "object", "properties": {"x": {"type": "string"}}}) is False

    def test_object_without_properties(self) -> None:
        assert is_leaf({"type": "object"}) is True

    def test_array_with_items(self) -> None:
        assert is_leaf({"type": "array", "items": {"type": "string"}}) is False

    def test_array_without_items(self) -> None:
        assert is_leaf({"type": "array"}) is True

    def test_empty_schema(self) -> None:
        assert is_leaf({}) is True

    def test_object_with_empty_properties(self) -> None:
        assert is_leaf({"type": "object", "properties": {}}) is True


# --- get_children ---


class TestGetChildren:
    def test_flat_object(self) -> None:
        children = get_children(FLAT_SCHEMA)
        names = [name for name, _, _ in children]
        assert set(names) == {"name", "age", "active"}

    def test_child_paths_flat(self) -> None:
        children = get_children(FLAT_SCHEMA)
        paths = [path for _, _, path in children]
        assert "name" in paths
        assert "age" in paths

    def test_field_names(self) -> None:
        children = get_children(FLAT_SCHEMA)
        name_to_path = {name: path for name, _, path in children}
        assert name_to_path["name"] == "name"
        assert name_to_path["age"] == "age"
        assert name_to_path["active"] == "active"

    def test_child_paths_with_parent(self) -> None:
        inner = NESTED_SCHEMA["properties"]["experiment"]  # type: ignore[index]
        children = get_children(inner, path="experiment")
        name_to_path = {name: path for name, _, path in children}
        assert name_to_path["name"] == "experiment.name"
        assert name_to_path["temp"] == "experiment.temp"

    def test_array_items(self) -> None:
        tags = ARRAY_OF_PRIMITIVES["properties"]["tags"]  # type: ignore[index]
        children = get_children(tags, path="tags")
        assert len(children) == 1
        name, schema, path = children[0]
        assert name == "[]"
        assert path == "tags[]"
        assert schema == {"type": "string"}

    def test_leaf_returns_empty(self) -> None:
        assert get_children({"type": "string"}) == []

    def test_empty_path(self) -> None:
        children = get_children(FLAT_SCHEMA, path="")
        paths = [path for _, _, path in children]
        assert "name" in paths  # no leading dot


# --- get_node_at_path ---


class TestGetNodeAtPath:
    def test_empty_path_returns_root(self) -> None:
        assert get_node_at_path(FLAT_SCHEMA, "") is FLAT_SCHEMA

    def test_flat_property(self) -> None:
        node = get_node_at_path(FLAT_SCHEMA, "name")
        assert node == {"type": "string"}

    def test_nested_property(self) -> None:
        node = get_node_at_path(NESTED_SCHEMA, "experiment.name")
        assert node == {"type": "string"}

    def test_array_items(self) -> None:
        node = get_node_at_path(ARRAY_OF_PRIMITIVES, "tags[]")
        assert node == {"type": "string"}

    def test_array_of_objects_field(self) -> None:
        node = get_node_at_path(ARRAY_OF_OBJECTS, "samples[].id")
        assert node == {"type": "string"}

    def test_deeply_nested_leaf(self) -> None:
        node = get_node_at_path(DEEPLY_NESTED, "experiment.samples[].measurements[].value")
        assert node == {"type": "number"}

    def test_nonexistent_path(self) -> None:
        assert get_node_at_path(FLAT_SCHEMA, "nonexistent") is None

    def test_partial_nonexistent_path(self) -> None:
        assert get_node_at_path(NESTED_SCHEMA, "experiment.bogus") is None

    def test_array_path_on_non_array(self) -> None:
        assert get_node_at_path(FLAT_SCHEMA, "name[]") is None

    def test_root_array_path(self) -> None:
        schema: dict[str, object] = {"type": "array", "items": {"type": "string"}}
        node = get_node_at_path(schema, "[]")
        assert node == {"type": "string"}


