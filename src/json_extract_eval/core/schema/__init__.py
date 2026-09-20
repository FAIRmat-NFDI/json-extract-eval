"""Schema handling: the SchemaNode tree, inference, traversal, x-eval config.

Modules:
- ``tree``       -- SchemaNode dataclass and parse_eval_schema()
- ``inference``  -- infer a resolved schema from gold, normalize $ref/allOf/anyOf
- ``utils``      -- traversal helpers for raw dict-schemas
- ``xeval``      -- x-eval-* annotation defaults and parsing
- ``validation`` -- validate gold instances against an eval schema
"""

from json_extract_eval.core.schema.inference import (
    collapse_multi_type_anyof,
    infer_schema,
    merge_all_of,
    remove_null_anyof,
    resolve_schema_references,
)
from json_extract_eval.core.schema.tree import (
    SchemaError,
    SchemaNode,
    parse_eval_schema,
)
from json_extract_eval.core.schema.utils import (
    get_children,
    get_leaf_paths,
    get_node_at_path,
    is_leaf,
    iter_schema,
    load_schema,
    non_null_types,
    resolve_type,
)
from json_extract_eval.core.schema.validation import (
    GoldValidationError,
    validate_gold,
)
from json_extract_eval.core.schema.xeval import (
    annotate_xeval,
    parse_xeval_entry,
    reset_type_defaults,
    set_type_default,
)

__all__ = [
    "GoldValidationError",
    "SchemaError",
    "SchemaNode",
    "annotate_xeval",
    "collapse_multi_type_anyof",
    "get_children",
    "get_leaf_paths",
    "get_node_at_path",
    "infer_schema",
    "is_leaf",
    "iter_schema",
    "load_schema",
    "merge_all_of",
    "non_null_types",
    "parse_eval_schema",
    "parse_xeval_entry",
    "remove_null_anyof",
    "reset_type_defaults",
    "resolve_schema_references",
    "resolve_type",
    "set_type_default",
    "validate_gold",
]
