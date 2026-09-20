from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class TransformSpec:
    """Reference to a transform in a chain: name + params.

    Stored on a SchemaNode after parsing. A chain of TransformSpecs
    represents the full ``x-eval-transform`` config for a field.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


class Transform(Protocol):
    """Interface for transform functions.

    A transform preprocesses a value before comparison.
    Applied to both gold and extracted values.
    """

    def __call__(self, value: Any, params: dict[str, Any]) -> Any: ...
