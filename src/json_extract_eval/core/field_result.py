"""Per-field evaluation result.

``FieldResult`` is the atomic unit of evaluation output -- one result per
leaf field compared between gold and extracted. It is defined here (rather
than in ``scoring``) so that both the scoring layer and the batch dispatch
layer can import it without circular dependencies.
"""

from dataclasses import dataclass
from typing import Literal

FieldStatus = Literal[
    "match", "mismatch", "omission", "hallucination", "skipped",
    "pending", "batch_error",
]


@dataclass
class FieldResult:
    """Result of comparing a single field between gold and extracted.

    Fields with ``status="pending"`` use a BatchComparator. The scoring layer
    leaves these with ``score=0.0`` and the comparator name in ``comparator``.
    A later pass via ``process_batches`` dispatches them to the registered
    batch handler and updates score/status/reason in place.

    ``gold_compared`` and ``extracted_compared`` are the values the comparator
    actually saw, after transforms were applied. When there are no transforms,
    they reference the same objects as ``gold_value`` and ``extracted_value``.
    These are used by batch handlers so the LLM/embedding/etc. sees the
    normalized values, not the raw ones.

    ``reason`` carries a short human-readable explanation, propagated from
    ``ComparatorResult.reason`` (per-field) or set by the batch handler.
    """

    path: str
    score: float
    comparator: str
    gold_value: object
    extracted_value: object
    status: FieldStatus
    reason: str | None = None
    gold_compared: object | None = None
    extracted_compared: object | None = None
