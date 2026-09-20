"""Core comparator types.

Two kinds of comparator:

- ``Comparator``: per-field comparator.

- ``BatchComparator``: compare many fields at once.

- ``CompoundComparator``: is a special kind of BatchComparator, it compares multiple compound fields.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ComparatorResult:
    """Result of comparing a gold value to an extracted value (single field).

    score: float in [0.0, 1.0]
    comparator: name of the comparator that produced this result
    reason: human-readable explanation, propagated to FieldResult.reason
    skip: if True, process_batches sets status="skipped" instead of match/mismatch.
        Used by compound comparators to mark supporting fields that contributed
        to a primary field's score but should not be counted in metrics themselves.
    """

    score: float
    comparator: str
    reason: str | None = field(default=None)
    skip: bool = field(default=False)


@dataclass(frozen=True)
class BatchItem:
    """One item in a batch passed to a BatchComparator.

    Carries everything a batch handler needs:
    - path: field path in the schema (e.g. "experiment.method")
    - gold_raw / extracted_raw: original values, as they appeared in the data
    - gold_compared / extracted_compared: post-transform values (what the
      comparator should actually compare). Same object as gold_raw/extracted_raw
      when there are no transforms.
    - params: per-field parameters from the comparator's x-eval-compare config
    """

    path: str
    gold_raw: object
    extracted_raw: object
    gold_compared: object
    extracted_compared: object
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparatorSpec:
    """Reference to a comparator: name + params.

    Stored on a SchemaNode after parsing. Container nodes (objects/arrays)
    use the empty default since they are scored via their children.
    """

    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)


class Comparator(Protocol):
    """Per-field comparator. One pair in, one result out."""

    def __call__(
        self, gold: Any, extracted: Any, params: dict[str, Any],
    ) -> ComparatorResult: ...


class BatchComparator(Protocol):
    """Batch comparator. Many fields in, many results out (one per input).

    Implementations are classes that set ``is_batch = True`` as a class
    attribute. The scoring dispatcher uses this attribute to decide whether
    to call inline (per-field) or defer to ``process_batches``.

    The returned list MUST be **positional**: same length as ``items``, and
    each entry corresponds to the input item at the same index. Each entry is:

    - ``ComparatorResult``: the handler decided this item -- score becomes
      the field's final score, status becomes match/mismatch
    - ``None``: the handler couldn't decide for this item -- ``process_batches``
      marks the corresponding field as ``batch_error``. Use this for per-item
      failures (LLM returned an invalid value, lookup failed, etc.) so a single
      bad item doesn't poison the rest of the batch.

    If the WHOLE batch fails (e.g. the handler raises), it can also return an
    empty list -- ``process_batches`` then marks every item in the batch as
    ``batch_error``.
    """

    is_batch: bool

    def __call__(
        self, items: list[BatchItem],
    ) -> list[ComparatorResult | None]: ...


class CompoundComparator:
    """Score multiple sibling fields together as one logical unit.

    Problem
    -------
    Some fields only make sense together. For example, ``surname`` and ``name``
    form a person's full name. Scoring them independently gives misleading
    results: if the extractor writes ``{"surname": "Smith", "name": "Jane"}``
    when gold is ``{"surname": "Smith", "name": "John"}``, independent scoring
    gives ``surname=1.0`` even though "Smith, Jane" is a different person from
    "Smith, John".

    A compound comparator evaluates the full name as one unit: if ANY part is
    wrong, the whole compound scores 0.

    How it works
    ------------
    1. Tag all sibling fields with the same comparator name in the schema.
    2. The scoring layer defers them as ``pending`` (like any batch comparator).
    3. ``process_batches`` sends all pending fields to this handler at once.
    4. This base class groups items by **parent path** (e.g., the object fields
    `surname` and `name` appear multiple times under different paths), so each
    object's fields are evaluated as their own unit.
    5. For each group, it calls your ``compare()`` method with the gold and
       extracted values as simple dicts.
    6. The **primary** field gets the compound score (0 or 1).
    7. All other fields get ``status="skipped"`` -- they contributed to the
       comparison but are excluded from precision/recall/F1 metrics.
       The compound counts as **1 field** in the metrics, not N.

    What you write
    --------------
    Subclass ``CompoundComparator`` and override ``compare()``.
    The base class handles grouping, field extraction, incomplete groups,
    and the primary/skip logic.

    Example: full name comparator
    -----------------------------
        class NameComparator(CompoundComparator):
            def __init__(self):
                super().__init__(
                    fields=["surname", "name"],   # sibling fields in the compound
                    primary="surname",             # this field gets the score
                    name="name_compound",          # comparator name in results
                )

            def compare(self, gold, extracted):
                # gold = {"surname": "Smith", "name": "John"}
                # extracted = {"surname": "Smith", "name": "Jane"}
                g = f"{gold['name']} {gold['surname']}"
                e = f"{extracted['name']} {extracted['surname']}"
                return 1.0 if g == e else 0.0

        register("name_compound", NameComparator())

    Schema -- tag both fields with the same comparator name::

        {
          "surname": {"type": "string", "x-eval-compare": "name_compound"},
          "name":    {"type": "string", "x-eval-compare": "name_compound"}
        }

    Works inside arrays too.

    Other use cases
    ---------------
    - **Quantity + unit**: ``fields=["value", "unit"]`` -- convert units before comparing
    - **Coordinates**: ``fields=["lat", "lon"]`` -- compute distance
    - **Date + timezone**: ``fields=["date", "tz"]`` -- normalize to UTC

    Parameters
    ----------
    fields : list[str]
        Sibling field names that form the compound (e.g. ``["surname", "name"]``).
    primary : str
        Which field gets the compound score. Must be in ``fields``.
        All other fields get ``status="skipped"`` (excluded from metrics).
    name : str
        Comparator name used in ``ComparatorResult.comparator`` and error messages.
    """

    is_batch = True

    def __init__(self, fields: list[str], primary: str, name: str = "compound") -> None:
        if primary not in fields:
            raise ValueError(
                f"primary field {primary!r} must be in fields {fields}"
            )
        self.fields = fields
        self.primary = primary
        self.name = name

    def compare(
        self, gold: dict[str, object], extracted: dict[str, object],
    ) -> float:
        """Override this with your comparison logic. Return a score in [0.0, 1.0].

        You receive two dicts, each mapping field name to the post-transform
        value. The keys are exactly the ``fields`` you declared in ``__init__``.

        Example -- if ``fields=["surname", "name"]``::

            gold      = {"surname": "Smith", "name": "John"}
            extracted = {"surname": "Smith", "name": "Jane"}

        Return 1.0 if the compound matches, 0.0 if it doesn't.
        The base class applies this score to the primary field and skips the rest.
        """
        raise NotImplementedError

    def __call__(self, items: list[BatchItem]) -> list[ComparatorResult | None]:
        # Group items by parent path of this compound key
        by_parent: dict[str, list[tuple[int, BatchItem]]] = {} # int is used for mapping the by_parent back to the items
        for i, item in enumerate(items):
            parent = item.path.rsplit(".", 1)[0] if "." in item.path else ""
            by_parent.setdefault(parent, []).append((i, item))

        result_by_index: dict[int, ComparatorResult] = {}

        for parent, group in by_parent.items():
            # Extract field name -> (index, item) for this group
            fields: dict[str, tuple[int, BatchItem]] = {}
            for idx, item in group:
                field_name = (
                    item.path.rsplit(".", 1)[-1] if "." in item.path else item.path
                )
                fields[field_name] = (idx, item)

            # Check all expected fields are present
            # todo: rethink if do it this way or its ok to have missing key as long as the gold==extracted
            missing = [f for f in self.fields if f not in fields]
            if missing:
                for idx, _ in group:
                    result_by_index[idx] = ComparatorResult(
                        score=0.0,
                        comparator=self.name,
                        reason=f"incomplete compound (missing {missing})",
                    )
                continue

            # Build gold/extracted dicts from post-transform values
            gold_dict = {
                f: fields[f][1].gold_compared for f in self.fields
            }
            extracted_dict = {
                f: fields[f][1].extracted_compared for f in self.fields
            }

            score = self.compare(gold_dict, extracted_dict)

            # Primary field gets the score, others get skipped
            for field_name, (idx, _) in fields.items():
                if field_name == self.primary:
                    result_by_index[idx] = ComparatorResult(
                        score=score,
                        comparator=self.name,
                        reason="compound: match" if score >= 1.0 else "compound: mismatch",
                    )
                else:
                    result_by_index[idx] = ComparatorResult(
                        score=0.0,
                        comparator=self.name,
                        reason=f"compound with {parent + '.' if parent else ''}{self.primary}",
                        skip=True,
                    )

        return [result_by_index.get(i) for i in range(len(items))]
