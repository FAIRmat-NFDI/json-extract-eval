"""Content scoring: walk SchemaNode tree with gold + extracted, produce per-field scores.

Supports flat objects, nested objects, and arrays (ordered and unordered).
Array alignment strategies are controlled by ``x-eval-align`` on the array
node. Default is ordered (positional). Key-field alignment matches elements
by a unique identifier field. Hungarian bipartite matching pairs elements to
maximize total F1.
"""

import logging
from typing import Literal

from struct_extract_eval.core.comparators.batch import process_batches
from struct_extract_eval.core.comparators.registry import get_comparator, is_batch
from struct_extract_eval.core.field_result import FieldResult
from struct_extract_eval.core.schema import SchemaNode
from struct_extract_eval.core.transforms.registry import get_transform
from struct_extract_eval.core.transforms.transform import TransformSpec

logger = logging.getLogger(__name__)


def score_record(
    schema: SchemaNode,
    gold: dict[str, object],
    extracted: dict[str, object],
) -> list[FieldResult]:
    """Walk the SchemaNode tree, comparing gold and extracted at each field.

    Returns a flat list of FieldResult entries -- mostly leaf fields, plus the
    occasional container-level result (an empty-array match, or a container
    type mismatch). Fields marked ``x-eval-skip`` are also included with status
    ``"skipped"`` for visibility, but are excluded from all metric calculations.
    """
    return _score_node(schema, gold, extracted)


def _score_node(
    node: SchemaNode,
    gold_value: object,
    extracted_value: object,
) -> list[FieldResult]:
    """Recursively score a single schema node against gold and extracted values."""
    # A node is scored structurally only when it is a container (has children)
    # with NO explicit comparator. Everything else is scored as a single unit
    # by its comparator via _score_leaf:
    #   - leaves (annotate_xeval assigns a default comparator)
    #   - skip nodes (handled inside _score_leaf)
    #   - a container with an explicit x-eval-compare: the comparator receives the
    #   whole value and apply comparator.
    #
    # We intentionally gate on node.children, not node.json_type: json_type is
    # only a reference (it can be wrong for a polymorphic field), while the
    # actual gold/extracted values determine the real type below.
    if not node.children or node.comparator.name:
        return [_score_leaf(node, gold_value, extracted_value)]


    expected = dict if node.json_type == "object" else list
    if not (isinstance(gold_value, expected) and isinstance(extracted_value, expected)):
        return _score_container_type_error(node, gold_value, extracted_value, expected)
    if node.json_type == "object":
        return _score_object(node, gold_value, extracted_value)
    return _score_array(node, gold_value, extracted_value)


def _score_object(
    node: SchemaNode,
    gold_value: object,
    extracted_value: object,
) -> list[FieldResult]:
    """Score an object node by iterating its children.

    _score_node guarantees both sides are real dicts before dispatching.
    """
    assert isinstance(gold_value, dict) and isinstance(extracted_value, dict)
    gold_dict: dict[str, object] = gold_value # for type checkers
    extracted_dict: dict[str, object] = extracted_value
    results: list[FieldResult] = []

    # Both empty objects: object-level match, mirroring the empty-array case
    # ([] vs [] -> match). Without this, the property loop below scores nothing
    # (every field is absent on both sides), so the agreement would be invisible.
    if len(gold_dict) == 0 and len(extracted_dict) == 0:
        return [FieldResult(
            path=node.path,
            score=1.0,
            comparator="",
            gold_value={},
            extracted_value={},
            status="match",
        )]

    for child in node.children:
        field_name = child.name

        # Skip fields are included in results for visibility but excluded
        # from all metric calculations (precision, recall, F1, total_fields).
        if child.skip:
            gold_val = gold_dict.get(field_name) # only get the current layer
            extracted_val = extracted_dict.get(field_name)
            results.append(FieldResult(
                path=child.path,
                score=0.0,
                comparator="",
                gold_value=gold_val,
                extracted_value=extracted_val,
                status="skipped",
                reason="skipped by x-eval-skip",
            ))
            continue

        gold_has = field_name in gold_dict
        extracted_has = field_name in extracted_dict

        if gold_has and extracted_has:
            # Schema has key, gold has key, extracted has key -> compare values
            results.extend(_score_node(child, gold_dict[field_name], extracted_dict[field_name]))
        elif gold_has and not extracted_has:
            # Schema has key, gold has key, extracted missing -> omission
            results.extend(_omission_results(child, gold_dict[field_name]))
        elif extracted_has and not gold_has:
            # Schema has key, gold missing, extracted has key -> hallucination
            results.extend(_hallucination_results(child, extracted_dict[field_name]))
        # else: schema has key, gold missing, extracted missing -> skip (nothing to score)

    # Gold fields without a schema comparator remain visible, but cannot be
    # scored. Consume the matching extracted key when present so reproducing
    # an intentionally unscored gold field is not classified as hallucination.
    # The schema loop above handles extracted keys that ARE in the schema
    # (matched against gold to decide match/mismatch/hallucination/skip).
    schema_fields = {child.name for child in node.children}
    for key in sorted(gold_dict):
        if key not in schema_fields:
            extracted_has = key in extracted_dict
            path = f"{node.path}.{key}" if node.path else key
            reason = "gold field not in schema, no comparator to apply"
            if extracted_has:
                reason += "; extracted also has this field"
            results.append(FieldResult(
                path=path,
                score=0.0,
                comparator="",
                gold_value=gold_dict[key],
                extracted_value=extracted_dict.get(key),
                status="skipped",
                reason=reason,
            ))

    # An extracted key is a hallucination only when neither schema nor gold
    # contains it.
    for key in sorted(extracted_dict):
        if key not in schema_fields and key not in gold_dict:
            path = f"{node.path}.{key}" if node.path else key
            results.append(FieldResult(
                path=path,
                score=0.0,
                comparator="",
                gold_value=None,
                extracted_value=extracted_dict[key],
                status="hallucination",
            ))

    return results


def _score_array(
    node: SchemaNode,
    gold_value: object,
    extracted_value: object,
) -> list[FieldResult]:
    """Dispatch to the appropriate array scoring strategy based on node.align.

    Supports ordered (positional) and unordered matching. For unordered,
    key-field alignment matches elements by a unique identifier field.
    Hungarian bipartite matching finds the optimal pairing by maximizing
    total F1 across all pairs.
    """
    align = node.align
    if align is None or align.get("ordered") is True:
        return _score_array_ordered(node, gold_value, extracted_value)
    match_by = align.get("match_by")
    if match_by == "key_field":
        return _score_array_matched_by_key_field(
            node, gold_value, extracted_value, key=str(align["key"])
        )
    if match_by == "hungarian":
        return _score_array_hungarian(node, gold_value, extracted_value)
    logger.warning(
        "Unknown x-eval-align match_by='%s' at '%s'. "
        "Falling back to ordered matching.",
        match_by, node.path,
    )
    return _score_array_ordered(node, gold_value, extracted_value)


def _score_array_ordered(
    node: SchemaNode,
    gold_value: object,
    extracted_value: object,
) -> list[FieldResult]:
    """Score an array node using ordered (positional) matching."""
    # _score_node guarantees both sides are real lists before dispatching.
    assert isinstance(gold_value, list) and isinstance(extracted_value, list)
    gold_list: list[object] = gold_value
    extracted_list: list[object] = extracted_value
    items_node = node.children[0]

    # Both empty lists: array-level match
    if len(gold_list) == 0 and len(extracted_list) == 0:
        return [FieldResult(
            path=node.path,
            score=1.0,
            comparator="",
            gold_value=[],
            extracted_value=[],
            status="match",
        )]

    results: list[FieldResult] = []

    # Matched pairs: compare element by element
    matched_count = min(len(gold_list), len(extracted_list))
    for i in range(matched_count):
        element_results = _score_node(items_node, gold_list[i], extracted_list[i])
        _rewrite_element_paths(element_results, items_node.path, i)
        results.extend(element_results)

    # Extra gold elements: omissions
    for i in range(matched_count, len(gold_list)):
        element_results = _omission_results(items_node, gold_list[i])
        _rewrite_element_paths(element_results, items_node.path, i)
        results.extend(element_results)

    # Extra extracted elements: hallucinations. Negative indices count down
    # (-1, -2, ...) so each has no gold counterpart yet stays distinct.
    halluc_index = -1
    for i in range(matched_count, len(extracted_list)):
        element_results = _hallucination_results(items_node, extracted_list[i])
        _rewrite_element_paths(element_results, items_node.path, halluc_index)
        results.extend(element_results)
        halluc_index -= 1

    return results


# Maximum number of (gold, extracted) pairs to score for Hungarian matching.
# Beyond this, the O(n*m*fields) cost matrix computation is too expensive.
_MAX_HUNGARIAN_PAIRS = 2500  # 50 x 50


def _score_array_hungarian(
    node: SchemaNode,
    gold_value: object,
    extracted_value: object,
) -> list[FieldResult]:
    """Score an array using Hungarian (bipartite) matching.

    Finds the optimal pairing of gold and extracted elements that
    maximizes total F1 across all pairs. Uses
    ``scipy.optimize.linear_sum_assignment`` on a cost matrix built
    by scoring every (gold_i, extracted_j) pair.

    For primitives, each pair has one field, so F1 equals the
    comparator score. For objects, F1 aggregates across multiple fields.

    Falls back to ordered matching with a warning if the number of
    pairs exceeds ``_MAX_HUNGARIAN_PAIRS``.
    """
    from struct_extract_eval.core.record import build_record_result

    # _score_node guarantees both sides are real lists before dispatching.
    assert isinstance(gold_value, list) and isinstance(extracted_value, list)
    gold_list: list[object] = gold_value
    extracted_list: list[object] = extracted_value
    items_node = node.children[0]

    # Both actual empty lists: array-level match
    if len(gold_list) == 0 and len(extracted_list) == 0:
        return [FieldResult(
            path=node.path,
            score=1.0,
            comparator="",
            gold_value=[],
            extracted_value=[],
            status="match",
        )]

    results: list[FieldResult] = []
    n = len(gold_list)
    m = len(extracted_list)

    # Size guard: fall back to ordered for very large arrays
    if n * m > _MAX_HUNGARIAN_PAIRS:
        logger.warning(
            "Array at '%s' has %d x %d = %d pairs, exceeding "
            "Hungarian threshold (%d)",
            node.path, n, m, n * m, _MAX_HUNGARIAN_PAIRS,
        )
        return _score_array_ordered(node, gold_value, extracted_value)

    # One side empty: no matching needed, just omissions/hallucinations
    if n == 0:
        halluc_index = -1
        for elem in extracted_list:
            element_results = _hallucination_results(items_node, elem)
            _rewrite_element_paths(element_results, items_node.path, halluc_index)
            results.extend(element_results)
            halluc_index -= 1
        return results
    if m == 0:
        for idx, elem in enumerate(gold_list):
            element_results = _omission_results(items_node, elem)
            _rewrite_element_paths(element_results, items_node.path, idx)
            results.extend(element_results)
        return results

    # score ALL n*m pairs, build cost matrix and results cache
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError(
            "Hungarian matching requires numpy and scipy. "
            "Install with: pip install numpy scipy"
        ) from exc

    score_matrix = np.zeros((n, m))
    # an array element gets a list of FieldResults
    results_matrix: list[list[list[FieldResult]]] = [
        [[] for _ in range(m)] for _ in range(n)
    ]

    has_pending = False
    for i, g in enumerate(gold_list):
        for j, e in enumerate(extracted_list):
            pair_results = _score_node(items_node, g, e)
            results_matrix[i][j] = pair_results
            if not has_pending and any(r.status == "pending" for r in pair_results):
                has_pending = True

    # Resolve batch comparators before computing the cost matrix.
    # This makes n*m batch calls -- the user opted into this by combining
    # batch comparators with Hungarian matching.
    if has_pending:
        logger.warning(
            "Array at '%s' uses Hungarian matching with batch comparators. "
            "Resolving %d x %d = %d pairs via batch comparator. "
            "Consider switching to key-field alignment "
            "(x-eval-align: {match_by: 'key_field', key: ...}) or adding "
            "deterministic/scorable fields to reduce ambiguity and cost.",
            node.path, n, m, n * m,
        )
        for i in range(n):
            for j in range(m):
                process_batches(results_matrix[i][j], items_node)

    # Build cost matrix from F1 scores.
    # Note: post-processors (e.g. reclassify_nulls) are NOT applied here --
    # they run after scoring in evaluate(). See array.md for details.
    for i in range(n):
        for j in range(m):
            record = build_record_result(0, results_matrix[i][j], {}, {})
            score_matrix[i][j] = record.f1

    # Hungarian finds optimal matching (maximize total F1)
    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

    matched_gold: set[int] = set()
    matched_ext: set[int] = set()
    for i, j in zip(row_ind, col_ind):
        # Only count as matched if the pair has a positive score.
        # A pair with F1=0 means no fields matched — treat as
        # unmatched (omission + hallucination) rather than a bad match.
        if score_matrix[i][j] > 0:
            matched_gold.add(i)
            matched_ext.add(j)
            element_results = results_matrix[i][j]
            _rewrite_element_paths(element_results, items_node.path, i)
            results.extend(element_results)
        # else: leave both unmatched

    # Unmatched gold -> omissions
    for i in range(n):
        if i not in matched_gold:
            element_results = _omission_results(items_node, gold_list[i])
            _rewrite_element_paths(element_results, items_node.path, i)
            results.extend(element_results)

    # Unmatched extracted -> hallucinations (distinct negative indices).
    halluc_index = -1
    for j in range(m):
        if j not in matched_ext:
            element_results = _hallucination_results(items_node, extracted_list[j])
            _rewrite_element_paths(element_results, items_node.path, halluc_index)
            results.extend(element_results)
            halluc_index -= 1

    return results


def _score_array_matched_by_key_field(
    node: SchemaNode,
    gold_value: object,
    extracted_value: object,
    key: str,
) -> list[FieldResult]:
    """Score an array using key-field alignment.

    Matches gold and extracted elements by the value of a shared key field
    (e.g. "name"). Order doesn't matter. Elements with the same key value
    are paired and scored recursively. Unmatched gold elements produce
    omissions; unmatched extracted elements produce hallucinations.
    """
    # _score_node guarantees both sides are real lists before dispatching.
    assert isinstance(gold_value, list) and isinstance(extracted_value, list)
    gold_list: list[object] = gold_value
    extracted_list: list[object] = extracted_value
    items_node = node.children[0]

    results: list[FieldResult] = []

    # Both empty lists: array-level match
    if len(gold_list) == 0 and len(extracted_list) == 0:
        return [FieldResult(
            path=node.path,
            score=1.0,
            comparator="",
            gold_value=[],
            extracted_value=[],
            status="match",
        )]

    # Build lookup: key_value -> first element for extracted.
    # Duplicate keys in extracted: first occurrence wins the match,
    # subsequent duplicates are treated as unmatched (hallucinations).
    extracted_by_key: dict[str | int | float, object] = {}
    extracted_unmatched: list[object] = []
    for elem in extracted_list:
        if not isinstance(elem, dict) or key not in elem:
            extracted_unmatched.append(elem)
            continue
        k = elem[key]
        if isinstance(k, bool) or not isinstance(k, (str, int, float)):
            # Unhashable, non-primitive, or bool key value — can't match.
            # bool is excluded because True == 1 and False == 0 in Python,
            # which causes silent key collisions in the lookup dict.
            logger.warning(
                "Key field '%s' at '%s' has non-matchable value %r (%s). "
                "Element treated as unmatched.",
                key, node.path, k, type(k).__name__,
            )
            extracted_unmatched.append(elem)
            continue
        if k in extracted_by_key:
            # Duplicate key — first wins, rest are unmatched
            logger.warning(
                "Duplicate key '%s'=%r in extracted at '%s'. "
                "Only the first occurrence is matched.",
                key, k, node.path,
            )
            extracted_unmatched.append(elem)
            continue
        extracted_by_key[k] = elem

    matched_keys: set[str | int | float] = set()

    # Match gold elements against extracted by key
    for idx, gold_elem in enumerate(gold_list):

        # todo rethink if the gold missing the key, what to do ?
        if not isinstance(gold_elem, dict) or key not in gold_elem:
            # Gold element missing the key field — omission
            element_results = _omission_results(items_node, gold_elem)
            _rewrite_element_paths(element_results, items_node.path, idx)
            results.extend(element_results)
            continue

        k = gold_elem[key]
        if isinstance(k, bool) or not isinstance(k, (str, int, float)):
            logger.warning(
                "Key field '%s' at '%s' has non-matchable value %r (%s) "
                "in gold. Element treated as unmatched.",
                key, node.path, k, type(k).__name__,
            )
            # todo rethink when key is not hashable, what todo ?
            element_results = _omission_results(items_node, gold_elem)
            _rewrite_element_paths(element_results, items_node.path, idx)
            results.extend(element_results)
            continue
        if k in matched_keys:
            # Duplicate key in gold — first already consumed the match
            logger.warning(
                "Duplicate key '%s'=%r in gold at '%s'. "
                "Only the first occurrence is matched.",
                key, k, node.path,
            )

            # todo rethink when key not unique, what to do ?
            element_results = _omission_results(items_node, gold_elem)
            _rewrite_element_paths(element_results, items_node.path, idx)
            results.extend(element_results)
        elif k in extracted_by_key:
            # Matched pair: score recursively
            matched_keys.add(k)
            element_results = _score_node(
                items_node, gold_elem, extracted_by_key[k]
            )
            _rewrite_element_paths(element_results, items_node.path, idx)
            results.extend(element_results)
        else:
            # No match in extracted — omission
            element_results = _omission_results(items_node, gold_elem)
            _rewrite_element_paths(element_results, items_node.path, idx)
            results.extend(element_results)

    # Hallucinations (no gold counterpart) get distinct negative indices
    # counting down. Both sources below share one counter so every hallucinated
    # element in this array has a unique path.
    halluc_index = -1

    # Unmatched extracted elements (key not in gold).
    for k, elem in extracted_by_key.items():
        if k not in matched_keys:
            element_results = _hallucination_results(items_node, elem)
            _rewrite_element_paths(element_results, items_node.path, halluc_index)
            results.extend(element_results)
            halluc_index -= 1

    # Extracted elements without the key field or with
    # unhashable/duplicate keys.
    for elem in extracted_unmatched:
        element_results = _hallucination_results(items_node, elem)
        _rewrite_element_paths(element_results, items_node.path, halluc_index)
        results.extend(element_results)
        halluc_index -= 1

    return results


def _score_container_type_error(
    node: SchemaNode,
    gold_value: object,
    extracted_value: object,
    expected: type,
) -> list[FieldResult]:
    """Wrong-type / missing policy when a container isn't the expected type.

    Reached only for a container without a comparator (one with a comparator is
    handled by ``_score_node``), so there is no comparator to consult here.

    - both present, not both the expected type -> ``match`` if the raw values
      are equal (the extractor reproduced gold exactly, even off-shape), else
      ``mismatch``. With well-formed gold (a real container) an off-type
      extracted value can never be equal, so this only ever rewards faithfully
      reproducing already-off-shape gold.
    - one side present, the other absent (``None``) -> omissions /
      hallucinations, expanded element/field-wise when the present side IS the
      expected type, otherwise a single node-level result.
    - both absent -> nothing scorable.
    """
    gold_present = gold_value is not None
    extracted_present = extracted_value is not None

    if gold_present and extracted_present:
        match = gold_value == extracted_value
        return [FieldResult(
            path=node.path,
            score=1.0 if match else 0.0,
            comparator="",
            gold_value=gold_value,
            extracted_value=extracted_value,
            status="match" if match else "mismatch",
            reason=None if match else (
                f"type mismatch: gold {type(gold_value).__name__}, "
                f"extracted {type(extracted_value).__name__}"
            ),
        )]

    if gold_present:
        if isinstance(gold_value, expected):
            return _omission_results(node, gold_value)
        return [FieldResult(
            path=node.path,
            score=0.0,
            comparator="",
            gold_value=gold_value,
            extracted_value=None,
            status="omission",
        )]
    if extracted_present:
        if isinstance(extracted_value, expected):
            return _hallucination_results(node, extracted_value)
        return [FieldResult(
            path=node.path,
            score=0.0,
            comparator="",
            gold_value=None,
            extracted_value=extracted_value,
            status="hallucination",
        )]
    return []  # both absent


def _score_leaf(
    node: SchemaNode,
    gold_value: object,
    extracted_value: object,
) -> FieldResult:
    """Score a leaf node: apply transforms, then dispatch to per-field or batch comparator.

    Returns a single FieldResult. For per-field comparators, the result is final.
    For batch comparators, the result is provisional (pending_batch set, score=0.0)
    and process_batches will fill in the real score later.
    """
    if node.skip:
        return FieldResult(
            path=node.path,
            score=0.0,
            comparator="",
            gold_value=gold_value,
            extracted_value=extracted_value,
            status="skipped",
        )

    gold_transformed = _apply_transforms(gold_value, node.transforms)
    extracted_transformed = _apply_transforms(extracted_value, node.transforms)

    comparator_fn = get_comparator(node.comparator.name)

    if is_batch(comparator_fn):
        # Defer: build a provisional FieldResult, mark pending. process_batches
        # will dispatch this to the registered batch handler later.
        # The comparator name identifies which batch handler to use.
        return FieldResult(
            path=node.path,
            score=0.0,
            comparator=node.comparator.name,
            gold_value=gold_value,
            extracted_value=extracted_value,
            status="pending",
            gold_compared=gold_transformed,
            extracted_compared=extracted_transformed,
        )

    # Per-field comparator: call inline
    result = comparator_fn(gold_transformed, extracted_transformed, node.comparator.params)

    status = "match" if result.score == 1.0 else "mismatch"

    return FieldResult(
        path=node.path,
        score=result.score,
        comparator=node.comparator.name,
        gold_value=gold_value,
        extracted_value=extracted_value,
        status=status,
        reason=result.reason,
        gold_compared=gold_transformed,
        extracted_compared=extracted_transformed,
    )


def _apply_transforms(value: object, transforms: list[TransformSpec]) -> object:
    """Apply a chain of transforms to a value.

    None is passed through to the transforms like any other value -- a transform
    decides for itself what None means (the built-ins no-op on None; a custom
    transform may rewrite it, e.g. None -> ""). This is what lets the transform
    layer normalize nulls.
    """
    if not transforms:
        return value
    for spec in transforms:
        fn = get_transform(spec.name)
        value = fn(value, spec.params)
    return value


def _omission_results(node: SchemaNode, gold_value: object) -> list[FieldResult]:
    """Omission results for a subtree that gold has and extracted lacks."""
    return _one_sided_results(node, gold_value, "omission")


def _hallucination_results(node: SchemaNode, extracted_value: object) -> list[FieldResult]:
    """Hallucination results for a subtree that extracted has and gold lacks."""
    return _one_sided_results(node, extracted_value, "hallucination")


def _one_sided_results(
    node: SchemaNode,
    value: object,
    status: Literal["omission", "hallucination"],
) -> list[FieldResult]:
    """Generate FieldResults for a subtree that is present on only one side.

    ``value`` is the gold subtree for an omission or the extracted subtree for
    a hallucination.
    the value is passed as arg for counting how many FieldResult with status should
    be appended.
    """
    if node.skip:
        return []
    side = "gold" if status == "omission" else "extracted"

    # Every child of the node gets the same one-sided status.
    if node.json_type == "object" and node.children:
        if value is not None and not isinstance(value, dict):
            logger.warning(
                "Expected dict at '%s', got %s in %s", node.path, type(value).__name__, side
            )
        value_dict = value if isinstance(value, dict) else {}
        results: list[FieldResult] = []
        for child in node.children:
            if child.name not in value_dict:
                continue
            results.extend(_one_sided_results(child, value_dict[child.name], status))
        return results

    if node.json_type == "array" and node.children:
        if value is not None and not isinstance(value, list):
            logger.warning(
                "Expected list at '%s', got %s in %s", node.path, type(value).__name__, side
            )
        value_list = value if isinstance(value, list) else []
        items_node = node.children[0]  # arrays have exactly one child: the items schema
        if len(value_list) == 0: # empty one side
            return [_one_sided_result(node, value, status, comparator="")]
        item_results: list[FieldResult] = []
        for position, elem in enumerate(value_list):
            index = position if status == "omission" else -(position + 1)
            elem_results = _one_sided_results(items_node, elem, status)
            _rewrite_element_paths(elem_results, items_node.path, index)
            item_results.extend(elem_results)
        return item_results

    # leaf
    return [_one_sided_result(node, value, status, comparator=node.comparator.name)]


def _one_sided_result(
    node: SchemaNode,
    value: object,
    status: Literal["omission", "hallucination"],
    comparator: str,
) -> FieldResult:
    """One zero-score result with ``value`` on whichever side has it."""
    return FieldResult(
        path=node.path,
        score=0.0,
        comparator=comparator,
        gold_value=value if status == "omission" else None,
        extracted_value=value if status == "hallucination" else None,
        status=status,
    )


def _rewrite_element_paths(
    results: list["FieldResult"],
    items_path: str,
    element_index: int,
) -> None:
    """Rewrite schema paths to instance paths for array element results.

    ``items_path`` is the items node's schema path (e.g. ``"steps[]"`` or
    ``"layers[].steps[]"``). This function replaces the LAST ``[]`` in that
    prefix with the element index, then applies the same replacement to every
    FieldResult path that starts with the prefix.

    Examples (items_path -> what happens to FieldResult.path):

    - items_path=``"steps[]"``, index=0:
      ``"steps[]"``      -> ``"steps[0]"``
      ``"steps[].name"`` -> ``"steps[0].name"``

    - items_path=``"layers[].steps[]"``, index=1:
      ``"layers[].steps[]"``      -> ``"layers[].steps[1]"``
      ``"layers[].steps[].name"`` -> ``"layers[].steps[1].name"``
      (the parent ``layers[]`` is left for the outer array scorer to resolve)
    """
    last_bracket = items_path.rfind("[]")
    if last_bracket == -1:
        return  # no [] in path — nothing to rewrite
    instance_path = (
        items_path[:last_bracket] + f"[{element_index}]" + items_path[last_bracket + 2:]
    )
    prefix_len = len(items_path)
    for r in results:
        r.path = instance_path + r.path[prefix_len:]
