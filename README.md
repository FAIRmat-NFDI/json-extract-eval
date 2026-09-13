# Struct Extract Eval

extract-eval gives you per-field precision/recall/F1 for LLM JSON extraction.

## Why This Package

When an LLM extracts structured data from text, you need to know: how good is it?
Exact match against a gold JSON is useless -- `"New York"` vs `"NYC"`, `42` vs `42.0`
are semantically equivalent but fail string equality. And a single overall score tells
you nothing about which fields are wrong or how they are wrong.

This package helps you:
- Optimize prompts for LLM data extraction
- Compare models for extraction quality
- Compare extraction pipelines end to end

## What It Provides

- **Per-field evaluation.** Each field is scored independently with its own comparator
  and transform chain. Custom comparators can be registered for domain-specific needs.
- **Type-aware comparison.** Strings, numbers, booleans, arrays, and nested objects are
  each handled appropriately (`exact`, `numeric`, `oneof`, `semantic` via LLM judge).
- **Semantic equivalence.** Free-text fields can be judged by an LLM -- paraphrases
  count as correct, factual disagreements don't.
- **Diagnostic metrics.** Precision, recall, and F1 at per-record and per-field level.
  Trace exactly which fields were missed (omissions), invented (hallucinations), or
  wrong (mismatches). Post-processors can reclassify results before metrics are computed
  (e.g. treating null values as absent for constrained-output tools).
- **Array alignment.** Ordered, key-field, or Hungarian bipartite matching for arrays
  where element order may differ.
- **Post-processing.** Plug in custom logic to adjust how fields are scored before
  final metrics are computed. Built-in and custom post-processors supported.
- **Single source of truth.** All evaluation config lives in the schema as `x-eval-*`
  extension keys. One file, no drift.

## Installation

```bash
pip install -e .                       # core only
pip install -e ".[dev]"                # core + dev tools
pip install -e ".[dev,methodology]"    # + IAA, contamination, shift detection
```

Requires Python >= 3.10.

## Quick Start

```python
from struct_extract_eval import evaluate, infer_schema, annotate_xeval, parse_eval_schema, validate_gold

gold = [
    {"method": "sputtering", "temperature": 300, "lab_id": "A1"},
    {"method": "evaporation", "temperature": 450, "lab_id": "B2"},
]
extracted = [
    {"method": "sputtering", "temperature": 301, "lab_id": "A1"},
    {"method": "evaporation", "temperature": 460, "lab_id": "B3"},
]

# 1. Infer schema from gold, add eval defaults
eval_schema = infer_schema(gold)
annotate_xeval(eval_schema)

# 2. Review and customize eval_schema (edit x-eval-* keys)

# 3. Validate gold against eval_schema to catch errors and omissions
# and parse schema to ensure no errors before evaluation.
parse_eval_schema(eval_schema)
validate_gold(gold, eval_schema)

# 4. Evaluate
result = evaluate(gold, extracted, schema=eval_schema)
print(f"F1: {result.mean_f1:.2f}")
print(f"Precision: {result.mean_precision:.2f}")
print(f"Recall: {result.mean_recall:.2f}")

for path, agg in result.per_field.items():
    print(f"  {path}: score={agg.mean_score:.2f}  matches={agg.matches}  "
          f"mismatches={agg.mismatches}  omissions={agg.omissions}")
```

For more details, please check examples in `examples/`.
---

## Key Concepts

### Terminology

| Term | Meaning |
|------|---------|
| **Gold** | Ground truth JSON instances (what the correct extraction looks like) |
| **Extracted** | LLM-produced JSON instances (what the extractor actually output) |
| **Resolved schema** | A simplified JSON schema with only `type`, `properties`, and `items` -- no `$ref`, `allOf`, `anyOf`, etc. |
| **Eval schema** | A resolved schema annotated with `x-eval-*` keys that tells the evaluator how to compare each field |
| **Transform** | A preprocessing step applied to both gold and extracted values before comparison (e.g. `lowercase`, `strip`). Configured via `x-eval-transform`. |
| **Comparator** | A function that scores one field by comparing gold and extracted values. Built-ins: `exact`, `numeric`, `oneof`. Custom comparators can be registered. |
| **Batch comparator** | A comparator that receives all fields in a record that use it and scores them together (e.g. LLM judge for semantic comparison, or compound comparators for grouped fields like name parts). |
| **Post-processor** | A function that reclassifies field results after scoring but before metrics are computed (e.g. treating null values as absent). |

### Default Field Statuses

Every leaf field gets one of these statuses after scoring:

| Status | What happened                                                               | Effect on metrics |
|--------|-----------------------------------------------------------------------------|-------------------|
| **match** | Both sides present, values equivalent                                       | Raises both precision and recall |
| **mismatch** | Both sides present, values differ                                           | Lowers both precision and recall |
| **omission** | In gold but missing key from extracted                                      | Lowers recall only |
| **hallucination** | Key not in gold but present in extracted, or not in schema but in extracted | Lowers precision only |
| **skipped** | Field marked `x-eval-skip: true`                                            | Excluded from all metrics |
| **pending** | Awaiting batch comparator dispatch (internal, resolved before metrics)       | Excluded from all metrics |
| **batch_error** | Batch comparator failed for this field                                       | Excluded from all metrics |

Field statuses can be changed by post-processors before metrics are computed.

### Scoring: Precision, Recall, F1

**Precision** = matches / (matches + mismatches + hallucinations).
"Of what the extractor produced, how much is correct?"

**Recall** = matches / (matches + mismatches + omissions).
"Of what gold expected, how much did the extractor get right?"

**F1** = harmonic mean of precision and recall.

Run-level metrics (`mean_precision`, `mean_recall`, `mean_f1`) are the arithmetic mean across all records.

### The Scoring Table

| In schema? | Gold has field? | Extracted has field? | Result |
|------------|-----------------|----------------------|--------|
| Yes | Yes | Yes | Compare using the field's comparator (match/mismatch) |
| Yes | Yes | No | **Omission** -- penalizes recall |
| Yes | No | Yes | **Hallucination** -- penalizes precision |
| Yes | No | No | Nothing -- field doesn't exist for this record |
| No | Yes | Yes/No | **Skipped** -- no comparator exists; excluded from metrics |
| No | No | Yes | **Hallucination** -- extractor invented an unknown field |
| No | No | No | Nothing -- field doesn't exist for this record |

**Important:** By default, all gold fields must be defined in the eval schema, so the evaluator knows
how to score them. A gold instance can omit a field that is in the schema (it simply won't
be scored for that record). `validate_gold()` raises an error if gold has fields not in
the schema. Pass `allow_extra_fields=True` to `validate_gold()` when gold records intentionally carry
unscored metadata. Those fields remain visible with `status="skipped"` and a diagnostic
reason, whether or not the extractor also returned them.
Property names containing `.` are rejected because dots separate nested field
paths in evaluation output. Gold property names are checked during the existing
schema traversal, before reporting an unknown field; comparator-owned values
remain opaque to structural validation.




---

## The Workflow

```
  Gold Instances          Existing JSON Schema
       |                        |
       v                        v
 infer_schema()      resolve_schema_references()
       |                        |
       +----------+-------------+
                  |
                  v
           Resolved Schema
          (type, properties,
           items only)
                  |
                  v
         annotate_xeval()
                  |
                  v
          Eval Schema (with x-eval-*)
                  |
                  v
          User reviews and edits
                  |
                  v
          validate_gold()          <-- recommended, otherwise, scoring may be inaccurate
                  |
                  v
     evaluate(gold, extracted, schema)
                  |
                  v
             RunResult
       (precision, recall, F1,
        per-field breakdown,
        per-record detail)
```

---


## Comparators

Built-in comparators (registered by default):

| Comparator | Use case | Score |
|------------|----------|-------|
| [`exact`](src/struct_extract_eval/core/comparators/exact.py#L6) | Booleans, enums, IDs, short strings | 0 or 1. Strict type and value equality. |
| [`numeric`](src/struct_extract_eval/core/comparators/numeric.py#L7) | Numbers | 0 or 1. Within tolerance = 1, outside = 0. Default: exact equality. |
| [`oneof`](src/struct_extract_eval/core/comparators/oneof.py#L6) | Fields with known acceptable synonyms | 1 if extracted matches any value in list, 0 otherwise. |

Provided batch comparator (must be registered by the user before use):

| Comparator | Use case | Score |
|------------|----------|-------|
| [`semantic`](src/struct_extract_eval/batch/semantic_comparator.py#L28) | Free-text fields (paraphrases, synonyms) | 0 or 1. Uses an LLM judge. Short-circuits on exact string match. See `examples/04_example_semantic`. |

Schema examples:

```json
"x-eval-compare": "exact"
"x-eval-compare": {"numeric": {"tolerance": {"rel": 0.01}}}
"x-eval-compare": {"oneof": {"values": ["PVD", "Sputtering", "CVD"]}}
"x-eval-compare": "semantic"
```

### Custom Comparators

Write a function that takes `(gold, extracted, params)` and returns a `ComparatorResult`,
then register it:

```python
from struct_extract_eval import ComparatorResult, register

def compare_date(gold, extracted, params):
    """Compare dates regardless of format."""
    from datetime import datetime
    formats = params.get("formats", ["%Y-%m-%d", "%b %d, %Y"])
    def parse(val):
        for fmt in formats:
            try:
                return datetime.strptime(str(val), fmt)
            except ValueError:
                continue
        return None
    g, e = parse(gold), parse(extracted)
    return ComparatorResult(
        score=1.0 if (g and e and g == e) else 0.0,
        comparator="date",
    )

register("date", compare_date)
```

Then in the schema: `"x-eval-compare": {"date": {"formats": ["%Y-%m-%d", "%b %d, %Y"]}}`

Use `overwrite=True` to replace an existing registration:
`register("date", compare_date, overwrite=True)`

### Batch Comparators

Per-field comparators score one field at a time. **Batch comparators** receive all fields
in a record that use them and score them together in one call. Two use cases:

- **LLM judge** (`semantic`): batches multiple free-text fields into one API call for
  cost efficiency and consistency.
- **Compound comparators** ([`CompoundComparator`](src/struct_extract_eval/core/comparators/comparator.py#L104)): groups sibling fields (e.g. `surname` + `name`) and scores
  them as a unit.

Batch comparators are not registered by default. See `examples/04_example_semantic.ipynb` and
`examples/05_example_compound.ipynb` in the examples.

Use `overwrite=True` to replace an existing custom registration (e.g. in notebooks):
`register("date", compare_date, overwrite=True)`. Built-in comparators (`exact`,
`numeric`, `oneof`) cannot be overwritten.

---

## Transforms

Preprocess both gold and extracted values before comparison. Chained left to right.
Transforms receive every value, including `null`: the built-ins below no-op on
`null` (return it unchanged), while a custom transform may rewrite it (e.g.
`null` -> `""`).

| Transform | Params | What it does |
|-----------|--------|-------------|
| [`lowercase`](src/struct_extract_eval/core/transforms/builtins.py#L5) | -- | Convert to lowercase |
| [`strip`](src/struct_extract_eval/core/transforms/builtins.py#L13) | -- | Strip leading/trailing whitespace |
| [`normalize_whitespace`](src/struct_extract_eval/core/transforms/builtins.py#L22) | -- | Collapse multiple spaces/newlines to single space |
| [`sort_tokens`](src/struct_extract_eval/core/transforms/builtins.py#L33) | -- | Alphabetize whitespace-separated tokens |
| [`round_digits`](src/struct_extract_eval/core/transforms/builtins.py#L44) | `{"digits": int}` | Round numeric value to N decimal places |
| [`type_convert`](src/struct_extract_eval/core/transforms/builtins.py#L99) | `{"to": "float"\|"int"\|"str"\|"bool"}` | Convert value to the given type |

Schema: `"x-eval-transform": ["strip", "lowercase"]`

---

## Array Alignment

Arrays need an alignment step before scoring: which gold element pairs with which
extracted element?

| Strategy | Config | When to use |
|----------|--------|-------------|
| **Ordered** (default) | No `x-eval-align` needed | Order matters (time series, steps) |
| **Key-field** | `{"match_by": "key_field", "key": "name"}` | Elements have a unique ID |
| **Hungarian** | `{"match_by": "hungarian"}` | No unique key, order doesn't matter |

After alignment, matched pairs are scored recursively. Unmatched gold elements are
omissions. Unmatched extracted elements are hallucinations.

See `examples/03_example_arrays.ipynb` in the examples.

---

## Post-Processing

Post-processors run after field scoring and can reclassify field results and influence the score matrix. Pass them to
`evaluate(post_process=...)`.

See the `reclassify_nulls` and `propagate_batch_errors` APIs.

---

## `x-eval-*` Extension Keys

All evaluation config lives in the JSON schema. No separate config file.

| Key | Purpose | Default | Example |
|-----|---------|---------|---------|
| `x-eval-compare` | Which comparator to use | inferred from type | `"exact"`, `{"numeric": {"tolerance": {"rel": 0.01}}}` |
| `x-eval-skip` | Exclude field from scoring | `false` | `true` |
| `x-eval-transform` | Preprocessing chain | none | `["lowercase", "strip"]` |
| `x-eval-align` | Array alignment strategy | ordered | `{"match_by": "key_field", "key": "name"}` |

Config syntax: both `x-eval-compare` and `x-eval-transform` entries use the same two shapes:
- String: `"exact"` (no parameters)
- Single-key dict: `{"numeric": {"tolerance": {"rel": 0.01}}}` (with parameters, value must be a dict)

### Field types (`type`)

`type` may be a single string or a **list** (JSON Schema allows both):

- `{"type": "string"}` -- a normal single-type field.
- `{"type": ["string", "null"]}` -- nullable; the `null` is dropped, so this is just a `string` field (null is handled by value presence, not type).
- `{"type": ["string", "object"]}` -- a **polymorphic** field that may take several shapes. It is scored as one value by its comparator (default `exact`; set `x-eval-compare` to a custom comparator that understands all the shapes). It is not scored structurally even if it also declares `properties`/`items`. See `examples/08_example_polymorphic.ipynb`.

---

## Results

### `RunResult`

| Field | Type | Description |
|-------|------|-------------|
| `records` | `list[RecordResult]` | All record results |
| `mean_precision` | `float` | Mean across records |
| `mean_recall` | `float` | Mean across records |
| `mean_f1` | `float` | Mean across records |
| `total_records` | `int` | Number of records evaluated |
| `total_fields` | `int` | Total scored fields |
| `total_omissions` | `int` | Fields missing from extracted |
| `total_hallucinations` | `int` | Extra fields in extracted |
| `total_batch_errors` | `int` | Fields where batch comparator failed |
| `per_field` | `dict[str, FieldAggregation]` | Per-field-path breakdown |

### `RecordResult`

| Field | Type | Description |
|-------|------|-------------|
| `record_id` | `str \| int` | Record identifier |
| `field_results` | `list[FieldResult]` | Per-field scores and statuses |
| `precision` | `float` | |
| `recall` | `float` | |
| `f1` | `float` | |

### `FieldAggregation`

| Field | Type | Description |
|-------|------|-------------|
| `mean_score` | `float` | Average score for this field path |
| `matches` | `int` | Correct extractions |
| `mismatches` | `int` | Incorrect extractions |
| `omissions` | `int` | Times this field was missing |
| `hallucinations` | `int` | Times this field was hallucinated |

The `per_field` breakdown is the primary diagnostic view -- it tells you which specific
fields your extractor struggles with.

---

## Examples

Step-by-step Jupyter notebooks in `examples/`:

| Example | What it covers |
|---------|---------------|
| `00_example_schema` | Getting a resolved schema (infer from gold or resolve existing schema) |
| `01_example_simple` | Simplest evaluation -- 3 steps, no customization |
| `02_example_customize` | Customizing the eval schema (oneof, tolerance, transforms, skip, custom comparator) |
| `03_example_arrays` | Array alignment (ordered, key-field, Hungarian) |
| `04_example_semantic` | Batch comparators and the LLM semantic judge |
| `05_example_compound` | Compound comparators (grouping sibling fields) |

---

## API Reference

| Function | Purpose |
|----------|---------|
| `infer_schema(instances)` | Infer resolved schema from gold instances |
| `resolve_schema_references(schema)` | Simplify a complex JSON Schema into a resolved schema |
| `annotate_xeval(schema)` | Add `x-eval-*` defaults to a resolved schema (in-place) |
| `set_type_default(json_type, comparator)` | Change the default comparator for a JSON type |
| `reset_type_defaults()` | Reset type-defaults mapping to built-in defaults |
| `parse_eval_schema(schema)` | Parse and validate eval schema, returns SchemaNode tree |
| `validate_gold(gold, schema, ..., strict_types=False)` | Validate gold against schema (extra-field errors; type mismatches warn by default, or raise when `strict_types=True`) |
| `evaluate(gold, extracted, schema)` | Evaluate gold vs extracted using a reviewed eval schema |
| `register(name, fn, overwrite=False)` | Register a custom comparator |
---
