"""LLM judge clients used by the semantic batch comparator.

This module owns ONLY the LLM-facing interface. The dispatch logic that hooks
into the scoring layer lives in:

- ``batch/semantic_comparator.py`` -- the BatchComparator wrapper
- ``core/comparators/batch.py`` -- the generic process_batches dispatcher

The Judge Protocol is intentionally minimal:
``judge_batch(items) -> list[float | None]``. Implementations:

- ``GroqJudge`` -- real client backed by Groq's free API (requires the ``groq``
  package and a ``GROQ_API_KEY`` env var)
- ``FakeJudge`` -- offline, deterministic, used in tests

Both return positional lists of 0.0, 1.0, or None per item.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


DEFAULT_PROMPT_TEMPLATE = """\
You judge whether two values mean the same thing in a JSON extraction task.
For each numbered pair, decide if the EXTRACTED value is semantically
equivalent to the GOLD value. Be strict but fair:

- Capitalization, whitespace, and punctuation differences ARE equivalent.
- Synonyms, abbreviations, and paraphrases ARE equivalent
  (e.g. "PVD" = "physical vapor deposition", "sputter process" = "standard sputtering procedure").
- Different facts, numbers, units, or specifications are NOT equivalent.
- When in doubt, return 0.

Each pair includes the FIELD path, which tells you what kind of value
it is. Use the field name as context (e.g. a "method" field vs a
"temperature" field need different judgment).

Pairs to judge:
{pairs}

Return STRICT JSON in this exact format and nothing else:
{{"results": [<0 or 1>, <0 or 1>, ...]}}

The results array MUST have exactly the same number of entries as
input pairs, in the same order.
"""


@dataclass(frozen=True)
class JudgeItem:
    """A single (gold, extracted) pair sent to the LLM judge.

    Carries the field path so the LLM can use it as context.
    """

    path: str
    gold: object
    extracted: object


class Judge(Protocol):
    """Interface for an LLM judge.

    Takes a list of JudgeItem and returns a positional list of scores in the
    same order. Each entry is 0.0, 1.0, or None.

    - 0.0 / 1.0 -> the judge decided
    - None      -> the judge couldn't produce a valid score for this item (e.g.
                   the LLM returned 0.5 or "maybe"). The caller marks this
                   field as batch_error.

    If the whole batch fails (network error, parse failure of the entire
    response), the implementation may return an empty list -- the caller
    marks all items as batch_error.
    """

    def judge_batch(self, items: list[JudgeItem]) -> list[float | None]: ...


class GroqJudge:
    """LLM judge backed by Groq's free API.

    Uses the official ``groq`` Python SDK. Default model is ``openai/gpt-oss-120b``.
    The API key is read from ``GROQ_API_KEY`` unless passed explicitly.

    Caching: in-memory dict keyed by ``(model, path, gold_str, extracted_str)``.
    The path is included because the prompt tells the LLM to use the field
    name as context -- the same (gold, extracted) pair at different paths
    (e.g. ``"method"`` vs ``"description"``) could get different judgments.
    Cache survives the lifetime of the process only -- no disk persistence.
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        api_key: str | None = None,
        prompt_template: str | None = None,
    ):
        """
        Args:
            model: Groq model name.
            api_key: Groq API key. Falls back to ``GROQ_API_KEY`` env var.
            prompt_template: The full prompt sent to the LLM. Must contain
                ``{pairs}`` which gets replaced with the rendered
                gold/extracted pairs. If None, uses
                ``DEFAULT_PROMPT_TEMPLATE``. Example::

                    prompt_template=(
                        "You are a materials science judge.\\n\\n"
                        "{pairs}\\n\\n"
                        "PVD and sputtering are synonyms.\\n"
                        "Return JSON: {{\\\"results\\\": [0 or 1, ...]}}"
                    )

                Note: literal braces in the template must be doubled
                (``{{`` / ``}}``) because ``.format()`` uses ``{...}``
                for placeholders.
        """
        try:
            from groq import Groq  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "GroqJudge requires the 'groq' package. "
                "Install with: pip install 'struct-extract-eval[batch]'"
            ) from exc

        self.model = model
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        if "{pairs}" not in self.prompt_template:
            raise ValueError(
                "prompt_template must contain a '{pairs}' placeholder "
                "for the rendered gold/extracted pairs"
            )
        self._client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        # Cache key includes path because the prompt tells the LLM to use
        # the field name as context -- same (gold, extracted) at different
        # paths could get different judgments.
        self._cache: dict[tuple[str, str, str, str], float] = {}

    def judge_batch(self, items: list[JudgeItem]) -> list[float | None]:
        if not items:
            return []

        # Positional scratch list. Each slot is filled either from the cache
        # or from the call result. Slots stay None when the judge couldn't
        # decide for that item.
        scores: list[float | None] = [None] * len(items)
        pending_indices: list[int] = []
        for i, item in enumerate(items):
            cache_key = (
                self.model,
                item.path,
                _stringify(item.gold),
                _stringify(item.extracted),
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                scores[i] = cached
            else:
                pending_indices.append(i)

        if pending_indices:
            pending_items = [items[i] for i in pending_indices]
            fresh_scores = self._call_groq(pending_items)
            # fresh_scores may be shorter than pending_items (whole-response
            # parse failure or batch error) or contain None entries (per-item
            # invalid values). Cache only successful 0/1 scores.
            for j, idx in enumerate(pending_indices):
                if j < len(fresh_scores):
                    score = fresh_scores[j]
                    scores[idx] = score
                    if score is not None:
                        cache_key = (
                            self.model,
                            items[idx].path,
                            _stringify(items[idx].gold),
                            _stringify(items[idx].extracted),
                        )
                        self._cache[cache_key] = score
                # else: leave scores[idx] as None -> caller marks batch_error

        # Return the full positional list, including None entries.
        return scores

    def _call_groq(self, items: list[JudgeItem]) -> list[float | None]:
        """Make one Groq API call for a batch of items.

        Returns one entry per item: 0.0, 1.0, or None (invalid). If the API
        call raises, the exception propagates to the caller. If the whole
        response is unparseable, returns an empty list. If individual items
        come back with invalid values (e.g. 0.5), those positions are None.
        """
        pairs_text = _render_pairs(items)
        prompt = self.prompt_template.format(pairs=pairs_text)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        return _parse_judge_response(content, expected_count=len(items))


class FakeJudge:
    """Deterministic offline judge for tests and dry runs.

    Without configuration, returns 1.0 for items where gold and extracted
    have the same string representation (case-insensitive), 0.0 otherwise.

    For finer control, pass a ``responses`` dict mapping
    ``(gold_str, extracted_str)`` -> a score (0.0, 1.0, or None to simulate
    a per-item judge failure), OR pass ``default_score`` to override the
    fallback.
    """

    def __init__(
        self,
        responses: dict[tuple[str, str], float | None] | None = None,
        default_score: float | None = None,
    ):
        self.responses = responses or {}
        self.default_score = default_score
        self.calls: list[list[JudgeItem]] = []  # for test inspection

    def judge_batch(self, items: list[JudgeItem]) -> list[float | None]:
        self.calls.append(list(items))
        scores: list[float | None] = []
        for item in items:
            key = (_stringify(item.gold), _stringify(item.extracted))
            if key in self.responses:
                scores.append(self.responses[key])
            elif self.default_score is not None:
                scores.append(self.default_score)
            else:
                # Fallback: case-insensitive equality
                g = _stringify(item.gold).strip().lower()
                e = _stringify(item.extracted).strip().lower()
                scores.append(1.0 if g == e else 0.0)
        return scores


# --- helpers ---


def _stringify(value: object) -> str:
    """Stable string representation for cache keys and prompt rendering.

    Strings are returned as-is (no JSON-quoting noise). Other types use
    json.dumps with sorted keys for stable ordering.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _render_pairs(items: list[JudgeItem]) -> str:
    """Format items as a numbered list.

    Pure data rendering -- no header like "Pairs to judge:". Any framing
    text belongs in the prompt template, not here.
    """
    lines: list[str] = []
    for i, item in enumerate(items, start=1):
        lines.append(
            f"{i}. field={item.path!r} "
            f"gold={_stringify(item.gold)!r} "
            f"extracted={_stringify(item.extracted)!r}"
        )
    return "\n".join(lines)


def _parse_judge_response(content: str, expected_count: int) -> list[float | None]:
    """Parse the JSON response from the judge into a positional list of 0/1 scores.

    Returns a list with one entry per parsed value (in order). Each entry is:

    - ``1.0`` if the value is unambiguously 1 / 1.0 / True / "1" / "true"
    - ``0.0`` if the value is unambiguously 0 / 0.0 / False / "0" / "false"
    - ``None`` if the value is anything else (0.5, 2, "maybe", null, ...)

    The contract with the LLM is "binary 0 or 1." Anything else is a contract
    violation -- coercing it silently would hide bad LLM behavior. We return
    None for those positions and the caller (SemanticBatchComparator ->
    process_batches) marks the corresponding field as ``batch_error``.

    If the whole response is unparseable (not valid JSON, no ``results`` key,
    or ``results`` isn't a list), returns an empty list -- the caller marks
    all items as ``batch_error``. ``expected_count`` is used only for logging.
    """
    content = content.strip()
    # Strip code fences if present.
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error(
            "Judge returned non-JSON response (expected %d results): %s; content=%r",
            expected_count,
            exc,
            content[:200],
        )
        return []

    results = parsed.get("results") if isinstance(parsed, dict) else None
    if not isinstance(results, list):
        logger.error(
            "Judge response missing 'results' list (expected %d): %r",
            expected_count,
            parsed,
        )
        return []

    scores: list[float | None] = []
    for index, value in enumerate(results):
        parsed_score = _coerce_binary_score(value)
        if parsed_score is None:
            logger.warning(
                "Judge returned invalid score at index %d: %r (expected 0 or 1). "
                "Marking this item as batch_error.",
                index, value,
            )
        scores.append(parsed_score)
    return scores


def _coerce_binary_score(value: object) -> float | None:
    """Strict coercion to 0.0 or 1.0. Anything ambiguous returns None.

    Accepts:
    - bool: True -> 1.0, False -> 0.0
    - int / float: exactly 0 or 1 -> 0.0/1.0; anything else -> None
    - str: "0"/"1"/"true"/"false" (case-insensitive, whitespace tolerated) -> 0.0/1.0;
      anything else -> None

    Note: ``isinstance(value, bool)`` is checked BEFORE ``isinstance(value, int)``
    because bool is a subclass of int in Python.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        if value == 0:
            return 0.0
        if value == 1:
            return 1.0
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("0", "false"):
            return 0.0
        if normalized in ("1", "true"):
            return 1.0
        return None
    return None
