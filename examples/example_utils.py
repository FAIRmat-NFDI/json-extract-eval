"""Printing helpers shared by the example notebooks.

Keeps result formatting out of the notebooks, so each cell shows only the
evaluator call it is demonstrating. Notebooks run with ``examples/`` as the
working directory, so they import this with ``from example_utils import
show_run``. If you copy a notebook elsewhere, copy this file next to it or
replace ``show_run`` with your own loop over ``run.records``.
"""

from collections.abc import Iterable

from struct_extract_eval.core.record import RunResult

_MAX_VALUE_WIDTH = 40


def show_run(
    run: RunResult,
    title: str | None = None,
    *,
    values: bool = True,
    reason: bool = True,
    fields: Iterable[str] | None = None,
) -> None:
    """Print a RunResult: a summary line, then one field table per record.

    Args:
        run: The result of ``evaluate(...)``.
        title: Optional heading printed first.
        values: Include the gold and extracted columns. Values are shown with
            ``repr`` so ``None``, ``""`` and ``"None"`` stay distinguishable.
        reason: Append the comparator's reason when it gave one.
        fields: Only show these field paths. Useful when a notebook is about
            one or two fields of a larger record.
    """
    wanted = set(fields) if fields is not None else None
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for record in run.records:
        for fr in record.field_results:
            if wanted is not None and fr.path not in wanted:
                continue
            rows.append((
                str(record.record_id),
                fr.path,
                _clip(repr(fr.gold_value)),
                _clip(repr(fr.extracted_value)),
                f"{fr.score:.1f}",
                fr.status,
                fr.reason or "",
            ))

    headers = ("record", "path", "gold", "extracted", "score", "status", "reason")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    shown = [0, 1] + ([2, 3] if values else []) + [4, 5] + ([6] if reason else [])

    if title:
        print(title)
    print(
        f"  mean P={run.mean_precision:.2f}  R={run.mean_recall:.2f}  "
        f"F1={run.mean_f1:.2f}   ({len(run.records)} record(s))"
    )
    print("  " + "  ".join(headers[i].ljust(widths[i]) for i in shown).rstrip())
    for row in rows:
        print("  " + "  ".join(row[i].ljust(widths[i]) for i in shown).rstrip())


def _clip(text: str) -> str:
    if len(text) <= _MAX_VALUE_WIDTH:
        return text
    return text[: _MAX_VALUE_WIDTH - 3] + "..."


def show_per_field(run: RunResult, title: str | None = None) -> None:
    """Print the per-field aggregate: mean score and status counts per path."""
    headers = ("path", "score", "match", "mismatch", "omission", "hallucination")
    rows = [
        (
            path,
            f"{agg.mean_score:.2f}",
            str(agg.matches),
            str(agg.mismatches),
            str(agg.omissions),
            str(agg.hallucinations),
        )
        for path, agg in sorted(run.per_field.items())
    ]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: tuple[str, ...]) -> str:
        # first column left-aligned, numeric columns right-aligned
        first = cells[0].ljust(widths[0])
        rest = [cells[i].rjust(widths[i]) for i in range(1, len(cells))]
        return "  " + "  ".join([first, *rest])

    if title:
        print(title)
    print(line(headers))
    for row in rows:
        print(line(row))
