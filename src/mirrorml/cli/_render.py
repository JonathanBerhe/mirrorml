"""Rich rendering for the CLI subcommands. INTERNAL.

Kept separate from the typer app so the command bodies in ``app.py`` stay
focused on argument parsing and exit codes. The renderers write to a
caller-provided :class:`rich.console.Console`, which the tests use to
capture output without binding it to ``sys.stdout``.
"""

from __future__ import annotations

from collections.abc import Iterable

from rich.console import Console
from rich.table import Table
from rich.text import Text

from mirrorml.diff import Divergence


def render_divergences(
    console: Console,
    divergences: Iterable[Divergence],
    *,
    left_label: str,
    right_label: str,
) -> None:
    """Render a sequence of divergences as a rich table.

    Empty input prints a single equivalence line so the user sees a
    positive confirmation rather than blank output.
    """

    divergences = tuple(divergences)
    if not divergences:
        console.print(
            Text.assemble(
                ("equivalent", "bold green"),
                ("  ", ""),
                (left_label, "cyan"),
                ("  ==  ", "dim"),
                (right_label, "cyan"),
            )
        )
        return

    header = Text.assemble(
        (f"{len(divergences)} divergence", "bold red"),
        ("s" if len(divergences) != 1 else "", "bold red"),
        ("  ", ""),
        (left_label, "cyan"),
        ("  vs  ", "dim"),
        (right_label, "cyan"),
    )
    console.print(header)

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("category", style="yellow", no_wrap=True)
    table.add_column("left op", style="dim", no_wrap=True)
    table.add_column("right op", style="dim", no_wrap=True)
    table.add_column("detail", overflow="fold")
    for d in divergences:
        table.add_row(
            d.category,
            _short_op_id(d.left_op_id),
            _short_op_id(d.right_op_id),
            d.detail,
        )
    console.print(table)


def _short_op_id(op_id: str | None) -> str:
    """Truncate a 64-char op_id to its first 8 chars for readable display.

    The full id is still recoverable via grep in the fingerprint JSON;
    8 chars is enough to disambiguate the ops within a single pipeline.
    """

    if op_id is None:
        return "-"
    return op_id[:8] if len(op_id) > 8 else op_id


def render_verify_result(
    console: Console,
    *,
    pair_name: str,
    bucket: str,
    category: str,
    expected: Iterable[str],
    predicted: Iterable[Divergence],
) -> bool:
    """Render the outcome of ``mirrorml verify`` and return whether it passed.

    A pair passes iff the set of predicted divergence categories equals
    the set of expected ones. Extras (false positives) and misses (false
    negatives) are both surfaced explicitly so the user can see which
    direction the disagreement runs.
    """

    expected_set = set(expected)
    predicted_seq = tuple(predicted)
    predicted_set = {d.category for d in predicted_seq}
    extras = sorted(predicted_set - expected_set)
    missing = sorted(expected_set - predicted_set)
    passed = not extras and not missing

    status = Text("PASS", style="bold green") if passed else Text("FAIL", style="bold red")
    console.print(
        Text.assemble(
            status,
            ("  ", ""),
            (pair_name, "bold"),
            ("  ", ""),
            (f"({bucket} / {category})", "dim"),
        )
    )
    console.print(
        Text.assemble(
            ("  expected: ", "dim"),
            (", ".join(sorted(expected_set)) or "(none)", "yellow" if expected_set else "dim"),
        )
    )
    console.print(
        Text.assemble(
            ("  found:    ", "dim"),
            (", ".join(sorted(predicted_set)) or "(none)", "yellow" if predicted_set else "dim"),
        )
    )
    if extras:
        console.print(
            Text.assemble(
                ("  extra:    ", "dim"),
                (", ".join(extras), "red"),
            )
        )
    if missing:
        console.print(
            Text.assemble(
                ("  missing:  ", "dim"),
                (", ".join(missing), "red"),
            )
        )

    if predicted_seq:
        render_divergences(
            console,
            predicted_seq,
            left_label="offline",
            right_label="online",
        )

    return passed
