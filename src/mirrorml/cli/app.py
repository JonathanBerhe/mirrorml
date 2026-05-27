"""Typer entry point. The program installed as ``mirrorml`` on PATH.

Three subcommands:

* :func:`trace` reads one side of a MirrorBench-style pair directory and
  emits the canonical fingerprint JSON to stdout (or ``--output``).
* :func:`diff` reads two fingerprint JSONs from disk, runs the diff
  engine, and renders the divergences. Exit 0 iff the fingerprints are
  equivalent.
* :func:`verify` traces both sides of a pair directory, diffs them, and
  compares the result against ``expected_divergences`` in
  ``meta.yaml``. Exit 0 iff the predicted and expected category sets
  match. This is the CI primitive.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer
from rich.console import Console

from mirrorml import __version__, diff
from mirrorml.cli._pair import discover_pairs, load_pair
from mirrorml.cli._render import (
    render_divergences,
    render_verify_compact,
    render_verify_result,
    render_verify_summary,
)
from mirrorml.fingerprint.schema import Fingerprint


class Side(str, Enum):
    """Which side of a pair to trace."""

    offline = "offline"
    online = "online"


app = typer.Typer(
    name="mirrorml",
    help="Static detection of training-serving skew in ML feature pipelines.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the MirrorML version and exit.",
        is_eager=True,
    ),
) -> None:
    """MirrorML. Static detection of training-serving skew."""

    if version:
        typer.echo(f"mirrorml {__version__}")
        raise typer.Exit()
    # ``no_args_is_help=True`` on the Typer app handles the bare
    # ``mirrorml`` invocation (prints help, exits with the standard
    # usage-error code).


@app.command()
def trace(
    pair_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Path to a pair directory containing a meta.yaml.",
    ),
    side: Side = typer.Option(
        ...,
        "--side",
        case_sensitive=False,
        help="Which side of the pair to trace.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write fingerprint JSON to this path instead of stdout.",
    ),
    indent: int = typer.Option(
        2,
        "--indent",
        min=0,
        help="JSON indent level. Pass 0 for compact output.",
    ),
) -> None:
    """Trace one side of a pair and emit its canonical fingerprint JSON."""

    try:
        pair = load_pair(pair_dir)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None

    fingerprint = pair.offline if side is Side.offline else pair.online
    text = fingerprint.model_dump_json(indent=indent if indent > 0 else None)
    if output is not None:
        output.write_text(text + "\n")
    else:
        typer.echo(text)


def diff_command(
    left: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Fingerprint JSON file (offline side).",
    ),
    right: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Fingerprint JSON file (online side).",
    ),
) -> None:
    """Diff two on-disk fingerprints and render divergences.

    Exits 0 if the fingerprints are equivalent (no divergences), 1
    otherwise. Designed to compose in shell pipelines:
    ``mirrorml diff a.json b.json && echo equivalent``.
    """

    try:
        left_fp = Fingerprint.model_validate_json(left.read_text())
        right_fp = Fingerprint.model_validate_json(right.read_text())
    except ValueError as exc:
        typer.echo(f"error: failed to load fingerprint: {exc}", err=True)
        raise typer.Exit(2) from None

    divergences = diff(left_fp, right_fp)
    console = Console()
    render_divergences(
        console,
        divergences,
        left_label=str(left),
        right_label=str(right),
    )
    raise typer.Exit(1 if divergences else 0)


# The CLI verb is ``diff`` but the function is named ``diff_command`` to
# avoid shadowing the imported ``diff`` symbol inside this module; the
# explicit ``name="diff"`` keeps the user-facing surface unchanged.
app.command(name="diff")(diff_command)


@app.command()
def verify(
    path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="A pair directory (with meta.yaml), or a directory of pairs to verify recursively.",
    ),
) -> None:
    """Trace both sides of each pair, diff, and check against expected_divergences.

    If PATH is a single pair (it contains a ``meta.yaml``) only that pair is
    checked, with full divergence detail. Otherwise every pair found beneath
    PATH is checked with a one-line result each plus a summary. Exits 0 iff
    every pair's predicted divergence categories match its expected set
    exactly; extras and misses both fail.
    """

    console = Console()

    if (path / "meta.yaml").is_file():
        pair_dirs = [path]
    else:
        pair_dirs = discover_pairs(path)
        if not pair_dirs:
            typer.echo(f"error: no pairs (meta.yaml) found under {path}", err=True)
            raise typer.Exit(2) from None

    single = len(pair_dirs) == 1
    passed_count = 0
    for pair_dir in pair_dirs:
        try:
            pair = load_pair(pair_dir)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from None

        divergences = diff(pair.offline, pair.online)
        render = render_verify_result if single else render_verify_compact
        if render(
            console,
            pair_name=pair.name,
            bucket=pair.bucket,
            category=pair.category,
            expected=(e.category for e in pair.expected),
            predicted=divergences,
        ):
            passed_count += 1

    if not single:
        render_verify_summary(console, total=len(pair_dirs), passed=passed_count)
    raise typer.Exit(0 if passed_count == len(pair_dirs) else 1)


if __name__ == "__main__":  # pragma: no cover
    app()
