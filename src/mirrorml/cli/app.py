"""Typer entry point — the program installed as ``mirrorml`` on PATH.

The v0.0.1 surface is just ``--version`` and ``--help`` so the entry point
is wirable into editor integrations and CI checks before any subcommands
exist. Subcommands (``trace``, ``diff``, ``verify``) land in M5.
"""

from __future__ import annotations

import typer

from mirrorml import __version__

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
    """MirrorML — static detection of training-serving skew."""

    if version:
        typer.echo(f"mirrorml {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


if __name__ == "__main__":  # pragma: no cover
    app()
