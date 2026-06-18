"""CLI smoke test: the entry point exists and ``--version`` prints the
package version."""

from __future__ import annotations

from typer.testing import CliRunner

import mirrorml
from mirrorml.cli.app import app


def test_version_flag_prints_package_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert mirrorml.__version__ in result.stdout


def test_no_args_shows_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, [])
    # With no_args_is_help=True / fallback, typer prints help on bare invocation.
    assert "MirrorML" in result.stdout or "Usage" in result.stdout
