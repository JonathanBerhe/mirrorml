"""MirrorML command-line interface.

In v0.0.1 the CLI exposes only ``--version`` and ``--help``. Subcommands
(``trace``, ``diff``, ``verify``) land in M5 once the tracers and diff
engine are implemented.
"""

from __future__ import annotations

from mirrorml.cli.app import app

__all__ = ["app"]
