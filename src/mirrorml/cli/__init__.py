"""MirrorML command-line interface.

The CLI exposes the ``trace``, ``diff``, and ``verify`` subcommands (plus
``--version`` and ``--help``).
"""

from __future__ import annotations

from mirrorml.cli.app import app

__all__ = ["app"]
