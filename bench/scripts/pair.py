"""Bench-facing re-export of the pair format + loader.

The implementation lives in :mod:`mirrorml.cli._pair` so the in-package
CLI subcommands (``trace``, ``verify``) can use it without bench needing
to be on ``sys.path``. This module preserves the historical import path
(``bench.scripts.pair``) for the bench harness and external callers.

See :mod:`mirrorml.cli._pair` for the full ``meta.yaml`` schema.
"""

from __future__ import annotations

from mirrorml.cli._pair import (
    ExpectedDivergence,
    Pair,
    discover_pairs,
    load_pair,
)

__all__ = ["ExpectedDivergence", "Pair", "discover_pairs", "load_pair"]
