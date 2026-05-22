"""Polars tracer — produces fingerprints from Polars feature pipelines.

**Not implemented in v0.0.1.** The full implementation lands in M2.

Polars's lazy-frame API gives us a cleaner tracing surface than pandas: the
``LazyFrame`` execution plan is itself close to a fingerprint, so the M2
work is mostly a translator from the polars plan to MirrorML's
:class:`~mirrorml.fingerprint.schema.Operation` union.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from mirrorml.fingerprint.schema import ColumnSpec, Fingerprint

__all__ = ["trace_polars"]


# EXPERIMENTAL: signature will be finalized in M2.
def trace_polars(
    pipeline: Callable[..., object],
    /,
    *,
    input_schema: Iterable[ColumnSpec] | None = None,
) -> Fingerprint:
    """Trace a Polars feature pipeline; return its canonical fingerprint.

    Not implemented in v0.0.1. The full implementation lands in M2.

    Raises:
        NotImplementedError: Always.
    """

    raise NotImplementedError(
        "trace_polars: not yet implemented in v0.0.1 (lands in M2). "
        "Track progress in the project's issue tracker."
    )
