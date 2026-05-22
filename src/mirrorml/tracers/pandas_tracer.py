"""pandas tracer — produces fingerprints from pandas feature pipelines.

**Not implemented in v0.0.1.** The full implementation lands in M2 along
with the pandas tracing harness (libcst-based source capture plus wrapper-
object runtime tracing).

This module must not import pandas eagerly — see CLAUDE.md's < 200ms
import-time budget. The function below has no pandas reference at all
because it does not need one to raise :class:`NotImplementedError`; the M2
implementation will import pandas inside the function body.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from mirrorml.fingerprint.schema import ColumnSpec, Fingerprint

__all__ = ["trace_pandas"]


# EXPERIMENTAL: the signature will be finalized in M2 when the tracing
# harness lands. Only the function name and return type are stable as of
# v0.0.1.
def trace_pandas(
    pipeline: Callable[..., object],
    /,
    *,
    input_schema: Iterable[ColumnSpec] | None = None,
) -> Fingerprint:
    """Trace a pandas feature pipeline; return its canonical fingerprint.

    Not implemented in v0.0.1. The full implementation lands in M2.

    Raises:
        NotImplementedError: Always.
    """

    raise NotImplementedError(
        "trace_pandas: not yet implemented in v0.0.1 (lands in M2). "
        "Track progress in the project's issue tracker."
    )
