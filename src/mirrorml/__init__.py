"""MirrorML — static detection of training-serving skew in ML feature pipelines.

The seven names below are the entire stable public surface as of v0.1.0.
Anything not in :data:`__all__` is internal and may change without notice.

- :class:`Fingerprint` — canonical representation of a pipeline.
- :func:`fingerprint` — construct a :class:`Fingerprint`.
- :class:`Divergence` — a classified disagreement between two fingerprints.
- :func:`diff` — compute divergences between two fingerprints.
- :func:`trace_pandas`, :func:`trace_polars`, :func:`trace_sql` — per-
  framework tracers.

All seven names are implemented: the tracers lower pandas, Polars, and SQL
pipelines into fingerprints, :func:`diff` classifies and localizes their
divergences, and the CLI exposes ``trace`` / ``diff`` / ``verify``.

Note: ``mirrorml.fingerprint`` is both the public constructor function and
the implementation subpackage. Attribute access (``mirrorml.fingerprint``)
gives the function; ``import mirrorml.fingerprint.schema`` still works for
internal access.
"""

from __future__ import annotations

from mirrorml.diff import Divergence, diff
from mirrorml.fingerprint import Fingerprint
from mirrorml.fingerprint import build_fingerprint as fingerprint
from mirrorml.tracers import trace_pandas, trace_polars, trace_sql

__all__ = [
    "Divergence",
    "Fingerprint",
    "diff",
    "fingerprint",
    "trace_pandas",
    "trace_polars",
    "trace_sql",
]
__version__ = "0.1.0"
