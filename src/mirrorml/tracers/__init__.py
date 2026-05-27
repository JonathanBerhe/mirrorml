"""Per-framework tracers — turn feature pipelines into fingerprints.

The three public tracers are :func:`trace_pandas`, :func:`trace_polars`, and
:func:`trace_sql`. All three are stubs in v0.0.1 and land in M2 along with
the tracing harnesses they need (libcst-based source capture for pandas /
Polars; sqlglot for SQL).

Tracers must remain lazy with respect to their target frameworks: importing
this package must not import pandas or polars eagerly (the < 200ms
import-time budget depends on it).
"""

from __future__ import annotations

from mirrorml.tracers.pandas_tracer import trace_pandas
from mirrorml.tracers.polars_tracer import trace_polars
from mirrorml.tracers.sql_tracer import trace_sql

__all__ = ["trace_pandas", "trace_polars", "trace_sql"]
