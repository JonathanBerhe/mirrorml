"""SQL tracer — produces fingerprints from SQL feature pipelines.

**Not implemented in v0.0.1.** The full implementation lands in M2 atop
sqlglot's dialect-aware AST.

Unlike the dataframe tracers, the SQL tracer does not need a runtime
tracing harness — sqlglot parses the SQL text statically and produces an
AST that maps cleanly onto our :class:`~mirrorml.fingerprint.schema.Operation`
union.
"""

from __future__ import annotations

from mirrorml.fingerprint.schema import Fingerprint

__all__ = ["trace_sql"]


# EXPERIMENTAL: signature will be finalized in M2.
def trace_sql(query: str, /, *, dialect: str | None = None) -> Fingerprint:
    """Trace a SQL feature pipeline; return its canonical fingerprint.

    Not implemented in v0.0.1. The full implementation lands in M2.

    Args:
        query: A SQL string. Will be parsed via sqlglot in the M2
            implementation.
        dialect: Optional sqlglot dialect name (``"snowflake"``,
            ``"bigquery"``, etc.); ``None`` means dialect auto-detection.

    Raises:
        NotImplementedError: Always.
    """

    raise NotImplementedError(
        "trace_sql: not yet implemented in v0.0.1 (lands in M2). "
        "Track progress in the project's issue tracker."
    )
