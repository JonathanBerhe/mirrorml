"""SQL tracer entry point.

sqlglot is imported lazily inside :func:`trace_sql` so that
``import mirrorml`` stays under the 200ms cold-start budget for users who
never touch a SQL pipeline.

The current surface accepts a single-table ``SELECT`` with optional
``WHERE``, a projection of bare column references (with optional ``AS``
aliasing), and optional ``ORDER BY``. JOINs, GROUP BY, HAVING, LIMIT,
DISTINCT, UNION, subqueries, CTEs, and any expression in the projection
or ORDER BY land in later M2 phases; everything outside the current
surface raises :class:`~mirrorml.exceptions.UnsupportedOperationError`
with an actionable message. See :mod:`mirrorml.tracers._sql_walker` for
the implementation and ``docs/concepts/dtype_vocabulary.md`` for the
canonical-dtype mapping that SQL types are normalized into.
"""

from __future__ import annotations

from collections.abc import Mapping

from mirrorml.fingerprint.schema import ColumnSpec, Fingerprint

__all__ = ["trace_sql"]


def trace_sql(
    query: str,
    /,
    *,
    schemas: Mapping[str, tuple[ColumnSpec, ...]] | None = None,
    dialect: str | None = None,
) -> Fingerprint:
    """Trace a SQL feature pipeline; return its canonical fingerprint.

    Args:
        query: A SQL string. Parsed via sqlglot.
        schemas: Mapping from table name to the table's column list, e.g.
            ``{"events": (("uid", "int64"), ("ts", "timestamp[ns, UTC]"))}``.
            Required for every table referenced in ``FROM``. Dtypes must
            be in the canonical vocabulary (see
            ``docs/concepts/dtype_vocabulary.md``).
        dialect: sqlglot dialect name (``"snowflake"``, ``"bigquery"``,
            ``"postgres"``, ...) or ``None`` for auto-detection. The
            dialect affects parsing only; the resulting fingerprint is
            dialect-independent.

    Returns:
        A canonical :class:`~mirrorml.fingerprint.schema.Fingerprint`.

    Raises:
        UnsupportedOperationError: For SQL constructs outside the M2
            phase 1 surface, or when ``schemas`` does not cover a
            table referenced in ``FROM``.

    Examples:
        >>> fp = trace_sql(
        ...     "SELECT uid, score FROM events WHERE score > 0",
        ...     schemas={"events": (("uid", "int64"), ("score", "float64"))},
        ... )
        >>> fp.framework
        'sql'
        >>> len(fp.operations)
        3
        >>> fp.output_schema
        (('uid', 'int64'), ('score', 'float64'))
    """

    from mirrorml.tracers._sql_walker import trace_sql_impl

    return trace_sql_impl(query, schemas=schemas, dialect=dialect)
