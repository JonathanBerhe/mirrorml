"""Map sqlglot ``DataType`` nodes to canonical dtype strings.

The mapping is intentionally narrow in M2 phase 1. Tracers that encounter
a SQL type with no entry here raise :class:`UnsupportedOperationError` so
the divergence does not silently appear as ``schema_drift`` later.

This module is internal; the public entry point is
:func:`mirrorml.tracers.trace_sql`. sqlglot is imported at module top
because callers reach this module only after they have already loaded
sqlglot, so there is no extra import-time cost vs the lazy boundary at
:mod:`mirrorml.tracers.sql_tracer`.
"""

from __future__ import annotations

from typing import Any, Final

import sqlglot.expressions as exp

from mirrorml.exceptions import UnsupportedOperationError

__all__ = ["DEFAULT_TIMESTAMP_UNIT", "map_sql_dtype"]

# sqlglot's nested ``DataType.Type`` enum is not usable as a static type
# under mypy strict, so the key type widens to ``Any`` here. Values are
# still validated against the enum at runtime by the dict lookups below.
_TypeKey = Any

DEFAULT_TIMESTAMP_UNIT: Final[str] = "us"
"""SQL timestamps without an explicit precision default to microsecond
resolution in the canonical vocabulary. This matches PostgreSQL,
Snowflake, BigQuery, and DuckDB; pandas defaults to ``ns`` and Polars to
``us``. M2 phase 1 keeps the SQL side on ``us`` and lets the diff
classifier surface ``rounding_precision`` if a pandas pipeline picked a
finer resolution."""

_SCALAR_MAP: Final[dict[_TypeKey, str]] = {
    exp.DataType.Type.NULL: "null",
    exp.DataType.Type.BOOLEAN: "bool",
    exp.DataType.Type.TINYINT: "int8",
    exp.DataType.Type.SMALLINT: "int16",
    exp.DataType.Type.INT: "int32",
    exp.DataType.Type.BIGINT: "int64",
    exp.DataType.Type.UTINYINT: "uint8",
    exp.DataType.Type.USMALLINT: "uint16",
    exp.DataType.Type.UINT: "uint32",
    exp.DataType.Type.UBIGINT: "uint64",
    exp.DataType.Type.FLOAT: "float32",
    exp.DataType.Type.DOUBLE: "float64",
    exp.DataType.Type.VARCHAR: "utf8",
    exp.DataType.Type.CHAR: "utf8",
    exp.DataType.Type.NCHAR: "utf8",
    exp.DataType.Type.NVARCHAR: "utf8",
    exp.DataType.Type.TEXT: "utf8",
    exp.DataType.Type.LONGTEXT: "utf8",
    exp.DataType.Type.MEDIUMTEXT: "utf8",
    exp.DataType.Type.BINARY: "binary",
    exp.DataType.Type.VARBINARY: "binary",
    exp.DataType.Type.BLOB: "binary",
    exp.DataType.Type.LONGBLOB: "binary",
    exp.DataType.Type.MEDIUMBLOB: "binary",
    exp.DataType.Type.DATE: "date",
}


def map_sql_dtype(dt: exp.DataType) -> str:
    """Return the canonical dtype string for a sqlglot ``DataType``.

    Raises :class:`UnsupportedOperationError` for type kinds not yet
    mapped (struct, map, interval-with-fields, dialect-specific scalars).
    The message names the offending kind so the user can either rewrite
    the SQL or file a tracking issue.
    """

    kind = dt.this

    scalar = _SCALAR_MAP.get(kind)
    if scalar is not None:
        return scalar

    if kind == exp.DataType.Type.TIME:
        return f"time[{DEFAULT_TIMESTAMP_UNIT}]"
    if kind == exp.DataType.Type.TIMETZ:
        return f"time[{DEFAULT_TIMESTAMP_UNIT}]"

    if kind in (
        exp.DataType.Type.TIMESTAMP,
        exp.DataType.Type.TIMESTAMPNTZ,
        exp.DataType.Type.DATETIME,
    ):
        return f"timestamp[{DEFAULT_TIMESTAMP_UNIT}]"
    if kind in (exp.DataType.Type.TIMESTAMPTZ, exp.DataType.Type.TIMESTAMPLTZ):
        return f"timestamp[{DEFAULT_TIMESTAMP_UNIT}, UTC]"

    if kind == exp.DataType.Type.INTERVAL:
        return f"duration[{DEFAULT_TIMESTAMP_UNIT}]"

    # NUMERIC is normalized to DECIMAL by sqlglot, so only DECIMAL is checked here.
    if kind == exp.DataType.Type.DECIMAL:
        precision, scale = _decimal_params(dt)
        return f"decimal[{precision}, {scale}]"

    if kind == exp.DataType.Type.ARRAY:
        if len(dt.expressions) != 1:
            raise UnsupportedOperationError(
                f"SQL ARRAY must have exactly one element type; "
                f"got {len(dt.expressions)} in {dt.sql()!r}"
            )
        inner = dt.expressions[0]
        if not isinstance(inner, exp.DataType):
            raise UnsupportedOperationError(
                f"SQL ARRAY element type must be a DataType, "
                f"got {type(inner).__name__} in {dt.sql()!r}"
            )
        return f"list[{map_sql_dtype(inner)}]"

    raise UnsupportedOperationError(
        f"SQL type {dt.sql()!r} (kind {kind.name!r}) is not yet supported "
        f"by the SQL tracer. File a tracking issue if this is needed."
    )


def _decimal_params(dt: exp.DataType) -> tuple[int, int]:
    """Extract (precision, scale) from a NUMERIC/DECIMAL ``DataType``.

    Defaults follow ANSI SQL: precision 38 (the max widely supported),
    scale 0. Specific dialect defaults can drift; the M2 work surfaces
    them by treating any divergence as ``rounding_precision`` rather
    than silently coercing.
    """

    params = dt.expressions
    precision = 38
    scale = 0

    if params:
        try:
            precision = int(params[0].name)
        except (AttributeError, ValueError) as e:
            raise UnsupportedOperationError(
                f"SQL DECIMAL precision could not be parsed from {dt.sql()!r}: {e}"
            ) from e

    if len(params) >= 2:
        try:
            scale = int(params[1].name)
        except (AttributeError, ValueError) as e:
            raise UnsupportedOperationError(
                f"SQL DECIMAL scale could not be parsed from {dt.sql()!r}: {e}"
            ) from e

    if precision <= 0 or scale < 0 or scale > precision:
        raise UnsupportedOperationError(
            f"SQL DECIMAL has invalid precision/scale in {dt.sql()!r}: "
            f"precision={precision}, scale={scale}"
        )

    return precision, scale
