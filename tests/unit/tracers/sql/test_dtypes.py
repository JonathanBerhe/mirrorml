"""SQL DataType to canonical dtype mapping."""

from __future__ import annotations

import pytest
import sqlglot
import sqlglot.expressions as exp

from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.tracers._sql_dtypes import DEFAULT_TIMESTAMP_UNIT, map_sql_dtype


def _parse_dtype(sql_type: str) -> exp.DataType:
    """Parse a SQL type via a dummy CAST and return the resulting DataType node."""

    ast = sqlglot.parse_one(f"SELECT CAST(x AS {sql_type}) FROM t")
    cast = ast.find(exp.Cast)
    assert cast is not None, f"no CAST found for {sql_type!r}"
    return cast.to


@pytest.mark.parametrize(
    "sql_type,canonical",
    [
        ("TINYINT", "int8"),
        ("SMALLINT", "int16"),
        ("INT", "int32"),
        ("INTEGER", "int32"),
        ("BIGINT", "int64"),
        ("FLOAT", "float32"),
        ("DOUBLE", "float64"),
        ("BOOLEAN", "bool"),
        ("VARCHAR", "utf8"),
        ("VARCHAR(100)", "utf8"),
        ("CHAR(10)", "utf8"),
        ("TEXT", "utf8"),
        ("BINARY", "binary"),
        ("VARBINARY", "binary"),
        ("DATE", "date"),
    ],
)
def test_scalar_mappings(sql_type: str, canonical: str) -> None:
    assert map_sql_dtype(_parse_dtype(sql_type)) == canonical


def test_timestamp_naive_uses_default_unit() -> None:
    assert map_sql_dtype(_parse_dtype("TIMESTAMP")) == f"timestamp[{DEFAULT_TIMESTAMP_UNIT}]"


def test_timestamp_with_time_zone_carries_utc_default() -> None:
    assert map_sql_dtype(_parse_dtype("TIMESTAMP WITH TIME ZONE")) == (
        f"timestamp[{DEFAULT_TIMESTAMP_UNIT}, UTC]"
    )


def test_datetime_is_naive_timestamp() -> None:
    assert map_sql_dtype(_parse_dtype("DATETIME")) == f"timestamp[{DEFAULT_TIMESTAMP_UNIT}]"


def test_interval_is_duration() -> None:
    assert map_sql_dtype(_parse_dtype("INTERVAL")) == f"duration[{DEFAULT_TIMESTAMP_UNIT}]"


def test_decimal_with_explicit_precision_and_scale() -> None:
    assert map_sql_dtype(_parse_dtype("DECIMAL(18, 2)")) == "decimal[18, 2]"
    assert map_sql_dtype(_parse_dtype("NUMERIC(38, 10)")) == "decimal[38, 10]"


def test_decimal_with_only_precision_defaults_scale_to_zero() -> None:
    assert map_sql_dtype(_parse_dtype("DECIMAL(10)")) == "decimal[10, 0]"


def test_decimal_with_no_params_uses_ansi_default() -> None:
    assert map_sql_dtype(_parse_dtype("DECIMAL")) == "decimal[38, 0]"


def test_array_recurses_into_element_type() -> None:
    assert map_sql_dtype(_parse_dtype("ARRAY<BIGINT>")) == "list[int64]"
    assert map_sql_dtype(_parse_dtype("ARRAY<VARCHAR>")) == "list[utf8]"


def test_nested_array() -> None:
    assert map_sql_dtype(_parse_dtype("ARRAY<ARRAY<INT>>")) == "list[list[int32]]"


def test_unsupported_type_raises_actionable_error() -> None:
    """sqlglot may parse types we have not mapped yet. The walker must
    surface them as UnsupportedOperationError with the offending kind in
    the message, not silently fall through."""

    # STRUCT / OBJECT are not in the M2 phase 1 surface.
    with pytest.raises(UnsupportedOperationError, match="not yet supported"):
        map_sql_dtype(_parse_dtype("STRUCT<a INT, b VARCHAR>"))
