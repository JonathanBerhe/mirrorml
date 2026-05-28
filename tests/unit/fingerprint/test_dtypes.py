"""The canonical dtype vocabulary is a public contract — the grammar and
the schema-level validation hook are pinned by these tests."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from mirrorml.fingerprint import build_fingerprint
from mirrorml.fingerprint.dtypes import (
    BINARY,
    BOOL,
    DATE,
    FLOAT32,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    INT64,
    NULL,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    UTF8,
    bit_width,
    element_dtype,
    is_float,
    is_integer,
    is_numeric,
    is_temporal,
    is_timezone_aware,
    parse_dtype,
    timezone_of,
    unit_of,
    validate_dtype,
)
from mirrorml.fingerprint.operations import Source

# --- parser: positive cases --------------------------------------------------


@pytest.mark.parametrize(
    "dtype,kind,bits",
    [
        ("int8", "int", 8),
        ("int16", "int", 16),
        ("int32", "int", 32),
        ("int64", "int", 64),
        ("uint8", "uint", 8),
        ("uint64", "uint", 64),
        ("float16", "float", 16),
        ("float32", "float", 32),
        ("float64", "float", 64),
    ],
)
def test_parse_numeric(dtype: str, kind: str, bits: int) -> None:
    parsed = parse_dtype(dtype)
    assert parsed.kind == kind
    assert parsed.bits == bits


@pytest.mark.parametrize("dtype", ["null", "bool", "utf8", "binary", "date"])
def test_parse_simple_scalars(dtype: str) -> None:
    parsed = parse_dtype(dtype)
    assert parsed.kind == dtype


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_parse_time(unit: str) -> None:
    parsed = parse_dtype(f"time[{unit}]")
    assert parsed.kind == "time"
    assert parsed.unit == unit


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_parse_timestamp_naive(unit: str) -> None:
    parsed = parse_dtype(f"timestamp[{unit}]")
    assert parsed.kind == "timestamp"
    assert parsed.unit == unit
    assert parsed.timezone is None


@pytest.mark.parametrize(
    "tz",
    ["UTC", "US/Pacific", "America/Argentina/Buenos_Aires", "Etc/GMT+12", "+05:30"],
)
def test_parse_timestamp_with_timezone(tz: str) -> None:
    parsed = parse_dtype(f"timestamp[ns, {tz}]")
    assert parsed.kind == "timestamp"
    assert parsed.timezone == tz


def test_parse_duration() -> None:
    assert parse_dtype("duration[s]").unit == "s"
    assert parse_dtype("duration[ns]").unit == "ns"


def test_parse_list() -> None:
    parsed = parse_dtype("list[int64]")
    assert parsed.kind == "list"
    assert parsed.element is not None
    assert parsed.element.kind == "int"
    assert parsed.element.bits == 64


def test_parse_nested_list() -> None:
    parsed = parse_dtype("list[list[float32]]")
    assert parsed.kind == "list"
    assert parsed.element is not None
    assert parsed.element.kind == "list"
    assert parsed.element.element is not None
    assert parsed.element.element.kind == "float"


def test_parse_list_of_timestamp() -> None:
    parsed = parse_dtype("list[timestamp[ns, UTC]]")
    assert parsed.kind == "list"
    assert parsed.element is not None
    assert parsed.element.kind == "timestamp"
    assert parsed.element.timezone == "UTC"


def test_parse_decimal() -> None:
    parsed = parse_dtype("decimal[18, 2]")
    assert parsed.kind == "decimal"
    assert parsed.precision == 18
    assert parsed.scale == 2


# --- parser: measurement-unit annotation ------------------------------------


@pytest.mark.parametrize(
    "dtype,unit",
    [
        ("float64{meters}", "meters"),
        ("float32{USD}", "USD"),
        ("int64{seconds}", "seconds"),
        ("uint32{bytes}", "bytes"),
        ("decimal[18, 2]{USD}", "USD"),
        ("float64{kg/m^2}", "kg/m^2"),
        ("float64{m.s^-1}", "m.s^-1"),
    ],
)
def test_parse_measurement_unit(dtype: str, unit: str) -> None:
    parsed = parse_dtype(dtype)
    assert parsed.measurement_unit == unit


def test_parse_dtype_without_unit_has_none() -> None:
    assert parse_dtype("float64").measurement_unit is None


@pytest.mark.parametrize(
    "bad",
    [
        "utf8{label}",  # only numeric base dtypes can carry a unit
        "bool{flag}",
        "timestamp[ns]{seconds}",  # already temporal
        "float64{}",  # empty unit
        "float64{has space}",  # invalid char (space)
        "float64{a,b}",  # invalid char (comma)
        "float64{",  # unmatched brace
    ],
)
def test_parse_rejects_invalid_measurement_unit(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_dtype(bad)


def test_measurement_unit_helpers_round_trip() -> None:
    from mirrorml.fingerprint.dtypes import measurement_unit_of, strip_measurement_unit

    assert measurement_unit_of("float64{meters}") == "meters"
    assert measurement_unit_of("float64") is None
    assert strip_measurement_unit("float64{meters}") == "float64"
    assert strip_measurement_unit("float64") == "float64"


# --- parser: negative cases --------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "int",  # no width
        "int128",  # unsupported width
        "INT64",  # case matters
        "Int64",
        "string",  # use utf8
        "varchar",
        "datetime64[ns]",  # pandas-style; canonical is timestamp[ns]
        "timestamp[ns",  # unbalanced
        "timestamp[xs]",  # bad unit
        "timestamp[]",
        "timestamp[ns,UTC]",  # missing space after comma
        "timestamp[ns, US Pacific]",  # space in tz
        "list[]",
        "list[int128]",  # invalid element
        "decimal[10]",  # missing scale
        "decimal[10, 20]",  # scale > precision
        "decimal[0, 0]",  # zero precision
        "decimal[a, b]",
        "unknown[ns]",
    ],
)
def test_parse_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_dtype(bad)


# --- validate_dtype + Pydantic wiring ----------------------------------------


def test_validate_dtype_returns_input_on_success() -> None:
    assert validate_dtype("int64") == "int64"
    assert validate_dtype("timestamp[ns, UTC]") == "timestamp[ns, UTC]"


def test_validate_dtype_raises_on_invalid() -> None:
    with pytest.raises(ValueError):
        validate_dtype("string")


def test_pydantic_rejects_invalid_dtype_in_column_spec() -> None:
    """The Annotated[str, AfterValidator(validate_dtype)] hook must reject
    bad strings at construction time, not at comparison time."""

    with pytest.raises(ValidationError):
        Source(op_id="s", name="t", columns=(("x", "datetime64[ns]"),))


def test_pydantic_rejects_invalid_dtype_in_input_schema() -> None:
    with pytest.raises(ValidationError):
        build_fingerprint(
            framework="pandas",
            input_schema=(("x", "varchar"),),
            output_schema=(("x", "varchar"),),
            operations=[Source(op_id="s", name="t", columns=(("x", "int64"),))],
        )


# --- named constants ---------------------------------------------------------


def test_named_constants_are_canonical() -> None:
    """Every named constant must be a self-canonical dtype string."""

    for name in (
        NULL,
        BOOL,
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        FLOAT32,
        FLOAT64,
        UTF8,
        BINARY,
        DATE,
    ):
        assert validate_dtype(name) == name


# --- helpers -----------------------------------------------------------------


def test_is_numeric() -> None:
    assert is_numeric("int64")
    assert is_numeric("uint8")
    assert is_numeric("float32")
    assert is_numeric("decimal[10, 2]")
    assert not is_numeric("bool")
    assert not is_numeric("utf8")
    assert not is_numeric("timestamp[ns]")


def test_is_integer_excludes_floats_and_decimals() -> None:
    assert is_integer("int64")
    assert is_integer("uint16")
    assert not is_integer("float64")
    assert not is_integer("decimal[10, 2]")


def test_is_float_is_floats_only() -> None:
    assert is_float("float16")
    assert is_float("float64")
    assert not is_float("int64")
    assert not is_float("decimal[10, 2]")


def test_is_temporal() -> None:
    assert is_temporal("date")
    assert is_temporal("time[ns]")
    assert is_temporal("timestamp[ns, UTC]")
    assert is_temporal("duration[s]")
    assert not is_temporal("int64")


def test_is_timezone_aware() -> None:
    assert is_timezone_aware("timestamp[ns, UTC]")
    assert not is_timezone_aware("timestamp[ns]")
    assert not is_timezone_aware("date")
    assert not is_timezone_aware("int64")


def test_timezone_of_returns_iana_or_none() -> None:
    assert timezone_of("timestamp[ns, US/Pacific]") == "US/Pacific"
    assert timezone_of("timestamp[ns]") is None
    assert timezone_of("date") is None
    assert timezone_of("int64") is None


def test_unit_of_returns_unit_or_none() -> None:
    assert unit_of("timestamp[ns, UTC]") == "ns"
    assert unit_of("duration[ms]") == "ms"
    assert unit_of("time[us]") == "us"
    assert unit_of("date") is None
    assert unit_of("int64") is None


def test_element_dtype_returns_inner_or_none() -> None:
    assert element_dtype("list[int64]") == "int64"
    assert element_dtype("list[timestamp[ns, UTC]]") == "timestamp[ns, UTC]"
    assert element_dtype("list[list[float32]]") == "list[float32]"
    assert element_dtype("int64") is None


def test_bit_width_for_fixed_width_numerics() -> None:
    assert bit_width("int8") == 8
    assert bit_width("uint64") == 64
    assert bit_width("float16") == 16
    assert bit_width("decimal[10, 2]") is None
    assert bit_width("bool") is None
    assert bit_width("utf8") is None


# --- property: serialize ∘ parse_dtype is identity ---------------------------


_unit = st.sampled_from(["s", "ms", "us", "ns"])
_tz = st.sampled_from(["UTC", "US/Pacific", "Europe/Berlin", "Etc/GMT+5"])
_scalar = st.sampled_from(
    [
        "null",
        "bool",
        "utf8",
        "binary",
        "date",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint64",
        "float16",
        "float32",
        "float64",
    ]
)


@st.composite
def _canonical_dtype(draw: st.DrawFn) -> str:
    choice = draw(st.integers(min_value=0, max_value=6))
    if choice == 0:
        return draw(_scalar)
    if choice == 1:
        return f"time[{draw(_unit)}]"
    if choice == 2:
        return f"timestamp[{draw(_unit)}]"
    if choice == 3:
        return f"timestamp[{draw(_unit)}, {draw(_tz)}]"
    if choice == 4:
        return f"duration[{draw(_unit)}]"
    if choice == 5:
        precision = draw(st.integers(min_value=1, max_value=38))
        scale = draw(st.integers(min_value=0, max_value=precision))
        return f"decimal[{precision}, {scale}]"
    inner = draw(_scalar)
    return f"list[{inner}]"


@given(_canonical_dtype())
def test_validate_accepts_any_canonical_string(dtype: str) -> None:
    assert validate_dtype(dtype) == dtype
