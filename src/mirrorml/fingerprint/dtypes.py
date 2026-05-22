"""Canonical dtype vocabulary for MirrorML.

The dtype vocabulary is Arrow-flavored: it borrows Apache Arrow's logical
type system because Arrow is the lingua franca every tracer can map into
cheaply — Polars is Arrow-native, pandas 2.x ships Arrow extension dtypes,
and most modern SQL engines expose Arrow IPC for column metadata.

The canonical form is a string in a closed grammar — see
``docs/concepts/dtype_vocabulary.md`` for the definitive spec. Every dtype
that appears in a :class:`~mirrorml.fingerprint.schema.Fingerprint` is one
of these strings; the M2 tracers normalize their framework-native types
into this form before constructing operations.

This module is the single source of truth for what counts as a valid
dtype. :func:`validate_dtype` is wired into the schema via :data:`Dtype`
so Pydantic rejects bad strings at fingerprint-construction time rather
than letting them propagate into diffs that silently disagree because two
tracers spelled ``int64`` differently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Final, TypeAlias

from pydantic import AfterValidator

__all__ = [
    "BINARY",
    "BOOL",
    "DATE",
    "DURATION_MS",
    "DURATION_NS",
    "DURATION_S",
    "DURATION_US",
    "FLOAT16",
    "FLOAT32",
    "FLOAT64",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "NULL",
    "TIMESTAMP_MS",
    "TIMESTAMP_NS",
    "TIMESTAMP_S",
    "TIMESTAMP_US",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "UTF8",
    "Dtype",
    "ParsedDtype",
    "bit_width",
    "element_dtype",
    "is_float",
    "is_integer",
    "is_numeric",
    "is_temporal",
    "is_timezone_aware",
    "parse_dtype",
    "timezone_of",
    "unit_of",
    "validate_dtype",
]

_UNITS: Final[frozenset[str]] = frozenset({"s", "ms", "us", "ns"})
_TIMEZONE_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_+/\-:]+$")


@dataclass(frozen=True)
class ParsedDtype:
    """Structured form of a canonical dtype string.

    Returned by :func:`parse_dtype`; intended for introspection (the diff
    classifier needs to compare units / timezones / element types to label
    a divergence). For storage and equality use the canonical string itself
    — it is byte-stable and JSON-native.
    """

    kind: str
    """One of: ``null``, ``bool``, ``int``, ``uint``, ``float``, ``utf8``,
    ``binary``, ``date``, ``time``, ``timestamp``, ``duration``, ``list``,
    ``decimal``."""

    bits: int | None = None
    """Bit width for fixed-width numerics (``int``, ``uint``, ``float``)."""

    unit: str | None = None
    """Time unit (``s`` / ``ms`` / ``us`` / ``ns``) for ``time``,
    ``timestamp``, ``duration``."""

    timezone: str | None = None
    """IANA timezone name for ``timestamp``; ``None`` for naive timestamps
    and all non-timestamp dtypes."""

    element: ParsedDtype | None = None
    """Element type for ``list``; ``None`` otherwise."""

    precision: int | None = None
    """Total digit count for ``decimal``."""

    scale: int | None = None
    """Digit count after the decimal point for ``decimal``."""


_SCALARS: Final[dict[str, ParsedDtype]] = {
    "null": ParsedDtype("null"),
    "bool": ParsedDtype("bool"),
    "int8": ParsedDtype("int", bits=8),
    "int16": ParsedDtype("int", bits=16),
    "int32": ParsedDtype("int", bits=32),
    "int64": ParsedDtype("int", bits=64),
    "uint8": ParsedDtype("uint", bits=8),
    "uint16": ParsedDtype("uint", bits=16),
    "uint32": ParsedDtype("uint", bits=32),
    "uint64": ParsedDtype("uint", bits=64),
    "float16": ParsedDtype("float", bits=16),
    "float32": ParsedDtype("float", bits=32),
    "float64": ParsedDtype("float", bits=64),
    "utf8": ParsedDtype("utf8"),
    "binary": ParsedDtype("binary"),
    "date": ParsedDtype("date"),
}


@lru_cache(maxsize=1024)
def parse_dtype(s: str) -> ParsedDtype:
    """Parse a canonical dtype string into its structured form.

    Examples:
        >>> parse_dtype("int64").bits
        64
        >>> parse_dtype("timestamp[ns, UTC]").timezone
        'UTC'
        >>> parse_dtype("list[int64]").element.kind
        'int'
        >>> parse_dtype("decimal[18, 2]").precision
        18

    Raises:
        ValueError: For any string not in canonical form. The message
            names the offending string and points to the spec.
    """

    if not s:
        raise ValueError(
            "dtype is empty; expected a canonical name (see docs/concepts/dtype_vocabulary.md)"
        )

    cached = _SCALARS.get(s)
    if cached is not None:
        return cached

    open_bracket = s.find("[")
    if open_bracket == -1:
        raise ValueError(
            f"dtype {s!r} is not a recognized canonical name; "
            f"see docs/concepts/dtype_vocabulary.md for the catalog"
        )
    if not s.endswith("]"):
        raise ValueError(
            f"dtype {s!r} has an unmatched opening bracket; expected the form `kind[parameters]`"
        )

    kind = s[:open_bracket]
    inner = s[open_bracket + 1 : -1]

    if kind == "time":
        return _parse_time(inner, s)
    if kind == "timestamp":
        return _parse_timestamp(inner, s)
    if kind == "duration":
        return _parse_duration(inner, s)
    if kind == "list":
        return ParsedDtype("list", element=parse_dtype(inner))
    if kind == "decimal":
        return _parse_decimal(inner, s)

    raise ValueError(
        f"dtype {s!r}: unrecognized parameterized kind {kind!r}; "
        f"valid kinds are: time, timestamp, duration, list, decimal"
    )


def _parse_time(inner: str, original: str) -> ParsedDtype:
    if inner not in _UNITS:
        raise ValueError(
            f"dtype {original!r}: invalid time unit {inner!r}; expected one of {sorted(_UNITS)}"
        )
    return ParsedDtype("time", unit=inner)


def _parse_timestamp(inner: str, original: str) -> ParsedDtype:
    if ", " in inner:
        unit, tz = inner.split(", ", 1)
    else:
        unit, tz = inner, None
    if unit not in _UNITS:
        raise ValueError(
            f"dtype {original!r}: invalid timestamp unit {unit!r}; expected one of {sorted(_UNITS)}"
        )
    if tz is not None and not _TIMEZONE_RE.fullmatch(tz):
        raise ValueError(
            f"dtype {original!r}: timezone {tz!r} contains invalid "
            f"characters; expected an IANA name like 'UTC' or 'US/Pacific'"
        )
    return ParsedDtype("timestamp", unit=unit, timezone=tz)


def _parse_duration(inner: str, original: str) -> ParsedDtype:
    if inner not in _UNITS:
        raise ValueError(
            f"dtype {original!r}: invalid duration unit {inner!r}; expected one of {sorted(_UNITS)}"
        )
    return ParsedDtype("duration", unit=inner)


def _parse_decimal(inner: str, original: str) -> ParsedDtype:
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) != 2:
        raise ValueError(f"dtype {original!r}: decimal must be `decimal[precision, scale]`")
    try:
        precision = int(parts[0])
        scale = int(parts[1])
    except ValueError:
        raise ValueError(
            f"dtype {original!r}: decimal precision and scale must be integers"
        ) from None
    if precision <= 0 or scale < 0 or scale > precision:
        raise ValueError(
            f"dtype {original!r}: invalid (precision={precision}, scale={scale}); "
            f"require precision > 0 and 0 <= scale <= precision"
        )
    return ParsedDtype("decimal", precision=precision, scale=scale)


def validate_dtype(s: str) -> str:
    """Return ``s`` if it is a valid canonical dtype; raise ``ValueError`` otherwise.

    Used as a Pydantic :class:`AfterValidator` on :data:`Dtype` so every
    dtype string flowing into a fingerprint is validated at construction
    time, not deferred until comparison.

    Examples:
        >>> validate_dtype("int64")
        'int64'
        >>> validate_dtype("timestamp[ns, UTC]")
        'timestamp[ns, UTC]'
    """

    parse_dtype(s)
    return s


Dtype: TypeAlias = Annotated[str, AfterValidator(validate_dtype)]
"""Canonical dtype string with Pydantic-enforced grammar validation."""


NULL: Final[Dtype] = "null"
BOOL: Final[Dtype] = "bool"

INT8: Final[Dtype] = "int8"
INT16: Final[Dtype] = "int16"
INT32: Final[Dtype] = "int32"
INT64: Final[Dtype] = "int64"

UINT8: Final[Dtype] = "uint8"
UINT16: Final[Dtype] = "uint16"
UINT32: Final[Dtype] = "uint32"
UINT64: Final[Dtype] = "uint64"

FLOAT16: Final[Dtype] = "float16"
FLOAT32: Final[Dtype] = "float32"
FLOAT64: Final[Dtype] = "float64"

UTF8: Final[Dtype] = "utf8"
BINARY: Final[Dtype] = "binary"
DATE: Final[Dtype] = "date"

TIMESTAMP_S: Final[Dtype] = "timestamp[s]"
TIMESTAMP_MS: Final[Dtype] = "timestamp[ms]"
TIMESTAMP_US: Final[Dtype] = "timestamp[us]"
TIMESTAMP_NS: Final[Dtype] = "timestamp[ns]"

DURATION_S: Final[Dtype] = "duration[s]"
DURATION_MS: Final[Dtype] = "duration[ms]"
DURATION_US: Final[Dtype] = "duration[us]"
DURATION_NS: Final[Dtype] = "duration[ns]"


def is_numeric(dtype: str) -> bool:
    """Whether the dtype is integer, unsigned-integer, float, or decimal."""

    return parse_dtype(dtype).kind in ("int", "uint", "float", "decimal")


def is_integer(dtype: str) -> bool:
    """Whether the dtype is a signed or unsigned integer."""

    return parse_dtype(dtype).kind in ("int", "uint")


def is_float(dtype: str) -> bool:
    """Whether the dtype is a binary floating-point number."""

    return parse_dtype(dtype).kind == "float"


def is_temporal(dtype: str) -> bool:
    """Whether the dtype carries time semantics (date / time / timestamp / duration)."""

    return parse_dtype(dtype).kind in ("date", "time", "timestamp", "duration")


def is_timezone_aware(dtype: str) -> bool:
    """Whether the dtype is a timestamp with an attached timezone."""

    parsed = parse_dtype(dtype)
    return parsed.kind == "timestamp" and parsed.timezone is not None


def timezone_of(dtype: str) -> str | None:
    """The IANA timezone of a timestamp dtype; ``None`` if naive or non-temporal."""

    parsed = parse_dtype(dtype)
    return parsed.timezone if parsed.kind == "timestamp" else None


def unit_of(dtype: str) -> str | None:
    """The time unit of a temporal dtype, or ``None`` otherwise.

    Returns ``None`` for ``date`` (which has implicit day resolution) and
    for any non-temporal dtype.
    """

    parsed = parse_dtype(dtype)
    if parsed.kind in ("time", "timestamp", "duration"):
        return parsed.unit
    return None


def element_dtype(dtype: str) -> str | None:
    """The element dtype string of a ``list[...]`` dtype, or ``None`` otherwise."""

    parsed = parse_dtype(dtype)
    if parsed.kind == "list" and parsed.element is not None:
        return _serialize(parsed.element)
    return None


def bit_width(dtype: str) -> int | None:
    """Bit width of a fixed-width numeric dtype (int / uint / float), else ``None``."""

    parsed = parse_dtype(dtype)
    if parsed.kind in ("int", "uint", "float"):
        return parsed.bits
    return None


def _serialize(parsed: ParsedDtype) -> str:
    """Reverse of :func:`parse_dtype`: render a :class:`ParsedDtype` as its
    canonical string. Used by :func:`element_dtype` to surface list element
    types without forcing callers to deal with the structured form."""

    if parsed.kind in ("null", "bool", "utf8", "binary", "date"):
        return parsed.kind
    if parsed.kind in ("int", "uint", "float"):
        return f"{parsed.kind}{parsed.bits}"
    if parsed.kind == "time":
        return f"time[{parsed.unit}]"
    if parsed.kind == "timestamp":
        if parsed.timezone is not None:
            return f"timestamp[{parsed.unit}, {parsed.timezone}]"
        return f"timestamp[{parsed.unit}]"
    if parsed.kind == "duration":
        return f"duration[{parsed.unit}]"
    if parsed.kind == "list":
        assert parsed.element is not None
        return f"list[{_serialize(parsed.element)}]"
    if parsed.kind == "decimal":
        return f"decimal[{parsed.precision}, {parsed.scale}]"
    raise AssertionError(f"unreachable: unknown kind {parsed.kind!r}")
