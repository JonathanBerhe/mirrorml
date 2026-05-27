# Canonical Dtype Vocabulary

Every dtype that appears in a MirrorML fingerprint is a string in the
closed canonical grammar specified here. The vocabulary is Arrow-flavored:
it borrows Apache Arrow's logical type system because Arrow is the lingua
franca every supported tracer can map into cheaply. Polars is
Arrow-native, pandas 2.x ships Arrow extension dtypes, and SQL engines
that matter (DuckDB, BigQuery, Snowflake, Postgres) expose Arrow metadata
for column types.

This file is the source of truth. The implementation lives at
`src/mirrorml/fingerprint/dtypes.py`; a unit test in
`tests/unit/fingerprint/test_dtypes.py` enforces that the implementation
matches what is written here.

## Why a canonical vocabulary at all

A fingerprint compares pipelines across frameworks. Without a canonical
form for dtypes, two tracers will produce strings that disagree
syntactically even when they agree semantically — pandas's
`datetime64[ns, UTC]`, Polars's `Datetime(time_unit='ns', time_zone='UTC')`,
and SQL's `TIMESTAMP WITH TIME ZONE` mean the same thing but compare
unequal. Every such mismatch becomes either a spurious divergence (precision
tanks) or an outright wrong classification (a `schema_drift` that is really
a `type_coercion`). The canonical vocabulary is the choke point that
makes cross-framework precision possible.

## Grammar

```
dtype       := scalar | parameterized | composite

scalar      := "null"
             | "bool"
             | "utf8"
             | "binary"
             | "date"
             | int_width | uint_width | float_width

int_width   := "int8" | "int16" | "int32" | "int64"
uint_width  := "uint8" | "uint16" | "uint32" | "uint64"
float_width := "float16" | "float32" | "float64"

parameterized := "time" "[" unit "]"
               | "timestamp" "[" unit ("," " " timezone)? "]"
               | "duration" "[" unit "]"
               | "decimal" "[" precision "," " " scale "]"

composite   := "list" "[" dtype "]"

unit        := "s" | "ms" | "us" | "ns"
timezone    := /[A-Za-z0-9_+/\-:]+/
precision   := /[1-9][0-9]*/
scale       := /0|[1-9][0-9]*/                 (where 0 ≤ scale ≤ precision)
```

Whitespace and case are **significant**. The canonical form is exactly
what the grammar emits — no leading/trailing whitespace, no alternate
casings (`INT64` is not a synonym for `int64`), one space after `,` in
`timestamp[unit, tz]` and `decimal[p, s]`.

## Catalog

### Null and boolean

| Canonical | Notes |
|---|---|
| `null` | Arrow's null type; only ever holds null. |
| `bool` | Eight-bit boolean. |

### Integers

| Canonical | Bits | Signed | Notes |
|---|---|---|---|
| `int8` … `int64` | 8 / 16 / 32 / 64 | yes | Two's complement, native byte order. |
| `uint8` … `uint64` | 8 / 16 / 32 / 64 | no |  |

### Floating point

| Canonical | Bits | IEEE-754 form |
|---|---|---|
| `float16` | 16 | binary16 |
| `float32` | 32 | binary32 |
| `float64` | 64 | binary64 |

### Strings and bytes

| Canonical | Notes |
|---|---|
| `utf8` | UTF-8 encoded string. Use this for any text column — there is no `string` / `varchar` synonym. |
| `binary` | Raw bytes. |

### Temporal

| Canonical | Holds | Examples |
|---|---|---|
| `date` | Calendar date, no time. | `2026-05-22` |
| `time[unit]` | Time of day with `unit` resolution. | `time[us]` |
| `timestamp[unit]` | Naive timestamp (no timezone). | `timestamp[ns]` |
| `timestamp[unit, tz]` | Timezone-aware. | `timestamp[ns, UTC]` |
| `duration[unit]` | Elapsed-time interval. | `duration[s]` |

`unit` is one of `s`, `ms`, `us`, `ns`. Timezones are IANA names; the
parser also accepts fixed offset strings like `+05:30` and the `Etc/GMT*`
forms. The parser does *not* verify that the timezone resolves against a
live IANA database — that responsibility lies with the tracer.

### Decimal

| Canonical | Holds |
|---|---|
| `decimal[precision, scale]` | Fixed-point decimal with `precision` total digits and `scale` digits after the point. |

Constraints: `precision ≥ 1`, `0 ≤ scale ≤ precision`. A common
financial decimal is `decimal[18, 2]`.

### Lists

| Canonical | Holds |
|---|---|
| `list[<dtype>]` | Variable-length list whose elements are themselves canonical dtypes. |

Lists nest. `list[list[float32]]` and `list[timestamp[ns, UTC]]` are
valid. There is no fixed-length list yet; if needed we will add
`fixed_size_list[<dtype>, <n>]` in a future minor schema version.

### Reserved for future minor versions

- `struct[<field>: <dtype>, ...]` — heterogeneous record type.
- `map[<key dtype>, <value dtype>]` — key-value pairs.
- `dictionary[<index dtype>, <value dtype>]` (Arrow dictionary-encoded
  type). Useful for categoricals; future categorical-encoding support may
  promote this.
- `fixed_size_binary[<n>]`.

Adding any of these is **additive** — minor-version bump on `SCHEMA_VERSION`
plus an entry here and an `_SCALARS` / parser branch in `dtypes.py`. No
existing fingerprint is invalidated.

## Tracer mapping

Each tracer normalizes its framework's native dtype repertoire into this
canonical form. Authoritative mappings live next to the tracers; the
high-level intent is sketched below for reference.

### pandas → canonical

| pandas | canonical |
|---|---|
| `int64`, `Int64` | `int64` |
| `float64`, `Float64` | `float64` |
| `bool`, `boolean` | `bool` |
| `object` (containing str) | `utf8` |
| `string`, `string[pyarrow]` | `utf8` |
| `datetime64[ns]` | `timestamp[ns]` |
| `datetime64[ns, UTC]` | `timestamp[ns, UTC]` |
| `timedelta64[ns]` | `duration[ns]` |
| `category` | (TBD — see "Reserved for future minor versions") |

### Polars → canonical

| Polars | canonical |
|---|---|
| `Int64`, `UInt64`, etc. | `int64`, `uint64`, etc. |
| `Float32`, `Float64` | `float32`, `float64` |
| `Boolean` | `bool` |
| `Utf8` | `utf8` |
| `Binary` | `binary` |
| `Date` | `date` |
| `Time` | `time[ns]` (Polars resolution) |
| `Datetime(time_unit, time_zone)` | `timestamp[<unit>]` / `timestamp[<unit>, <tz>]` |
| `Duration(time_unit)` | `duration[<unit>]` |
| `List(inner)` | `list[<canonical(inner)>]` |
| `Decimal(precision, scale)` | `decimal[<precision>, <scale>]` |

### SQL → canonical

Dialect-specific. The SQL tracer dispatches via sqlglot's `DataType` AST.
Examples for common dialects:

| SQL (Snowflake / BigQuery / Postgres) | canonical |
|---|---|
| `BIGINT` / `INT64` / `BIGINT` | `int64` |
| `DOUBLE` / `FLOAT64` / `DOUBLE PRECISION` | `float64` |
| `BOOLEAN` | `bool` |
| `VARCHAR` / `STRING` / `TEXT` | `utf8` |
| `BINARY` / `BYTES` / `BYTEA` | `binary` |
| `DATE` | `date` |
| `TIMESTAMP` / `TIMESTAMP` / `TIMESTAMP` | `timestamp[us]` (or dialect default unit) |
| `TIMESTAMP WITH TIME ZONE` / `TIMESTAMP` / `TIMESTAMPTZ` | `timestamp[us, UTC]` |
| `INTERVAL` / `INTERVAL` | `duration[us]` |
| `NUMERIC(p, s)` / `NUMERIC(p, s)` / `NUMERIC(p, s)` | `decimal[p, s]` |
| `ARRAY<T>` / `ARRAY<T>` / `T[]` | `list[<canonical(T)>]` |

## Comparison semantics

Two canonical dtypes are equal iff their strings are byte-equal.

The diff classifier consults the structured form (`parse_dtype`) when a
divergence touches a dtype:

- `int8` vs `int64` → `type_coercion` (same kind, different width).
- `int64` vs `float64` → `type_coercion`.
- `timestamp[ns]` vs `timestamp[ns, UTC]` → `timezone_mismatch`.
- `timestamp[ns, UTC]` vs `timestamp[ns, US/Pacific]` → `timezone_mismatch`.
- `timestamp[ns, UTC]` vs `timestamp[ms, UTC]` → `rounding_precision`.
- `decimal[18, 2]` vs `decimal[18, 4]` → `rounding_precision`.
- `list[int64]` vs `list[int32]` → `type_coercion` (recursive).
- `utf8` vs `int64` → `schema_drift` (kind difference).

The diff classifier implements these rules. This list is the design
contract those rules satisfy.

## Limits

- No struct, map, dictionary, or fixed-size types in v1.0.0 — see
  "Reserved for future minor versions" above. Tracers that encounter
  these must raise `UnsupportedOperationError` until the schema catches
  up.
- The vocabulary does not encode nullability. Every dtype is implicitly
  nullable; nullability differences are surfaced by the diff classifier
  via `null_handling`, not via dtype strings.
- Endianness, alignment, and physical-layout details are out of scope.
  MirrorML works at the logical-type level only.
