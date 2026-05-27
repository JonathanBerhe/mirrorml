# MirrorML

> Great Expectations is for your data. MirrorML is for your pipelines.

MirrorML is a static-analysis library for detecting **training-serving skew** in
machine learning feature pipelines. Given two pipelines that should compute the
same features (typically one offline for training and one online for serving),
MirrorML produces a *semantic fingerprint* of each and reports whether they are
equivalent. When they diverge, MirrorML localizes the divergence to the
responsible operation and classifies it into one of fifteen well-defined
categories (window boundary mismatches, timezone handling, as-of join direction,
aggregation-function swaps, null handling, and so on).

## Status

Pre-alpha (`v0.0.1`). The fingerprint schema is the locked public contract. All
three tracers (pandas, Polars, SQL), the diff engine, the MirrorBench evaluation
harness, and the `trace` / `diff` / `verify` CLI commands are implemented. The
cross-framework promise holds end to end: an equivalent pandas, Polars, or SQL
pipeline produces fingerprints that `diff()` to empty, and a real difference is
classified and localized to the responsible operation.

The diff engine currently detects 11 of the 15 taxonomy categories. The
remaining four (categorical encoding, unit mismatch, temporal feature leakage,
seed mismatch) need runtime or whole-graph information and are planned alongside
the statistical companion check. Benchmark numbers to date are in-distribution
(the synthetic corpus); a real-world and replayed-incident corpus is in progress.

## Install

MirrorML is not yet published to PyPI. From a clone:

```bash
uv sync --all-extras --dev
```

Once published:

```bash
uv add mirrorml              # core
uv add 'mirrorml[pandas]'    # pandas tracer
uv add 'mirrorml[polars]'    # Polars tracer
```

The core install does not depend on pandas or Polars. Tracers are lazy-imported
so `import mirrorml` stays under the 200ms cold-start budget.

## Quickstart

```python
from mirrorml import diff, trace_pandas, trace_sql

EVENTS = (("uid", "int64"), ("score", "float64"))


def offline(df):
    return df[df["score"] > 0].groupby("uid").agg({"score": "mean"})


pandas_fp = trace_pandas(offline, input_schema=EVENTS, source_name="events")
sql_fp = trace_sql(
    "SELECT uid, AVG(score) AS score FROM events WHERE score > 0 GROUP BY uid",
    schemas={"events": EVENTS},
)

assert diff(pandas_fp, sql_fp) == ()  # the two pipelines are equivalent

# Now the online side sums instead of averaging:
sql_skewed = trace_sql(
    "SELECT uid, SUM(score) AS score FROM events WHERE score > 0 GROUP BY uid",
    schemas={"events": EVENTS},
)
for d in diff(pandas_fp, sql_skewed):
    print(d.category, "|", d.detail)
# aggregation_function | aggregation 'score': function 'mean' vs 'sum'
```

The Polars tracer takes the same shape; its pipeline receives the frame plus a
`pl` expression namespace:

```python
from mirrorml import trace_polars


def offline(lf, pl):
    return lf.filter(pl.col("score") > 0).group_by("uid").agg(pl.col("score").mean())


polars_fp = trace_polars(offline, input_schema=EVENTS, source_name="events")
```

## CLI

```bash
# Emit one side of a pair as canonical fingerprint JSON
mirrorml trace path/to/pair --side offline -o offline.json

# Diff two on-disk fingerprints (exit 1 if they diverge)
mirrorml diff offline.json online.json

# Trace both sides of a pair, diff, and check against its expected
# divergences; exits non-zero on mismatch (the CI primitive)
mirrorml verify path/to/pair
```

A "pair" is a directory containing a `meta.yaml` (which names each side's
language, source file, and schema) plus the source files themselves. The
MirrorBench pairs under `bench/pairs/` are runnable examples.

## Public API

The seven names exported from `mirrorml` are the entire stable surface:

| Name | Kind | Status |
|---|---|---|
| `Fingerprint` | Pydantic model | Stable |
| `fingerprint(...)` | constructor | Stable |
| `Divergence` | Pydantic model | Stable |
| `diff(...)` | function | Implemented |
| `trace_pandas(...)` | function | Implemented |
| `trace_sql(...)` | function | Implemented |
| `trace_polars(...)` | function | Implemented (experimental) |

Anything not listed is internal and may change without notice.

## Development

```bash
uv sync --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
```

## License

Apache-2.0. See [LICENSE](./LICENSE).
