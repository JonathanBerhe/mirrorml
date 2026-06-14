# MirrorML

> Find the bugs that make a machine learning model behave differently in testing
> than it does in production.

MirrorML checks whether two data pipelines that are supposed to compute the same
thing actually do.

Here is the problem it solves. A machine learning model learns from "features":
the input numbers fed into it. Those numbers are usually computed twice, by two
separate pieces of code: once over saved historical data while the model is being
trained, and again over live data while the model is running for real users.
These two pieces of code are often written by different people, in different
languages, at different times. When they fall out of sync even slightly (one
rounds a number differently, one fills in missing values differently, one
averages over a different time range), the model quietly makes worse predictions
in production than it did in testing. This mismatch is called **training-serving
skew**. Because nothing crashes and no error is raised, it can go unnoticed for a
long time.

MirrorML reads both pieces of code and builds a short summary of what each one
does, called a *fingerprint*. It then compares the two summaries. If they match,
the pipelines are equivalent and there is no skew. If they do not, MirrorML points
to the exact step that differs and labels what kind of difference it is (for
example: a different time zone, a different rounding, adding numbers up where the
other averages them, or handling missing values differently). It works by reading
the code, so it does not need to run the pipelines or have any real data.

## Status

Early development (`v0.0.1`). The core works end to end. MirrorML can read
pipelines written in pandas, Polars, or SQL (three common ways to work with
tables of data in Python and databases), summarize each one, compare them, and
point to the step that differs. The same pipeline written in any of the three
produces the same summary, so a training pipeline written in one language can be
checked against a production pipeline written in another.

MirrorML recognizes all fifteen kinds of difference it sets out to catch. As a
second, independent check, it can also run both pipelines on a small batch of
generated data and compare the actual results. So far it has been tested mainly
on examples we created ourselves, plus four real-world bugs rebuilt from
published reports. Testing on pipelines collected from public projects is still
to come, so there are not yet accuracy numbers from real-world use.

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

The core install does not require pandas or Polars. Each one is loaded only when
you actually use its tracer, so `import mirrorml` stays fast.

## Quickstart

The example below takes a training pipeline written in pandas and a production
pipeline written in SQL, and confirms they compute the same thing. Then it
changes the SQL side to add scores up instead of averaging them, and MirrorML
reports the difference and where it is.

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

# Trace both sides of a pair, diff them, and check the result against the
# differences the pair says to expect; exits with an error code on a
# mismatch, so it can run as an automated check
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
