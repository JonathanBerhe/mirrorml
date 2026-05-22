# MirrorML

> Great Expectations is for your data. MirrorML is for your pipelines.

MirrorML is a static-analysis library for detecting **training-serving skew** in
machine learning feature pipelines. Given two pipelines that should compute the
same features — typically one offline (training) and one online (serving) —
MirrorML produces a *semantic fingerprint* of each and reports whether they are
equivalent. When they diverge, MirrorML localizes the divergence to the
responsible operation and classifies it into one of fifteen well-defined
categories (window boundary mismatches, timezone handling, as-of join direction,
categorical encoding drift, and so on).

## Status

Pre-alpha (`v0.0.1`). The fingerprint schema is locked as the public contract;
the tracers (pandas, Polars, SQL), diff engine, MirrorBench, and CLI commands
beyond `--help` / `--version` are scheduled for upcoming milestones. The
public-API symbols exist and import cleanly; calling the unimplemented ones
raises `NotImplementedError` with a milestone tag.

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
import mirrorml

# The fingerprint schema is real and stable as of v0.0.1.
# Tracers (M2), diff engine (M3), and CLI commands (M5) follow.
```

## Public API

The seven names exported from `mirrorml` are the entire stable surface:

| Name | Kind | Status |
|---|---|---|
| `Fingerprint` | Pydantic model | Stable |
| `fingerprint(...)` | constructor | Stable |
| `Divergence` | Pydantic model | Stable |
| `diff(...)` | function | M3 — raises `NotImplementedError` |
| `trace_pandas(...)` | function | M2 — raises `NotImplementedError` |
| `trace_polars(...)` | function | M2 — raises `NotImplementedError` |
| `trace_sql(...)` | function | M2 — raises `NotImplementedError` |

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
