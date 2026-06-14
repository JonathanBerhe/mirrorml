# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project will follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches
1.0.0; until then, minor pre-1.0 versions may include breaking changes.

The fingerprint schema is versioned independently of the package (see
`SCHEMA_VERSION`); it is currently at 1.1.0.

## [Unreleased]

The project has not yet been published to PyPI. This section describes the
state that will become the first published release.

### Added

- Canonical fingerprint format for feature pipelines: a content-addressed
  summary of a pipeline's operation graph, parameters, schema effects, and
  temporal semantics, with a stable `fingerprint_id` safe for equality checks.
- Three tracers that lower a pipeline to the canonical fingerprint: `trace_sql`
  (via sqlglot), `trace_pandas`, and `trace_polars` (wrapper-object tracing; no
  pandas/polars import unless the tracer is called).
- Cross-framework equivalence: a pandas, Polars, or SQL pipeline that computes
  the same thing produces fingerprints that `diff()` to empty.
- Diff engine that compares two fingerprints, classifies each disagreement into
  the divergence taxonomy, and localizes it to the responsible operation.
- Coverage of all fifteen taxonomy categories, including `unit_mismatch`
  (via an optional `{measurement_unit}` dtype suffix), `seed_mismatch` (via a
  `Sample` operation that carries the random seed), `categorical_encoding`
  (via the `Encode` operation), and `feature_leakage_temporal` (a whole-graph
  check using `event_time_column` on the source).
- User-defined-function support: `df.apply(...)` and `lf.map_batches(...)` lower
  to a `Udf` operation keyed on a normalized libcst source-hash
  (`libcst-norm-v1`), so two callables that differ only in formatting or
  comments are treated as equivalent.
- Statistical companion check: runs both pipelines on a small generated fixture
  and compares outputs within tolerance, including a polars-based fallback for
  trailing-window SQL queries that the SQL executor does not support.
- Command-line interface: `mirrorml trace`, `mirrorml diff`, and
  `mirrorml verify` (the last checks a pair, or a whole directory of pairs,
  against its expected divergences and exits non-zero on a mismatch).
- MirrorBench: an in-repo benchmark of pipeline pairs spanning every category,
  plus four incidents reconstructed from peer-reviewed reports of production
  skew, with an evaluation harness (`bench/scripts/run_eval.py`) that reports
  precision, recall, localization accuracy, and static-versus-statistical
  agreement.
- Schema versioning and migration: fingerprints carry a `schema_version`, and
  loading an older fingerprint routes through a registered migration path.

### Notes

- Benchmark accuracy figures to date are in-distribution (measured on the
  generated synthetic corpus and the reconstructed incidents). Accuracy on
  feature pipelines mined from public projects has not been measured yet.
