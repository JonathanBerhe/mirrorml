# Category Support Matrix

The [divergence taxonomy](./divergence_taxonomy.md) defines *what* each of the
fifteen categories means. This document describes *how* MirrorML detects each
one today: the signal it keys on, which tracers can produce that signal, and
which pipeline patterns are not yet traced.

The distinction matters. The diff engine can emit all fifteen categories, but a
category is only reported when a tracer lowers the relevant pipeline pattern
into the fingerprint. For example, `categorical_encoding` is detected from an
`Encode` operation, and today only the Polars `to_dummies` call produces one.
A pandas `get_dummies` pipeline will not raise a `categorical_encoding`
divergence yet, because that pattern is not traced.

## How detection works

Every divergence comes from one of three signals:

- **Schema dtype diff.** MirrorML compares the dtype of each column on the input
  and output schemas. These categories work the same across all three tracers,
  because they depend on the declared schema, not on a particular operation.
- **Operation diff.** Two aligned operations of the same kind differ in a
  parameter (an aggregation function, a window size, a fill value, a random
  seed). These depend on the tracer emitting that operation.
- **Whole-graph rule.** The engine reasons over the whole operation graph rather
  than a single operation. Today this powers `feature_leakage_temporal`.

## The matrix

| Category | Signal | Tracers that can trigger it | Not yet traced |
|---|---|---|---|
| `timezone_mismatch` | schema dtype diff (timestamp timezone) | pandas, Polars, SQL | — |
| `type_coercion` | schema dtype diff (base dtype) | pandas, Polars, SQL | — |
| `rounding_precision` | schema dtype diff (time unit, or decimal precision/scale) | pandas, Polars, SQL | — |
| `unit_mismatch` | schema dtype diff (`{measurement_unit}` suffix) | pandas, Polars, SQL | automatic unit inference; units must be declared in the schema |
| `aggregation_function` | `Aggregate` op (reduction function) | pandas, Polars, SQL | — |
| `ordering_dependence` | `Sort` op (keys or direction) | pandas, Polars, SQL | — |
| `schema_drift` | column add / drop / rename, operation-count mismatch, or a same-kind op difference no finer category fits | pandas, Polars, SQL | — |
| `null_handling` | `FillNa` op (value, columns, strategy), or a filter predicate referencing NULL | pandas, Polars (FillNa); any tracer (NULL filter) | imputers that fit at runtime (for example sklearn `SimpleImputer`) |
| `window_size_mismatch` | `Window` op (frame size) | Polars (`rolling`), SQL (trailing `ROWS` frame) | pandas |
| `window_boundary` | `Window` op (`closed` boundary) | Polars (`rolling(closed=...)`) | pandas; SQL `ROWS` frames do not carry a closed-boundary semantic |
| `as_of_join_direction` | `AsOfJoin` op (direction / strategy) | Polars (`join_asof`) | pandas `merge_asof`; SQL `ASOF JOIN` |
| `join_key_mismatch` | `Join` op (join keys) | SQL | pandas `merge`; Polars equi-`join` |
| `categorical_encoding` | `Encode` op (method, columns, categories) | Polars (`to_dummies`) | pandas `get_dummies`; sklearn encoders |
| `seed_mismatch` | `Sample` op (random seed) | pandas (`sample`), Polars (`sample`) | train/test split; stochastic encoders |
| `feature_leakage_temporal` | whole-graph rule: one side guards an aggregation inside a time `Window`, the other does not, with `event_time_column` declared on both sources | pandas + Polars (the guarded side uses a Polars time window) | leakage shapes beyond the window-versus-plain-aggregation asymmetry; SQL sources |

## Reading the matrix

The first eight rows are detected at the schema or common-operation level and
behave consistently across every tracer. The remaining seven depend on a
specific operation that a specific tracer emits today; the "Not yet traced"
column lists the patterns that fall outside current coverage.

Broadening a category (adding a new trigger, such as pandas `get_dummies` for
`categorical_encoding`) is demand-driven: it lands when a pipeline that needs it
appears, rather than being built ahead of need. When a pipeline uses a pattern a
tracer does not yet model, the tracer raises `UnsupportedOperationError` with the
operation name, so the gap is explicit rather than silent.
