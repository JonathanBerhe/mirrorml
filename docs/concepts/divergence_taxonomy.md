# Divergence Taxonomy

This document is the definitive specification of MirrorML's divergence
categories. The set is **closed** and **flat** — fifteen labels, no
hierarchy. Adding a category requires (per `CLAUDE.md`) a definition in this
file, a classifier rule in `src/mirrorml/diff/classify.py`, and at least
five MirrorBench examples. Renaming or removing a category is a breaking
change.

The static type `DivergenceCategory` in `src/mirrorml/_taxonomy.py` and the
runtime tuple `DIVERGENCE_CATEGORIES` must match the ordering below. A unit
test in `tests/unit/test_taxonomy.py` enforces this.

## The fifteen categories

### `window_boundary`

A rolling, sliding, or session window includes (or excludes) its endpoints
differently in the two pipelines. The mathematics of the aggregation is
identical; the set of rows aggregated is not.

> *Example.* Offline pipeline uses `df.rolling("5min", closed="left")`;
> online uses `closed="right"`. A request landing exactly on a minute
> boundary lands in different windows in the two pipelines, producing a
> consistently biased feature.

### `window_size_mismatch`

The window aggregation function and boundary semantics are identical; the
window size itself differs.

> *Example.* Offline uses a 7-day rolling mean; online uses a 6-day rolling
> mean because the configuration key was misspelled.

### `timezone_mismatch`

Two pipelines disagree on the timezone applied to a timestamp column, or
one localizes naive timestamps to a different zone than the other.

> *Example.* Offline reads UTC timestamps and applies `tz_convert("US/Pacific")`
> before bucketing by hour-of-day; online assumes raw timestamps are
> already local. Hour-of-day features are shifted by the local UTC offset.

### `null_handling`

Two pipelines disagree on what happens when a null value reaches an
aggregation, a filter, or a downstream consumer. Includes nulls treated as
zero, propagated, dropped, or filled with a sentinel.

> *Example.* Offline drops rows where `price` is null before computing the
> mean; online includes them, treating null as zero. The online mean is
> consistently lower.

### `categorical_encoding`

Two pipelines produce different encodings for the same categorical input —
different category orders, different label-to-index maps, different
handling of unseen categories, or a different encoding scheme entirely
(one-hot vs. label vs. target).

> *Example.* Offline one-hot-encodes `country` against a vocabulary fit on
> the training set including `{US, GB, CA}`; online builds its own
> vocabulary per request and emits `{CA, GB, US}` in alphabetical order.
> Column positions in the resulting feature vector are permuted.

### `join_key_mismatch`

Two pipelines join the same logical tables but disagree on the join keys —
different columns, different cast strategy, different handling of
case-sensitivity in string keys.

> *Example.* Offline joins on `(user_id, day)` with `day` materialized as a
> `date`; online joins on `(user_id, day)` with `day` left as `datetime`,
> silently producing zero matches at the hour-granular timestamps the
> serving path receives.

### `as_of_join_direction`

An as-of join is performed in a different direction (forward vs. backward
vs. nearest) or with a different tolerance. The most common source of
silent skew in pipelines that mix point-in-time correctness with serving
performance.

> *Example.* Offline runs `merge_asof(direction="backward")` to attach the
> last-known feature value; online runs `direction="forward"`, attaching a
> feature value computed *after* the prediction time — a leakage bug in
> training that disappears at serving, producing an apparently correct
> model that degrades the moment it ships.

### `aggregation_function`

Two pipelines compute different aggregations on the same grouping. Includes
`mean` vs `median`, `count` vs `count_distinct`, weighted vs. unweighted
forms.

> *Example.* Offline computes `count_distinct(user_id)`; online computes
> `count(user_id)`. For groups with repeat visits, the online feature is
> systematically larger.

### `type_coercion`

Two pipelines coerce the same column to different types, or one preserves
the source type and the other coerces. Manifests as float-vs-int
truncation, string-vs-numeric comparisons, or NaN-vs-NULL semantics
differing between the engines.

> *Example.* Offline keeps `age_in_days` as `int64`; online coerces it to
> `float32` via a serialization round-trip, losing the bottom bits for
> values above 2²³.

### `ordering_dependence`

A pipeline's output depends on the order of its input, and the two
pipelines order it differently. Includes `first` / `last` aggregations
applied to unordered groups, top-N filters, and `head()` / `tail()`
operations on unsorted frames.

> *Example.* Offline takes `groupby('uid').last()` after a stable
> `sort_values('ts')`; online relies on the natural order from a streaming
> source which happens to be reverse-chronological. The "last" value is
> systematically the wrong one.

### `seed_mismatch`

A stochastic operation (sampling, hashing-based encoding, random
projection) uses a different seed in the two pipelines, producing different
outputs for the same input.

> *Example.* Offline uses `hashing_trick(n=2**10, seed=42)`; online uses
> `seed=0` because the deployment configuration's seed field was renamed
> and the migration script missed this site.

### `schema_drift`

Two pipelines produce different output schemas — extra columns, missing
columns, reordered columns. The most common cause is one side adding a
feature that the other has not yet shipped.

> *Example.* The offline pipeline now produces a new `country_tier` column;
> the online pipeline still emits the previous 47-column feature vector.
> The model trained on the new schema receives mis-aligned features at
> serving time.

### `rounding_precision`

Two pipelines round numeric values at different precisions or with
different rounding modes (banker's vs. away-from-zero), producing values
that are arbitrarily close but not identical and that occasionally cross a
threshold differently.

> *Example.* Offline rounds `score` to 4 decimal places before bucketing by
> a threshold of 0.5; online rounds to 3 places. Records with raw scores of
> 0.4998 fall on opposite sides of the threshold.

### `feature_leakage_temporal`

The pipeline's output for a row at time *t* depends, in one direction but
not the other, on data observable only after *t*. Closely related to
`as_of_join_direction` but covers leakage paths beyond explicit
as-of joins: windowed aggregations that include the current row, look-ahead
joins, target encoding that pools the target column itself.

> *Example.* Offline trains a target-encoded feature on the full training
> set including the row being encoded; online cannot leak future targets
> because they do not exist yet. The model's training-time accuracy
> overstates its real performance.

### `unit_mismatch`

Two pipelines produce the same column with different units of measure —
seconds vs. milliseconds, dollars vs. cents, meters vs. miles.

> *Example.* Offline emits `elapsed_time` in seconds; online emits it in
> milliseconds because the source schema was updated and the consumer was
> not. The model treats events of 1ms duration as if they were 1s long.

## Ordering

The order above is the order of `DIVERGENCE_CATEGORIES` and the
`DivergenceCategory` `Literal`. Do not rearrange — downstream sort-stability
relies on it.
