"""Divergence classifier. Maps structural fingerprint differences to one
of the fifteen closed taxonomy categories in
``docs/concepts/divergence_taxonomy.md``.

This module is the home of all rule-style logic. The engine in
:mod:`mirrorml.diff.engine` orchestrates the walk over two fingerprints
and calls the helpers here for each interesting comparison. Keeping the
rules separate lets the test suite exercise them in isolation and lets
the engine stay focused on alignment.

In M3 phase 1 the classifier covers seven categories that are reachable
from SQL-only fingerprints: ``schema_drift``, ``type_coercion``,
``timezone_mismatch``, ``rounding_precision``, ``aggregation_function``,
``join_key_mismatch``, and ``ordering_dependence``. Window-boundary,
as-of join direction, null handling, categorical encoding, seed,
feature-leakage-temporal, and unit-mismatch land in later phases when
the relevant ops are emitted by a tracer or when whole-graph reasoning
is added.
"""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict, Field

from mirrorml._taxonomy import DivergenceCategory
from mirrorml.fingerprint._typing import ColumnName, OpId
from mirrorml.fingerprint.dtypes import parse_dtype
from mirrorml.fingerprint.schema import ColumnSpec, Operation


class Divergence(BaseModel):
    """A single classified disagreement between two fingerprints.

    ``category`` is drawn from the closed taxonomy of fifteen labels (see
    ``docs/concepts/divergence_taxonomy.md``). ``left_op_id`` and
    ``right_op_id`` locate the responsible operation on each side; either
    may be ``None`` when the responsible op exists on only one side, or
    when the divergence is schema-level rather than op-level.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: DivergenceCategory
    left_op_id: OpId | None = None
    right_op_id: OpId | None = None
    detail: str = Field(
        default="",
        description="Human-readable explanation suitable for CLI output.",
    )


# --- dtype-level classification ----------------------------------------------


def classify_dtype_difference(
    column: ColumnName, left_dtype: str, right_dtype: str, *, location: str
) -> Divergence:
    """Classify the difference between two dtypes assigned to the same column.

    The mapping is:

    * Both ``timestamp`` with different timezones -> ``timezone_mismatch``.
    * Same kind, different unit (``time`` / ``timestamp`` / ``duration``)
      -> ``rounding_precision``.
    * Both ``decimal`` with different precision or scale ->
      ``rounding_precision``.
    * Anything else -> ``type_coercion``.

    ``location`` is a free-form label (``"input"``, ``"output"``, or
    ``"op:<op_id>"``) that gets embedded in the diagnostic.
    """

    left = parse_dtype(left_dtype)
    right = parse_dtype(right_dtype)

    if left.kind == "timestamp" and right.kind == "timestamp" and left.timezone != right.timezone:
        return Divergence(
            category="timezone_mismatch",
            detail=(
                f"column {column!r} ({location}): timezone "
                f"{left.timezone!r} vs {right.timezone!r} "
                f"({left_dtype} vs {right_dtype})"
            ),
        )

    if left.kind == right.kind and left.unit is not None and left.unit != right.unit:
        return Divergence(
            category="rounding_precision",
            detail=(
                f"column {column!r} ({location}): {left.kind} unit "
                f"{left.unit!r} vs {right.unit!r} "
                f"({left_dtype} vs {right_dtype})"
            ),
        )

    if (
        left.kind == "decimal"
        and right.kind == "decimal"
        and (left.precision, left.scale) != (right.precision, right.scale)
    ):
        return Divergence(
            category="rounding_precision",
            detail=(
                f"column {column!r} ({location}): decimal precision/scale "
                f"({left.precision}, {left.scale}) vs "
                f"({right.precision}, {right.scale})"
            ),
        )

    return Divergence(
        category="type_coercion",
        detail=(f"column {column!r} ({location}): dtype {left_dtype} vs {right_dtype}"),
    )


# --- schema-level comparison -------------------------------------------------


def compare_schemas(
    left: tuple[ColumnSpec, ...],
    right: tuple[ColumnSpec, ...],
    *,
    location: str,
    left_op_id: OpId | None = None,
    right_op_id: OpId | None = None,
) -> Iterator[Divergence]:
    """Diff two column lists.

    Columns present on only one side produce ``schema_drift`` divergences;
    common columns with different dtypes are routed through
    :func:`classify_dtype_difference`.

    ``left_op_id`` and ``right_op_id`` are the ops responsible for the
    schemas under comparison: the engine passes the Source op_ids when
    diffing input schemas, and the terminal-op op_ids when diffing output
    schemas. Embedding them lets downstream consumers (the CLI renderer,
    the bench localization metric) point at the op that owns the column
    list rather than treating schema divergences as un-localizable.

    Order of the output: drops (left-only) first, then adds (right-only),
    then per-common-column dtype divergences in left order. The ordering
    is deterministic so diff output is reproducible.
    """

    left_dict = dict(left)
    right_dict = dict(right)

    for col, dtype in left:
        if col not in right_dict:
            yield Divergence(
                category="schema_drift",
                left_op_id=left_op_id,
                right_op_id=right_op_id,
                detail=(
                    f"column {col!r} ({location}, dtype {dtype}) is present "
                    f"on the left but not the right"
                ),
            )

    for col, dtype in right:
        if col not in left_dict:
            yield Divergence(
                category="schema_drift",
                left_op_id=left_op_id,
                right_op_id=right_op_id,
                detail=(
                    f"column {col!r} ({location}, dtype {dtype}) is present "
                    f"on the right but not the left"
                ),
            )

    for col, left_dtype in left:
        right_dtype = right_dict.get(col)
        if right_dtype is None or right_dtype == left_dtype:
            continue
        div = classify_dtype_difference(col, left_dtype, right_dtype, location=location)
        yield div.model_copy(update={"left_op_id": left_op_id, "right_op_id": right_op_id})


# --- op-pair classification --------------------------------------------------


def classify_op_pair(left: Operation, right: Operation) -> Iterator[Divergence]:
    """Yield divergences for two ops aligned at the same pipeline position.

    Skips identical pairs. Dispatches on ``kind``; kinds without specific
    rules emit a fallback ``schema_drift`` divergence so the disagreement
    is at least surfaced (a missed divergence is worse than a false
    positive the user can suppress).
    """

    if left == right:
        return

    if left.kind != right.kind:
        yield Divergence(
            category="schema_drift",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=(f"op kinds differ at the same position: {left.kind!r} vs {right.kind!r}"),
        )
        return

    kind = left.kind
    if kind == "aggregate":
        yield from _classify_aggregate(left, right)
    elif kind == "join":
        yield from _classify_join(left, right)
    elif kind == "as_of_join":
        yield from _classify_as_of_join(left, right)
    elif kind == "sort":
        yield from _classify_sort(left, right)
    elif kind == "window":
        yield from _classify_window(left, right)
    elif kind == "source":
        yield from _classify_source(left, right)
    elif kind == "project":
        # Project column lists are reflected in output_schema, so the
        # schema diff catches add/drop/reorder. Renames live in
        # schema_delta.renamed and are surfaced here when they differ
        # (the schema diff alone cannot distinguish rename from
        # drop+add when both names happen to be unique).
        yield from _classify_project(left, right)
    elif kind == "filter":
        yield from _classify_filter(left, right)
    elif kind == "fill_na":
        yield from _classify_fill_na(left, right)
    elif kind == "udf":
        yield from _classify_udf(left, right)
    else:
        yield Divergence(
            category="schema_drift",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=f"{kind!r} ops differ but classifier rule is not yet implemented",
        )


def _classify_source(left: Operation, right: Operation) -> Iterator[Divergence]:
    """Compare two Source ops. Column / dtype changes are surfaced via the
    schema-level diff; the only op-local change worth flagging is the
    table name (a different source table is a meaningful semantic change)."""

    assert left.kind == "source" and right.kind == "source"
    if left.name != right.name:
        yield Divergence(
            category="schema_drift",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=f"source table name: {left.name!r} vs {right.name!r}",
        )


def _classify_filter(left: Operation, right: Operation) -> Iterator[Divergence]:
    """Compare two Filter ops. A change to the predicate is a real
    divergence that does not fit any single taxonomy category cleanly:
    if the predicate mentions ``NULL`` / ``IS NULL`` we map it to
    ``null_handling``; otherwise the predicate change is surfaced as a
    ``schema_drift`` fallback so it cannot silently vanish (a missed
    divergence is worse than an imperfect classification)."""

    assert left.kind == "filter" and right.kind == "filter"
    if left.predicate == right.predicate:
        return

    detail = f"filter predicate: {left.predicate!r} vs {right.predicate!r}"
    if _mentions_null(left.predicate) or _mentions_null(right.predicate):
        yield Divergence(
            category="null_handling",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=detail,
        )
        return
    yield Divergence(
        category="schema_drift",
        left_op_id=left.op_id,
        right_op_id=right.op_id,
        detail=detail,
    )


def _classify_project(left: Operation, right: Operation) -> Iterator[Divergence]:
    """Compare two Project ops. Most differences are caught by the
    schema diff; what is op-local is the rename mapping in
    schema_delta. A rename change without a column-set change still
    matters semantically (the same column is now exposed under a
    different name)."""

    assert left.kind == "project" and right.kind == "project"
    if left.schema_delta.renamed != right.schema_delta.renamed:
        yield Divergence(
            category="schema_drift",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=(
                f"project renames: {left.schema_delta.renamed} vs {right.schema_delta.renamed}"
            ),
        )


def _classify_udf(left: Operation, right: Operation) -> Iterator[Divergence]:
    """Compare two :class:`~mirrorml.fingerprint.operations.Udf` ops.

    The static side cannot inspect a callable's semantics, so the
    canonical signal is the :class:`UdfRef` source-hash: same hash means
    the two callables normalize to the same Python source (modulo
    whitespace / comments / docstrings). Different hashes mean the
    bodies actually differ, which is a real divergence the diff layer
    must surface. We route it through ``schema_drift`` as the
    "couldn't-pin-it-to-a-finer-category" bucket; the statistical
    companion check is the right way to upgrade the diagnosis when the
    user wants to know whether the value-level outputs also disagree.

    Input / output column-set changes are op-local and surface here too
    (they fall under ``schema_drift`` by definition).
    """

    assert left.kind == "udf" and right.kind == "udf"

    if left.ref.source_hash_algorithm != right.ref.source_hash_algorithm:
        yield Divergence(
            category="schema_drift",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=(
                f"udf source-hash algorithm: "
                f"{left.ref.source_hash_algorithm!r} vs "
                f"{right.ref.source_hash_algorithm!r} (cannot compare hashes "
                f"across algorithm versions; run a migration first)"
            ),
        )
        return

    if left.ref.source_hash != right.ref.source_hash:
        yield Divergence(
            category="schema_drift",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=(
                f"udf body differs: {left.ref.qualname!r} "
                f"({left.ref.source_hash[:12]}...) vs "
                f"{right.ref.qualname!r} ({right.ref.source_hash[:12]}...). "
                f"Static analysis cannot decide whether they compute the "
                f"same value; run the statistical companion check to find out."
            ),
        )

    if left.input_columns != right.input_columns:
        yield Divergence(
            category="schema_drift",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=f"udf input_columns: {left.input_columns} vs {right.input_columns}",
        )

    if left.output_columns != right.output_columns:
        yield Divergence(
            category="schema_drift",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=f"udf output_columns: {left.output_columns} vs {right.output_columns}",
        )


def _classify_fill_na(left: Operation, right: Operation) -> Iterator[Divergence]:
    """Compare two FillNa ops. Every difference (filled columns, fill value,
    or fill strategy) is a ``null_handling`` divergence: filling nulls with
    0 offline and the mean online is a textbook training-serving skew."""

    assert left.kind == "fill_na" and right.kind == "fill_na"
    if left.columns != right.columns:
        yield Divergence(
            category="null_handling",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=f"fill_na columns: {left.columns} vs {right.columns}",
        )
    if left.value != right.value:
        yield Divergence(
            category="null_handling",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=f"fill_na value: {left.value!r} vs {right.value!r}",
        )
    if left.strategy != right.strategy:
        yield Divergence(
            category="null_handling",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=f"fill_na strategy: {left.strategy!r} vs {right.strategy!r}",
        )


def _mentions_null(predicate: object) -> bool:
    """Heuristic: does a predicate reference null handling? Used to route
    Filter divergences to the ``null_handling`` category when applicable."""

    if not isinstance(predicate, str):
        return False
    upper = predicate.upper()
    return "NULL" in upper or "COALESCE" in upper or "IFNULL" in upper


def _classify_aggregate(left: Operation, right: Operation) -> Iterator[Divergence]:
    """Compare two Aggregate ops. Different group keys are
    ``join_key_mismatch`` (the grouping is the equivalent concept).
    Different functions on the same output column are
    ``aggregation_function``."""

    assert left.kind == "aggregate" and right.kind == "aggregate"
    if left.by != right.by:
        yield Divergence(
            category="join_key_mismatch",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=f"GROUP BY keys differ: {left.by} vs {right.by}",
        )

    left_aggs = {output: (input_col, func) for output, input_col, func in left.aggregations}
    right_aggs = {output: (input_col, func) for output, input_col, func in right.aggregations}

    for output_col, (left_input, left_func) in left_aggs.items():
        if output_col not in right_aggs:
            yield Divergence(
                category="schema_drift",
                left_op_id=left.op_id,
                right_op_id=right.op_id,
                detail=f"aggregation {output_col!r} present on left but not right",
            )
            continue
        right_input, right_func = right_aggs[output_col]
        if left_func != right_func:
            yield Divergence(
                category="aggregation_function",
                left_op_id=left.op_id,
                right_op_id=right.op_id,
                detail=(f"aggregation {output_col!r}: function {left_func!r} vs {right_func!r}"),
            )
        if left_input != right_input:
            yield Divergence(
                category="aggregation_function",
                left_op_id=left.op_id,
                right_op_id=right.op_id,
                detail=(
                    f"aggregation {output_col!r}: input column {left_input!r} vs {right_input!r}"
                ),
            )

    for output_col in right_aggs:
        if output_col not in left_aggs:
            yield Divergence(
                category="schema_drift",
                left_op_id=left.op_id,
                right_op_id=right.op_id,
                detail=f"aggregation {output_col!r} present on right but not left",
            )


def _classify_join(left: Operation, right: Operation) -> Iterator[Divergence]:
    """Compare two Join ops."""

    assert left.kind == "join" and right.kind == "join"
    if left.how != right.how:
        yield Divergence(
            category="schema_drift",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=f"join kind differs: {left.how!r} vs {right.how!r}",
        )
    if left.left_keys != right.left_keys or left.right_keys != right.right_keys:
        yield Divergence(
            category="join_key_mismatch",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=(
                f"join keys: ({left.left_keys}, {left.right_keys}) vs "
                f"({right.left_keys}, {right.right_keys})"
            ),
        )


def _classify_as_of_join(left: Operation, right: Operation) -> Iterator[Divergence]:
    """Compare two AsOfJoin ops. The temporal ``direction`` is the dominant
    skew source for these joins (the canonical
    ``as_of_join_direction`` category)."""

    assert left.kind == "as_of_join" and right.kind == "as_of_join"
    if left.temporal.direction != right.temporal.direction:
        yield Divergence(
            category="as_of_join_direction",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=(
                f"as-of join direction: {left.temporal.direction!r} vs {right.temporal.direction!r}"
            ),
        )
    if left.temporal.tolerance != right.temporal.tolerance:
        yield Divergence(
            category="as_of_join_direction",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=(
                f"as-of join tolerance: {left.temporal.tolerance!r} vs {right.temporal.tolerance!r}"
            ),
        )
    if left.left_keys != right.left_keys or left.right_keys != right.right_keys:
        yield Divergence(
            category="join_key_mismatch",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=(
                f"as-of join keys: ({left.left_keys}, {left.right_keys}) vs "
                f"({right.left_keys}, {right.right_keys})"
            ),
        )


def _classify_sort(left: Operation, right: Operation) -> Iterator[Divergence]:
    """Compare two Sort ops. ``ordering_dependence`` covers ``by`` differences."""

    assert left.kind == "sort" and right.kind == "sort"
    if left.by != right.by:
        yield Divergence(
            category="ordering_dependence",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=f"sort by: {left.by} vs {right.by}",
        )


def _classify_window(left: Operation, right: Operation) -> Iterator[Divergence]:
    """Compare two Window ops. ``window_size_mismatch`` and
    ``window_boundary`` are the two dominant skew sources."""

    assert left.kind == "window" and right.kind == "window"
    if left.size != right.size:
        yield Divergence(
            category="window_size_mismatch",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=f"window size: {left.size!r} vs {right.size!r}",
        )
    if left.temporal.closed != right.temporal.closed:
        yield Divergence(
            category="window_boundary",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=(
                f"window boundary: closed={left.temporal.closed!r} vs "
                f"closed={right.temporal.closed!r}"
            ),
        )
    if left.over != right.over:
        yield Divergence(
            category="join_key_mismatch",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=f"window partition keys: {left.over} vs {right.over}",
        )
    if left.order_by != right.order_by:
        yield Divergence(
            category="ordering_dependence",
            left_op_id=left.op_id,
            right_op_id=right.op_id,
            detail=f"window ORDER BY: {left.order_by} vs {right.order_by}",
        )
    yield from _classify_windowed_aggregations(left, right)


def _classify_windowed_aggregations(left: Operation, right: Operation) -> Iterator[Divergence]:
    """Compare the per-column aggregations of two Window ops.

    Mirrors :func:`_classify_aggregate`: a differing reduction function or
    input column on the same output column is an ``aggregation_function``
    divergence; an output column present on only one side is
    ``schema_drift``. Without this, a window whose aggregation changed (mean
    vs sum) but whose frame is identical would slip through silently."""

    assert left.kind == "window" and right.kind == "window"
    left_aggs = {out: (inp, func) for out, inp, func in left.aggregations}
    right_aggs = {out: (inp, func) for out, inp, func in right.aggregations}

    for out, (left_input, left_func) in left_aggs.items():
        if out not in right_aggs:
            yield Divergence(
                category="schema_drift",
                left_op_id=left.op_id,
                right_op_id=right.op_id,
                detail=f"windowed aggregation {out!r} present on left but not right",
            )
            continue
        right_input, right_func = right_aggs[out]
        if left_func != right_func:
            yield Divergence(
                category="aggregation_function",
                left_op_id=left.op_id,
                right_op_id=right.op_id,
                detail=f"windowed aggregation {out!r}: function {left_func!r} vs {right_func!r}",
            )
        if left_input != right_input:
            yield Divergence(
                category="aggregation_function",
                left_op_id=left.op_id,
                right_op_id=right.op_id,
                detail=f"windowed aggregation {out!r}: input column {left_input!r} vs {right_input!r}",
            )

    for out in right_aggs:
        if out not in left_aggs:
            yield Divergence(
                category="schema_drift",
                left_op_id=left.op_id,
                right_op_id=right.op_id,
                detail=f"windowed aggregation {out!r} present on right but not left",
            )
