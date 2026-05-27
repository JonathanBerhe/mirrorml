"""Wrapper objects for the Polars tracer.

Mirrors the pandas wrapper tracer: the user's pipeline runs against proxy
``_TraceLazyFrame`` and expression objects that intercept the Polars API
and record canonical :class:`~mirrorml.fingerprint.schema.Operation`
instances. Predicate and aggregation rendering is shared with the pandas
tracer via :mod:`mirrorml.tracers._trace_common`, so an equivalent Polars
and pandas / SQL pipeline produce byte-identical ``Filter.predicate``
strings and identical ``Aggregate`` ops. That parity is what lets the
diff engine return ``()`` for cross-framework equivalent pipelines
(PAPER.md C4).

The pipeline is invoked as ``pipeline(frame, pl)`` where ``pl`` is a
tracing namespace exposing ``col`` and ``lit``. Polars expressions are
namespace-level (``pl.col("x")``), not frame-attached, so the namespace
is passed explicitly. This also keeps the wrappers from importing the
real ``polars`` package (the pandas wrappers likewise never import
pandas), preserving the < 200ms import budget.

Phase 1 scope mirrors pandas phase 1a/1b: ``Source``, ``Filter`` (via
``pl.col(...) <op> literal``), ``Project`` (``select`` / ``rename``), and
``Aggregate`` (``group_by(...).agg(...)``). ``with_columns``, joins,
sorts, and window functions land in later phases.
"""

from __future__ import annotations

from typing import Literal

from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint.operations import Aggregate, Filter, Project, Sort, Source, Window
from mirrorml.fingerprint.schema import ColumnSpec, Operation, SchemaDelta, TemporalSemantics
from mirrorml.tracers._trace_common import (
    TracePredicate,
    aggregation_output_dtype,
    next_op_index,
    render_literal,
    resolve_agg_func,
    sort_directions,
)

# Polars rolling `closed` values -> our TemporalSemantics.closed vocabulary.
_CLOSED_MAP: dict[str, Literal["left", "right", "both", "neither"]] = {
    "left": "left",
    "right": "right",
    "both": "both",
    "none": "neither",
}

# Polars reduction-method names -> canonical reduction names. The canonical
# set is shared with the pandas and SQL tracers, so an Aggregate emitted
# from any side diffs to () when the structure matches.
_CANONICAL_AGG: dict[str, str] = {
    "sum": "sum",
    "mean": "mean",
    "min": "min",
    "max": "max",
    "count": "count",
    "n_unique": "count_distinct",
    "median": "median",
    "first": "first",
    "last": "last",
    "std": "std",
    "var": "var",
}


class _TraceLit:
    """A literal wrapped by ``pl.lit(...)``. Unwrapped at predicate render."""

    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value


def _unwrap(other: object) -> object:
    return other.value if isinstance(other, _TraceLit) else other


def _flatten(items: tuple[object, ...]) -> list[object]:
    """Flatten one level of list/tuple args. Polars accepts both
    ``select("a", "b")`` and ``select(["a", "b"])``; normalize to a flat
    list so callers handle one shape."""

    out: list[object] = []
    for item in items:
        if isinstance(item, list | tuple):
            out.extend(item)
        else:
            out.append(item)
    return out


class _TraceAggExpr:
    """A captured aggregation: input column, canonical func, output name.

    Output name defaults to the input column name, matching Polars's own
    default (``pl.col("score").mean()`` yields a column named ``score``);
    ``.alias(...)`` overrides it.
    """

    __slots__ = ("_func", "_input", "_output")

    def __init__(self, *, input_col: str, func: str, output: str) -> None:
        self._input = input_col
        self._func = func
        self._output = output

    def alias(self, name: str) -> _TraceAggExpr:
        if not isinstance(name, str):
            raise UnsupportedOperationError(
                f"polars tracer: alias() expects a string; got {type(name).__name__}"
            )
        return _TraceAggExpr(input_col=self._input, func=self._func, output=name)


class _TraceColExpr:
    """A captured column reference (``pl.col("x")``).

    Comparison operators build predicates (SQL form, shared with the
    pandas tracer); reduction methods build aggregations; ``alias``
    renames the column for projection.
    """

    __slots__ = ("_name", "_output")

    def __init__(self, name: str, *, output: str | None = None) -> None:
        self._name = name
        self._output = output if output is not None else name

    @property
    def input_name(self) -> str:
        return self._name

    @property
    def output_name(self) -> str:
        return self._output

    # --- projection / rename -------------------------------------------------

    def alias(self, name: str) -> _TraceColExpr:
        if not isinstance(name, str):
            raise UnsupportedOperationError(
                f"polars tracer: alias() expects a string; got {type(name).__name__}"
            )
        return _TraceColExpr(self._name, output=name)

    # --- predicates ----------------------------------------------------------

    def __gt__(self, other: object) -> TracePredicate:
        return TracePredicate(f"{self._name} > {render_literal(_unwrap(other))}")

    def __lt__(self, other: object) -> TracePredicate:
        return TracePredicate(f"{self._name} < {render_literal(_unwrap(other))}")

    def __ge__(self, other: object) -> TracePredicate:
        return TracePredicate(f"{self._name} >= {render_literal(_unwrap(other))}")

    def __le__(self, other: object) -> TracePredicate:
        return TracePredicate(f"{self._name} <= {render_literal(_unwrap(other))}")

    def __eq__(self, other: object) -> TracePredicate:  # type: ignore[override]
        return TracePredicate(f"{self._name} = {render_literal(_unwrap(other))}")

    def __ne__(self, other: object) -> TracePredicate:  # type: ignore[override]
        return TracePredicate(f"{self._name} <> {render_literal(_unwrap(other))}")

    __hash__ = None  # type: ignore[assignment]

    # --- aggregations --------------------------------------------------------

    def _agg(self, polars_name: str) -> _TraceAggExpr:
        canonical = resolve_agg_func(polars_name, name_map=_CANONICAL_AGG, framework="polars")
        return _TraceAggExpr(input_col=self._name, func=canonical, output=self._output)

    def sum(self) -> _TraceAggExpr:
        return self._agg("sum")

    def mean(self) -> _TraceAggExpr:
        return self._agg("mean")

    def min(self) -> _TraceAggExpr:
        return self._agg("min")

    def max(self) -> _TraceAggExpr:
        return self._agg("max")

    def count(self) -> _TraceAggExpr:
        return self._agg("count")

    def n_unique(self) -> _TraceAggExpr:
        return self._agg("n_unique")

    def median(self) -> _TraceAggExpr:
        return self._agg("median")

    def std(self) -> _TraceAggExpr:
        return self._agg("std")

    def var(self) -> _TraceAggExpr:
        return self._agg("var")

    def first(self) -> _TraceAggExpr:
        return self._agg("first")

    def last(self) -> _TraceAggExpr:
        return self._agg("last")


class _TraceExprNamespace:
    """The injected ``pl`` namespace. Exposes the supported slice of the
    Polars expression API (``col``, ``lit``)."""

    def col(self, name: str, *more: str) -> _TraceColExpr:
        if more:
            raise UnsupportedOperationError(
                "polars tracer: multi-column pl.col('a', 'b') is not yet "
                "supported; reference one column per pl.col(...)."
            )
        if not isinstance(name, str):
            raise UnsupportedOperationError(
                f"polars tracer: pl.col(...) expects a column name string; "
                f"got {type(name).__name__}"
            )
        return _TraceColExpr(name)

    def lit(self, value: object) -> _TraceLit:
        return _TraceLit(value)


class _TraceLazyFrame:
    """Proxy ``LazyFrame``. Each supported method records an Operation and
    returns a derived frame carrying the post-op schema.

    The frame carries a mutable schema dict (current column -> dtype), a
    shared operations list that all derived frames append to, and the
    op_id of the operation that produced this frame.
    """

    __slots__ = ("_last_op_id", "_operations", "_schema")

    def __init__(
        self,
        *,
        schema: dict[str, str],
        operations: list[Operation],
        last_op_id: str,
    ) -> None:
        self._schema = schema
        self._operations = operations
        self._last_op_id = last_op_id

    @property
    def columns(self) -> list[str]:
        return list(self._schema)

    @property
    def schema(self) -> dict[str, str]:
        return dict(self._schema)

    @property
    def dtypes(self) -> dict[str, str]:
        return dict(self._schema)

    def _derive(self, schema: dict[str, str], last_op_id: str) -> _TraceLazyFrame:
        return _TraceLazyFrame(
            schema=schema,
            operations=self._operations,
            last_op_id=last_op_id,
        )

    def filter(self, *predicates: object) -> _TraceLazyFrame:
        if not predicates:
            raise UnsupportedOperationError("polars tracer: filter() needs at least one predicate")
        preds: list[TracePredicate] = []
        for p in _flatten(predicates):
            if not isinstance(p, TracePredicate):
                raise UnsupportedOperationError(
                    f"polars tracer: filter() expects pl.col(...) comparison "
                    f"predicates; got {type(p).__name__}. Keyword-style filters "
                    f"and boolean Series masks are not yet supported."
                )
            preds.append(p)
        combined = preds[0]
        for p in preds[1:]:
            combined = combined & p

        op_id = f"filter_{next_op_index(self._operations)}"
        self._operations.append(
            Filter(
                op_id=op_id,
                dependencies=(self._last_op_id,),
                predicate=combined.render(),
            )
        )
        return self._derive(dict(self._schema), op_id)

    def select(self, *exprs: object) -> _TraceLazyFrame:
        cols = _flatten(exprs)
        if not cols:
            raise UnsupportedOperationError("polars tracer: select() needs at least one column")

        new_columns: list[str] = []
        new_schema: dict[str, str] = {}
        renames: list[tuple[str, str]] = []
        for c in cols:
            if isinstance(c, str):
                in_name, out_name = c, c
            elif isinstance(c, _TraceColExpr):
                in_name, out_name = c.input_name, c.output_name
            else:
                raise UnsupportedOperationError(
                    f"polars tracer: select() expects column names or "
                    f"pl.col(...) expressions; got {type(c).__name__}. "
                    f"Computed expressions are not yet supported."
                )
            if in_name not in self._schema:
                raise UnsupportedOperationError(
                    f"polars tracer: select references column {in_name!r}, "
                    f"which is not in the frame. Available: {sorted(self._schema)}"
                )
            new_columns.append(out_name)
            new_schema[out_name] = self._schema[in_name]
            if out_name != in_name:
                renames.append((in_name, out_name))

        op_id = f"project_{next_op_index(self._operations)}"
        self._operations.append(
            Project(
                op_id=op_id,
                dependencies=(self._last_op_id,),
                columns=tuple(new_columns),
                schema_delta=SchemaDelta(renamed=tuple(renames)),
            )
        )
        return self._derive(new_schema, op_id)

    def rename(self, mapping: object) -> _TraceLazyFrame:
        """Rename via a ``{old: new}`` mapping. Emits a :class:`Project`
        that carries every current column (in order, renames applied) and
        records the mapping in ``schema_delta.renamed`` (byte-aligned with
        the pandas ``rename`` and SQL ``AS`` paths)."""

        if not isinstance(mapping, dict):
            raise UnsupportedOperationError(
                f"polars tracer: rename expects a {{old: new}} mapping; "
                f"got {type(mapping).__name__}"
            )
        for old, new in mapping.items():
            if not isinstance(old, str) or not isinstance(new, str):
                raise UnsupportedOperationError(
                    f"polars tracer: rename mapping must use string keys and "
                    f"values; got {type(old).__name__}->{type(new).__name__}"
                )
            if old not in self._schema:
                raise UnsupportedOperationError(
                    f"polars tracer: rename source column {old!r} not in frame. "
                    f"Available: {sorted(self._schema)}"
                )

        new_columns: list[str] = []
        new_schema: dict[str, str] = {}
        renames: list[tuple[str, str]] = []
        for col, dtype in self._schema.items():
            output_name = mapping.get(col, col)
            new_columns.append(output_name)
            new_schema[output_name] = dtype
            if output_name != col:
                renames.append((col, output_name))

        op_id = f"project_{next_op_index(self._operations)}"
        self._operations.append(
            Project(
                op_id=op_id,
                dependencies=(self._last_op_id,),
                columns=tuple(new_columns),
                schema_delta=SchemaDelta(renamed=tuple(renames)),
            )
        )
        return self._derive(new_schema, op_id)

    def group_by(self, *by: object) -> _TraceGroupBy:
        keys = _flatten(by)
        if not keys:
            raise UnsupportedOperationError("polars tracer: group_by() needs at least one key")
        key_names: list[str] = []
        for k in keys:
            if isinstance(k, str):
                name = k
            elif isinstance(k, _TraceColExpr):
                name = k.input_name
            else:
                raise UnsupportedOperationError(
                    f"polars tracer: group_by key must be a column name or "
                    f"pl.col(...); got {type(k).__name__}"
                )
            if name not in self._schema:
                raise UnsupportedOperationError(
                    f"polars tracer: group_by key {name!r} not in frame. "
                    f"Available: {sorted(self._schema)}"
                )
            key_names.append(name)
        return _TraceGroupBy(frame=self, by=tuple(key_names))

    def groupby(self, *by: object) -> _TraceGroupBy:
        """Alias for :meth:`group_by` (Polars < 0.20 spelling)."""
        return self.group_by(*by)

    def sort(self, by: object, *more_by: object, descending: object = False) -> _TraceLazyFrame:
        """Emit a :class:`Sort` op. ``by`` (plus any ``*more_by``) are column
        names or ``pl.col(...)`` expressions; ``descending`` is a bool or a
        per-column list of bools (Polars semantics). Output schema is
        unchanged."""

        names: list[str] = []
        for key in _flatten((by, *more_by)):
            if isinstance(key, str):
                names.append(key)
            elif isinstance(key, _TraceColExpr):
                names.append(key.input_name)
            else:
                raise UnsupportedOperationError(
                    f"polars tracer: sort key must be a column name or pl.col(...); "
                    f"got {type(key).__name__}"
                )
        for name in names:
            if name not in self._schema:
                raise UnsupportedOperationError(
                    f"polars tracer: sort column {name!r} not in frame. "
                    f"Available: {sorted(self._schema)}"
                )

        if isinstance(descending, bool):
            ascending_flags = [not descending] * len(names)
        elif isinstance(descending, list):
            if len(descending) != len(names):
                raise UnsupportedOperationError(
                    "polars tracer: sort descending list length must match the key count"
                )
            if not all(isinstance(d, bool) for d in descending):
                raise UnsupportedOperationError(
                    "polars tracer: sort descending list must contain bools"
                )
            ascending_flags = [not d for d in descending]
        else:
            raise UnsupportedOperationError(
                f"polars tracer: sort descending must be a bool or list of bools; "
                f"got {type(descending).__name__}"
            )

        op_id = f"sort_{next_op_index(self._operations)}"
        self._operations.append(
            Sort(
                op_id=op_id,
                dependencies=(self._last_op_id,),
                by=sort_directions(names, ascending_flags),
            )
        )
        return self._derive(dict(self._schema), op_id)

    def rolling(
        self,
        index_column: object,
        *,
        period: object,
        offset: object = None,
        closed: object = "right",
        group_by: object = None,
    ) -> _TraceRolling:
        """Open a time-based rolling window. ``.agg(...)`` emits the
        :class:`Window` op. Mirrors ``polars.LazyFrame.rolling``; the
        explicit ``closed`` boundary is what makes ``window_boundary``
        detectable. ``offset`` is not yet modeled and must be left unset."""

        if isinstance(index_column, _TraceColExpr):
            index_name = index_column.input_name
        elif isinstance(index_column, str):
            index_name = index_column
        else:
            raise UnsupportedOperationError(
                f"polars tracer: rolling index_column must be a column name or "
                f"pl.col(...); got {type(index_column).__name__}"
            )
        if index_name not in self._schema:
            raise UnsupportedOperationError(
                f"polars tracer: rolling index_column {index_name!r} not in frame. "
                f"Available: {sorted(self._schema)}"
            )
        if not isinstance(period, str):
            raise UnsupportedOperationError(
                "polars tracer: rolling period must be a duration string like '3d'."
            )
        if offset is not None:
            raise UnsupportedOperationError(
                "polars tracer: rolling offset is not yet modeled; leave it unset."
            )
        if not isinstance(closed, str) or closed not in _CLOSED_MAP:
            raise UnsupportedOperationError(
                f"polars tracer: rolling closed must be one of {sorted(_CLOSED_MAP)}; "
                f"got {closed!r}"
            )

        over: list[str] = []
        if group_by is not None:
            for key in _flatten((group_by,)):
                if isinstance(key, str):
                    name = key
                elif isinstance(key, _TraceColExpr):
                    name = key.input_name
                else:
                    raise UnsupportedOperationError(
                        f"polars tracer: rolling group_by must be a column name or "
                        f"pl.col(...); got {type(key).__name__}"
                    )
                if name not in self._schema:
                    raise UnsupportedOperationError(
                        f"polars tracer: rolling group_by key {name!r} not in frame. "
                        f"Available: {sorted(self._schema)}"
                    )
                over.append(name)

        return _TraceRolling(
            frame=self,
            index_column=index_name,
            period=period,
            closed=_CLOSED_MAP[closed],
            over=tuple(over),
        )


class _TraceGroupBy:
    """A captured group_by. ``.agg(*exprs)`` emits the :class:`Aggregate`."""

    __slots__ = ("_by", "_frame")

    def __init__(self, *, frame: _TraceLazyFrame, by: tuple[str, ...]) -> None:
        self._frame = frame
        self._by = by

    def agg(self, *exprs: object) -> _TraceLazyFrame:
        agg_exprs = _flatten(exprs)
        if not agg_exprs:
            raise UnsupportedOperationError(
                "polars tracer: group_by().agg() needs at least one aggregation"
            )

        aggregations: list[tuple[str, str | None, str]] = []
        for e in agg_exprs:
            if not isinstance(e, _TraceAggExpr):
                raise UnsupportedOperationError(
                    f"polars tracer: group_by().agg() expects aggregation "
                    f"expressions like pl.col('x').mean(); got {type(e).__name__}"
                )
            if e._input not in self._frame._schema:
                raise UnsupportedOperationError(
                    f"polars tracer: agg references column {e._input!r}, which "
                    f"is not in the frame. Available: {sorted(self._frame._schema)}"
                )
            if e._input in self._by:
                raise UnsupportedOperationError(
                    f"polars tracer: agg target {e._input!r} is also a group_by "
                    f"key. Aggregate a non-key column."
                )
            aggregations.append((e._output, e._input, e._func))

        op_id = f"aggregate_{next_op_index(self._frame._operations)}"
        self._frame._operations.append(
            Aggregate(
                op_id=op_id,
                dependencies=(self._frame._last_op_id,),
                by=self._by,
                aggregations=tuple(aggregations),
            )
        )

        output_schema: dict[str, str] = {key: self._frame._schema[key] for key in self._by}
        for output_col, input_col, func in aggregations:
            output_schema[output_col] = aggregation_output_dtype(
                func, input_col, self._frame._schema
            )

        return self._frame._derive(output_schema, op_id)


class _TraceRolling:
    """A captured time-based rolling window (``LazyFrame.rolling``).

    ``.agg(*exprs)`` emits the :class:`Window` op. The output schema, like
    Polars's, is the partition keys, then the index column, then the
    aggregation outputs."""

    __slots__ = ("_closed", "_frame", "_index_column", "_over", "_period")

    def __init__(
        self,
        *,
        frame: _TraceLazyFrame,
        index_column: str,
        period: str,
        closed: Literal["left", "right", "both", "neither"],
        over: tuple[str, ...],
    ) -> None:
        self._frame = frame
        self._index_column = index_column
        self._period = period
        self._closed = closed
        self._over = over

    def agg(self, *exprs: object) -> _TraceLazyFrame:
        agg_exprs = _flatten(exprs)
        if not agg_exprs:
            raise UnsupportedOperationError(
                "polars tracer: rolling().agg() needs at least one aggregation"
            )

        aggregations: list[tuple[str, str | None, str]] = []
        for e in agg_exprs:
            if not isinstance(e, _TraceAggExpr):
                raise UnsupportedOperationError(
                    f"polars tracer: rolling().agg() expects aggregation expressions "
                    f"like pl.col('x').mean(); got {type(e).__name__}"
                )
            if e._input not in self._frame._schema:
                raise UnsupportedOperationError(
                    f"polars tracer: agg references column {e._input!r}, which is not "
                    f"in the frame. Available: {sorted(self._frame._schema)}"
                )
            aggregations.append((e._output, e._input, e._func))

        op_id = f"window_{next_op_index(self._frame._operations)}"
        self._frame._operations.append(
            Window(
                op_id=op_id,
                dependencies=(self._frame._last_op_id,),
                over=self._over,
                order_by=(self._index_column,),
                size=self._period,
                aggregations=tuple(aggregations),
                temporal=TemporalSemantics(closed=self._closed),
            )
        )

        output_schema: dict[str, str] = {key: self._frame._schema[key] for key in self._over}
        output_schema[self._index_column] = self._frame._schema[self._index_column]
        for output_col, input_col, func in aggregations:
            output_schema[output_col] = aggregation_output_dtype(
                func, input_col, self._frame._schema
            )

        return self._frame._derive(output_schema, op_id)


def build_initial_frame(
    *,
    source_name: str,
    input_schema: tuple[ColumnSpec, ...],
) -> tuple[_TraceLazyFrame, list[Operation]]:
    """Build the initial ``_TraceLazyFrame`` and its Source operation.

    The returned operations list is shared with the frame; derived frames
    append to it as the pipeline runs.
    """

    operations: list[Operation] = []
    source = Source(op_id="source_0", name=source_name, columns=input_schema)
    operations.append(source)

    frame = _TraceLazyFrame(
        schema=dict(input_schema),
        operations=operations,
        last_op_id=source.op_id,
    )
    return frame, operations
