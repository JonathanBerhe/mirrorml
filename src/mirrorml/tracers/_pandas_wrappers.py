"""Wrapper objects for the pandas tracer.

The pandas tracer is a wrapper-object tracer: the user's pipeline runs
against proxy ``_TraceFrame`` / ``_TraceSeries`` / ``TracePredicate``
instances that intercept the standard pandas operations and record them
as canonical :class:`~mirrorml.fingerprint.schema.Operation` instances.

Phase 1a scope: ``df[bool_mask]`` (``Filter``), ``df[col_list]`` (``Project``),
``df['col']`` (returns a ``_TraceSeries`` so comparison operators can
build predicates), and the comparison + boolean operators needed to
write realistic filter expressions. Aggregations, joins, sorts, and
column writes land in later phases.

This module does not import pandas. The wrapper API is what pandas
users *write against*, so we need the surface to look DataFrame-like,
but the tracer never sees a real ``pd.DataFrame``. pandas types only
become relevant in phase 1b when we add dtype inference from a
runtime ``example_df``.

Predicate rendering follows SQL form (``=``, ``<>``, ``AND``, ``OR``,
``NOT (...)``) so a pandas filter and the equivalent SQL filter produce
byte-identical ``Filter.predicate`` strings. This is what lets the diff
engine return ``()`` for cross-framework equivalent pipelines.
"""

from __future__ import annotations

import inspect
from typing import Any

from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint.operations import Aggregate, FillNa, Filter, Project, Sort, Source, Udf
from mirrorml.fingerprint.schema import ColumnSpec, Operation, SchemaDelta, UdfRef
from mirrorml.fingerprint.udf_hash import SOURCE_HASH_ALGORITHM, source_hash
from mirrorml.tracers._trace_common import (
    TracePredicate,
    aggregation_output_dtype,
    next_op_index,
    render_literal,
    resolve_agg_func,
    sort_directions,
)

# Maps pandas-side agg names to canonical reduction names. The SQL tracer
# uses the same canonical names so an Aggregate emitted from either side
# diffs to () when the structure matches.
_CANONICAL_AGG: dict[str, str] = {
    "sum": "sum",
    "mean": "mean",
    "min": "min",
    "max": "max",
    "count": "count",
    "nunique": "count_distinct",
    "median": "median",
    "first": "first",
    "last": "last",
    "std": "std",
    "var": "var",
}


class _TraceSeries:
    """A captured column reference. Comparison operators build predicates.

    Returning a :class:`TracePredicate` from ``__eq__`` / ``__ne__``
    mirrors pandas's own broadcast-comparison semantics; the side effect
    is that ``_TraceSeries`` is not hashable (which is correct: a Series
    is not a dict key in real pandas either).
    """

    __slots__ = ("_dtype", "_name")

    def __init__(self, name: str, dtype: str) -> None:
        self._name = name
        self._dtype = dtype

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> str:
        return self._dtype

    def __gt__(self, other: object) -> TracePredicate:
        return TracePredicate(f"{self._name} > {render_literal(other)}")

    def __lt__(self, other: object) -> TracePredicate:
        return TracePredicate(f"{self._name} < {render_literal(other)}")

    def __ge__(self, other: object) -> TracePredicate:
        return TracePredicate(f"{self._name} >= {render_literal(other)}")

    def __le__(self, other: object) -> TracePredicate:
        return TracePredicate(f"{self._name} <= {render_literal(other)}")

    # __eq__ / __ne__ deliberately diverge from object identity so
    # comparison-style usage builds predicates. _TraceSeries is treated as
    # unhashable to keep this safe.
    def __eq__(self, other: object) -> TracePredicate:  # type: ignore[override]
        return TracePredicate(f"{self._name} = {render_literal(other)}")

    def __ne__(self, other: object) -> TracePredicate:  # type: ignore[override]
        return TracePredicate(f"{self._name} <> {render_literal(other)}")

    __hash__ = None  # type: ignore[assignment]


class _TraceFrame:
    """Proxy DataFrame. ``__getitem__`` routes to Filter / Project / Series
    depending on the key type.

    The frame carries a mutable schema dict (current post-op column ->
    dtype map), a shared operations list that all derived frames append
    to, and the op_id of the operation that produced this frame.
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
    def dtypes(self) -> dict[str, str]:
        return dict(self._schema)

    def __getitem__(self, key: object) -> _TraceFrame | _TraceSeries:
        if isinstance(key, TracePredicate):
            return self._apply_filter(key)
        if isinstance(key, str):
            return self._select_column(key)
        if isinstance(key, list):
            return self._project(key)
        raise UnsupportedOperationError(
            f"pandas tracer: unsupported __getitem__ key {type(key).__name__!r}. "
            f"Supported: boolean mask, column name (str), column list."
        )

    def _apply_filter(self, predicate: TracePredicate) -> _TraceFrame:
        op_id = f"filter_{next_op_index(self._operations)}"
        self._operations.append(
            Filter(
                op_id=op_id,
                dependencies=(self._last_op_id,),
                predicate=predicate.render(),
            )
        )
        return _TraceFrame(
            schema=dict(self._schema),
            operations=self._operations,
            last_op_id=op_id,
        )

    def _select_column(self, name: str) -> _TraceSeries:
        if name not in self._schema:
            raise UnsupportedOperationError(
                f"pandas tracer: column {name!r} not in current frame. "
                f"Available: {sorted(self._schema)}"
            )
        return _TraceSeries(name, self._schema[name])

    def _project(self, columns: list[Any]) -> _TraceFrame:
        for col in columns:
            if not isinstance(col, str):
                raise UnsupportedOperationError(
                    f"pandas tracer: column list must contain strings; got {type(col).__name__}"
                )
            if col not in self._schema:
                raise UnsupportedOperationError(
                    f"pandas tracer: column {col!r} not in current frame. "
                    f"Available: {sorted(self._schema)}"
                )
        op_id = f"project_{next_op_index(self._operations)}"
        self._operations.append(
            Project(
                op_id=op_id,
                dependencies=(self._last_op_id,),
                columns=tuple(columns),
            )
        )
        return _TraceFrame(
            schema={c: self._schema[c] for c in columns},
            operations=self._operations,
            last_op_id=op_id,
        )

    def groupby(self, by: object) -> _TraceGroupBy:
        """Return a :class:`_TraceGroupBy` that records the groups and waits
        for an aggregation call to emit the :class:`Aggregate` op.
        """

        if isinstance(by, str):
            keys: tuple[str, ...] = (by,)
        elif isinstance(by, list):
            for k in by:
                if not isinstance(k, str):
                    raise UnsupportedOperationError(
                        f"pandas tracer: groupby key list must contain "
                        f"strings; got {type(k).__name__}"
                    )
            keys = tuple(by)
        else:
            raise UnsupportedOperationError(
                f"pandas tracer: groupby key must be a string or list of "
                f"strings; got {type(by).__name__}"
            )

        for k in keys:
            if k not in self._schema:
                raise UnsupportedOperationError(
                    f"pandas tracer: groupby key {k!r} not in frame. "
                    f"Available: {sorted(self._schema)}"
                )

        return _TraceGroupBy(frame=self, by=keys, selection=None)

    def rename(self, columns: dict[str, str] | None = None) -> _TraceFrame:
        """Rename columns via a mapping. Emits a :class:`Project` op that
        carries every current column (in order, with renames applied) and
        records the rename mapping in ``schema_delta.renamed``.
        """

        if columns is None:
            raise UnsupportedOperationError(
                "pandas tracer: rename requires the columns= mapping; "
                "positional / index renames are not supported."
            )
        if not isinstance(columns, dict):
            raise UnsupportedOperationError(
                f"pandas tracer: rename columns= must be a dict; got {type(columns).__name__}"
            )
        for old, new in columns.items():
            if not isinstance(old, str) or not isinstance(new, str):
                raise UnsupportedOperationError(
                    f"pandas tracer: rename mapping must use string keys "
                    f"and values; got {type(old).__name__}->"
                    f"{type(new).__name__}"
                )
            if old not in self._schema:
                raise UnsupportedOperationError(
                    f"pandas tracer: rename source column {old!r} not in "
                    f"frame. Available: {sorted(self._schema)}"
                )

        new_columns: list[str] = []
        new_schema: dict[str, str] = {}
        renames: list[tuple[str, str]] = []
        for col, dtype in self._schema.items():
            output_name = columns.get(col, col)
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
        return _TraceFrame(
            schema=new_schema,
            operations=self._operations,
            last_op_id=op_id,
        )

    def sort_values(self, by: object, ascending: object = True) -> _TraceFrame:
        """Emit a :class:`Sort` op. ``by`` is a column name or list of names;
        ``ascending`` is a bool or a per-column list of bools (pandas
        semantics). The output schema is unchanged."""

        keys = _as_column_list(by, what="sort_values by")
        for key in keys:
            if key not in self._schema:
                raise UnsupportedOperationError(
                    f"pandas tracer: sort_values column {key!r} not in frame. "
                    f"Available: {sorted(self._schema)}"
                )

        if isinstance(ascending, bool):
            flags = [ascending] * len(keys)
        elif isinstance(ascending, list):
            if len(ascending) != len(keys):
                raise UnsupportedOperationError(
                    "pandas tracer: sort_values ascending list length must match by"
                )
            if not all(isinstance(a, bool) for a in ascending):
                raise UnsupportedOperationError(
                    "pandas tracer: sort_values ascending list must contain bools"
                )
            flags = ascending
        else:
            raise UnsupportedOperationError(
                f"pandas tracer: sort_values ascending must be a bool or list of "
                f"bools; got {type(ascending).__name__}"
            )

        op_id = f"sort_{next_op_index(self._operations)}"
        self._operations.append(
            Sort(
                op_id=op_id,
                dependencies=(self._last_op_id,),
                by=sort_directions(keys, flags),
            )
        )
        return _TraceFrame(
            schema=dict(self._schema),
            operations=self._operations,
            last_op_id=op_id,
        )

    def fillna(self, value: object = None) -> _TraceFrame:
        """Emit a :class:`FillNa` op for a constant fill. ``value`` is a
        scalar (fills every column) or a ``{column: value}`` dict whose
        values are all the same constant (FillNa carries a single value).
        Method/strategy-based fills are deferred. Output schema unchanged."""

        if isinstance(value, dict):
            if not value:
                raise UnsupportedOperationError("pandas tracer: fillna dict is empty")
            columns: tuple[str, ...] = tuple(value)
            rendered: set[str] = set()
            for col, fill in value.items():
                if not isinstance(col, str):
                    raise UnsupportedOperationError(
                        f"pandas tracer: fillna dict keys must be column names; "
                        f"got {type(col).__name__}"
                    )
                if col not in self._schema:
                    raise UnsupportedOperationError(
                        f"pandas tracer: fillna column {col!r} not in frame. "
                        f"Available: {sorted(self._schema)}"
                    )
                rendered.add(render_literal(fill))
            if len(rendered) != 1:
                raise UnsupportedOperationError(
                    "pandas tracer: fillna with differing per-column values is not yet "
                    "supported (FillNa carries one value); use one value per fillna call."
                )
            fill_value = rendered.pop()
        elif value is None:
            raise UnsupportedOperationError(
                "pandas tracer: fillna needs a scalar value or a {column: value} dict; "
                "method-based fills (ffill/bfill) are deferred."
            )
        elif isinstance(value, int | float | str | bool):
            columns = tuple(self._schema)
            fill_value = render_literal(value)
        else:
            raise UnsupportedOperationError(
                f"pandas tracer: fillna value must be a scalar or a {{column: value}} dict; "
                f"got {type(value).__name__}"
            )

        op_id = f"fill_na_{next_op_index(self._operations)}"
        self._operations.append(
            FillNa(
                op_id=op_id,
                dependencies=(self._last_op_id,),
                columns=columns,
                value=fill_value,
                strategy="constant",
            )
        )
        return _TraceFrame(
            schema=dict(self._schema),
            operations=self._operations,
            last_op_id=op_id,
        )

    def apply(
        self,
        func: object,
        *,
        axis: object = 0,
        result_type: object = None,
    ) -> _TraceFrame:
        """Emit a :class:`Udf` op for ``df.apply(callable, axis=...)``.

        Static analysis cannot decide what the callable computes, so the
        UDF body is captured as a normalized source-hash on the
        :class:`~mirrorml.fingerprint.schema.UdfRef`. Two pipelines that
        apply the same callable (modulo formatting / comments) produce the
        same fingerprint; two pipelines that apply different callables
        produce different fingerprints, and the diff layer surfaces the
        mismatch.

        Output schema: ``axis=0`` reduces each column to a scalar (we model
        the result as keeping the same column set, since pandas returns a
        Series indexed by column name); ``axis=1`` reduces each row, which
        cannot be modeled without inspecting the function and is rejected.
        """

        if axis not in (0, "index"):
            raise UnsupportedOperationError(
                f"pandas tracer: df.apply(axis={axis!r}) is not supported; "
                f"axis=0 (per-column) is the only modeled form. axis=1 / "
                f"'columns' requires inspecting the callable body and is "
                f"deferred."
            )
        if result_type is not None:
            raise UnsupportedOperationError(
                f"pandas tracer: df.apply(result_type={result_type!r}) is "
                f"not supported; only the default is modeled."
            )
        if not callable(func):
            raise UnsupportedOperationError(
                f"pandas tracer: df.apply(...) needs a callable; got {type(func).__name__}"
            )
        ref = _udf_ref(func)
        input_columns = tuple(self._schema)
        output_columns = input_columns  # per-column reduction keeps the column set
        op_id = f"udf_{next_op_index(self._operations)}"
        self._operations.append(
            Udf(
                op_id=op_id,
                dependencies=(self._last_op_id,),
                ref=ref,
                input_columns=input_columns,
                output_columns=output_columns,
            )
        )
        return _TraceFrame(
            schema=dict(self._schema),
            operations=self._operations,
            last_op_id=op_id,
        )


def _udf_ref(func: object) -> UdfRef:
    """Build a :class:`UdfRef` for ``func``: source-hash plus a signature
    string. Lambdas and C extensions raise via :func:`source_hash`.
    """

    assert callable(func)
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", None) or repr(func)
    try:
        sig = str(inspect.signature(func))
    except (TypeError, ValueError):
        sig = "(?)"
    return UdfRef(
        qualname=str(qualname),
        source_hash=source_hash(func),
        signature=sig,
        source_hash_algorithm=SOURCE_HASH_ALGORITHM,
    )


def _as_column_list(value: object, *, what: str) -> list[str]:
    """Normalize a column-or-list argument to a list of column-name strings."""

    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, str):
                raise UnsupportedOperationError(
                    f"pandas tracer: {what} must contain strings; got {type(item).__name__}"
                )
        return list(value)
    raise UnsupportedOperationError(
        f"pandas tracer: {what} must be a column name or list of names; got {type(value).__name__}"
    )


class _TraceGroupBy:
    """A captured groupby. Holds the keys and an optional column selection;
    actual :class:`Aggregate` emission happens on the next method call
    (``.agg(...)``, ``.sum()``, ``.mean()``, etc.).

    The selection follows pandas semantics: ``df.groupby('k')['col']``
    narrows subsequent aggregation to one column; ``df.groupby('k')[['c1', 'c2']]``
    to several. ``df.groupby('k').agg({...})`` ignores any selection and
    uses the dict keys directly.
    """

    __slots__ = ("_by", "_frame", "_selection")

    def __init__(
        self,
        *,
        frame: _TraceFrame,
        by: tuple[str, ...],
        selection: tuple[str, ...] | None,
    ) -> None:
        self._frame = frame
        self._by = by
        self._selection = selection

    def __getitem__(self, key: object) -> _TraceGroupBy:
        if isinstance(key, str):
            selection: tuple[str, ...] = (key,)
        elif isinstance(key, list):
            for col in key:
                if not isinstance(col, str):
                    raise UnsupportedOperationError(
                        f"pandas tracer: groupby selection list must "
                        f"contain strings; got {type(col).__name__}"
                    )
            selection = tuple(key)
        else:
            raise UnsupportedOperationError(
                f"pandas tracer: groupby selection must be a string or "
                f"list of strings; got {type(key).__name__}"
            )

        for col in selection:
            if col not in self._frame._schema:
                raise UnsupportedOperationError(
                    f"pandas tracer: groupby selection {col!r} not in "
                    f"frame. Available: {sorted(self._frame._schema)}"
                )
            if col in self._by:
                raise UnsupportedOperationError(
                    f"pandas tracer: groupby selection {col!r} is also a "
                    f"groupby key. Select a non-key column to aggregate."
                )

        return _TraceGroupBy(frame=self._frame, by=self._by, selection=selection)

    def agg(self, spec: object) -> _TraceFrame:
        """Build and emit the :class:`Aggregate` op.

        Accepts:

        * ``{col: func}`` -- per-column aggregations. ``func`` is one of
          the canonical reduction names (``"sum"``, ``"mean"``, etc.).
        * ``"sum"`` (or another canonical name) -- applies the same
          aggregation to each column in the current selection. Requires
          a column selection (``groupby('k')[...].agg("sum")``).
        """

        aggregations: list[tuple[str, str | None, str]] = []

        if callable(spec) and not isinstance(spec, str):
            raise UnsupportedOperationError(
                "pandas tracer: callable aggregations (UDFs) are not yet "
                "supported. Pass a canonical reduction name like 'sum' "
                "or an agg dict instead."
            )

        if isinstance(spec, dict):
            for output_col, func in spec.items():
                if not isinstance(output_col, str):
                    raise UnsupportedOperationError(
                        f"pandas tracer: agg dict keys must be column "
                        f"names; got {type(output_col).__name__}"
                    )
                input_col = output_col
                if input_col not in self._frame._schema:
                    raise UnsupportedOperationError(
                        f"pandas tracer: agg references column "
                        f"{input_col!r}, which is not in the frame. "
                        f"Available: {sorted(self._frame._schema)}"
                    )
                aggregations.append(
                    (
                        output_col,
                        input_col,
                        resolve_agg_func(func, name_map=_CANONICAL_AGG, framework="pandas"),
                    )
                )
        elif isinstance(spec, str):
            if self._selection is None:
                raise UnsupportedOperationError(
                    f"pandas tracer: groupby.agg({spec!r}) needs a "
                    f"column selection. Use groupby(k)[col].agg(...) or "
                    f"groupby(k).agg({{col: {spec!r}}})."
                )
            canonical = resolve_agg_func(spec, name_map=_CANONICAL_AGG, framework="pandas")
            for col in self._selection:
                aggregations.append((col, col, canonical))
        else:
            raise UnsupportedOperationError(
                f"pandas tracer: agg argument must be a dict or a "
                f"canonical reduction name string; got "
                f"{type(spec).__name__}"
            )

        return self._emit_aggregate(aggregations)

    def sum(self) -> _TraceFrame:
        return self.agg("sum")

    def mean(self) -> _TraceFrame:
        return self.agg("mean")

    def min(self) -> _TraceFrame:
        return self.agg("min")

    def max(self) -> _TraceFrame:
        return self.agg("max")

    def count(self) -> _TraceFrame:
        return self.agg("count")

    def nunique(self) -> _TraceFrame:
        return self.agg("nunique")

    def _emit_aggregate(self, aggregations: list[tuple[str, str | None, str]]) -> _TraceFrame:
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

        return _TraceFrame(
            schema=output_schema,
            operations=self._frame._operations,
            last_op_id=op_id,
        )


def build_initial_frame(
    *,
    source_name: str,
    input_schema: tuple[ColumnSpec, ...],
) -> tuple[_TraceFrame, list[Operation]]:
    """Build the initial ``_TraceFrame`` and its Source operation.

    The returned operations list is shared with the frame; derived frames
    (from ``__getitem__``) append to it as the pipeline runs.
    """

    operations: list[Operation] = []
    source = Source(op_id="source_0", name=source_name, columns=input_schema)
    operations.append(source)

    frame = _TraceFrame(
        schema=dict(input_schema),
        operations=operations,
        last_op_id=source.op_id,
    )
    return frame, operations
