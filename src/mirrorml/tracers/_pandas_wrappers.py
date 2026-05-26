"""Wrapper objects for the pandas tracer.

The pandas tracer is a wrapper-object tracer: the user's pipeline runs
against proxy ``_TraceFrame`` / ``_TraceSeries`` / ``_TracePredicate``
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

from typing import Any

from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint.operations import Filter, Project, Source
from mirrorml.fingerprint.schema import ColumnSpec, Operation


class _TracePredicate:
    """A captured boolean expression. Used as the key in ``df[mask]``.

    Implements ``&``, ``|``, ``~`` to compose with SQL precedence so
    ``(df.a > 0) & (df.b < 10)`` renders as ``"a > 0 AND b < 10"``.
    """

    __slots__ = ("_sql",)

    def __init__(self, sql: str) -> None:
        self._sql = sql

    def render(self) -> str:
        return self._sql

    def __and__(self, other: _TracePredicate) -> _TracePredicate:
        return _TracePredicate(f"{self._sql} AND {other._sql}")

    def __or__(self, other: _TracePredicate) -> _TracePredicate:
        return _TracePredicate(f"{self._sql} OR {other._sql}")

    def __invert__(self) -> _TracePredicate:
        return _TracePredicate(f"NOT ({self._sql})")


class _TraceSeries:
    """A captured column reference. Comparison operators build predicates.

    Returning a :class:`_TracePredicate` from ``__eq__`` / ``__ne__``
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

    def __gt__(self, other: object) -> _TracePredicate:
        return _TracePredicate(f"{self._name} > {_render_literal(other)}")

    def __lt__(self, other: object) -> _TracePredicate:
        return _TracePredicate(f"{self._name} < {_render_literal(other)}")

    def __ge__(self, other: object) -> _TracePredicate:
        return _TracePredicate(f"{self._name} >= {_render_literal(other)}")

    def __le__(self, other: object) -> _TracePredicate:
        return _TracePredicate(f"{self._name} <= {_render_literal(other)}")

    # __eq__ / __ne__ deliberately diverge from object identity so
    # comparison-style usage builds predicates. _TraceSeries is treated as
    # unhashable to keep this safe.
    def __eq__(self, other: object) -> _TracePredicate:  # type: ignore[override]
        return _TracePredicate(f"{self._name} = {_render_literal(other)}")

    def __ne__(self, other: object) -> _TracePredicate:  # type: ignore[override]
        return _TracePredicate(f"{self._name} <> {_render_literal(other)}")

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
        if isinstance(key, _TracePredicate):
            return self._apply_filter(key)
        if isinstance(key, str):
            return self._select_column(key)
        if isinstance(key, list):
            return self._project(key)
        raise UnsupportedOperationError(
            f"pandas tracer: unsupported __getitem__ key {type(key).__name__!r}. "
            f"Supported: boolean mask, column name (str), column list."
        )

    def _apply_filter(self, predicate: _TracePredicate) -> _TraceFrame:
        op_id = f"filter_{_next_op_index(self._operations)}"
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
        op_id = f"project_{_next_op_index(self._operations)}"
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


def _next_op_index(operations: list[Operation]) -> int:
    """Position-based op_id seed. Canonicalization rewrites these later,
    so the specific seed does not affect fingerprint stability; uniqueness
    within a fingerprint is what matters."""

    return len(operations)


def _render_literal(value: object) -> str:
    """Render a Python literal as a SQL-like value.

    Matches sqlglot's default rendering for the common cases so the
    pandas ``Filter.predicate`` and the SQL ``Filter.predicate`` byte-
    match for cross-framework equivalence.
    """

    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, int | float):
        return str(value)
    raise UnsupportedOperationError(
        f"pandas tracer: cannot render literal {value!r} ({type(value).__name__}) "
        f"in a predicate; supported: int, float, str, bool, None."
    )
