"""Shared machinery for the wrapper-object tracers (pandas, Polars).

Both wrapper tracers render Filter predicates and Aggregate ops into the
*same* canonical forms the SQL tracer uses, so an equivalent pandas /
Polars pipeline and SQL query produce byte-identical ``Filter.predicate``
strings and identical ``Aggregate`` ops. That byte-parity is the basis
for the cross-framework ``diff(a, b) == ()`` claim (PAPER.md C4).

Keeping the rendering here, shared, makes the parity *structural* rather
than something maintained by hand in two places. The SQL tracer keeps its
own copies of the dtype rules (it walks sqlglot trees, not wrappers); the
values are mirrored and a cross-framework test guards the equivalence.

Predicate strings follow SQL form: ``=``, ``<>``, ``AND``, ``OR``,
``NOT (...)`` and SQL literal rendering.
"""

from __future__ import annotations

from collections.abc import Mapping

from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint.schema import Operation

# Output dtype rules for aggregations. Mirrors _sql_walker._FIXED_DTYPE_FOR_FUNC
# so a cross-framework Aggregate op produces an identical output_schema.
FIXED_DTYPE_FOR_FUNC: dict[str, str] = {
    "count": "int64",
    "count_distinct": "int64",
    "mean": "float64",
}


class TracePredicate:
    """A captured boolean expression rendered in SQL form.

    Implements ``&``, ``|``, ``~`` so ``(col("a") > 0) & (col("b") < 10)``
    renders as ``"a > 0 AND b < 10"`` regardless of which wrapper tracer
    built it.
    """

    __slots__ = ("_sql",)

    def __init__(self, sql: str) -> None:
        self._sql = sql

    def render(self) -> str:
        return self._sql

    def __and__(self, other: TracePredicate) -> TracePredicate:
        return TracePredicate(f"{self._sql} AND {other._sql}")

    def __or__(self, other: TracePredicate) -> TracePredicate:
        return TracePredicate(f"{self._sql} OR {other._sql}")

    def __invert__(self) -> TracePredicate:
        return TracePredicate(f"NOT ({self._sql})")


def render_literal(value: object) -> str:
    """Render a Python literal as a SQL-like value.

    Matches sqlglot's default rendering for the common cases so a wrapper
    tracer's ``Filter.predicate`` and the SQL tracer's ``Filter.predicate``
    byte-match for cross-framework equivalence.
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
        f"cannot render literal {value!r} ({type(value).__name__}) in a "
        f"predicate; supported: int, float, str, bool, None."
    )


def next_op_index(operations: list[Operation]) -> int:
    """Position-based op_id seed. Canonicalization rewrites these later,
    so the specific seed does not affect fingerprint stability; uniqueness
    within a fingerprint is what matters."""

    return len(operations)


def resolve_agg_func(func: object, *, name_map: Mapping[str, str], framework: str) -> str:
    """Map a framework-side agg-function value to the canonical reduction name.

    ``name_map`` translates the framework's own reduction names (e.g.
    pandas ``"nunique"`` / Polars ``"n_unique"``) to the canonical set;
    ``framework`` only flavors the error message.
    """

    if callable(func) and not isinstance(func, str):
        raise UnsupportedOperationError(
            f"{framework} tracer: callable aggregations (UDFs) are not yet "
            f"supported; pass a canonical reduction name like 'sum'."
        )
    if not isinstance(func, str):
        raise UnsupportedOperationError(
            f"{framework} tracer: agg function must be a canonical reduction "
            f"name string; got {type(func).__name__}"
        )
    canonical = name_map.get(func)
    if canonical is None:
        raise UnsupportedOperationError(
            f"{framework} tracer: agg function {func!r} is not in the canonical "
            f"reduction set ({sorted(name_map)})"
        )
    return canonical


def aggregation_output_dtype(
    func: str, input_col: str | None, source_dtypes: Mapping[str, str]
) -> str:
    """Output dtype of a canonical aggregation. Mirrors the SQL walker's rule."""

    fixed = FIXED_DTYPE_FOR_FUNC.get(func)
    if fixed is not None:
        return fixed
    if input_col is None:
        raise UnsupportedOperationError(f"aggregation {func!r} has no input column")
    return source_dtypes[input_col]
