"""SQL AST walker. Converts a sqlglot ``Select`` into a tuple of canonical
:class:`~mirrorml.fingerprint.schema.Operation` instances and computes the
input and output schemas.

Current scope:

* Single-table ``SELECT``.
* Optional ``WHERE``.
* Projection mixing bare column references (with optional ``AS`` aliasing)
  and canonical aggregate function calls (``COUNT``, ``SUM``, ``AVG``,
  ``MIN``, ``MAX``, ``COUNT(DISTINCT col)``).
* Optional ``GROUP BY`` plus ``HAVING``.
* Optional ``ORDER BY`` on output columns (``ASC`` / ``DESC``).

Anything else (CTEs, JOINs, LIMIT, DISTINCT row sets, UNION, subqueries,
non-canonical aggregates, expressions inside aggregates or ORDER BY)
raises :class:`~mirrorml.exceptions.UnsupportedOperationError` with a
message naming the unsupported feature and the offending SQL fragment.

This module is internal. The public entry point is
:func:`mirrorml.tracers.trace_sql`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

import sqlglot
import sqlglot.expressions as exp

from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint import build_fingerprint
from mirrorml.fingerprint.operations import Aggregate, Filter, Project, Sort, Source
from mirrorml.fingerprint.schema import ColumnSpec, Fingerprint, Operation, SchemaDelta

# Map sqlglot's aggregate function classes to canonical reduction names.
_AGG_FUNCS: dict[type[exp.AggFunc], str] = {
    exp.Count: "count",
    exp.Sum: "sum",
    exp.Avg: "mean",
    exp.Min: "min",
    exp.Max: "max",
}

# Function output-dtype rules. Captures SQL standard behavior for the
# canonical reductions we support; cross-framework differences (e.g. SUM
# widening int32 to int64) are the diff classifier's job in M3.
_FIXED_DTYPE_FOR_FUNC: dict[str, str] = {
    "count": "int64",
    "count_distinct": "int64",
    "mean": "float64",
}

__all__ = ["trace_sql_impl"]


def trace_sql_impl(
    query: str,
    *,
    schemas: Mapping[str, tuple[ColumnSpec, ...]] | None,
    dialect: str | None,
) -> Fingerprint:
    """Parse ``query`` and produce a canonical fingerprint."""

    if schemas is None:
        schemas = {}

    try:
        ast = cast(exp.Expression, sqlglot.parse_one(query, dialect=dialect))
    except sqlglot.errors.ParseError as e:
        raise UnsupportedOperationError(
            f"SQL tracer could not parse the query (sqlglot dialect {dialect or 'auto'!r}): {e}"
        ) from e

    if ast is None:
        raise UnsupportedOperationError(f"SQL tracer received an empty query string: {query!r}")

    operations, input_schema, output_schema = _walk_select(ast, schemas)

    return build_fingerprint(
        framework="sql",
        input_schema=input_schema,
        output_schema=output_schema,
        operations=operations,
    )


def _walk_select(
    node: exp.Expression,
    schemas: Mapping[str, tuple[ColumnSpec, ...]],
) -> tuple[list[Operation], tuple[ColumnSpec, ...], tuple[ColumnSpec, ...]]:
    _reject_unsupported_toplevel(node)

    if not isinstance(node, exp.Select):
        raise UnsupportedOperationError(
            f"SQL tracer phase 1 only handles top-level SELECT; got "
            f"{type(node).__name__} in {node.sql()!r}"
        )

    source_table = _extract_single_table(node)
    if source_table not in schemas:
        raise UnsupportedOperationError(
            f"SQL FROM references table {source_table!r}, but no schema was "
            f"provided. Pass schemas={{{source_table!r}: ((col, dtype), ...)}} "
            f"to trace_sql so the Source operation can be built."
        )

    source_columns = schemas[source_table]
    column_dtype = dict(source_columns)

    operations: list[Operation] = []
    source = Source(
        op_id="source",
        name=source_table,
        columns=source_columns,
    )
    operations.append(source)
    last_op_id = source.op_id

    where = node.args.get("where")
    if where is not None:
        predicate = where.this.sql()
        flt = Filter(
            op_id="filter",
            dependencies=(last_op_id,),
            predicate=predicate,
        )
        operations.append(flt)
        last_op_id = flt.op_id

    group_by_keys = _extract_group_by_keys(node)
    has_aggregations = any(_is_aggregation_call(_unwrap_alias(e)) for e in node.expressions)

    if group_by_keys is not None or has_aggregations:
        agg_op, output_schema = _build_aggregate(
            node=node,
            group_keys=group_by_keys or (),
            dependency=last_op_id,
            source_dtypes=column_dtype,
        )
        operations.append(agg_op)
        last_op_id = agg_op.op_id

        having = node.args.get("having")
        if having is not None:
            having_filter = Filter(
                op_id="having",
                dependencies=(last_op_id,),
                predicate=having.this.sql(),
            )
            operations.append(having_filter)
            last_op_id = having_filter.op_id
    else:
        projection = _extract_projection(node)
        if projection is None:
            output_schema = source_columns
        else:
            for source_col, _ in projection:
                if source_col not in column_dtype:
                    raise UnsupportedOperationError(
                        f"SELECT references column {source_col!r}, which is not "
                        f"in the schema of table {source_table!r}. Available "
                        f"columns: {sorted(column_dtype)}"
                    )
            renames = tuple((source, output) for source, output in projection if source != output)
            prj = Project(
                op_id="project",
                dependencies=(last_op_id,),
                columns=tuple(output for _, output in projection),
                schema_delta=SchemaDelta(renamed=renames),
            )
            operations.append(prj)
            last_op_id = prj.op_id
            output_schema = tuple((output, column_dtype[source]) for source, output in projection)

    order = node.args.get("order")
    if order is not None:
        sort_by = _extract_order_by(order, output_schema)
        srt = Sort(
            op_id="sort",
            dependencies=(last_op_id,),
            by=sort_by,
        )
        operations.append(srt)

    return operations, source_columns, output_schema


def _reject_unsupported_toplevel(node: exp.Expression) -> None:
    if isinstance(node, exp.Union):
        raise UnsupportedOperationError(
            "SQL UNION (and INTERSECT / EXCEPT) is not supported in M2 phase 1"
        )

    if isinstance(node, exp.Select):
        if node.args.get("with_") or node.args.get("with"):
            raise UnsupportedOperationError(
                "SQL CTEs (WITH clauses) are not supported in M2 phase 1"
            )
        if node.args.get("joins"):
            raise UnsupportedOperationError("SQL JOINs are not supported in M2 phase 1")
        if node.args.get("limit"):
            raise UnsupportedOperationError("SQL LIMIT is not supported in M2 phase 1")
        if node.args.get("distinct"):
            raise UnsupportedOperationError("SQL SELECT DISTINCT is not supported in M2 phase 1")


def _extract_single_table(node: exp.Select) -> str:
    from_clause = node.args.get("from_") or node.args.get("from")
    if from_clause is None:
        raise UnsupportedOperationError("SQL SELECT without FROM is not supported in M2 phase 1")

    target = from_clause.this
    if not isinstance(target, exp.Table):
        raise UnsupportedOperationError(
            f"FROM target must be a plain table reference in M2 phase 1; "
            f"got {type(target).__name__} in {target.sql()!r}. Subqueries "
            f"and table-valued functions are deferred."
        )

    return target.name


def _extract_projection(node: exp.Select) -> tuple[tuple[str, str], ...] | None:
    """Return ``(source_column, output_name)`` pairs, or ``None`` for ``SELECT *``.

    A bare column reference yields ``(name, name)``; ``col AS alias`` yields
    ``(col, alias)``. Aliases on non-column expressions (function calls,
    arithmetic) raise :class:`UnsupportedOperationError` until later M2
    phases extend the surface.
    """

    expressions = node.expressions
    if not expressions:
        raise UnsupportedOperationError(f"SELECT has no projection list in {node.sql()!r}")

    if len(expressions) == 1 and isinstance(expressions[0], exp.Star):
        return None

    columns: list[tuple[str, str]] = []
    for expression in expressions:
        if isinstance(expression, exp.Column):
            columns.append((expression.name, expression.name))
        elif isinstance(expression, exp.Alias):
            underlying = expression.this
            if not isinstance(underlying, exp.Column):
                raise UnsupportedOperationError(
                    f"SELECT alias {expression.sql()!r} renames a non-column "
                    f"expression. Expressions, function calls, and "
                    f"aggregations are deferred to a later M2 phase."
                )
            columns.append((underlying.name, expression.alias))
        else:
            raise UnsupportedOperationError(
                f"SELECT projection {expression.sql()!r} is not a bare "
                f"column reference. Expressions, function calls, and "
                f"aggregations are deferred to a later M2 phase."
            )

    return tuple(columns)


def _unwrap_alias(expression: exp.Expression) -> exp.Expression:
    """Return the underlying expression beneath an Alias node, or the node itself."""

    return expression.this if isinstance(expression, exp.Alias) else expression


def _is_aggregation_call(expression: exp.Expression) -> bool:
    """Whether ``expression`` is one of the canonical aggregate function calls."""

    return any(isinstance(expression, cls) for cls in _AGG_FUNCS)


def _extract_group_by_keys(node: exp.Select) -> tuple[str, ...] | None:
    """Return the GROUP BY column names, or ``None`` if no GROUP BY clause."""

    group = node.args.get("group")
    if group is None:
        return None

    keys: list[str] = []
    for item in group.expressions:
        if not isinstance(item, exp.Column):
            raise UnsupportedOperationError(
                f"GROUP BY {item.sql()!r} is not a bare column reference. "
                f"Expressions and function calls in GROUP BY are deferred."
            )
        keys.append(item.name)
    return tuple(keys)


def _build_aggregate(
    *,
    node: exp.Select,
    group_keys: tuple[str, ...],
    dependency: str,
    source_dtypes: Mapping[str, str],
) -> tuple[Aggregate, tuple[ColumnSpec, ...]]:
    """Construct the :class:`Aggregate` op and the post-aggregate output schema.

    Each SELECT-list item must be either a bare column reference that appears
    in GROUP BY (the group keys) or a supported aggregate function call
    (``COUNT`` / ``SUM`` / ``AVG`` / ``MIN`` / ``MAX``) with an optional
    ``AS`` alias. Anything else raises
    :class:`UnsupportedOperationError`.
    """

    if not node.expressions:
        raise UnsupportedOperationError(f"SELECT has no projection list in {node.sql()!r}")

    if len(node.expressions) == 1 and isinstance(node.expressions[0], exp.Star):
        raise UnsupportedOperationError(
            "SELECT * is not supported with GROUP BY / aggregations; "
            "name each output column explicitly."
        )

    output_schema_builder: list[ColumnSpec] = []
    aggregations: list[tuple[str, str | None, str]] = []

    group_key_set = set(group_keys)

    for item in node.expressions:
        underlying = _unwrap_alias(item)
        alias = item.alias if isinstance(item, exp.Alias) else None

        if isinstance(underlying, exp.Column):
            if underlying.name not in group_key_set:
                raise UnsupportedOperationError(
                    f"SELECT references {underlying.name!r}, which is neither a "
                    f"GROUP BY key nor an aggregate. Add it to GROUP BY or wrap "
                    f"it in an aggregate function."
                )
            if underlying.name not in source_dtypes:
                raise UnsupportedOperationError(
                    f"SELECT references column {underlying.name!r}, which is "
                    f"not in the source schema. Available: {sorted(source_dtypes)}"
                )
            output_name = alias or underlying.name
            output_schema_builder.append((output_name, source_dtypes[underlying.name]))
            continue

        if _is_aggregation_call(underlying):
            input_col, func = _interpret_aggregation_call(underlying)
            output_name = alias or _default_aggregation_name(func, input_col)
            output_dtype = _aggregation_output_dtype(func, input_col, source_dtypes)
            aggregations.append((output_name, input_col, func))
            output_schema_builder.append((output_name, output_dtype))
            continue

        raise UnsupportedOperationError(
            f"SELECT item {item.sql()!r} is not a GROUP BY key or a supported "
            f"aggregation. Supported aggregates: "
            f"COUNT, SUM, AVG, MIN, MAX, COUNT(DISTINCT ...)."
        )

    # Validate every GROUP BY key actually exists in the source schema.
    for key in group_keys:
        if key not in source_dtypes:
            raise UnsupportedOperationError(
                f"GROUP BY references column {key!r}, which is not in the "
                f"source schema. Available: {sorted(source_dtypes)}"
            )

    agg_op = Aggregate(
        op_id="aggregate",
        dependencies=(dependency,),
        by=group_keys,
        aggregations=tuple(aggregations),
    )
    return agg_op, tuple(output_schema_builder)


def _interpret_aggregation_call(node: exp.Expression) -> tuple[str | None, str]:
    """Return ``(input_column_or_None, canonical_function_name)``.

    ``input_column`` is ``None`` for ``COUNT(*)``. ``COUNT(DISTINCT col)``
    maps to function ``"count_distinct"``. Aggregates whose argument is not
    a bare column, a Star, or ``DISTINCT col`` raise
    :class:`UnsupportedOperationError`.
    """

    for cls, canonical_name in _AGG_FUNCS.items():
        if isinstance(node, cls):
            inner = node.this
            if isinstance(inner, exp.Star):
                if canonical_name != "count":
                    raise UnsupportedOperationError(
                        f"{cls.__name__.upper()}(*) is not a supported aggregation; "
                        f"only COUNT(*) is."
                    )
                return None, "count"
            if isinstance(inner, exp.Distinct):
                if canonical_name != "count":
                    raise UnsupportedOperationError(
                        f"DISTINCT is only supported inside COUNT; "
                        f"got {cls.__name__.upper()}(DISTINCT ...)."
                    )
                distinct_args = inner.expressions
                if len(distinct_args) != 1 or not isinstance(distinct_args[0], exp.Column):
                    raise UnsupportedOperationError(
                        f"COUNT(DISTINCT ...) must wrap a single bare column; got {node.sql()!r}."
                    )
                return distinct_args[0].name, "count_distinct"
            if isinstance(inner, exp.Column):
                return inner.name, canonical_name
            raise UnsupportedOperationError(
                f"Aggregate {node.sql()!r} argument must be a bare column "
                f"(or '*' for COUNT); expressions inside aggregates are deferred."
            )

    raise UnsupportedOperationError(
        f"Aggregation {node.sql()!r} is not a supported canonical reduction; "
        f"supported: COUNT, SUM, AVG, MIN, MAX, COUNT(DISTINCT ...)."
    )


def _default_aggregation_name(func: str, input_col: str | None) -> str:
    """Generate a deterministic output column name for an unaliased aggregate."""

    if input_col is None:
        return f"{func}(*)"
    return f"{func}({input_col})"


def _aggregation_output_dtype(
    func: str, input_col: str | None, source_dtypes: Mapping[str, str]
) -> str:
    """Output dtype of a canonical aggregation.

    ``COUNT`` and ``COUNT(DISTINCT)`` always return ``int64``; ``AVG``
    always returns ``float64``; ``SUM`` / ``MIN`` / ``MAX`` propagate the
    input column's dtype. SQL widening semantics for SUM (e.g. int32 to
    int64 in Postgres) are the diff classifier's territory in M3.
    """

    fixed = _FIXED_DTYPE_FOR_FUNC.get(func)
    if fixed is not None:
        return fixed
    if input_col is None:
        raise UnsupportedOperationError(
            f"Aggregation {func!r} has no input column; only COUNT(*) is allowed without an input."
        )
    if input_col not in source_dtypes:
        raise UnsupportedOperationError(
            f"Aggregation {func}({input_col}) references column not in the "
            f"source schema. Available: {sorted(source_dtypes)}"
        )
    return source_dtypes[input_col]


def _extract_order_by(
    order: exp.Order, output_schema: tuple[ColumnSpec, ...]
) -> tuple[tuple[str, Literal["asc", "desc"]], ...]:
    """Build the ``Sort.by`` tuple from an ORDER BY clause.

    ORDER BY references must be plain column names; expressions, function
    calls, and positional references (``ORDER BY 1``) raise
    :class:`UnsupportedOperationError`. The referenced column must appear
    in the projection's output schema, matching SQL's scoping rules where
    ORDER BY sees the post-projection names.
    """

    available = {name for name, _ in output_schema}

    by: list[tuple[str, Literal["asc", "desc"]]] = []
    for ordered in order.expressions:
        if not isinstance(ordered, exp.Ordered):
            raise UnsupportedOperationError(
                f"ORDER BY item {ordered.sql()!r} is not a recognized "
                f"ordered expression (sqlglot produced {type(ordered).__name__})"
            )
        underlying = ordered.this
        if not isinstance(underlying, exp.Column):
            raise UnsupportedOperationError(
                f"ORDER BY {ordered.sql()!r} is not a bare column reference. "
                f"Expressions, function calls, and positional ORDER BY are "
                f"deferred to a later M2 phase."
            )
        col = underlying.name
        if col not in available:
            raise UnsupportedOperationError(
                f"ORDER BY references column {col!r}, which is not in the "
                f"output schema. Available: {sorted(available)}"
            )
        direction: Literal["asc", "desc"] = "desc" if ordered.args.get("desc") else "asc"
        by.append((col, direction))

    return tuple(by)
