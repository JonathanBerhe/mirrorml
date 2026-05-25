"""SQL AST walker. Converts a sqlglot ``Select`` into a tuple of canonical
:class:`~mirrorml.fingerprint.schema.Operation` instances and computes the
input and output schemas.

Current scope:

* ``SELECT`` over one or more tables joined by ``INNER`` / ``LEFT`` /
  ``RIGHT`` / ``FULL OUTER`` joins. Joins use ``ON`` (USING is deferred)
  and equality conjunctions only. Multi-way joins are supported via
  chained binary Join ops; the resulting schema combines left + right
  with ``_right`` suffix on column-name collisions.
* Table aliases in ``FROM`` and ``JOIN`` (``FROM events AS e``).
* Optional ``WHERE``.
* Projection mixing bare column references (with optional ``AS`` aliasing)
  and canonical aggregate function calls (``COUNT``, ``SUM``, ``AVG``,
  ``MIN``, ``MAX``, ``COUNT(DISTINCT col)``).
* Optional ``GROUP BY`` plus ``HAVING``.
* Optional ``ORDER BY`` on output columns (``ASC`` / ``DESC``).

Anything else (CTEs, ``CROSS JOIN``, ``USING``, LIMIT, DISTINCT row sets,
UNION, subqueries, non-canonical aggregates, expressions inside
aggregates or ORDER BY) raises
:class:`~mirrorml.exceptions.UnsupportedOperationError` with a message
naming the unsupported feature and the offending SQL fragment.

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
from mirrorml.fingerprint.operations import Aggregate, Filter, Join, Project, Sort, Source
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

    operations, column_dtype, source_columns, last_op_id = _build_from_chain(node, schemas)

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
            output_schema = tuple(column_dtype.items())
        else:
            for source_col, _ in projection:
                if source_col not in column_dtype:
                    raise UnsupportedOperationError(
                        f"SELECT references column {source_col!r}, which is not "
                        f"in the post-FROM schema. Available columns: "
                        f"{sorted(column_dtype)}"
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
        if node.args.get("limit"):
            raise UnsupportedOperationError("SQL LIMIT is not supported in M2 phase 1")
        if node.args.get("distinct"):
            raise UnsupportedOperationError("SQL SELECT DISTINCT is not supported in M2 phase 1")


def _build_from_chain(
    node: exp.Select, schemas: Mapping[str, tuple[ColumnSpec, ...]]
) -> tuple[list[Operation], dict[str, str], tuple[ColumnSpec, ...], str]:
    """Build the Source + (Join, Source)* chain from a Select's FROM and joins.

    Returns:
        operations: the new ops in the order they should appear.
        current_schema: the post-chain column->dtype map (downstream ops use this).
        left_source_columns: the left-most source table's column list (used as
            the Fingerprint's input_schema; multi-source pipelines collapse to
            the leftmost since the schema carries only one).
        last_op_id: the id of the final op in the chain (Source if no joins,
            otherwise the last Join).

    Raises :class:`UnsupportedOperationError` for unknown tables, non-table
    FROM targets, joins without ON, CROSS / USING joins, and ambiguous or
    impossible-to-resolve ON keys.
    """

    from_clause = node.args.get("from_") or node.args.get("from")
    if from_clause is None:
        raise UnsupportedOperationError("SQL SELECT without FROM is not supported")

    left_target = from_clause.this
    if not isinstance(left_target, exp.Table):
        raise UnsupportedOperationError(
            f"FROM target must be a plain table reference; got "
            f"{type(left_target).__name__} in {left_target.sql()!r}. Subqueries "
            f"and table-valued functions are deferred."
        )

    left_name = left_target.name
    left_alias = left_target.alias or left_name
    if left_name not in schemas:
        raise UnsupportedOperationError(
            f"FROM references table {left_name!r}, but no schema was provided. "
            f"Pass schemas={{{left_name!r}: ((col, dtype), ...)}} to trace_sql."
        )
    left_columns = schemas[left_name]

    operations: list[Operation] = []
    left_source = Source(op_id="source_0", name=left_name, columns=left_columns)
    operations.append(left_source)
    last_op_id = left_source.op_id

    current_schema: dict[str, str] = dict(left_columns)
    left_qualifiers: set[str] = {left_alias, left_name}

    joins = node.args.get("joins") or []
    for index, join_node in enumerate(joins):
        right_target = join_node.this
        if not isinstance(right_target, exp.Table):
            raise UnsupportedOperationError(
                f"JOIN target must be a plain table reference; got "
                f"{type(right_target).__name__} in {right_target.sql()!r}. "
                f"Subqueries are deferred."
            )

        right_name = right_target.name
        right_alias = right_target.alias or right_name
        if right_name not in schemas:
            raise UnsupportedOperationError(
                f"JOIN references table {right_name!r}, but no schema was provided. "
                f"Pass it in schemas= to trace_sql."
            )
        right_columns = schemas[right_name]
        right_dtype_map = dict(right_columns)

        right_source = Source(op_id=f"source_{index + 1}", name=right_name, columns=right_columns)
        operations.append(right_source)

        how = _determine_join_kind(join_node)

        on_expr = join_node.args.get("on")
        if on_expr is None:
            raise UnsupportedOperationError(
                "JOIN without ON clause is not supported (use ON, not USING / CROSS)"
            )

        left_keys, right_keys = _resolve_on_keys(
            on_expr=on_expr,
            left_qualifiers=left_qualifiers,
            right_qualifier=right_alias,
            right_table_name=right_name,
            left_schema=current_schema,
            right_schema=right_dtype_map,
        )

        join_op = Join(
            op_id=f"join_{index}",
            dependencies=(last_op_id, right_source.op_id),
            how=how,
            left_keys=left_keys,
            right_keys=right_keys,
        )
        operations.append(join_op)
        last_op_id = join_op.op_id

        current_schema = _combine_schemas_with_suffix(
            current_schema, right_columns, suffix_right="_right"
        )
        left_qualifiers.add(right_alias)
        left_qualifiers.add(right_name)

    return operations, current_schema, left_columns, last_op_id


def _determine_join_kind(
    join_node: exp.Join,
) -> Literal["inner", "left", "right", "outer"]:
    """Map a sqlglot Join node to one of the canonical join kinds."""

    if join_node.args.get("using"):
        raise UnsupportedOperationError("JOIN ... USING (...) is not yet supported; rewrite as ON")

    kind = join_node.args.get("kind")
    side = join_node.args.get("side")

    if kind == "CROSS":
        raise UnsupportedOperationError("CROSS JOIN is not supported")

    if side is None:
        return "inner"
    if side == "LEFT":
        return "left"
    if side == "RIGHT":
        return "right"
    if side == "FULL":
        return "outer"

    raise UnsupportedOperationError(f"unrecognized join: side={side!r}, kind={kind!r}")


def _resolve_on_keys(
    *,
    on_expr: exp.Expression,
    left_qualifiers: set[str],
    right_qualifier: str,
    right_table_name: str,
    left_schema: Mapping[str, str],
    right_schema: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Parse an ON predicate into ``(left_keys, right_keys)``.

    The predicate must be a chain of equalities joined by ``AND``; each
    equality must compare a column from the left side to a column from the
    right side. Qualifier-based resolution uses the table name or alias;
    unqualified columns are resolved by membership in the left vs right
    schema, with ambiguous matches rejected.
    """

    right_qualifiers = {right_qualifier, right_table_name}

    left_keys: list[str] = []
    right_keys: list[str] = []

    for eq in _flatten_and_chain(on_expr):
        if not isinstance(eq, exp.EQ):
            raise UnsupportedOperationError(
                f"only equi-joins are supported; ON predicate {eq.sql()!r} is not an equality"
            )
        lhs = eq.this
        rhs = eq.expression
        if not isinstance(lhs, exp.Column) or not isinstance(rhs, exp.Column):
            raise UnsupportedOperationError(
                f"ON predicate {eq.sql()!r} must use plain column references on both sides"
            )

        lhs_side = _side_of(lhs, left_qualifiers, right_qualifiers, left_schema, right_schema)
        rhs_side = _side_of(rhs, left_qualifiers, right_qualifiers, left_schema, right_schema)

        if lhs_side == "left" and rhs_side == "right":
            left_keys.append(lhs.name)
            right_keys.append(rhs.name)
        elif lhs_side == "right" and rhs_side == "left":
            left_keys.append(rhs.name)
            right_keys.append(lhs.name)
        else:
            raise UnsupportedOperationError(
                f"ON predicate {eq.sql()!r} does not equate a left-side column "
                f"with a right-side column"
            )

    return tuple(left_keys), tuple(right_keys)


def _side_of(
    col: exp.Column,
    left_qualifiers: set[str],
    right_qualifiers: set[str],
    left_schema: Mapping[str, str],
    right_schema: Mapping[str, str],
) -> Literal["left", "right"]:
    """Decide whether a column reference belongs to the left or right side."""

    qualifier = col.table
    if qualifier:
        if qualifier in right_qualifiers:
            return "right"
        if qualifier in left_qualifiers:
            return "left"
        raise UnsupportedOperationError(
            f"ON references qualifier {qualifier!r} that does not match the "
            f"left side {sorted(left_qualifiers)} or the right side "
            f"{sorted(right_qualifiers)}"
        )

    name = col.name
    in_left = name in left_schema
    in_right = name in right_schema
    if in_left and not in_right:
        return "left"
    if in_right and not in_left:
        return "right"
    if in_left and in_right:
        raise UnsupportedOperationError(
            f"ON column {name!r} is ambiguous (exists on both sides). "
            f"Qualify it with the table name or alias."
        )
    raise UnsupportedOperationError(f"ON column {name!r} does not exist on either side of the join")


def _flatten_and_chain(expr: exp.Expression) -> list[exp.Expression]:
    """Flatten a chain of AND expressions into the list of leaf conjuncts."""

    if isinstance(expr, exp.And):
        return _flatten_and_chain(expr.this) + _flatten_and_chain(expr.expression)
    return [expr]


def _combine_schemas_with_suffix(
    left: Mapping[str, str], right: tuple[ColumnSpec, ...], *, suffix_right: str
) -> dict[str, str]:
    """Combine two schemas. Right-side columns colliding with anything
    already in the combined schema get ``suffix_right`` appended repeatedly
    until unique, so multi-way chains do not silently overwrite columns
    from earlier joins.

    Right is iterated in declaration order to keep the output deterministic.
    """

    combined: dict[str, str] = dict(left)
    for col, dtype in right:
        candidate = col
        while candidate in combined:
            candidate = f"{candidate}{suffix_right}"
        combined[candidate] = dtype
    return combined


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
