"""SQL AST walker. Converts a sqlglot ``Select`` into a tuple of canonical
:class:`~mirrorml.fingerprint.schema.Operation` instances and computes the
input and output schemas.

Phase 1 scope:

* Single-table ``SELECT``.
* Optional ``WHERE``.
* Projection: ``SELECT *`` or a list of bare column references.

Anything else (CTEs, JOINs, GROUP BY, ORDER BY, HAVING, LIMIT, DISTINCT,
UNION, subqueries, aliases, expressions in the projection list) raises
:class:`~mirrorml.exceptions.UnsupportedOperationError` with a message
that names the unsupported feature and the offending SQL fragment.

This module is internal. The public entry point is
:func:`mirrorml.tracers.trace_sql`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import sqlglot
import sqlglot.expressions as exp

from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint import build_fingerprint
from mirrorml.fingerprint.operations import Filter, Project, Source
from mirrorml.fingerprint.schema import ColumnSpec, Fingerprint, Operation

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

    projection = _extract_projection(node)

    if projection is None:
        output_schema = source_columns
    else:
        for col_name in projection:
            if col_name not in column_dtype:
                raise UnsupportedOperationError(
                    f"SELECT references column {col_name!r}, which is not "
                    f"in the schema of table {source_table!r}. Available "
                    f"columns: {sorted(column_dtype)}"
                )
        prj = Project(
            op_id="project",
            dependencies=(last_op_id,),
            columns=projection,
        )
        operations.append(prj)
        output_schema = tuple((c, column_dtype[c]) for c in projection)

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
        if node.args.get("group"):
            raise UnsupportedOperationError("SQL GROUP BY is not supported in M2 phase 1")
        if node.args.get("order"):
            raise UnsupportedOperationError("SQL ORDER BY is not supported in M2 phase 1")
        if node.args.get("having"):
            raise UnsupportedOperationError("SQL HAVING is not supported in M2 phase 1")
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


def _extract_projection(node: exp.Select) -> tuple[str, ...] | None:
    """Return projected column names, or ``None`` for ``SELECT *``."""

    expressions = node.expressions
    if not expressions:
        raise UnsupportedOperationError(f"SELECT has no projection list in {node.sql()!r}")

    if len(expressions) == 1 and isinstance(expressions[0], exp.Star):
        return None

    columns: list[str] = []
    for expression in expressions:
        if isinstance(expression, exp.Column):
            columns.append(expression.name)
        elif isinstance(expression, exp.Alias):
            raise UnsupportedOperationError(
                f"Column aliasing is deferred (got {expression.sql()!r}). "
                f"Reference columns by their underlying name in M2 phase 1."
            )
        else:
            raise UnsupportedOperationError(
                f"SELECT projection {expression.sql()!r} is not a bare "
                f"column reference. Expressions, function calls, and "
                f"aliases are deferred to a later M2 phase."
            )

    return tuple(columns)
