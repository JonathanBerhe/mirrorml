"""Statistical companion check.

The static fingerprint is the primary equivalence test. When two pipelines
have different fingerprints but might still be semantically equivalent (UDF
bodies that compute the same value, two formulations of the same
aggregation), the statistical companion runs both on a shared fixture and
compares the outputs within a tolerance.

This module provides:

* :func:`compare_frames` -- the comparison core. Do two output tables agree
  within tolerance? Order-insensitive; numeric columns are compared within
  ``rtol`` / ``atol``, others exactly. Accepts a plain ``{column: values}``
  mapping or a pandas / Polars DataFrame (lazily converted, never imported
  at module load).
* :func:`statistical_check` -- run two pipelines on a shared fixture and
  compare their outputs. For pandas / Polars the pipeline is a callable;
  for SQL it is a query string executed via sqlglot's built-in executor
  (no new runtime dependency). Window functions in SQL are not supported
  by that executor and raise :class:`UnsupportedOperationError`.

These names are internal / experimental and not part of the public API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mirrorml.exceptions import UnsupportedOperationError


@dataclass(frozen=True)
class StatComparison:
    """Outcome of comparing two pipeline outputs.

    ``equivalent`` is the headline; ``detail`` names the first disagreement
    (schema, row count, or a specific column) when they are not equivalent.
    """

    equivalent: bool
    detail: str = ""


def compare_frames(
    left: object,
    right: object,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> StatComparison:
    """Compare two pipeline output tables within tolerance.

    Numeric columns must agree within ``abs(a - b) <= atol + rtol * abs(b)``;
    non-numeric values must be equal. The comparison is order-insensitive:
    both tables are sorted before alignment, so a group-by that emits rows
    in a different order is still equivalent.

    Args:
        left, right: a ``{column: values}`` mapping, or a pandas / Polars
            ``DataFrame``.
        rtol, atol: relative and absolute tolerance for numeric columns.

    Examples:
        >>> compare_frames({"x": [1, 2]}, {"x": [2, 1]}).equivalent
        True
        >>> compare_frames({"x": [1.0]}, {"x": [1.0 + 1e-9]}).equivalent
        True
        >>> compare_frames({"x": [1]}, {"x": [2]}).equivalent
        False
    """

    left_cols = _to_columns(left)
    right_cols = _to_columns(right)

    if set(left_cols) != set(right_cols):
        return StatComparison(
            False, f"column sets differ: {sorted(left_cols)} vs {sorted(right_cols)}"
        )

    columns = sorted(left_cols)
    left_n = _row_count(left_cols)
    right_n = _row_count(right_cols)
    if left_n != right_n:
        return StatComparison(False, f"row counts differ: {left_n} vs {right_n}")

    left_rows = sorted(_rows(left_cols, columns), key=_row_sort_key)
    right_rows = sorted(_rows(right_cols, columns), key=_row_sort_key)

    for left_row, right_row in zip(left_rows, right_rows, strict=True):
        for column, left_value, right_value in zip(columns, left_row, right_row, strict=True):
            if not _values_close(left_value, right_value, rtol, atol):
                return StatComparison(
                    False,
                    f"column {column!r} differs after alignment: {left_value!r} vs {right_value!r}",
                )

    return StatComparison(True)


def statistical_check(
    left: object,
    right: object,
    fixture: object,
    *,
    framework: str,
    source_name: str = "input",
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> StatComparison:
    """Run both pipelines on a shared ``fixture`` and compare the outputs.

    ``framework`` selects how the pipelines are executed:

    * ``"pandas"`` -- ``left(df)`` / ``right(df)`` callables on a
      ``pandas.DataFrame``.
    * ``"polars"`` -- ``left(lf, pl)`` / ``right(lf, pl)`` callables on a
      ``polars.LazyFrame`` with the real ``polars`` module (the same
      signature the tracer uses, so an unchanged pipeline runs both ways).
    * ``"sql"`` -- ``left`` and ``right`` are SQL query strings, executed
      via sqlglot's built-in executor against the fixture (no new runtime
      dependency). ``source_name`` is the FROM table alias. Window
      functions are not supported by the executor and raise.
    """

    return compare_frames(
        run_pipeline(left, fixture, framework, source_name=source_name),
        run_pipeline(right, fixture, framework, source_name=source_name),
        rtol=rtol,
        atol=atol,
    )


def run_pipeline(
    pipeline: object,
    fixture: object,
    framework: str,
    *,
    source_name: str = "input",
    aux_sources: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
) -> dict[str, list[Any]]:
    """Execute one pipeline on a fixture and return ``{column: values}``.

    ``aux_sources`` (optional) names additional input tables the pipeline
    consumes. SQL pipelines that JOIN multiple tables get them passed to
    sqlglot's executor under their declared names. Polars pipelines that
    call ``pl.source(name, schema=...)`` (a tracing-namespace construct)
    get a thin wrapper namespace whose ``.source(...)`` returns a real
    ``LazyFrame`` built from the matching aux fixture, so the pipeline
    code runs unchanged under both the tracer and the statistical check.
    """

    columns = _to_columns(fixture)
    if framework == "pandas":
        if not callable(pipeline):
            raise UnsupportedOperationError(
                "statistical_check(framework='pandas'): pipeline must be a callable "
                "taking a pandas DataFrame."
            )
        import pandas as pd  # type: ignore[import-untyped]

        result: Any = pipeline(pd.DataFrame(columns))
        return _to_columns(result)
    if framework == "polars":
        if not callable(pipeline):
            raise UnsupportedOperationError(
                "statistical_check(framework='polars'): pipeline must be a callable "
                "taking (LazyFrame, pl)."
            )
        import warnings

        import polars as pl

        namespace: Any = _polars_namespace_with_aux(pl, aux_sources) if aux_sources else pl
        frame: Any = pl.LazyFrame(columns)
        output: Any = pipeline(frame, namespace)
        with warnings.catch_warnings():
            # Polars warns when ``join_asof(by=...)`` or ``rolling(group_by=...)``
            # is invoked on a frame whose sortedness it cannot verify. The
            # fixture is small and deterministic; the warning is informational,
            # not a correctness signal, and would otherwise clutter CI output.
            warnings.filterwarnings(
                "ignore",
                message=r"Sortedness of columns cannot be checked",
                category=UserWarning,
            )
            collected = output.collect() if hasattr(output, "collect") else output
        return _to_columns(collected)
    if framework == "sql":
        if not isinstance(pipeline, str):
            raise UnsupportedOperationError(
                "statistical_check(framework='sql'): pipeline must be a SQL query string."
            )
        return _run_sql(pipeline, columns, source_name=source_name, aux_sources=aux_sources)
    raise UnsupportedOperationError(
        f"statistical_check: framework {framework!r} is not supported; "
        f"'pandas', 'polars', and 'sql' are."
    )


def _polars_namespace_with_aux(
    pl_module: Any, aux_sources: Mapping[str, Mapping[str, Sequence[Any]]]
) -> Any:
    """Return a thin proxy over the real ``polars`` module that resolves
    ``pl.source(name, schema=...)`` to a real ``LazyFrame`` over the
    matching aux fixture. Everything else delegates to the real module."""

    fixtures = {
        name: {column: list(values) for column, values in columns.items()}
        for name, columns in aux_sources.items()
    }

    class _StatsPolarsNamespace:
        def __getattr__(self, item: str) -> Any:
            return getattr(pl_module, item)

        def source(self, name: str, *, schema: object) -> Any:
            del schema  # the real schema is whatever the fixture columns are
            if name not in fixtures:
                raise UnsupportedOperationError(
                    f"statistical_check(framework='polars'): pl.source({name!r}, ...) "
                    f"has no aux fixture; available aux sources are "
                    f"{sorted(fixtures)}"
                )
            return pl_module.LazyFrame(fixtures[name])

    return _StatsPolarsNamespace()


def _run_sql(
    query: str,
    columns: Mapping[str, Sequence[Any]],
    *,
    source_name: str,
    aux_sources: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
) -> dict[str, list[Any]]:
    """Execute ``query`` over the fixture using sqlglot's built-in executor.

    The executor handles SELECT / WHERE / GROUP BY / JOIN / ORDER BY but not
    window functions; the latter raise :class:`UnsupportedOperationError`
    rather than a sqlglot internal exception. ``aux_sources`` adds extra
    tables (each a ``{column: values}`` fixture) under their declared
    names so multi-table joins can run.
    """

    from sqlglot import executor
    from sqlglot.errors import ExecuteError

    tables = {source_name: _columns_to_rows(columns)}
    if aux_sources:
        for aux_name, aux_columns in aux_sources.items():
            if aux_name == source_name:
                continue
            tables[aux_name] = _columns_to_rows(aux_columns)

    try:
        result = executor.execute(query, tables=tables)
    except ExecuteError as exc:
        raise UnsupportedOperationError(
            f"statistical_check(sql): sqlglot's executor could not run the query "
            f"(window functions and some shapes are not supported): {exc}"
        ) from exc

    output_columns = tuple(result.columns)
    return {
        column: [row[index] for row in result.rows] for index, column in enumerate(output_columns)
    }


def _columns_to_rows(columns: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Reshape ``{column: values}`` into ``[{column: value} per row]`` for
    sqlglot's executor, which is row-oriented."""

    nrows = _row_count(columns)
    return [{column: columns[column][i] for column in columns} for i in range(nrows)]


def _to_columns(frame: object) -> dict[str, list[Any]]:
    """Normalize a table to ``{column: [values]}``. Accepts a mapping or a
    pandas / Polars DataFrame (detected by module name; not imported here)."""

    if isinstance(frame, Mapping):
        return {str(key): list(values) for key, values in frame.items()}

    module = type(frame).__module__.split(".")[0]
    if module == "pandas":
        df: Any = frame
        return {str(column): list(df[column]) for column in df.columns}
    if module == "polars":
        df_pl: Any = frame
        return {str(key): list(values) for key, values in df_pl.to_dict(as_series=False).items()}

    raise UnsupportedOperationError(
        f"compare_frames: cannot interpret {type(frame).__name__!r} as a table; "
        f"pass a {{column: values}} mapping or a pandas / Polars DataFrame."
    )


def _row_count(columns: Mapping[str, Sequence[Any]]) -> int:
    if not columns:
        return 0
    return len(next(iter(columns.values())))


def _rows(columns: Mapping[str, Sequence[Any]], order: list[str]) -> list[tuple[Any, ...]]:
    return list(zip(*(columns[column] for column in order), strict=True))


def _row_sort_key(row: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(_cell_key(value) for value in row)


def _cell_key(value: object) -> str:
    if value is None:
        return "\x00"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _values_close(left: object, right: object, rtol: float, atol: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, int | float) and isinstance(right, int | float):
        return abs(float(left) - float(right)) <= atol + rtol * abs(float(right))
    return left == right
