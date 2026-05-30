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

import math
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

        def _invoke(frame_arg: Any) -> Any:
            output_local: Any = pipeline(frame_arg, namespace)
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
                return output_local.collect() if hasattr(output_local, "collect") else output_local

        try:
            collected = _invoke(pl.LazyFrame(columns))
        except AttributeError as exc:
            # Several polars methods are ``DataFrame``-only (``sample``,
            # ``to_dummies``, etc.); the pipeline reaches for them on a
            # ``LazyFrame`` and explodes. Retry once with an eager frame
            # iff the failure is specifically that polars's LazyFrame
            # lacks the attribute. We do NOT swallow generic
            # AttributeErrors (e.g. a typo inside a user UDF) -- those
            # would silently re-run the pipeline. Side-effect-free
            # pipelines (the bench's are deterministic) are unaffected
            # by the retry.
            if "'LazyFrame' object has no attribute" not in str(exc):
                raise
            collected = _invoke(pl.DataFrame(columns))
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
    """Execute ``query`` over the fixture using sqlglot's built-in executor,
    falling back to a polars-based translator for trailing-ROWS-frame
    window functions that sqlglot's executor does not implement.

    Supported shapes:

    * Plain SELECT / WHERE / GROUP BY / JOIN / ORDER BY via sqlglot's executor.
    * Single trailing-ROWS-frame window function over a single
      ``PARTITION BY`` column and single ``ORDER BY`` column, via the
      polars fallback. Agg functions: AVG, SUM, MIN, MAX, COUNT.
    * ``aux_sources`` adds extra tables under their declared names so
      multi-table joins can run.

    Other window shapes (multiple windows, RANGE frames, complex partition
    keys) raise :class:`UnsupportedOperationError` so the bench reports
    them as honest skips rather than silently returning wrong values.
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
        # The bench's window-function pairs trip sqlglot's executor; try
        # the trailing-ROWS-frame polars translator before giving up.
        # Building the column-oriented view of every input table is only
        # needed for the fallback, so defer it until the executor fails.
        fallback_columns: dict[str, Mapping[str, Sequence[Any]]] = {source_name: columns}
        if aux_sources:
            for aux_name, aux_columns in aux_sources.items():
                if aux_name != source_name:
                    fallback_columns[aux_name] = aux_columns
        fallback = _try_sql_window_via_polars(query, fallback_columns)
        if fallback is not None:
            return fallback
        raise UnsupportedOperationError(
            f"statistical_check(sql): sqlglot's executor could not run the query "
            f"and no fallback shape matched: {exc}"
        ) from exc

    output_columns = tuple(result.columns)
    return {
        column: [row[index] for row in result.rows] for index, column in enumerate(output_columns)
    }


def _try_sql_window_via_polars(
    query: str,
    tables_columns: Mapping[str, Mapping[str, Sequence[Any]]],
) -> dict[str, list[Any]] | None:
    """Translate a single trailing-ROWS-frame window query to polars and
    execute it. Returns ``None`` if the query is not in the supported
    shape; raises only if the query *is* in shape but a downstream
    polars call fails.

    Supported shape:

        SELECT <passthrough_cols>, <agg>(<col>) OVER (
            PARTITION BY <part_col>
            ORDER BY <order_col>
            ROWS BETWEEN (<n> PRECEDING | UNBOUNDED PRECEDING) AND CURRENT ROW
        ) AS <alias>
        FROM <table>

    Aggregation support:

    * ``UNBOUNDED PRECEDING`` (cumulative): AVG, SUM, MIN, MAX, COUNT.
    * ``<n> PRECEDING`` (rolling, window_size = n + 1): AVG, SUM, MIN, MAX.
      Bounded ``COUNT`` would need ``rolling_map`` over a per-window
      length, which the translator does not yet implement; bounded
      ``COUNT`` queries fall through to the caller's
      ``UnsupportedOperationError``.
    """

    import polars as pl
    from sqlglot import exp, parse_one
    from sqlglot.errors import ParseError

    try:
        ast = parse_one(query)
    except ParseError:
        return None
    if not isinstance(ast, exp.Select):
        return None

    windows = list(ast.find_all(exp.Window))
    if len(windows) != 1:
        return None
    window = windows[0]

    spec = window.args.get("spec")
    if spec is None or spec.args.get("kind") != "ROWS":
        return None
    if spec.args.get("end") != "CURRENT ROW":
        return None
    if spec.args.get("start_side") != "PRECEDING":
        return None

    agg = window.this
    agg_kind_map = {
        exp.Avg: "avg",
        exp.Sum: "sum",
        exp.Min: "min",
        exp.Max: "max",
        exp.Count: "count",
    }
    agg_kind = next((k for cls, k in agg_kind_map.items() if isinstance(agg, cls)), None)
    if agg_kind is None:
        return None
    if not isinstance(agg.this, exp.Column):
        return None
    agg_col_name = agg.this.name

    partition_by = window.args.get("partition_by") or []
    if len(partition_by) != 1 or not isinstance(partition_by[0], exp.Column):
        return None
    partition_col = partition_by[0].name

    order = window.args.get("order")
    if order is None or len(order.expressions) != 1:
        return None
    ordered = order.expressions[0]
    if not isinstance(ordered.this, exp.Column):
        return None
    order_col = ordered.this.name

    start = spec.args.get("start")
    window_size: int | None
    if start == "UNBOUNDED":
        window_size = None
    elif isinstance(start, exp.Literal):
        try:
            window_size = int(start.this) + 1  # N PRECEDING + CURRENT ROW = N+1
        except (TypeError, ValueError):
            return None
    else:
        return None

    # sqlglot >= 11 uses "from_" as the args key (verified against the pinned
    # version). The historical "from" alias is no longer emitted, so we no
    # longer probe for it.
    from_ = ast.args.get("from_")
    if from_ is None or not isinstance(from_.this, exp.Table):
        return None
    table_name = from_.this.name
    if table_name not in tables_columns:
        return None

    output_cols: list[str] = []
    window_alias: str | None = None
    for select_expr in ast.expressions:
        if isinstance(select_expr, exp.Alias) and isinstance(select_expr.this, exp.Window):
            window_alias = select_expr.alias
            output_cols.append(window_alias)
        elif isinstance(select_expr, exp.Column):
            output_cols.append(select_expr.name)
        else:
            return None
    if window_alias is None:
        return None

    table_columns = {col: list(values) for col, values in tables_columns[table_name].items()}
    df = pl.DataFrame(table_columns).sort([partition_col, order_col])

    src = pl.col(agg_col_name)
    if window_size is None:
        cum_map = {
            "avg": (src.cum_sum() / src.cum_count()).over(partition_col),
            "sum": src.cum_sum().over(partition_col),
            "min": src.cum_min().over(partition_col),
            "max": src.cum_max().over(partition_col),
            "count": src.cum_count().over(partition_col),
        }
        agg_expr = cum_map[agg_kind]
    else:
        rolling_map = {
            "avg": src.rolling_mean(window_size=window_size).over(partition_col),
            "sum": src.rolling_sum(window_size=window_size).over(partition_col),
            "min": src.rolling_min(window_size=window_size).over(partition_col),
            "max": src.rolling_max(window_size=window_size).over(partition_col),
        }
        if agg_kind not in rolling_map:
            # No rolling_count in polars; punt rather than misimplement.
            return None
        agg_expr = rolling_map[agg_kind]

    result = df.with_columns(agg_expr.alias(window_alias)).select(output_cols)
    return {col: list(result.get_column(col)) for col in result.columns}


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
        if math.isnan(value):
            # Sort NaN with None so two outputs that produce NaN at the
            # same logical position align in the sorted comparison.
            return "\x00nan"
        return f"{value:.12g}"
    return str(value)


def _values_close(left: object, right: object, rtol: float, atol: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, int | float) and isinstance(right, int | float):
        left_f, right_f = float(left), float(right)
        # Two NaNs at the same position mean the pipelines agree on
        # "no value here"; that should not register as a divergence.
        if math.isnan(left_f) and math.isnan(right_f):
            return True
        if math.isnan(left_f) or math.isnan(right_f):
            return False
        return abs(left_f - right_f) <= atol + rtol * abs(right_f)
    return left == right
