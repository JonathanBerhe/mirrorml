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
* :func:`statistical_check` -- run two pandas / Polars pipelines on a shared
  fixture and compare their outputs. SQL execution is deferred because it
  needs a query engine; pass already-computed output frames to
  :func:`compare_frames` for the SQL case.

These names are internal / experimental and not part of the public API.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    left: Callable[..., object],
    right: Callable[..., object],
    fixture: object,
    *,
    framework: str,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> StatComparison:
    """Run both pipelines on a shared ``fixture`` and compare the outputs.

    ``framework`` selects how the pipelines are executed:

    * ``"pandas"`` -- ``left(df)`` / ``right(df)`` on a ``pandas.DataFrame``.
    * ``"polars"`` -- ``left(lf, pl)`` / ``right(lf, pl)`` on a
      ``polars.LazyFrame`` with the real ``polars`` module (the same
      signature the tracer uses, so an unchanged pipeline runs both ways).

    SQL pipelines are not executed here (that needs a query engine); compute
    their outputs separately and call :func:`compare_frames`.
    """

    return compare_frames(
        _run(left, fixture, framework),
        _run(right, fixture, framework),
        rtol=rtol,
        atol=atol,
    )


def _run(pipeline: Callable[..., object], fixture: object, framework: str) -> dict[str, list[Any]]:
    columns = _to_columns(fixture)
    if framework == "pandas":
        import pandas as pd  # type: ignore[import-untyped]

        result: Any = pipeline(pd.DataFrame(columns))
        return _to_columns(result)
    if framework == "polars":
        import polars as pl

        frame: Any = pl.LazyFrame(columns)
        output: Any = pipeline(frame, pl)
        collected = output.collect() if hasattr(output, "collect") else output
        return _to_columns(collected)
    raise UnsupportedOperationError(
        f"statistical_check: framework {framework!r} cannot be executed in-process; "
        f"'pandas' and 'polars' are supported. SQL needs a query engine, so compute "
        f"its outputs separately and use compare_frames."
    )


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
