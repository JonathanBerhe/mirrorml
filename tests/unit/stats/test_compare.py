"""Tests for the statistical companion check (compare_frames + statistical_check).

pandas and Polars are available in the dev environment, so the executor
paths run real pipelines on small in-memory fixtures.
"""

from __future__ import annotations

from typing import Any

import pytest

from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.stats import compare_frames, statistical_check

# --- compare_frames ---------------------------------------------------------


def test_identical_frames_are_equivalent() -> None:
    frame = {"uid": [1, 2], "score": [0.5, 0.25]}
    assert compare_frames(frame, dict(frame)).equivalent


def test_comparison_is_order_insensitive() -> None:
    left = {"uid": [1, 2, 3], "score": [10.0, 20.0, 30.0]}
    right = {"uid": [3, 1, 2], "score": [30.0, 10.0, 20.0]}
    assert compare_frames(left, right).equivalent


def test_numeric_within_tolerance_is_equivalent() -> None:
    assert compare_frames({"x": [1.0, 2.0]}, {"x": [1.0 + 1e-9, 2.0 - 1e-9]}).equivalent


def test_numeric_beyond_tolerance_is_not_equivalent() -> None:
    result = compare_frames({"x": [1.0]}, {"x": [1.5]})
    assert not result.equivalent
    assert "x" in result.detail


def test_nan_at_same_position_is_equivalent() -> None:
    """Two pipelines that both leave a cell as NaN agree; ``nan == nan``
    is ``False`` in Python so the comparator must special-case this or it
    would falsely flag identity-UDF pairs as divergent."""

    left = {"x": [1.0, float("nan"), 3.0]}
    right = {"x": [1.0, float("nan"), 3.0]}
    assert compare_frames(left, right).equivalent


def test_nan_vs_value_is_not_equivalent() -> None:
    """NaN on one side and a real number on the other is a real
    divergence."""

    result = compare_frames({"x": [1.0]}, {"x": [float("nan")]})
    assert not result.equivalent


def test_column_set_mismatch_is_not_equivalent() -> None:
    result = compare_frames({"a": [1]}, {"b": [1]})
    assert not result.equivalent
    assert "column sets differ" in result.detail


def test_row_count_mismatch_is_not_equivalent() -> None:
    result = compare_frames({"a": [1, 2]}, {"a": [1]})
    assert not result.equivalent
    assert "row counts differ" in result.detail


def test_non_numeric_values_compared_exactly() -> None:
    assert compare_frames({"k": ["a", "b"]}, {"k": ["a", "b"]}).equivalent
    assert not compare_frames({"k": ["a"]}, {"k": ["b"]}).equivalent


def test_bool_is_not_tolerance_matched_to_a_near_float() -> None:
    # A bool must not be considered "close" to a nearly-equal float; only
    # exact equality applies. (True == 1 is genuinely equal in Python, so
    # that case is intentionally not asserted here.)
    assert not compare_frames({"flag": [True]}, {"flag": [False]}).equivalent
    assert not compare_frames({"flag": [True]}, {"flag": [0.9999999]}).equivalent


def test_accepts_pandas_dataframe() -> None:
    pd = pytest.importorskip("pandas")
    left = pd.DataFrame({"uid": [1, 2], "score": [1.0, 2.0]})
    right = pd.DataFrame({"uid": [2, 1], "score": [2.0, 1.0]})
    assert compare_frames(left, right).equivalent


# --- statistical_check (execution) ------------------------------------------


def test_statistical_check_pandas_equivalent_pipelines() -> None:
    pytest.importorskip("pandas")
    fixture = {"uid": [1, 1, 2], "score": [1.0, 3.0, 5.0]}

    def left(df: Any) -> Any:
        return df[df["score"] > 0].groupby("uid").agg({"score": "mean"}).reset_index()

    def right(df: Any) -> Any:
        # Same computation, different but equivalent spelling.
        filtered = df[df["score"] > 0]
        return filtered.groupby("uid", as_index=False).agg({"score": "mean"})

    assert statistical_check(left, right, fixture, framework="pandas").equivalent


def test_statistical_check_pandas_divergent_pipelines() -> None:
    pytest.importorskip("pandas")
    fixture = {"uid": [1, 1, 2], "score": [1.0, 3.0, 5.0]}

    def left(df: Any) -> Any:
        return df.groupby("uid", as_index=False).agg({"score": "mean"})

    def right(df: Any) -> Any:
        return df.groupby("uid", as_index=False).agg({"score": "sum"})

    assert not statistical_check(left, right, fixture, framework="pandas").equivalent


def test_statistical_check_polars_equivalent_pipelines() -> None:
    pytest.importorskip("polars")
    fixture = {"uid": [1, 1, 2], "score": [1.0, 3.0, 5.0]}

    def left(lf: Any, pl: Any) -> Any:
        return lf.filter(pl.col("score") > 0).group_by("uid").agg(pl.col("score").mean())

    def right(lf: Any, pl: Any) -> Any:
        return lf.group_by("uid").agg(pl.col("score").mean())

    assert statistical_check(left, right, fixture, framework="polars").equivalent


def test_statistical_check_sql_equivalent_queries() -> None:
    fixture = {"uid": [1, 1, 2], "score": [1.0, 3.0, 5.0]}
    left = "SELECT uid, AVG(score) AS score FROM events GROUP BY uid"
    right = "SELECT uid, AVG(score) AS score FROM events WHERE 1 = 1 GROUP BY uid"
    assert statistical_check(left, right, fixture, framework="sql", source_name="events").equivalent


def test_statistical_check_sql_divergent_queries() -> None:
    fixture = {"uid": [1, 1, 2], "score": [1.0, 3.0, 5.0]}
    left = "SELECT uid, AVG(score) AS score FROM events GROUP BY uid"
    right = "SELECT uid, SUM(score) AS score FROM events GROUP BY uid"
    result = statistical_check(left, right, fixture, framework="sql", source_name="events")
    assert not result.equivalent
    assert "score" in result.detail


def test_statistical_check_sql_trailing_rows_window_runs_via_polars_fallback() -> None:
    """Trailing-ROWS-frame window functions used to be rejected because
    sqlglot's executor does not support them; the SQL stats path now
    falls back to a focused polars translator for that shape, so the
    same query on both sides comes back equivalent."""

    fixture = {
        "uid": [1, 1, 1, 2, 2, 2],
        "ts": [1, 2, 3, 1, 2, 3],
        "score": [10.0, 20.0, 30.0, 5.0, 15.0, 25.0],
    }
    query = (
        "SELECT uid, ts, AVG(score) OVER (PARTITION BY uid ORDER BY ts "
        "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS roll FROM events"
    )
    result = statistical_check(query, query, fixture, framework="sql", source_name="events")
    assert result.equivalent, result.detail


def test_statistical_check_sql_unbounded_preceding_window_uses_cumulative_branch() -> None:
    """The polars fallback covers two frame shapes: bounded
    ``<n> PRECEDING`` (``rolling_*``) and ``UNBOUNDED PRECEDING``
    (cumulative). Pin the cumulative branch end-to-end so an accidental
    integer-division or dtype-promotion regression in cum_sum/cum_count
    surfaces here rather than silently in the bench.
    """

    from collections.abc import Mapping, Sequence

    from mirrorml.stats import _try_sql_window_via_polars

    fixture: Mapping[str, Sequence[float]] = {
        "uid": [1, 1, 1, 2, 2, 2],
        "ts": [1, 2, 3, 1, 2, 3],
        "score": [10.0, 20.0, 30.0, 5.0, 15.0, 25.0],
    }
    query = (
        "SELECT uid, ts, AVG(score) OVER (PARTITION BY uid ORDER BY ts "
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS roll FROM events"
    )
    result = _try_sql_window_via_polars(query, {"events": fixture})
    assert result is not None
    # Per partition: cumulative mean of (10, 20, 30) = (10, 15, 20);
    # of (5, 15, 25) = (5, 10, 15). Confirms cum_sum / cum_count gives
    # float division, not int division.
    assert result["roll"] == [10.0, 15.0, 20.0, 5.0, 10.0, 15.0]


def test_statistical_check_sql_unsupported_window_shape_still_rejected() -> None:
    """RANGE frames (and other shapes outside the bench's vocabulary) are
    not translated; the stats path raises so the bench reports an honest
    skip rather than silently returning wrong values."""

    fixture = {"uid": [1, 2], "score": [1.0, 2.0]}
    # RANGE BETWEEN ... uses ts-distance semantics that the focused
    # translator deliberately does not handle.
    query = (
        "SELECT uid, AVG(score) OVER (PARTITION BY uid ORDER BY score "
        "RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) AS roll FROM events"
    )
    with pytest.raises(UnsupportedOperationError, match="no fallback shape matched"):
        statistical_check(query, query, fixture, framework="sql", source_name="events")


def test_statistical_check_unknown_framework_is_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="not supported"):
        statistical_check(lambda x: x, lambda x: x, {"a": [1]}, framework="duckdb")


def test_statistical_check_pandas_with_non_callable_is_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="must be a callable"):
        statistical_check("not a callable", "also not", {"a": [1]}, framework="pandas")


def test_statistical_check_sql_with_non_string_is_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="SQL query string"):
        statistical_check(lambda x: x, lambda x: x, {"a": [1]}, framework="sql")
