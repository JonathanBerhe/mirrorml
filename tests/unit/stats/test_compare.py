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


def test_statistical_check_sql_is_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="query engine"):
        statistical_check(lambda x: x, lambda x: x, {"a": [1]}, framework="sql")
