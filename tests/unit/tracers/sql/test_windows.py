"""SQL tracer: window functions (``<agg>(col) OVER (...)``).

Phase-1 surface is deliberately narrow: a trailing rows frame
(``ROWS BETWEEN {n|UNBOUNDED} PRECEDING AND CURRENT ROW``) with bare
passthrough columns alongside. Everything else is rejected rather than
fingerprinted approximately, so the diff engine never reports a false
equivalence between genuinely different windows.
"""

from __future__ import annotations

import pytest

from mirrorml import diff, trace_sql
from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint.operations import Window
from mirrorml.fingerprint.schema import Fingerprint

EVENTS = (
    ("uid", "int64"),
    ("ts", "timestamp[ns, UTC]"),
    ("score", "float64"),
)


def _trace(query: str) -> Fingerprint:
    return trace_sql(query, schemas={"events": EVENTS})


# --- emission ---------------------------------------------------------------


def test_window_op_emitted_with_expected_fields() -> None:
    fp = _trace(
        "SELECT uid, ts, AVG(score) OVER ("
        "PARTITION BY uid ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW"
        ") AS roll FROM events"
    )
    assert [op.kind for op in fp.operations] == ["source", "window"]
    win = fp.operations[1]
    assert isinstance(win, Window)
    assert win.over == ("uid",)
    assert win.order_by == ("ts",)
    assert win.size == "3rows"
    assert win.temporal.closed == "right"
    assert win.aggregations == (("roll", "score", "mean"),)


def test_window_passthrough_columns_in_output_schema() -> None:
    fp = _trace(
        "SELECT uid, ts, SUM(score) OVER ("
        "PARTITION BY uid ORDER BY ts ROWS BETWEEN 4 PRECEDING AND CURRENT ROW"
        ") AS roll FROM events"
    )
    assert fp.output_schema == (
        ("uid", "int64"),
        ("ts", "timestamp[ns, UTC]"),
        ("roll", "float64"),
    )


def test_unbounded_preceding_size() -> None:
    fp = _trace(
        "SELECT uid, AVG(score) OVER ("
        "PARTITION BY uid ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        ") AS roll FROM events"
    )
    win = fp.operations[1]
    assert isinstance(win, Window)
    assert win.size == "unbounded"


def test_window_without_partition_has_empty_over() -> None:
    fp = _trace(
        "SELECT AVG(score) OVER ("
        "ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW"
        ") AS roll FROM events"
    )
    win = fp.operations[1]
    assert isinstance(win, Window)
    assert win.over == ()
    assert win.order_by == ("ts",)


# --- divergences ------------------------------------------------------------


def test_identical_windows_diff_to_empty() -> None:
    q = (
        "SELECT uid, ts, AVG(score) OVER ("
        "PARTITION BY uid ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW"
        ") AS roll FROM events"
    )
    assert diff(_trace(q), _trace(q)) == ()


def test_different_window_size_surfaces_window_size_mismatch() -> None:
    base = (
        "SELECT uid, ts, AVG(score) OVER ("
        "PARTITION BY uid ORDER BY ts ROWS BETWEEN {n} PRECEDING AND CURRENT ROW"
        ") AS roll FROM events"
    )
    divs = diff(_trace(base.format(n=2)), _trace(base.format(n=5)))
    assert [d.category for d in divs] == ["window_size_mismatch"]


def test_different_window_aggregation_surfaces_aggregation_function() -> None:
    base = (
        "SELECT uid, ts, {agg}(score) OVER ("
        "PARTITION BY uid ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW"
        ") AS roll FROM events"
    )
    divs = diff(_trace(base.format(agg="AVG")), _trace(base.format(agg="SUM")))
    assert any(d.category == "aggregation_function" for d in divs)


def test_different_window_order_surfaces_ordering_dependence() -> None:
    base = (
        "SELECT uid, AVG(score) OVER ("
        "PARTITION BY uid ORDER BY {ord} ROWS BETWEEN 2 PRECEDING AND CURRENT ROW"
        ") AS roll FROM events"
    )
    divs = diff(_trace(base.format(ord="ts")), _trace(base.format(ord="score")))
    assert any(d.category == "ordering_dependence" for d in divs)


def test_different_partition_surfaces_join_key_mismatch() -> None:
    base = (
        "SELECT AVG(score) OVER ("
        "PARTITION BY {part} ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW"
        ") AS roll FROM events"
    )
    divs = diff(_trace(base.format(part="uid")), _trace(base.format(part="score")))
    assert any(d.category == "join_key_mismatch" for d in divs)


# --- rejections (conservative surface) --------------------------------------


def test_range_frame_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="ROWS"):
        _trace(
            "SELECT AVG(score) OVER ("
            "PARTITION BY uid ORDER BY ts RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            ") AS roll FROM events"
        )


def test_frameless_window_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="explicit ROWS frame"):
        _trace("SELECT AVG(score) OVER (PARTITION BY uid ORDER BY ts) AS roll FROM events")


def test_following_end_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="CURRENT ROW"):
        _trace(
            "SELECT AVG(score) OVER ("
            "PARTITION BY uid ORDER BY ts ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING"
            ") AS roll FROM events"
        )


def test_offset_preceding_end_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="CURRENT ROW"):
        _trace(
            "SELECT AVG(score) OVER ("
            "PARTITION BY uid ORDER BY ts ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING"
            ") AS roll FROM events"
        )


def test_window_with_group_by_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="GROUP BY"):
        _trace(
            "SELECT uid, AVG(score) OVER ("
            "PARTITION BY uid ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW"
            ") AS roll FROM events GROUP BY uid"
        )


def test_non_aggregate_window_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="non-aggregate"):
        _trace(
            "SELECT uid, ROW_NUMBER() OVER ("
            "PARTITION BY uid ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW"
            ") AS rn FROM events"
        )


def test_mixed_window_specs_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="different OVER specs"):
        _trace(
            "SELECT "
            "AVG(score) OVER (PARTITION BY uid ORDER BY ts "
            "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS a, "
            "SUM(score) OVER (PARTITION BY uid ORDER BY ts "
            "ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS b "
            "FROM events"
        )
