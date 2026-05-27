"""Tests for the latency harness + the performance-budget gate.

The budget assertions use the generous documented thresholds (500ms /
50ms) and measure the p95 over a handful of runs; the real numbers are
~0.4ms / ~0.2ms, so the assertions have a ~1000x safety margin and are
not timing-flaky. They run in the normal suite rather than being skipped.
"""

from __future__ import annotations

from bench.scripts.latency import (
    BUDGET_DIFF_MS,
    BUDGET_FINGERPRINT_MS,
    _fingerprint_n,
    _percentile,
    _time_ms,
    check_budgets,
    measure,
)

from mirrorml import diff


def test_percentile_handles_basic_and_empty() -> None:
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95) == 5.0
    assert _percentile([], 95) == 0.0


def test_measure_returns_expected_structure() -> None:
    results = measure(runs=3)
    assert set(results["fingerprint_ms_by_ops"]) == {"6", "11", "21"}
    assert "p50" in results["diff_ms"] and "p95" in results["diff_ms"]
    assert results["import_ms"] > 0


def test_fingerprint_20op_under_budget() -> None:
    timing = _time_ms(lambda: _fingerprint_n(20), runs=15, warmup=3)
    assert timing["p95"] < BUDGET_FINGERPRINT_MS


def test_diff_under_budget() -> None:
    left = _fingerprint_n(20)
    right = _fingerprint_n(20, start=1)
    timing = _time_ms(lambda: diff(left, right), runs=15, warmup=3)
    assert timing["p95"] < BUDGET_DIFF_MS


def test_check_budgets_passes_for_fast_results() -> None:
    fast = {
        "fingerprint_ms_by_ops": {"21": {"p95": 1.0}},
        "diff_ms": {"p95": 1.0},
        "import_ms": 50.0,
    }
    assert check_budgets(fast) == []


def test_check_budgets_flags_every_slow_result() -> None:
    slow = {
        "fingerprint_ms_by_ops": {"21": {"p95": 9999.0}},
        "diff_ms": {"p95": 9999.0},
        "import_ms": 9999.0,
    }
    failures = check_budgets(slow)
    assert len(failures) == 3
