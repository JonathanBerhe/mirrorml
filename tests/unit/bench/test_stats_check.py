"""Statistical-check tests on real bench pairs.

These exercise the full path: load a pair, generate a fixture from its
input schema, run both sides via the statistical companion check, and
compare. Identity pairs must come back equivalent; divergent pairs must
not (so the statistical companion agrees with the static diff on these
shapes).
"""

from __future__ import annotations

from pathlib import Path

from bench.scripts.stats_check import generate_fixture, statistically_check_pair

REPO_ROOT = Path(__file__).resolve().parents[3]
SYN = REPO_ROOT / "bench" / "pairs" / "synthetic"
REPLAYED = REPO_ROOT / "bench" / "pairs" / "replayed_bugs"


# --- generate_fixture -------------------------------------------------------


def test_generate_fixture_handles_common_dtypes() -> None:
    rows = 4
    fixture = generate_fixture(
        (
            ("uid", "int64"),
            ("score", "float64"),
            ("name", "utf8"),
            ("ts", "timestamp[ns, UTC]"),
            ("amount", "decimal[18, 2]"),
        ),
        n_rows=rows,
    )
    assert set(fixture) == {"uid", "score", "name", "ts", "amount"}
    assert all(len(values) == rows for values in fixture.values())


# --- identity pairs (must come back equivalent) -----------------------------


def test_synthetic_sql_identity_pair_is_statistically_equivalent() -> None:
    result, reason = statistically_check_pair(SYN / "identity" / "identity_simple_select")
    assert reason == "", reason
    assert result is not None and result.equivalent


def test_polars_pair_using_pl_source_is_skipped() -> None:
    # pl.source(...) is a tracing-namespace construct, not part of real
    # polars; pipelines that use it for a second input table cannot run
    # statistically here and are reported as a clean skip.
    result, reason = statistically_check_pair(REPLAYED / "pit_leakage_forward_asof")
    assert result is None
    assert "skipped" in reason


# --- divergent pairs (must come back not equivalent) ------------------------


def test_synthetic_aggregation_function_divergence_caught_statistically() -> None:
    result, reason = statistically_check_pair(
        SYN / "aggregation_function" / "aggregation_function_000"
    )
    assert reason == "", reason
    assert result is not None and not result.equivalent


def test_replayed_breck_fillna_divergence_caught_statistically() -> None:
    result, reason = statistically_check_pair(REPLAYED / "breck_rpc_error_sentinel")
    assert reason == "", reason
    assert result is not None and not result.equivalent


# --- skips (unsupported shapes) ---------------------------------------------


def test_window_pair_is_skipped_with_reason() -> None:
    result, reason = statistically_check_pair(
        SYN / "window_size_mismatch" / "window_size_mismatch_000"
    )
    assert result is None
    assert "window" in reason or "skipped" in reason


def test_multi_table_join_pair_is_skipped() -> None:
    # join pairs declare two source tables; the stats SQL path is currently
    # single-source so this side is skipped with a clear message.
    result, reason = statistically_check_pair(
        SYN / "join_key_mismatch" / "join_key_mismatch_single_key"
    )
    assert result is None
    assert "single-source" in reason or "skipped" in reason
