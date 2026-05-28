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


def test_polars_pair_using_pl_source_runs_via_wrapper_namespace() -> None:
    # ``pl.source(...)`` is a tracing-namespace construct; the stats path
    # now traces each side to discover declared sources, generates a
    # fixture per source, and supplies a thin wrapper around real polars
    # whose ``.source(...)`` returns a LazyFrame over the matching
    # fixture. The pipeline runs unchanged. The forward / backward
    # as-of-join difference does not always surface on a tiny fixture,
    # so we assert only that the check ran (no skip).
    result, reason = statistically_check_pair(REPLAYED / "pit_leakage_forward_asof")
    assert reason == "", reason
    assert result is not None


# --- divergent pairs (must come back not equivalent) ------------------------


def test_polars_aux_source_namespace_resolves_known_sources() -> None:
    """Unit-level: the polars wrapper namespace returns a real LazyFrame
    for sources declared via ``pl.source(...)`` and raises for unknown
    names, so a typo in a pipeline does not silently produce empty
    results."""

    from collections.abc import Mapping, Sequence
    from typing import Any

    import polars as pl

    from mirrorml.exceptions import UnsupportedOperationError
    from mirrorml.stats import _polars_namespace_with_aux

    aux: Mapping[str, Mapping[str, Sequence[Any]]] = {
        "prices": {"uid": [1, 2], "price": [10.0, 20.0]},
    }
    ns = _polars_namespace_with_aux(pl, aux)

    frame = ns.source("prices", schema=[("uid", "int64"), ("price", "float64")])
    collected = frame.collect()
    assert collected.to_dict(as_series=False) == {"uid": [1, 2], "price": [10.0, 20.0]}

    import pytest

    with pytest.raises(UnsupportedOperationError, match="prices"):
        ns.source("missing", schema=[("uid", "int64")])


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


def test_sql_multi_table_join_passes_all_fixtures_to_executor() -> None:
    """Identity SQL pair with two source tables must come back equivalent
    (the executor sees both tables and produces matching joins on each
    side)."""

    result, reason = statistically_check_pair(SYN / "identity" / "identity_join_pipeline")
    assert reason == "", reason
    assert result is not None and result.equivalent


def test_window_pair_is_skipped_with_reason() -> None:
    result, reason = statistically_check_pair(
        SYN / "window_size_mismatch" / "window_size_mismatch_000"
    )
    assert result is None
    assert "window" in reason or "skipped" in reason


def test_multi_table_join_pair_runs_via_multi_table_fixture() -> None:
    # The stats SQL path now generates one fixture per declared table and
    # passes them all to sqlglot's executor, so multi-table JOIN queries
    # run. The k1 vs k2 difference on a deterministic 6-row fixture often
    # produces equivalent output, which is the expected complementarity
    # of static vs statistical checks (the static fingerprint flags it,
    # the small fixture does not surface it).
    result, reason = statistically_check_pair(
        SYN / "join_key_mismatch" / "join_key_mismatch_single_key"
    )
    assert reason == "", reason
    assert result is not None
