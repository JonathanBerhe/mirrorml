"""Tests for the evaluator's metric aggregation + CLI threshold flags."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from bench.scripts.run_eval import PairResult, aggregate, main


def _result(
    *,
    name: str = "p",
    category: str = "timezone_mismatch",
    tp: int = 0,
    fp: int = 0,
    fn: int = 0,
    is_identity: bool = False,
    expected: tuple[str, ...] = (),
    predicted: tuple[str, ...] = (),
    has_localization_target: bool = False,
    localization_top1: bool = False,
    localization_top3: bool = False,
) -> PairResult:
    return PairResult(
        name=name,
        bucket="synthetic",
        category=category,
        expected_categories=expected,
        predicted_categories=predicted,
        tp=tp,
        fp=fp,
        fn=fn,
        is_identity=is_identity,
        has_localization_target=has_localization_target,
        localization_top1=localization_top1,
        localization_top3=localization_top3,
    )


def test_empty_input_returns_zero_metrics() -> None:
    summary = aggregate([])
    assert summary["headline"]["pairs"] == 0
    assert summary["headline"]["precision"] == 0.0
    assert summary["headline"]["recall"] == 0.0
    assert summary["headline"]["f1"] == 0.0


def test_all_correct_predictions_give_unit_metrics() -> None:
    results = [
        _result(
            tp=1, fp=0, fn=0, expected=("timezone_mismatch",), predicted=("timezone_mismatch",)
        ),
        _result(tp=1, fp=0, fn=0, expected=("type_coercion",), predicted=("type_coercion",)),
    ]
    summary = aggregate(results)
    assert summary["headline"]["precision"] == 1.0
    assert summary["headline"]["recall"] == 1.0
    assert summary["headline"]["f1"] == 1.0


def test_one_false_positive_lowers_precision_not_recall() -> None:
    results = [
        _result(
            tp=1,
            fp=1,
            fn=0,
            expected=("timezone_mismatch",),
            predicted=("timezone_mismatch", "type_coercion"),
        ),
    ]
    summary = aggregate(results)
    assert summary["headline"]["precision"] == 0.5
    assert summary["headline"]["recall"] == 1.0


def test_one_false_negative_lowers_recall_not_precision() -> None:
    results = [
        _result(
            tp=1,
            fp=0,
            fn=1,
            expected=("timezone_mismatch", "type_coercion"),
            predicted=("timezone_mismatch",),
        ),
    ]
    summary = aggregate(results)
    assert summary["headline"]["precision"] == 1.0
    assert summary["headline"]["recall"] == 0.5


def test_identity_precision_tracked_separately() -> None:
    results = [
        _result(name="id_a", is_identity=True, tp=0, fp=0, fn=0),
        _result(name="id_b", is_identity=True, tp=0, fp=1, fn=0, predicted=("schema_drift",)),
        _result(
            tp=1,
            fp=0,
            fn=0,
            expected=("timezone_mismatch",),
            predicted=("timezone_mismatch",),
        ),
    ]
    summary = aggregate(results)
    assert summary["identity"]["pairs"] == 2
    assert summary["identity"]["clean"] == 1
    assert summary["identity"]["precision"] == 0.5


def test_per_category_metrics_excluded_for_identity_pairs() -> None:
    """Identity pairs are not tied to a taxonomy category; they should
    not contribute to the per-category buckets."""

    results = [
        _result(name="id", is_identity=True, category="identity"),
        _result(
            tp=1,
            fp=0,
            fn=0,
            expected=("timezone_mismatch",),
            predicted=("timezone_mismatch",),
        ),
    ]
    summary = aggregate(results)
    assert "identity" not in summary["by_category"]
    assert summary["by_category"]["timezone_mismatch"]["pairs"] == 1


def test_per_category_metrics_aggregate_within_category() -> None:
    results = [
        _result(
            tp=1, fp=0, fn=0, expected=("timezone_mismatch",), predicted=("timezone_mismatch",)
        ),
        _result(
            tp=1,
            fp=1,
            fn=0,
            expected=("timezone_mismatch",),
            predicted=("timezone_mismatch", "type_coercion"),
        ),
    ]
    summary = aggregate(results)
    cat = summary["by_category"]["timezone_mismatch"]
    assert cat["pairs"] == 2
    assert cat["tp"] == 2
    assert cat["fp"] == 1
    assert cat["fn"] == 0
    assert cat["precision"] == pytest.approx(2 / 3)
    assert cat["recall"] == 1.0


import pytest  # noqa: E402  (kept at bottom so the helper above stays minimal)

# --- localization aggregation -----------------------------------------------


def test_localization_aggregation_counts_top1_and_top3() -> None:
    results = [
        _result(
            name="hit", has_localization_target=True, localization_top1=True, localization_top3=True
        ),
        _result(
            name="top3_only",
            has_localization_target=True,
            localization_top1=False,
            localization_top3=True,
        ),
        _result(name="miss", has_localization_target=True),
        _result(name="no_target"),  # untagged; excluded from denominator
    ]
    summary = aggregate(results)
    loc = summary["localization"]
    assert loc["pairs"] == 3
    assert loc["top1"] == 1
    assert loc["top3"] == 2
    assert loc["top1_accuracy"] == pytest.approx(1 / 3)
    assert loc["top3_accuracy"] == pytest.approx(2 / 3)


def test_localization_block_empty_when_no_pairs_tagged() -> None:
    results = [_result(name="untagged")]
    summary = aggregate(results)
    loc = summary["localization"]
    assert loc["pairs"] == 0
    assert loc["top1_accuracy"] == 0.0
    assert loc["top3_accuracy"] == 0.0


# --- CLI threshold flags ----------------------------------------------------


def _write_minimal_pair(
    dir_: Path,
    *,
    name: str,
    category: str,
    expected: list[dict[str, str]],
    same_tz: bool,
) -> None:
    """Build a one-pair bucket that either matches expectations exactly
    or produces a known mismatch, for exercising the threshold flags."""

    dir_.mkdir(parents=True)
    (dir_ / "offline.sql").write_text("SELECT ts FROM events\n")
    (dir_ / "online.sql").write_text("SELECT ts FROM events\n")
    meta = {
        "name": name,
        "bucket": "synthetic",
        "category": category,
        "description": "fixture",
        "expected_divergences": expected,
        "offline": {
            "language": "sql",
            "source": "offline.sql",
            "schemas": {"events": [["ts", "timestamp[ns, UTC]"]]},
        },
        "online": {
            "language": "sql",
            "source": "online.sql",
            "schemas": {
                "events": [["ts", "timestamp[ns, UTC]" if same_tz else "timestamp[ns, US/Pacific]"]]
            },
        },
    }
    with (dir_ / "meta.yaml").open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def _patch_paths(monkeypatch: pytest.MonkeyPatch, *, pairs_root: Path, results_dir: Path) -> None:
    """Redirect the evaluator's globals to a tmp bucket layout."""

    import bench.scripts.run_eval as run_eval

    monkeypatch.setattr(run_eval, "PAIRS_ROOT", pairs_root)
    monkeypatch.setattr(run_eval, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(run_eval, "REPO_ROOT", pairs_root.parent)


def test_threshold_flag_passes_when_metrics_meet_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs_root = tmp_path / "pairs"
    results_dir = tmp_path / "results"
    _write_minimal_pair(
        pairs_root / "synthetic" / "timezone_mismatch" / "tz_001",
        name="tz_001",
        category="timezone_mismatch",
        expected=[{"category": "timezone_mismatch"}],
        same_tz=False,
    )
    _patch_paths(monkeypatch, pairs_root=pairs_root, results_dir=results_dir)

    rc = main(
        [
            "--quick",
            "--out",
            str(results_dir),
            "--fail-under-precision",
            "0.95",
            "--fail-under-recall",
            "0.80",
        ]
    )
    assert rc == 0
    with (results_dir / "synthetic.json").open() as f:
        summary = json.load(f)
    assert summary["headline"]["precision"] == 1.0


def test_threshold_flag_fails_when_recall_below_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs_root = tmp_path / "pairs"
    results_dir = tmp_path / "results"
    # Build a pair whose expected divergence is timezone_mismatch but
    # whose two sides are byte-identical -- the engine emits no
    # divergences, so recall is 0.
    _write_minimal_pair(
        pairs_root / "synthetic" / "timezone_mismatch" / "tz_clean",
        name="tz_clean",
        category="timezone_mismatch",
        expected=[{"category": "timezone_mismatch"}],
        same_tz=True,
    )
    _patch_paths(monkeypatch, pairs_root=pairs_root, results_dir=results_dir)

    rc = main(
        [
            "--quick",
            "--out",
            str(results_dir),
            "--fail-under-recall",
            "0.80",
        ]
    )
    assert rc == 1


def test_threshold_flag_fails_when_precision_below_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs_root = tmp_path / "pairs"
    results_dir = tmp_path / "results"
    # Identity-ish pair with no expected divergences, but two sides that
    # genuinely differ -- engine emits a FP, precision drops to 0.
    _write_minimal_pair(
        pairs_root / "synthetic" / "identity" / "id_dirty",
        name="id_dirty",
        category="identity",
        expected=[],
        same_tz=False,
    )
    _patch_paths(monkeypatch, pairs_root=pairs_root, results_dir=results_dir)

    rc = main(
        [
            "--quick",
            "--out",
            str(results_dir),
            "--fail-under-precision",
            "0.95",
        ]
    )
    assert rc == 1


def test_localization_threshold_passes_when_engine_localizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic timezone pair with expected_localization=source: the
    engine now attaches the Source op_id to input-schema divergences, so
    top-1 should hit and the threshold should pass."""

    pairs_root = tmp_path / "pairs"
    results_dir = tmp_path / "results"
    pair_dir = pairs_root / "synthetic" / "timezone_mismatch" / "tz_001"
    pair_dir.mkdir(parents=True)
    (pair_dir / "offline.sql").write_text("SELECT ts FROM events\n")
    (pair_dir / "online.sql").write_text("SELECT ts FROM events\n")
    meta = {
        "name": "tz_001",
        "bucket": "synthetic",
        "category": "timezone_mismatch",
        "description": "tz",
        "expected_divergences": [{"category": "timezone_mismatch"}],
        "expected_localization": [{"op_kind": "source", "side": "both"}],
        "offline": {
            "language": "sql",
            "source": "offline.sql",
            "schemas": {"events": [["ts", "timestamp[ns, UTC]"]]},
        },
        "online": {
            "language": "sql",
            "source": "online.sql",
            "schemas": {"events": [["ts", "timestamp[ns, US/Pacific]"]]},
        },
    }
    with (pair_dir / "meta.yaml").open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    _patch_paths(monkeypatch, pairs_root=pairs_root, results_dir=results_dir)

    rc = main(
        [
            "--quick",
            "--out",
            str(results_dir),
            "--fail-under-localization-top1",
            "0.75",
        ]
    )
    assert rc == 0
    with (results_dir / "synthetic.json").open() as f:
        summary = json.load(f)
    assert summary["localization"]["top1_accuracy"] == 1.0


def test_localization_threshold_fails_when_kind_mismatched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pair whose meta.yaml claims the wrong responsible op_kind should
    miss top-1, dragging the metric below the threshold."""

    pairs_root = tmp_path / "pairs"
    results_dir = tmp_path / "results"
    pair_dir = pairs_root / "synthetic" / "timezone_mismatch" / "tz_001"
    pair_dir.mkdir(parents=True)
    (pair_dir / "offline.sql").write_text("SELECT ts FROM events\n")
    (pair_dir / "online.sql").write_text("SELECT ts FROM events\n")
    meta = {
        "name": "tz_001",
        "bucket": "synthetic",
        "category": "timezone_mismatch",
        "description": "tz",
        "expected_divergences": [{"category": "timezone_mismatch"}],
        # Deliberately wrong: the responsible op is the Source.
        "expected_localization": [{"op_kind": "aggregate", "side": "both"}],
        "offline": {
            "language": "sql",
            "source": "offline.sql",
            "schemas": {"events": [["ts", "timestamp[ns, UTC]"]]},
        },
        "online": {
            "language": "sql",
            "source": "online.sql",
            "schemas": {"events": [["ts", "timestamp[ns, US/Pacific]"]]},
        },
    }
    with (pair_dir / "meta.yaml").open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    _patch_paths(monkeypatch, pairs_root=pairs_root, results_dir=results_dir)

    rc = main(
        [
            "--quick",
            "--out",
            str(results_dir),
            "--fail-under-localization-top1",
            "0.75",
        ]
    )
    assert rc == 1


def test_step_summary_is_written_when_flag_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs_root = tmp_path / "pairs"
    results_dir = tmp_path / "results"
    summary_file = tmp_path / "step.md"
    _write_minimal_pair(
        pairs_root / "synthetic" / "timezone_mismatch" / "tz_001",
        name="tz_001",
        category="timezone_mismatch",
        expected=[{"category": "timezone_mismatch"}],
        same_tz=False,
    )
    _patch_paths(monkeypatch, pairs_root=pairs_root, results_dir=results_dir)

    rc = main(
        [
            "--quick",
            "--out",
            str(results_dir),
            "--github-step-summary",
            str(summary_file),
        ]
    )
    assert rc == 0
    content = summary_file.read_text()
    assert "MirrorBench" in content
    assert "Precision" in content
    assert "1.000" in content
