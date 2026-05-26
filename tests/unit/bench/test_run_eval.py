"""Tests for the evaluator's metric aggregation."""

from __future__ import annotations

from bench.scripts.run_eval import PairResult, aggregate


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
