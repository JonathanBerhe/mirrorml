"""MirrorBench evaluator. Walks ``bench/pairs/``, diffs each pair, and
writes per-pair + headline metrics to ``bench/results/<bucket>.json``.

Run directly:

.. code-block:: shell

    uv run python -m bench.scripts.run_eval                  # all buckets
    uv run python -m bench.scripts.run_eval --quick          # synthetic only

The "headline" block reports precision / recall / F1 computed at the
category level: a pair contributes a true positive when at least one
predicted divergence has a category in the expected set, a false
positive when it predicts a category not in the expected set, and a
false negative when an expected category is not in the predicted set.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.scripts.pair import Pair, discover_pairs, load_pair
from mirrorml import diff
from mirrorml._taxonomy import DIVERGENCE_CATEGORIES

REPO_ROOT = Path(__file__).resolve().parents[2]
PAIRS_ROOT = REPO_ROOT / "bench" / "pairs"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

BUCKETS = ("synthetic", "real_world", "replayed_bugs")

# An "identity" pair has no expected divergences; it tests precision.
# We give it its own marker category in meta.yaml so the evaluator can
# distinguish it from a real category that happens to have an empty
# expected list (which would be a malformed pair).
IDENTITY_CATEGORY = "identity"


@dataclass(frozen=True)
class PairResult:
    """Per-pair evaluation result."""

    name: str
    bucket: str
    category: str
    expected_categories: tuple[str, ...]
    predicted_categories: tuple[str, ...]
    tp: int
    fp: int
    fn: int
    is_identity: bool


def evaluate_pair(pair: Pair) -> PairResult:
    """Diff the pair's two fingerprints and score the prediction against
    the expected divergence categories.
    """

    divergences = diff(pair.offline, pair.online)
    predicted = {d.category for d in divergences}
    expected = {e.category for e in pair.expected}

    is_identity = pair.category == IDENTITY_CATEGORY or not expected

    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)

    return PairResult(
        name=pair.name,
        bucket=pair.bucket,
        category=pair.category,
        expected_categories=tuple(sorted(expected)),
        predicted_categories=tuple(sorted(predicted)),
        tp=tp,
        fp=fp,
        fn=fn,
        is_identity=is_identity,
    )


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def aggregate(results: list[PairResult]) -> dict[str, Any]:
    """Compute headline + per-category metrics from per-pair results."""

    total_tp = sum(r.tp for r in results)
    total_fp = sum(r.fp for r in results)
    total_fn = sum(r.fn for r in results)
    precision = _safe_div(total_tp, total_tp + total_fp)
    recall = _safe_div(total_tp, total_tp + total_fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    identity_pairs = [r for r in results if r.is_identity]
    identity_clean = sum(1 for r in identity_pairs if r.fp == 0)
    identity_precision = _safe_div(identity_clean, len(identity_pairs))

    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "pairs": 0}
    )
    for r in results:
        if r.is_identity:
            continue
        bucket = by_category[r.category]
        bucket["tp"] += r.tp
        bucket["fp"] += r.fp
        bucket["fn"] += r.fn
        bucket["pairs"] += 1

    category_metrics = {}
    for category in DIVERGENCE_CATEGORIES:
        if category not in by_category:
            continue
        b = by_category[category]
        category_metrics[category] = {
            "pairs": b["pairs"],
            "tp": b["tp"],
            "fp": b["fp"],
            "fn": b["fn"],
            "precision": _safe_div(b["tp"], b["tp"] + b["fp"]),
            "recall": _safe_div(b["tp"], b["tp"] + b["fn"]),
        }

    return {
        "headline": {
            "pairs": len(results),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "identity": {
            "pairs": len(identity_pairs),
            "clean": identity_clean,
            "precision": identity_precision,
        },
        "by_category": category_metrics,
        "by_pair": [
            {
                "name": r.name,
                "category": r.category,
                "expected": list(r.expected_categories),
                "predicted": list(r.predicted_categories),
                "tp": r.tp,
                "fp": r.fp,
                "fn": r.fn,
            }
            for r in results
        ],
    }


def evaluate_bucket(bucket: str) -> tuple[list[PairResult], dict[str, Any]]:
    bucket_root = PAIRS_ROOT / bucket
    pair_dirs = discover_pairs(bucket_root)
    results = [evaluate_pair(load_pair(d)) for d in pair_dirs]
    return results, aggregate(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_eval")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Evaluate only the synthetic bucket. Used in CI.",
    )
    parser.add_argument(
        "--bucket",
        choices=BUCKETS,
        action="append",
        help="Evaluate only the named bucket(s). Repeatable.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to write per-bucket result JSON. Defaults to bench/results/.",
    )
    args = parser.parse_args(argv)

    if args.quick and args.bucket:
        parser.error("--quick and --bucket are mutually exclusive")

    buckets: tuple[str, ...]
    if args.quick:
        buckets = ("synthetic",)
    elif args.bucket:
        buckets = tuple(args.bucket)
    else:
        buckets = BUCKETS

    args.out.mkdir(parents=True, exist_ok=True)

    overall_failed = False
    for bucket in buckets:
        _, summary = evaluate_bucket(bucket)
        out_path = args.out / f"{bucket}.json"
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=False)
            f.write("\n")
        head = summary["headline"]
        print(
            f"[{bucket}] pairs={head['pairs']} "
            f"precision={head['precision']:.3f} "
            f"recall={head['recall']:.3f} "
            f"f1={head['f1']:.3f} "
            f"-> {out_path.relative_to(REPO_ROOT)}"
        )
        if head["pairs"] == 0 and bucket == "synthetic":
            # synthetic should always have generated pairs; empty means
            # the generator was not run.
            overall_failed = True

    return 1 if overall_failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
