"""MirrorBench latency harness.

Measures the latencies in the project's performance budgets ("the system
is fast enough for CI use"):

* **fingerprint latency** for pipelines of growing size (the budget names
  a 20-op pandas pipeline at < 500ms),
* **diff latency** for two ~20-op fingerprints (budget < 50ms),

plus cold ``import mirrorml`` time (budget < 200ms), measured in a fresh
subprocess so it reflects a real first import.

Writes ``bench/results/latency.json`` and, with ``--check``, exits
non-zero if any p95 exceeds its budget.

Run:

.. code-block:: shell

    uv run python -m bench.scripts.latency
    uv run python -m bench.scripts.latency --check
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mirrorml import diff, trace_pandas
from mirrorml.fingerprint.schema import ColumnSpec, Fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"

_SCHEMA: tuple[ColumnSpec, ...] = (("uid", "int64"), ("score", "float64"))

# Budgets (milliseconds) from the project's performance budgets.
BUDGET_FINGERPRINT_MS = 500.0
BUDGET_DIFF_MS = 50.0
BUDGET_IMPORT_MS = 200.0


def _n_op_pipeline(n_ops: int, *, start: int = 0) -> Callable[[Any], Any]:
    """A pandas pipeline of ``n_ops`` chained filters with distinct
    thresholds (distinct so canonicalization keeps them as separate ops)."""

    def pipeline(df: Any) -> Any:
        frame = df
        for i in range(start, start + n_ops):
            frame = frame[frame["score"] > i]
        return frame

    return pipeline


def _fingerprint_n(n_ops: int, *, start: int = 0) -> Fingerprint:
    return trace_pandas(_n_op_pipeline(n_ops, start=start), input_schema=_SCHEMA, source_name="t")


def _percentile(samples: list[float], pct: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        return 0.0
    k = round((pct / 100.0) * (len(ordered) - 1))
    return ordered[max(0, min(len(ordered) - 1, k))]


def _time_ms(thunk: Callable[[], object], *, runs: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        thunk()
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        thunk()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {"p50": _percentile(samples, 50), "p95": _percentile(samples, 95)}


def _measure_import_ms() -> float:
    """Cold ``import mirrorml`` time, measured in a fresh subprocess."""

    code = (
        "import time; s=time.perf_counter(); import mirrorml; print((time.perf_counter()-s)*1000)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def measure(*, runs: int) -> dict[str, Any]:
    """Measure fingerprint, diff, and import latencies."""

    fingerprint: dict[str, dict[str, float]] = {}
    for n_ops in (5, 10, 20):
        # +1 for the Source op; "20-op" is the budget's reference size.
        fingerprint[str(n_ops + 1)] = _time_ms(
            functools.partial(_fingerprint_n, n_ops), runs=runs, warmup=2
        )

    left = _fingerprint_n(20)
    right = _fingerprint_n(20, start=1)
    diff_timing = _time_ms(lambda: diff(left, right), runs=runs, warmup=2)

    return {
        "fingerprint_ms_by_ops": fingerprint,
        "diff_ms": diff_timing,
        "import_ms": _measure_import_ms(),
        "budgets_ms": {
            "fingerprint_20op": BUDGET_FINGERPRINT_MS,
            "diff": BUDGET_DIFF_MS,
            "import": BUDGET_IMPORT_MS,
        },
        "runs": runs,
    }


def check_budgets(results: dict[str, Any]) -> list[str]:
    """Return a list of budget-violation messages (empty if all pass)."""

    failures: list[str] = []
    fp_20 = results["fingerprint_ms_by_ops"]["21"]["p95"]
    if fp_20 > BUDGET_FINGERPRINT_MS:
        failures.append(f"fingerprint 20-op p95 {fp_20:.1f}ms > {BUDGET_FINGERPRINT_MS:.0f}ms")
    diff_p95 = results["diff_ms"]["p95"]
    if diff_p95 > BUDGET_DIFF_MS:
        failures.append(f"diff p95 {diff_p95:.1f}ms > {BUDGET_DIFF_MS:.0f}ms")
    import_ms = results["import_ms"]
    if import_ms > BUDGET_IMPORT_MS:
        failures.append(f"import {import_ms:.1f}ms > {BUDGET_IMPORT_MS:.0f}ms")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="latency")
    parser.add_argument("--runs", type=int, default=50, help="Timed runs per measurement.")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="Output directory.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any p95 exceeds its performance budget.",
    )
    args = parser.parse_args(argv)

    results = measure(runs=args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "latency.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, sort_keys=False)
        f.write("\n")

    fp_20 = results["fingerprint_ms_by_ops"]["21"]
    print(
        f"fingerprint(20op) p50={fp_20['p50']:.1f}ms p95={fp_20['p95']:.1f}ms  "
        f"diff p95={results['diff_ms']['p95']:.2f}ms  "
        f"import={results['import_ms']:.1f}ms  -> {out_path.relative_to(REPO_ROOT)}"
    )

    if args.check:
        failures = check_budgets(results)
        for message in failures:
            print(f"FAIL: {message}", file=sys.stderr)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
