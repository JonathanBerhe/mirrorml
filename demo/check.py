"""Validate MirrorML end to end on the demo's feature pipelines.

Run from the repo root:

    uv run python demo/check.py

It traces the offline pandas pipeline and the online SQL pipeline, confirms
the correct versions are equivalent, then injects three realistic serving-side
mistakes and confirms MirrorML catches and localizes each one. The script is
self-checking: it exits non-zero if any expectation fails, so it can double as
a smoke test.
"""

import sys

from offline_features import ORDERS_SCHEMA, average_order_value
from online_features import CORRECT, SKEW_AGGREGATION, SKEW_MISSING_FILTER

from mirrorml import diff, trace_pandas, trace_sql

# Same orders table, but the warehouse column is in US/Pacific rather than UTC.
ORDERS_SCHEMA_PACIFIC = (
    ("customer_id", "int64"),
    ("amount", "float64"),
    ("ts", "timestamp[ns, US/Pacific]"),
)


def _online(query, schema):
    """Trace one online (SQL) pipeline against the orders schema."""
    return trace_sql(query, schemas={"orders": schema})


def _report(title, divergences):
    """Print one scenario's verdict and return the set of categories found."""
    print(f"\n{title}")
    if not divergences:
        print("    verdict: EQUIVALENT (diff is empty)")
        return set()
    print("    verdict: SKEW DETECTED")
    for d in divergences:
        print(f"      - {d.category}: {d.detail}")
    return {d.category for d in divergences}


def main():
    offline = trace_pandas(average_order_value, input_schema=ORDERS_SCHEMA, source_name="orders")

    print("MirrorML demo: churn feature pipelines (offline pandas vs online SQL)")
    print("=" * 70)

    failures = []

    # [1] Correct pipelines must be equivalent.
    cats = _report(
        "[1] Correct serving query (AVG of valid orders)",
        diff(offline, _online(CORRECT, ORDERS_SCHEMA)),
    )
    if cats:
        failures.append("correct pipelines were not reported equivalent")

    # [2] SUM instead of AVG -> aggregation_function.
    cats = _report(
        "[2] Serving bug: SUM instead of AVG",
        diff(offline, _online(SKEW_AGGREGATION, ORDERS_SCHEMA)),
    )
    if "aggregation_function" not in cats:
        failures.append("aggregation swap not caught")

    # [3] Dropped the `amount > 0` validity filter -> a missing-operation skew.
    cats = _report(
        "[3] Serving bug: dropped the `amount > 0` validity filter",
        diff(offline, _online(SKEW_MISSING_FILTER, ORDERS_SCHEMA)),
    )
    if "schema_drift" not in cats:
        failures.append("missing filter not caught")

    # [4] orders.ts read as US/Pacific instead of UTC -> timezone_mismatch.
    cats = _report(
        "[4] Serving bug: orders.ts read as US/Pacific, not UTC",
        diff(offline, _online(CORRECT, ORDERS_SCHEMA_PACIFIC)),
    )
    if "timezone_mismatch" not in cats:
        failures.append("timezone skew not caught")

    print("\n" + "=" * 70)
    if failures:
        for f in failures:
            print(f"FAILED: {f}")
        return 1
    print("All checks passed: every injected skew was flagged, the correct pair was not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
