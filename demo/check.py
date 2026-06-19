"""Validate MirrorML end to end on the demo's feature pipelines.

Run from the repo root:

    uv run python demo/check.py

Two layers of checking run, both self-asserting (the script exits non-zero if
any expectation fails, so it can double as a smoke test):

1. STATIC: trace the offline pandas pipeline and the online SQL pipeline, diff
   them, and confirm the correct pair is equivalent while three realistic
   serving-side mistakes are each caught and localized.

2. STATISTICAL: actually run both pipelines on a small in-memory fixture
   (pandas executes the offline side; the online SQL runs through sqlglot's
   built-in executor, so no database is needed) and compare the output values.
   This is the statistical companion check. It needs pandas, so it skips
   cleanly if pandas is not installed.

The timezone case is the interesting one: the static check flags it (the two
sides declare the column in different timezones), but the statistical check
reports the outputs as equal, because the feature does not aggregate on that
column. That is the hybrid design working as intended: the two checks are
complementary, not redundant.
"""

import sys

from offline_features import ORDERS_SCHEMA, average_order_value
from online_features import CORRECT, SKEW_AGGREGATION, SKEW_MISSING_FILTER

from mirrorml import diff, trace_pandas, trace_sql
from mirrorml.stats import compare_frames, run_pipeline

# Same orders table, but the warehouse column is in US/Pacific rather than UTC.
ORDERS_SCHEMA_PACIFIC = (
    ("customer_id", "int64"),
    ("amount", "float64"),
    ("ts", "timestamp[ns, US/Pacific]"),
)

# A small deterministic fixture for the statistical check. customer 1 has two
# valid orders and one refund (negative); customer 2 has one valid order and a
# zero. The `amount > 0` filter and AVG-vs-SUM differences therefore produce
# genuinely different numbers, so a value comparison can tell them apart. ts is
# present but unused by the feature, which is what makes the timezone skew
# invisible to the statistical check.
FIXTURE: dict[str, list[object]] = {
    "customer_id": [1, 1, 1, 2, 2],
    "amount": [10.0, 20.0, -5.0, 100.0, 0.0],
    "ts": [None, None, None, None, None],
}


def _online(query, schema):
    """Trace one online (SQL) pipeline against the orders schema."""
    return trace_sql(query, schemas={"orders": schema})


def _static_report(title, divergences):
    """Print one static scenario's verdict; return the categories found."""
    print(f"\n{title}")
    if not divergences:
        print("    static : EQUIVALENT (diff is empty)")
        return set()
    print("    static : SKEW DETECTED")
    for d in divergences:
        print(f"      - {d.category}: {d.detail}")
    return {d.category for d in divergences}


def run_static():
    """The static trace-and-diff layer. Returns a list of failure messages."""
    offline = trace_pandas(average_order_value, input_schema=ORDERS_SCHEMA, source_name="orders")
    failures = []

    cats = _static_report(
        "[1] Correct serving query (AVG of valid orders)",
        diff(offline, _online(CORRECT, ORDERS_SCHEMA)),
    )
    if cats:
        failures.append("static: correct pipelines were not reported equivalent")

    cats = _static_report(
        "[2] Serving bug: SUM instead of AVG",
        diff(offline, _online(SKEW_AGGREGATION, ORDERS_SCHEMA)),
    )
    if "aggregation_function" not in cats:
        failures.append("static: aggregation swap not caught")

    cats = _static_report(
        "[3] Serving bug: dropped the `amount > 0` validity filter",
        diff(offline, _online(SKEW_MISSING_FILTER, ORDERS_SCHEMA)),
    )
    if "schema_drift" not in cats:
        failures.append("static: missing filter not caught")

    cats = _static_report(
        "[4] Serving bug: orders.ts read as US/Pacific, not UTC",
        diff(offline, _online(CORRECT, ORDERS_SCHEMA_PACIFIC)),
    )
    if "timezone_mismatch" not in cats:
        failures.append("static: timezone skew not caught")

    return failures


def _offline_values():
    """Run the offline pandas pipeline on the fixture and return its output as
    a plain ``{column: values}`` mapping (the group key is the index, so it is
    reset into a column before comparison)."""
    import pandas as pd

    out = average_order_value(pd.DataFrame(FIXTURE)).reset_index()
    return {str(col): list(out[col]) for col in out.columns}


def _online_values(query):
    """Run an online SQL query on the fixture via sqlglot's executor."""
    return run_pipeline(query, FIXTURE, "sql", source_name="orders")


def _stat_report(title, result, expect_equivalent):
    """Print one statistical scenario's verdict; return a failure message or None."""
    verdict = "EQUIVALENT" if result.equivalent else "VALUES DIFFER"
    print(f"\n{title}")
    print(f"    stats  : {verdict}{('' if result.equivalent else ': ' + result.detail)}")
    if result.equivalent != expect_equivalent:
        return f"stats: {title!r} expected equivalent={expect_equivalent}, got {result.equivalent}"
    return None


def run_statistical():
    """The statistical layer: execute both pipelines on the fixture and compare
    output values. Returns a list of failure messages, or None if pandas is not
    installed (in which case this layer is skipped)."""
    try:
        import pandas  # noqa: F401
    except ModuleNotFoundError:
        return None

    offline = _offline_values()
    failures = []

    for title, query, expect_equivalent in [
        ("[5] Correct serving query: values should match", CORRECT, True),
        ("[6] SUM instead of AVG: values should differ", SKEW_AGGREGATION, False),
        ("[7] Dropped validity filter: values should differ", SKEW_MISSING_FILTER, False),
    ]:
        msg = _stat_report(title, compare_frames(offline, _online_values(query)), expect_equivalent)
        if msg:
            failures.append(msg)

    # The complementary case: the timezone skew changes only the declared dtype
    # of an unused column, so the computed values are identical. The static
    # check flagged it; the statistical check correctly sees no value difference.
    msg = _stat_report(
        "[8] Timezone skew: values match (static caught what stats cannot)",
        compare_frames(offline, _online_values(CORRECT)),
        expect_equivalent=True,
    )
    if msg:
        failures.append(msg)

    return failures


def main():
    print("MirrorML demo: churn feature pipelines (offline pandas vs online SQL)")
    print("=" * 70)
    print("\n--- STATIC: trace and diff (no data needed) ---")
    failures = run_static()

    print("\n\n--- STATISTICAL: run both pipelines on a fixture, compare values ---")
    stat_failures = run_statistical()
    if stat_failures is None:
        print("\n    (skipped: pandas is not installed)")
    else:
        failures.extend(stat_failures)

    print("\n" + "=" * 70)
    if failures:
        for f in failures:
            print(f"FAILED: {f}")
        return 1
    print("All checks passed: static detection and statistical verification agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
