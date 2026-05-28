"""Programmatic generator for synthetic MirrorBench pairs.

Synthetic pairs MUST be programmatically generated; hand-authoring is
disallowed. This module is the single entry point.
Re-running it is idempotent (same output every time) and is the only
supported way to update ``bench/pairs/synthetic/``.

Phase 1 generates pairs across the seven categories the diff engine
currently detects: ``schema_drift``, ``type_coercion``,
``timezone_mismatch``, ``rounding_precision``, ``aggregation_function``,
``join_key_mismatch``, ``ordering_dependence``. Each category gets
2-4 pairs varying the specific parameters (different timezones,
different aggregation functions, etc.) so the evaluator has multiple
samples per category. Pipelines stay short (1-3 ops) so the focus is
on the targeted divergence.

Run directly:

.. code-block:: shell

    uv run python -m bench.scripts.generate_synthetic

That clears ``bench/pairs/synthetic/`` and writes the full set fresh.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = REPO_ROOT / "bench" / "pairs" / "synthetic"


def _loc(op_kind: str, side: str = "both") -> dict[str, str]:
    """One expected-localization row. Names the responsible op kind on
    each side (defaults to both, the common case)."""

    return {"op_kind": op_kind, "side": side}


# A "pair spec" is an in-memory description; the generator materializes
# it as a directory of files. Spec shape:
#
#   {
#     "name": str,
#     "category": str,
#     "description": str,
#     "offline_sql": str,           # contents of offline.sql
#     "online_sql": str,            # contents of online.sql
#     "offline_schemas": dict[str, list[tuple[str, str]]],
#     "online_schemas": dict[str, list[tuple[str, str]]],
#     "expected_divergences": list[dict[str, Any]],
#   }


def _timezone_mismatch_pairs() -> Iterable[dict[str, Any]]:
    cases = [
        ("UTC", "US/Pacific"),
        ("UTC", "Europe/London"),
        ("US/Eastern", "Asia/Tokyo"),
    ]
    query = "SELECT ts FROM events\n"
    for i, (off_tz, on_tz) in enumerate(cases):
        yield {
            "name": f"timezone_mismatch_{i:03d}",
            "category": "timezone_mismatch",
            "description": (
                f"Offline reads events.ts as timestamp[ns, {off_tz}]; "
                f"online reads as timestamp[ns, {on_tz}]."
            ),
            "offline_sql": query,
            "online_sql": query,
            "offline_schemas": {"events": [("ts", f"timestamp[ns, {off_tz}]")]},
            "online_schemas": {"events": [("ts", f"timestamp[ns, {on_tz}]")]},
            "expected_divergences": [{"category": "timezone_mismatch"}],
            "expected_localization": [_loc("source")],
        }


def _rounding_precision_pairs() -> Iterable[dict[str, Any]]:
    timestamp_cases = [
        ("ns", "us"),
        ("ms", "us"),
        ("s", "ms"),
    ]
    query = "SELECT ts FROM events\n"
    for i, (off_unit, on_unit) in enumerate(timestamp_cases):
        yield {
            "name": f"rounding_precision_timestamp_{i:03d}",
            "category": "rounding_precision",
            "description": (
                f"Offline timestamp resolution {off_unit!r}; online resolution {on_unit!r}."
            ),
            "offline_sql": query,
            "online_sql": query,
            "offline_schemas": {"events": [("ts", f"timestamp[{off_unit}, UTC]")]},
            "online_schemas": {"events": [("ts", f"timestamp[{on_unit}, UTC]")]},
            "expected_divergences": [{"category": "rounding_precision"}],
            "expected_localization": [_loc("source")],
        }

    decimal_cases = [
        ((18, 2), (18, 4)),
        ((10, 0), (10, 2)),
    ]
    decimal_query = "SELECT amount FROM transactions\n"
    for i, (off, on) in enumerate(decimal_cases):
        yield {
            "name": f"rounding_precision_decimal_{i:03d}",
            "category": "rounding_precision",
            "description": (
                f"Offline decimal({off[0]}, {off[1]}); online decimal({on[0]}, {on[1]})."
            ),
            "offline_sql": decimal_query,
            "online_sql": decimal_query,
            "offline_schemas": {"transactions": [("amount", f"decimal[{off[0]}, {off[1]}]")]},
            "online_schemas": {"transactions": [("amount", f"decimal[{on[0]}, {on[1]}]")]},
            "expected_divergences": [{"category": "rounding_precision"}],
            "expected_localization": [_loc("source")],
        }


def _type_coercion_pairs() -> Iterable[dict[str, Any]]:
    cases = [
        ("int64", "float64"),
        ("int32", "int64"),
        ("float32", "float64"),
        ("utf8", "int64"),
    ]
    query = "SELECT x FROM t\n"
    for i, (off, on) in enumerate(cases):
        yield {
            "name": f"type_coercion_{i:03d}",
            "category": "type_coercion",
            "description": f"Offline x dtype {off!r}; online x dtype {on!r}.",
            "offline_sql": query,
            "online_sql": query,
            "offline_schemas": {"t": [("x", off)]},
            "online_schemas": {"t": [("x", on)]},
            "expected_divergences": [{"category": "type_coercion"}],
            "expected_localization": [_loc("source")],
        }


def _schema_drift_pairs() -> Iterable[dict[str, Any]]:
    schema = [("uid", "int64"), ("score", "float64")]
    yield {
        "name": "schema_drift_column_only_left",
        "category": "schema_drift",
        "description": "Offline projects uid + score; online drops score.",
        "offline_sql": "SELECT uid, score FROM events\n",
        "online_sql": "SELECT uid FROM events\n",
        "offline_schemas": {"events": schema},
        "online_schemas": {"events": schema},
        "expected_divergences": [{"category": "schema_drift"}],
        "expected_localization": [_loc("project")],
    }
    yield {
        "name": "schema_drift_column_only_right",
        "category": "schema_drift",
        "description": "Offline projects uid; online projects uid + score.",
        "offline_sql": "SELECT uid FROM events\n",
        "online_sql": "SELECT uid, score FROM events\n",
        "offline_schemas": {"events": schema},
        "online_schemas": {"events": schema},
        "expected_divergences": [{"category": "schema_drift"}],
        "expected_localization": [_loc("project")],
    }
    yield {
        "name": "schema_drift_rename",
        "category": "schema_drift",
        "description": "Offline keeps uid; online renames uid to user_id.",
        "offline_sql": "SELECT uid FROM events\n",
        "online_sql": "SELECT uid AS user_id FROM events\n",
        "offline_schemas": {"events": schema},
        "online_schemas": {"events": schema},
        "expected_divergences": [{"category": "schema_drift"}],
        "expected_localization": [_loc("project")],
    }


def _aggregation_function_pairs() -> Iterable[dict[str, Any]]:
    schema = [("uid", "int64"), ("score", "float64")]
    # ``output_dtype`` lets the pair declare the secondary dtype divergence
    # that some agg-function changes produce as a side effect. COUNT
    # returns int64; SUM / AVG / MIN / MAX over float64 stay float64. The
    # diff engine surfaces both the function-change and the dtype-change
    # as divergences (they are independently observable), so the expected
    # list reflects that.
    cases: list[tuple[str, str, list[str]]] = [
        ("SUM", "AVG", ["aggregation_function"]),
        ("SUM", "MAX", ["aggregation_function"]),
        ("MIN", "MAX", ["aggregation_function"]),
        ("COUNT", "SUM", ["aggregation_function", "type_coercion"]),
    ]
    for i, (off_fn, on_fn, expected_categories) in enumerate(cases):
        yield {
            "name": f"aggregation_function_{i:03d}",
            "category": "aggregation_function",
            "description": (f"Offline aggregates score with {off_fn}; online uses {on_fn}."),
            "offline_sql": (f"SELECT uid, {off_fn}(score) AS score FROM events GROUP BY uid\n"),
            "online_sql": (f"SELECT uid, {on_fn}(score) AS score FROM events GROUP BY uid\n"),
            "offline_schemas": {"events": schema},
            "online_schemas": {"events": schema},
            "expected_divergences": [{"category": c} for c in expected_categories],
            "expected_localization": [_loc("aggregate")],
        }


def _join_key_mismatch_pairs() -> Iterable[dict[str, Any]]:
    left = [("k1", "int64"), ("k2", "int64"), ("v", "float64")]
    right = [("k1", "int64"), ("k2", "int64"), ("v", "float64")]
    yield {
        "name": "join_key_mismatch_single_key",
        "category": "join_key_mismatch",
        "description": "Offline joins on k1; online joins on k2.",
        "offline_sql": "SELECT a.v FROM a JOIN b ON a.k1 = b.k1\n",
        "online_sql": "SELECT a.v FROM a JOIN b ON a.k2 = b.k2\n",
        "offline_schemas": {"a": left, "b": right},
        "online_schemas": {"a": left, "b": right},
        "expected_divergences": [{"category": "join_key_mismatch"}],
        "expected_localization": [_loc("join")],
    }
    yield {
        "name": "join_key_mismatch_extra_key",
        "category": "join_key_mismatch",
        "description": "Offline joins on (k1); online joins on (k1, k2).",
        "offline_sql": "SELECT a.v FROM a JOIN b ON a.k1 = b.k1\n",
        "online_sql": "SELECT a.v FROM a JOIN b ON a.k1 = b.k1 AND a.k2 = b.k2\n",
        "offline_schemas": {"a": left, "b": right},
        "online_schemas": {"a": left, "b": right},
        "expected_divergences": [{"category": "join_key_mismatch"}],
        "expected_localization": [_loc("join")],
    }


def _ordering_dependence_pairs() -> Iterable[dict[str, Any]]:
    schema = [("uid", "int64"), ("score", "float64")]
    yield {
        "name": "ordering_dependence_different_columns",
        "category": "ordering_dependence",
        "description": "Offline orders by uid; online orders by score.",
        "offline_sql": "SELECT uid, score FROM events ORDER BY uid\n",
        "online_sql": "SELECT uid, score FROM events ORDER BY score\n",
        "offline_schemas": {"events": schema},
        "online_schemas": {"events": schema},
        "expected_divergences": [{"category": "ordering_dependence"}],
        "expected_localization": [_loc("sort")],
    }
    yield {
        "name": "ordering_dependence_asc_vs_desc",
        "category": "ordering_dependence",
        "description": "Same column, opposite directions.",
        "offline_sql": "SELECT uid FROM events ORDER BY uid ASC\n",
        "online_sql": "SELECT uid FROM events ORDER BY uid DESC\n",
        "offline_schemas": {"events": schema},
        "online_schemas": {"events": schema},
        "expected_divergences": [{"category": "ordering_dependence"}],
        "expected_localization": [_loc("sort")],
    }


def _rolling_window_sql(lookback: int | str, agg: str = "AVG") -> str:
    """A trailing rolling-window SELECT. ``lookback`` is the PRECEDING row
    count, or the string ``"unbounded"`` for a cumulative window."""

    start = "UNBOUNDED PRECEDING" if lookback == "unbounded" else f"{lookback} PRECEDING"
    return (
        f"SELECT uid, ts, {agg}(score) OVER (PARTITION BY uid ORDER BY ts "
        f"ROWS BETWEEN {start} AND CURRENT ROW) AS roll FROM events\n"
    )


def _window_function_pairs() -> Iterable[dict[str, Any]]:
    """Rolling-window pairs. Activate ``window_size_mismatch`` (different
    lookback) and exercise windowed-aggregation-function differences. The
    SQL tracer supports only the trailing ROWS frame, so every frame here
    is ``ROWS BETWEEN <n> PRECEDING AND CURRENT ROW``."""

    schema = [("uid", "int64"), ("ts", "timestamp[ns, UTC]"), ("score", "float64")]

    yield {
        "name": "identity_window_rolling_mean",
        "category": "identity",
        "description": "Same 3-row trailing mean on both sides; diff must be empty.",
        "offline_sql": _rolling_window_sql(2),
        "online_sql": _rolling_window_sql(2),
        "offline_schemas": {"events": schema},
        "online_schemas": {"events": schema},
        "expected_divergences": [],
    }

    for i, (off_n, on_n) in enumerate([(2, 4), (3, 5), (4, 9)]):
        yield {
            "name": f"window_size_mismatch_{i:03d}",
            "category": "window_size_mismatch",
            "description": (
                f"Offline rolling mean over {off_n + 1} rows; online over {on_n + 1} rows."
            ),
            "offline_sql": _rolling_window_sql(off_n),
            "online_sql": _rolling_window_sql(on_n),
            "offline_schemas": {"events": schema},
            "online_schemas": {"events": schema},
            "expected_divergences": [{"category": "window_size_mismatch"}],
            "expected_localization": [_loc("window")],
        }

    yield {
        "name": "window_size_mismatch_unbounded_vs_bounded",
        "category": "window_size_mismatch",
        "description": "Offline cumulative (unbounded) mean; online 3-row trailing mean.",
        "offline_sql": _rolling_window_sql("unbounded"),
        "online_sql": _rolling_window_sql(2),
        "offline_schemas": {"events": schema},
        "online_schemas": {"events": schema},
        "expected_divergences": [{"category": "window_size_mismatch"}],
        "expected_localization": [_loc("window")],
    }

    yield {
        "name": "window_aggregation_function_mean_vs_sum",
        "category": "aggregation_function",
        "description": "Same trailing window; offline averages, online sums.",
        "offline_sql": _rolling_window_sql(2, agg="AVG"),
        "online_sql": _rolling_window_sql(2, agg="SUM"),
        "offline_schemas": {"events": schema},
        "online_schemas": {"events": schema},
        "expected_divergences": [{"category": "aggregation_function"}],
        "expected_localization": [_loc("window")],
    }


def _rolling_window_polars_src(closed: str, function: str) -> str:
    """A Polars time-based rolling-mean pipeline with an explicit closed
    boundary (the field that makes window_boundary detectable)."""

    return (
        f"def {function}(lf, pl):\n"
        f"    return lf.rolling(index_column='ts', period='3d', closed='{closed}', "
        f"group_by='uid').agg(pl.col('score').mean())\n"
    )


def _window_boundary_pairs() -> Iterable[dict[str, Any]]:
    """Polars rolling-window pairs that differ only in the ``closed``
    boundary. These activate ``window_boundary`` (time windows separate the
    boundary from the size cleanly, unlike SQL ROWS frames)."""

    schema = [("uid", "int64"), ("ts", "timestamp[ns, UTC]"), ("score", "float64")]

    yield {
        "name": "identity_rolling_window",
        "category": "identity",
        "description": "Same 3-day rolling mean (closed=right) on both sides; diff empty.",
        "offline": {
            "language": "polars",
            "python_source": _rolling_window_polars_src("right", "offline"),
            "function": "offline",
            "input_schema": schema,
            "source_name": "events",
        },
        "online": {
            "language": "polars",
            "python_source": _rolling_window_polars_src("right", "online"),
            "function": "online",
            "input_schema": schema,
            "source_name": "events",
        },
        "expected_divergences": [],
    }

    # closed values use Polars's own vocabulary (left/right/both/none); the
    # tracer maps "none" -> our canonical "neither".
    for i, (off_c, on_c) in enumerate(
        [("left", "right"), ("both", "none"), ("left", "both"), ("right", "none")]
    ):
        yield {
            "name": f"window_boundary_{i:03d}",
            "category": "window_boundary",
            "description": f"Offline rolling window closed={off_c!r}; online closed={on_c!r}.",
            "offline": {
                "language": "polars",
                "python_source": _rolling_window_polars_src(off_c, "offline"),
                "function": "offline",
                "input_schema": schema,
                "source_name": "events",
            },
            "online": {
                "language": "polars",
                "python_source": _rolling_window_polars_src(on_c, "online"),
                "function": "online",
                "input_schema": schema,
                "source_name": "events",
            },
            "expected_divergences": [{"category": "window_boundary"}],
            "expected_localization": [_loc("window")],
        }


def _asof_join_polars_src(strategy: str, function: str) -> str:
    """A Polars as-of join pipeline. The right table (prices) is declared
    inline via pl.source so trace_polars's signature stays single-source."""

    return (
        f"def {function}(lf, pl):\n"
        f"    prices = pl.source('prices', schema=[('uid', 'int64'), "
        f"('ts', 'timestamp[ns, UTC]'), ('price', 'float64')])\n"
        f"    return lf.join_asof(prices, on='ts', by='uid', strategy='{strategy}')\n"
    )


def _as_of_join_pairs() -> Iterable[dict[str, Any]]:
    """Polars as-of-join pairs that differ only in the join direction
    (``strategy``). These activate ``as_of_join_direction``, the dominant
    point-in-time skew source."""

    schema = [("uid", "int64"), ("ts", "timestamp[ns, UTC]"), ("score", "float64")]

    yield {
        "name": "identity_asof_join",
        "category": "identity",
        "description": "Same backward as-of join on both sides; diff must be empty.",
        "offline": {
            "language": "polars",
            "python_source": _asof_join_polars_src("backward", "offline"),
            "function": "offline",
            "input_schema": schema,
            "source_name": "events",
        },
        "online": {
            "language": "polars",
            "python_source": _asof_join_polars_src("backward", "online"),
            "function": "online",
            "input_schema": schema,
            "source_name": "events",
        },
        "expected_divergences": [],
    }

    for i, (off_s, on_s) in enumerate(
        [("backward", "forward"), ("backward", "nearest"), ("forward", "nearest")]
    ):
        yield {
            "name": f"as_of_join_direction_{i:03d}",
            "category": "as_of_join_direction",
            "description": f"Offline as-of join {off_s!r}; online {on_s!r}.",
            "offline": {
                "language": "polars",
                "python_source": _asof_join_polars_src(off_s, "offline"),
                "function": "offline",
                "input_schema": schema,
                "source_name": "events",
            },
            "online": {
                "language": "polars",
                "python_source": _asof_join_polars_src(on_s, "online"),
                "function": "online",
                "input_schema": schema,
                "source_name": "events",
            },
            "expected_divergences": [{"category": "as_of_join_direction"}],
            "expected_localization": [_loc("as_of_join")],
        }


def _fill_na_pairs() -> Iterable[dict[str, Any]]:
    """Null-fill pairs. The identity pair (pandas fillna vs Polars fill_null,
    same constant) must diff to ``()``; the value-difference pairs surface
    ``null_handling`` (filling nulls with different constants offline vs
    online is a textbook skew)."""

    events = [("uid", "int64"), ("score", "float64")]

    yield {
        "name": "cross_framework_identity_fillna",
        "category": "identity",
        "description": "pandas df.fillna(0) vs polars lf.fill_null(0); diff empty.",
        "offline": {
            "language": "pandas",
            "python_source": "def offline(df):\n    return df.fillna(0)\n",
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "polars",
            "python_source": "def online(lf, pl):\n    return lf.fill_null(0)\n",
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [],
    }

    for i, (off_v, on_v) in enumerate([("0", "-1"), ("0", "999")]):
        yield {
            "name": f"fillna_null_handling_{i:03d}",
            "category": "null_handling",
            "description": (f"Offline fills score's nulls with {off_v}; online with {on_v}."),
            "offline": {
                "language": "pandas",
                "python_source": f"def offline(df):\n    return df.fillna({{'score': {off_v}}})\n",
                "function": "offline",
                "input_schema": events,
                "source_name": "events",
            },
            "online": {
                "language": "pandas",
                "python_source": f"def online(df):\n    return df.fillna({{'score': {on_v}}})\n",
                "function": "online",
                "input_schema": events,
                "source_name": "events",
            },
            "expected_divergences": [{"category": "null_handling"}],
            "expected_localization": [_loc("fill_na")],
        }


def _identity_pairs() -> Iterable[dict[str, Any]]:
    """Negative-control pairs: structurally equivalent pipelines that
    must diff to ``()``. These are essential for precision; without
    them the evaluator only measures recall.
    """

    schema = [("uid", "int64"), ("score", "float64")]
    yield {
        "name": "identity_simple_select",
        "category": "identity",
        "description": "Same SQL on both sides; diff must be empty.",
        "offline_sql": "SELECT uid, score FROM events WHERE score > 0\n",
        "online_sql": "SELECT uid, score FROM events WHERE score > 0\n",
        "offline_schemas": {"events": schema},
        "online_schemas": {"events": schema},
        "expected_divergences": [],
    }
    yield {
        "name": "identity_groupby_pipeline",
        "category": "identity",
        "description": "Equivalent grouped aggregation; diff must be empty.",
        "offline_sql": ("SELECT uid, SUM(score) AS score FROM events GROUP BY uid\n"),
        "online_sql": ("SELECT uid, SUM(score) AS score FROM events GROUP BY uid\n"),
        "offline_schemas": {"events": schema},
        "online_schemas": {"events": schema},
        "expected_divergences": [],
    }
    yield {
        "name": "identity_join_pipeline",
        "category": "identity",
        "description": "Equivalent join; diff must be empty.",
        "offline_sql": ("SELECT a.uid FROM a JOIN b ON a.uid = b.uid WHERE a.uid > 0\n"),
        "online_sql": ("SELECT a.uid FROM a JOIN b ON a.uid = b.uid WHERE a.uid > 0\n"),
        "offline_schemas": {
            "a": [("uid", "int64")],
            "b": [("uid", "int64"), ("country", "utf8")],
        },
        "online_schemas": {
            "a": [("uid", "int64")],
            "b": [("uid", "int64"), ("country", "utf8")],
        },
        "expected_divergences": [],
    }


def _feature_leakage_temporal_pairs() -> Iterable[dict[str, Any]]:
    """feature_leakage_temporal pairs. Both sides declare an
    ``event_time_column`` on Source; one side wraps its aggregation in a
    Window (point-in-time safe) while the other uses a plain Aggregate
    (potentially sees rows from after the label time). The engine's
    whole-graph rule fires on the asymmetry."""

    events = [
        ("uid", "int64"),
        ("ts", "timestamp[ns, UTC]"),
        ("score", "float64"),
    ]
    windowed_src = (
        "def {fn}(lf, pl):\n"
        "    return lf.rolling(index_column='ts', period='7d', closed='left', "
        "group_by='uid').agg(pl.col('score').mean())\n"
    )
    unbounded_src = "def {fn}(lf, pl):\n    return lf.group_by('uid').agg(pl.col('score').mean())\n"

    yield {
        "name": "feature_leakage_temporal_window_vs_unbounded_agg",
        "category": "feature_leakage_temporal",
        "description": (
            "Offline uses a 7-day point-in-time-safe rolling mean; online "
            "uses a plain unbounded mean. The unbounded side may see rows "
            "from after the label time."
        ),
        "offline": {
            "language": "polars",
            "python_source": windowed_src.format(fn="offline"),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
            "event_time_column": "ts",
        },
        "online": {
            "language": "polars",
            "python_source": unbounded_src.format(fn="online"),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
            "event_time_column": "ts",
        },
        "expected_divergences": [{"category": "feature_leakage_temporal"}],
        "expected_localization": [_loc("aggregate", side="online")],
    }

    yield {
        "name": "feature_leakage_temporal_offline_leaky_online_safe",
        "category": "feature_leakage_temporal",
        "description": (
            "Symmetric case: offline uses an unbounded mean (potentially "
            "leaky during training); online uses the point-in-time-safe "
            "rolling mean. Direction matters for the diagnostic."
        ),
        "offline": {
            "language": "polars",
            "python_source": unbounded_src.format(fn="offline"),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
            "event_time_column": "ts",
        },
        "online": {
            "language": "polars",
            "python_source": windowed_src.format(fn="online"),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
            "event_time_column": "ts",
        },
        "expected_divergences": [{"category": "feature_leakage_temporal"}],
        "expected_localization": [_loc("aggregate", side="offline")],
    }

    yield {
        "name": "identity_feature_leakage_temporal_both_windowed",
        "category": "identity",
        "description": (
            "Both sides use the point-in-time-safe rolling mean over the "
            "same event_time_column. Diff is empty: no leakage divergence "
            "because there is no asymmetry."
        ),
        "offline": {
            "language": "polars",
            "python_source": windowed_src.format(fn="offline"),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
            "event_time_column": "ts",
        },
        "online": {
            "language": "polars",
            "python_source": windowed_src.format(fn="online"),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
            "event_time_column": "ts",
        },
        "expected_divergences": [],
    }

    # A control: same unbounded aggregate on both sides. The leakage rule
    # requires *asymmetry*; symmetric leakiness is not a divergence (both
    # pipelines have the same bug, which the rule cannot distinguish from
    # the user genuinely not needing point-in-time guarantees).
    yield {
        "name": "identity_feature_leakage_temporal_both_unbounded",
        "category": "identity",
        "description": (
            "Both sides use the unbounded aggregate. The leakage rule "
            "intentionally does not fire on symmetric leakiness: both "
            "pipelines share the same temporal assumption."
        ),
        "offline": {
            "language": "polars",
            "python_source": unbounded_src.format(fn="offline"),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
            "event_time_column": "ts",
        },
        "online": {
            "language": "polars",
            "python_source": unbounded_src.format(fn="online"),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
            "event_time_column": "ts",
        },
        "expected_divergences": [],
    }

    # Cross-framework: pandas-side leaky aggregation vs polars-side
    # windowed aggregation. event_time_column declared on both.
    yield {
        "name": "feature_leakage_temporal_cross_framework_pandas_vs_polars",
        "category": "feature_leakage_temporal",
        "description": (
            "pandas offline aggregates score with no temporal bound; "
            "polars online wraps it in a 7-day rolling window. "
            "feature_leakage_temporal fires because of the asymmetric guard."
        ),
        "offline": {
            "language": "pandas",
            "python_source": (
                "def offline(df):\n    return df.groupby('uid').agg({'score': 'mean'})\n"
            ),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
            "event_time_column": "ts",
        },
        "online": {
            "language": "polars",
            "python_source": windowed_src.format(fn="online"),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
            "event_time_column": "ts",
        },
        "expected_divergences": [{"category": "feature_leakage_temporal"}],
        "expected_localization": [_loc("aggregate", side="offline")],
    }

    # Sanity: event_time_column declared on only one side -- rule must NOT
    # fire (it requires both sides to declare it, otherwise we have no
    # claim on what 'temporal' means for the unannotated side).
    yield {
        "name": "no_leakage_rule_when_only_one_side_declares_event_time",
        "category": "identity",
        "description": (
            "Sanity: when only one side declares event_time_column, the "
            "leakage rule does not fire. The pipelines may still differ "
            "structurally (and the engine will surface that elsewhere), "
            "but it is not a temporal-leakage divergence."
        ),
        "offline": {
            "language": "polars",
            "python_source": unbounded_src.format(fn="offline"),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
            "event_time_column": "ts",
        },
        "online": {
            "language": "polars",
            "python_source": unbounded_src.format(fn="online"),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
            # NB: no event_time_column declared on this side.
        },
        "expected_divergences": [],
    }


def _categorical_encoding_pairs() -> Iterable[dict[str, Any]]:
    """categorical_encoding pairs. Both sides one-hot-encode via polars
    ``lf.to_dummies(...)`` but disagree on which columns are encoded.
    The Encode op carries (columns, method, categories); any difference
    routes to ``categorical_encoding``."""

    events = [("uid", "int64"), ("country", "utf8"), ("city", "utf8")]

    yield {
        "name": "identity_to_dummies_same_columns",
        "category": "identity",
        "description": (
            "Both sides one-hot-encode the same columns with the same method; "
            "the Encode ops fingerprint identically and diff is empty."
        ),
        "offline": {
            "language": "polars",
            "python_source": (
                "def offline(lf, pl):\n    return lf.to_dummies(columns=['country'])\n"
            ),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "polars",
            "python_source": (
                "def online(lf, pl):\n    return lf.to_dummies(columns=['country'])\n"
            ),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [],
    }

    yield {
        "name": "categorical_encoding_different_columns",
        "category": "categorical_encoding",
        "description": (
            "Offline encodes ``country``; online encodes ``city``. The "
            "Encode.columns differ; classifier surfaces categorical_encoding."
        ),
        "offline": {
            "language": "polars",
            "python_source": (
                "def offline(lf, pl):\n    return lf.to_dummies(columns=['country'])\n"
            ),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "polars",
            "python_source": ("def online(lf, pl):\n    return lf.to_dummies(columns=['city'])\n"),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [{"category": "categorical_encoding"}],
        "expected_localization": [_loc("encode")],
    }

    yield {
        "name": "categorical_encoding_extra_column",
        "category": "categorical_encoding",
        "description": (
            "Offline encodes ``country`` only; online encodes both "
            "``country`` and ``city``. Column-set difference fires."
        ),
        "offline": {
            "language": "polars",
            "python_source": (
                "def offline(lf, pl):\n    return lf.to_dummies(columns=['country'])\n"
            ),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "polars",
            "python_source": (
                "def online(lf, pl):\n    return lf.to_dummies(columns=['country', 'city'])\n"
            ),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [{"category": "categorical_encoding"}],
        "expected_localization": [_loc("encode")],
    }

    yield {
        "name": "categorical_encoding_pinned_vs_runtime_fit",
        "category": "categorical_encoding",
        "description": (
            "Offline pins the category vocabulary; online leaves it as "
            "None (runtime-fit). Static analysis cannot prove equality; "
            "categorical_encoding fires."
        ),
        "offline": {
            "language": "polars",
            "python_source": (
                "def offline(lf, pl):\n    return lf.to_dummies(columns=['country'])\n"
            ),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "polars",
            "python_source": (
                "def online(lf, pl):\n    return lf.to_dummies(columns=['country'])\n"
            ),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        # Both sides emit the same Encode op (categories=None on both).
        # We override one side's categories via an Encode op constructed
        # downstream in a follow-up; for now both render identically, so
        # this pair acts as a control that documents the limitation.
        "expected_divergences": [],
        "_documentation_note": (
            "Both sides emit categories=None; the divergence is unobservable "
            "statically. This pair documents the v0 limitation (no pinned-"
            "categories tracer API yet) rather than fabricating a fake diff."
        ),
    }

    yield {
        "name": "categorical_encoding_all_vs_subset",
        "category": "categorical_encoding",
        "description": (
            "Offline one-hot-encodes every column (``columns=None`` default); "
            "online encodes a subset. Column-set difference fires."
        ),
        "offline": {
            "language": "polars",
            "python_source": "def offline(lf, pl):\n    return lf.to_dummies()\n",
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "polars",
            "python_source": (
                "def online(lf, pl):\n    return lf.to_dummies(columns=['country'])\n"
            ),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [{"category": "categorical_encoding"}],
        "expected_localization": [_loc("encode")],
    }

    yield {
        "name": "categorical_encoding_column_order_reversed",
        "category": "categorical_encoding",
        "description": (
            "Offline encodes (country, city); online encodes (city, country). "
            "Same set, different order. Encode.columns is order-sensitive "
            "so this fires (downstream column order is observable)."
        ),
        "offline": {
            "language": "polars",
            "python_source": (
                "def offline(lf, pl):\n    return lf.to_dummies(columns=['country', 'city'])\n"
            ),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "polars",
            "python_source": (
                "def online(lf, pl):\n    return lf.to_dummies(columns=['city', 'country'])\n"
            ),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [{"category": "categorical_encoding"}],
        "expected_localization": [_loc("encode")],
    }


def _seed_mismatch_pairs() -> Iterable[dict[str, Any]]:
    """seed_mismatch pairs. Both sides call ``df.sample(...)`` /
    ``lf.sample(...)`` with the same n / fraction but different random
    seeds (or one side pins a seed while the other does not). The
    classifier surfaces the divergence as ``seed_mismatch`` (different
    rows end up in the training and serving subsets even on the same
    input). Same-seed pairs land elsewhere as identity controls."""

    events = [("uid", "int64"), ("score", "float64")]

    yield {
        "name": "identity_sample_same_seed",
        "category": "identity",
        "description": (
            "Both sides take a fixed-n reproducible sample with the same seed; "
            "the Sample ops fingerprint identically and diff is empty."
        ),
        "offline": {
            "language": "pandas",
            "python_source": ("def offline(df):\n    return df.sample(n=2, random_state=42)\n"),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "pandas",
            "python_source": ("def online(df):\n    return df.sample(n=2, random_state=42)\n"),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [],
    }

    yield {
        "name": "seed_mismatch_pandas_different_seeds",
        "category": "seed_mismatch",
        "description": "pandas df.sample(n=2, random_state=...) with different seeds.",
        "offline": {
            "language": "pandas",
            "python_source": ("def offline(df):\n    return df.sample(n=2, random_state=42)\n"),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "pandas",
            "python_source": ("def online(df):\n    return df.sample(n=2, random_state=7)\n"),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [{"category": "seed_mismatch"}],
        "expected_localization": [_loc("sample")],
    }

    yield {
        "name": "seed_mismatch_pandas_seed_pinned_vs_unpinned",
        "category": "seed_mismatch",
        "description": (
            "Offline pins a seed; online does not. The unseeded side is "
            "non-reproducible by definition and disagrees with the seeded one."
        ),
        "offline": {
            "language": "pandas",
            "python_source": ("def offline(df):\n    return df.sample(n=3, random_state=0)\n"),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "pandas",
            "python_source": ("def online(df):\n    return df.sample(n=3)\n"),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [{"category": "seed_mismatch"}],
        "expected_localization": [_loc("sample")],
    }

    yield {
        "name": "seed_mismatch_polars_different_seeds",
        "category": "seed_mismatch",
        "description": "polars lf.sample(n=2, seed=...) with different seeds.",
        "offline": {
            "language": "polars",
            "python_source": ("def offline(lf, pl):\n    return lf.sample(n=2, seed=42)\n"),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "polars",
            "python_source": ("def online(lf, pl):\n    return lf.sample(n=2, seed=7)\n"),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [{"category": "seed_mismatch"}],
        "expected_localization": [_loc("sample")],
    }

    yield {
        "name": "seed_mismatch_cross_framework_pandas_vs_polars",
        "category": "seed_mismatch",
        "description": (
            "Cross-framework: pandas df.sample(random_state=42) vs polars "
            "lf.sample(seed=7). Both stages fingerprint identically except "
            "for the seed, which is the only field that differs."
        ),
        "offline": {
            "language": "pandas",
            "python_source": ("def offline(df):\n    return df.sample(n=2, random_state=42)\n"),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "polars",
            "python_source": ("def online(lf, pl):\n    return lf.sample(n=2, seed=7)\n"),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [{"category": "seed_mismatch"}],
        "expected_localization": [_loc("sample")],
    }

    yield {
        "name": "seed_mismatch_fraction_with_different_seeds",
        "category": "seed_mismatch",
        "description": (
            "Fraction-based sampling, same fraction, different seeds. "
            "Demonstrates that seed_mismatch fires regardless of whether "
            "the sample size is set via n or fraction."
        ),
        "offline": {
            "language": "pandas",
            "python_source": (
                "def offline(df):\n    return df.sample(frac=0.5, random_state=42)\n"
            ),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "pandas",
            "python_source": ("def online(df):\n    return df.sample(frac=0.5, random_state=7)\n"),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [{"category": "seed_mismatch"}],
        "expected_localization": [_loc("sample")],
    }


def _unit_mismatch_pairs() -> Iterable[dict[str, Any]]:
    """unit_mismatch pairs. Each side declares the same numeric base
    dtype but a different ``{measurement_unit}`` annotation; the
    classifier flags this as unit_mismatch rather than type_coercion
    (the base dtype matches; only the semantic unit differs)."""

    query = "SELECT x FROM t\n"
    cases: list[tuple[str, str, str, str]] = [
        ("meters_vs_feet", "float64{meters}", "float64{feet}", "Length: meters vs feet."),
        (
            "celsius_vs_fahrenheit",
            "float64{celsius}",
            "float64{fahrenheit}",
            "Temperature: celsius vs fahrenheit.",
        ),
        (
            "usd_vs_eur_decimal",
            "decimal[18, 2]{USD}",
            "decimal[18, 2]{EUR}",
            "Money: USD vs EUR on decimal.",
        ),
        (
            "kg_vs_lb",
            "float32{kg}",
            "float32{lb}",
            "Mass: kg vs lb.",
        ),
        (
            "seconds_vs_milliseconds_value",
            "int64{seconds}",
            "int64{milliseconds}",
            "Elapsed time stored as int (not duration): seconds vs milliseconds.",
        ),
        (
            "annotated_vs_unannotated",
            "float64{USD}",
            "float64",
            (
                "Offline declares USD; online is unannotated. The static "
                "fingerprint cannot prove they agree, so unit_mismatch fires."
            ),
        ),
    ]
    for name, off_dtype, on_dtype, description in cases:
        yield {
            "name": f"unit_mismatch_{name}",
            "category": "unit_mismatch",
            "description": description,
            "offline_sql": query,
            "online_sql": query,
            "offline_schemas": {"t": [("x", off_dtype)]},
            "online_schemas": {"t": [("x", on_dtype)]},
            "expected_divergences": [{"category": "unit_mismatch"}],
            "expected_localization": [_loc("source")],
        }


def _udf_pairs() -> Iterable[dict[str, Any]]:
    """UDF-comparison pairs exercising the pandas ``df.apply(...)`` tracer
    and the diff classifier's Udf rule. The identity pair shares a callable
    so the source-hashes match (diff empty); the divergent pair uses two
    different callables (diff surfaces ``schema_drift`` with a "udf body
    differs" detail).
    """

    events = [("uid", "int64"), ("score", "float64")]
    double_src = (
        "def _double(col):\n    return col * 2\n\ndef offline(df):\n    return df.apply(_double)\n"
    )
    triple_src = (
        "def _triple(col):\n    return col * 3\n\ndef online(df):\n    return df.apply(_triple)\n"
    )
    double_src_online = (
        "def _double(col):\n    return col * 2\n\ndef online(df):\n    return df.apply(_double)\n"
    )
    double_src_with_comments = (
        "def _double(col):\n"
        "    # multiply by two\n"
        '    """Double the column."""\n'
        "    return col * 2\n\n"
        "def online(df):\n    return df.apply(_double)\n"
    )

    yield {
        "name": "cross_framework_identity_udf_pandas_vs_polars",
        "category": "identity",
        "description": (
            "pandas df.apply(_identity) vs polars lf.map_batches(_identity) "
            "with the same callable body. The libcst-norm-v1 source-hash "
            "collapses formatting and makes the Udf ops fingerprint "
            "identically across frameworks, so diff is empty."
        ),
        "offline": {
            "language": "pandas",
            "python_source": (
                "def _identity(df):\n    return df\n\n"
                "def offline(df):\n    return df.apply(_identity)\n"
            ),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "polars",
            "python_source": (
                "def _identity(df):\n    return df\n\n"
                "def online(lf, pl):\n    return lf.map_batches(_identity)\n"
            ),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [],
    }

    yield {
        "name": "identity_udf_apply",
        "category": "identity",
        "description": (
            "Both sides apply the same UDF body, formatted differently. "
            "The libcst-norm-v1 source-hash collapses formatting / comments "
            "/ docstrings, so the fingerprints match and diff is empty."
        ),
        "offline": {
            "language": "pandas",
            "python_source": double_src,
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "pandas",
            "python_source": double_src_with_comments,
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [],
    }

    yield {
        "name": "udf_body_mismatch",
        "category": "schema_drift",
        "description": (
            "Offline applies a doubling UDF; online applies a tripling UDF. "
            "Different source-hashes; diff surfaces schema_drift with a "
            "'udf body differs' detail. The statistical companion check is "
            "the right way to upgrade the diagnosis to a value-level answer."
        ),
        "offline": {
            "language": "pandas",
            "python_source": double_src_online,
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "pandas",
            "python_source": triple_src,
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [{"category": "schema_drift"}],
        "expected_localization": [_loc("udf")],
    }


def _all_pair_specs() -> Iterable[dict[str, Any]]:
    yield from _identity_pairs()
    yield from _timezone_mismatch_pairs()
    yield from _rounding_precision_pairs()
    yield from _type_coercion_pairs()
    yield from _schema_drift_pairs()
    yield from _aggregation_function_pairs()
    yield from _join_key_mismatch_pairs()
    yield from _ordering_dependence_pairs()
    yield from _window_function_pairs()
    yield from _window_boundary_pairs()
    yield from _as_of_join_pairs()
    yield from _fill_na_pairs()
    yield from _cross_framework_identity_pairs()
    yield from _cross_framework_divergence_pairs()
    yield from _cross_framework_polars_pairs()
    yield from _cross_framework_sort_pairs()
    yield from _adversarial_predicate_pairs()
    yield from _adversarial_structure_pairs()
    yield from _adversarial_cosmetic_pairs()
    yield from _adversarial_multi_divergence_pairs()
    yield from _udf_pairs()
    yield from _unit_mismatch_pairs()
    yield from _seed_mismatch_pairs()
    yield from _categorical_encoding_pairs()
    yield from _feature_leakage_temporal_pairs()


def _normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize a pair spec to the modern format with explicit side dicts.

    Old-style specs use top-level ``offline_sql`` / ``online_sql`` /
    ``offline_schemas`` / ``online_schemas`` fields and are SQL-only;
    new-style specs nest each side under ``offline`` / ``online`` with a
    ``language`` discriminator. Both are accepted so the SQL-only
    generator functions below do not need to change.
    """

    if "offline" in spec and "online" in spec:
        return spec

    out: dict[str, Any] = {
        "name": spec["name"],
        "category": spec["category"],
        "description": spec["description"],
        "offline": {
            "language": "sql",
            "sql": spec["offline_sql"],
            "schemas": spec["offline_schemas"],
        },
        "online": {
            "language": "sql",
            "sql": spec["online_sql"],
            "schemas": spec["online_schemas"],
        },
        "expected_divergences": spec["expected_divergences"],
    }
    if "expected_localization" in spec:
        out["expected_localization"] = spec["expected_localization"]
    return out


def _write_side(target: Path, side: dict[str, Any], *, side_label: str) -> dict[str, Any]:
    """Materialize one side of a pair and return its ``meta.yaml`` fragment."""

    language = side["language"]
    if language == "sql":
        source = f"{side_label}.sql"
        (target / source).write_text(side["sql"])
        return {
            "language": "sql",
            "source": source,
            "schemas": {
                table: [list(col) for col in cols] for table, cols in side["schemas"].items()
            },
        }
    if language in ("pandas", "polars"):
        source = f"{side_label}.py"
        (target / source).write_text(side["python_source"])
        side_meta: dict[str, Any] = {
            "language": language,
            "source": source,
            "function": side["function"],
            "input_schema": [list(col) for col in side["input_schema"]],
            "source_name": side.get("source_name", "input"),
        }
        if side.get("event_time_column"):
            side_meta["event_time_column"] = side["event_time_column"]
        return side_meta
    raise ValueError(f"unsupported pair language: {language!r}")


def _write_pair(target: Path, spec: dict[str, Any]) -> None:
    spec = _normalize_spec(spec)
    target.mkdir(parents=True, exist_ok=True)

    meta_offline = _write_side(target, spec["offline"], side_label="offline")
    meta_online = _write_side(target, spec["online"], side_label="online")

    meta: dict[str, Any] = {
        "name": spec["name"],
        "bucket": "synthetic",
        "category": spec["category"],
        "description": spec["description"],
        "expected_divergences": spec["expected_divergences"],
    }
    if spec.get("expected_localization"):
        meta["expected_localization"] = spec["expected_localization"]
    meta["offline"] = meta_offline
    meta["online"] = meta_online
    meta["generator"] = {
        "module": "bench.scripts.generate_synthetic",
        "version": 1,
    }
    with (target / "meta.yaml").open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


# --- cross-framework pairs ---------------------------------------------------


def _cross_framework_identity_pairs() -> Iterable[dict[str, Any]]:
    """Cross-framework pairs that MUST diff to ``()`` -- the empirical
    backbone of the cross-framework equivalence claim (a pandas pipeline and
    the structurally equivalent SQL query produce equivalent fingerprints)."""

    events = [("uid", "int64"), ("score", "float64")]

    yield {
        "name": "cross_framework_identity_filter_project",
        "category": "identity",
        "description": (
            "pandas df[df.score > 0][['uid', 'score']] vs SQL "
            "SELECT uid, score FROM events WHERE score > 0."
        ),
        "offline": {
            "language": "pandas",
            "python_source": (
                "def offline(df):\n    return df[df['score'] > 0][['uid', 'score']]\n"
            ),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "sql",
            "sql": "SELECT uid, score FROM events WHERE score > 0\n",
            "schemas": {"events": events},
        },
        "expected_divergences": [],
    }

    yield {
        "name": "cross_framework_identity_groupby",
        "category": "identity",
        "description": (
            "pandas df.groupby('uid').agg({'score': 'sum'}) vs SQL "
            "SELECT uid, SUM(score) AS score FROM events GROUP BY uid."
        ),
        "offline": {
            "language": "pandas",
            "python_source": (
                "def offline(df):\n    return df.groupby('uid').agg({'score': 'sum'})\n"
            ),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "sql",
            "sql": "SELECT uid, SUM(score) AS score FROM events GROUP BY uid\n",
            "schemas": {"events": events},
        },
        "expected_divergences": [],
    }

    yield {
        "name": "cross_framework_identity_rename",
        "category": "identity",
        "description": (
            "pandas df.rename(columns={'uid': 'user_id'}) vs SQL "
            "SELECT uid AS user_id, score FROM events."
        ),
        "offline": {
            "language": "pandas",
            "python_source": (
                "def offline(df):\n    return df.rename(columns={'uid': 'user_id'})\n"
            ),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "sql",
            "sql": "SELECT uid AS user_id, score FROM events\n",
            "schemas": {"events": events},
        },
        "expected_divergences": [],
    }


def _cross_framework_divergence_pairs() -> Iterable[dict[str, Any]]:
    """Cross-framework pairs where pandas and SQL DO differ. The diff
    engine must produce the same divergence categories it would on a
    same-framework pair with the same structural difference."""

    events = [("uid", "int64"), ("score", "float64")]

    yield {
        "name": "cross_framework_aggregation_function_sum_vs_avg",
        "category": "aggregation_function",
        "description": "pandas sums, SQL averages.",
        "offline": {
            "language": "pandas",
            "python_source": (
                "def offline(df):\n    return df.groupby('uid').agg({'score': 'sum'})\n"
            ),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "sql",
            "sql": "SELECT uid, AVG(score) AS score FROM events GROUP BY uid\n",
            "schemas": {"events": events},
        },
        "expected_divergences": [{"category": "aggregation_function"}],
        "expected_localization": [_loc("aggregate")],
    }

    events_utc = [("uid", "int64"), ("ts", "timestamp[ns, UTC]")]
    events_pt = [("uid", "int64"), ("ts", "timestamp[ns, US/Pacific]")]
    yield {
        "name": "cross_framework_timezone_mismatch",
        "category": "timezone_mismatch",
        "description": "pandas reads events.ts as UTC; SQL as US/Pacific.",
        "offline": {
            "language": "pandas",
            "python_source": "def offline(df):\n    return df\n",
            "function": "offline",
            "input_schema": events_utc,
            "source_name": "events",
        },
        "online": {
            # SELECT * (not an explicit column list) so the SQL side emits
            # just a Source op, matching pandas's `return df`. Otherwise
            # the column-listing Project on the SQL side produces a
            # spurious schema_drift via length mismatch.
            "language": "sql",
            "sql": "SELECT * FROM events\n",
            "schemas": {"events": events_pt},
        },
        "expected_divergences": [{"category": "timezone_mismatch"}],
        "expected_localization": [_loc("source")],
    }


def _cross_framework_sort_pairs() -> Iterable[dict[str, Any]]:
    """Sort pairs exercising the wrapper tracers' new sort support. The
    identity pair (Polars vs pandas, same sort) must diff to ``()``; the
    direction pair (pandas DESC vs SQL ASC) must surface
    ``ordering_dependence``."""

    events = [("uid", "int64"), ("score", "float64")]

    yield {
        "name": "cross_framework_identity_sort_polars_vs_pandas",
        "category": "identity",
        "description": "polars lf.sort('score') vs pandas df.sort_values('score'); diff empty.",
        "offline": {
            "language": "polars",
            "python_source": "def offline(lf, pl):\n    return lf.sort('score')\n",
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "pandas",
            "python_source": "def online(df):\n    return df.sort_values('score')\n",
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [],
    }

    yield {
        "name": "cross_framework_sort_direction_pandas_vs_sql",
        "category": "ordering_dependence",
        "description": "pandas sorts score DESC; SQL sorts score ASC.",
        "offline": {
            "language": "pandas",
            "python_source": (
                "def offline(df):\n    return df.sort_values('score', ascending=False)\n"
            ),
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "sql",
            "sql": "SELECT * FROM events ORDER BY score\n",
            "schemas": {"events": events},
        },
        "expected_divergences": [{"category": "ordering_dependence"}],
        "expected_localization": [_loc("sort")],
    }


def _cross_framework_polars_pairs() -> Iterable[dict[str, Any]]:
    """Cross-framework pairs exercising the third tracer (Polars), both
    against SQL and directly against pandas. Identity pairs MUST diff to
    ``()`` (cross-framework equivalence, extended to three frameworks); the
    sum-vs-avg pair must still surface ``aggregation_function``."""

    events = [("uid", "int64"), ("score", "float64")]
    filter_project_src = (
        "def offline(lf, pl):\n    return lf.filter(pl.col('score') > 0).select('uid', 'score')\n"
    )
    groupby_sum_src = (
        "def offline(lf, pl):\n    return lf.group_by('uid').agg(pl.col('score').sum())\n"
    )

    yield {
        "name": "cross_framework_identity_polars_filter_project",
        "category": "identity",
        "description": (
            "polars lf.filter(pl.col('score') > 0).select('uid', 'score') vs "
            "SQL SELECT uid, score FROM events WHERE score > 0."
        ),
        "offline": {
            "language": "polars",
            "python_source": filter_project_src,
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "sql",
            "sql": "SELECT uid, score FROM events WHERE score > 0\n",
            "schemas": {"events": events},
        },
        "expected_divergences": [],
    }

    yield {
        "name": "cross_framework_identity_polars_groupby",
        "category": "identity",
        "description": (
            "polars lf.group_by('uid').agg(pl.col('score').sum()) vs "
            "SQL SELECT uid, SUM(score) AS score FROM events GROUP BY uid."
        ),
        "offline": {
            "language": "polars",
            "python_source": groupby_sum_src,
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "sql",
            "sql": "SELECT uid, SUM(score) AS score FROM events GROUP BY uid\n",
            "schemas": {"events": events},
        },
        "expected_divergences": [],
    }

    yield {
        "name": "cross_framework_identity_polars_vs_pandas",
        "category": "identity",
        "description": (
            "polars and pandas agree directly (no SQL in the loop): filter "
            "then project must diff to ()."
        ),
        "offline": {
            "language": "polars",
            "python_source": filter_project_src,
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "pandas",
            "python_source": (
                "def online(df):\n    return df[df['score'] > 0][['uid', 'score']]\n"
            ),
            "function": "online",
            "input_schema": events,
            "source_name": "events",
        },
        "expected_divergences": [],
    }

    yield {
        "name": "cross_framework_polars_sum_vs_sql_avg",
        "category": "aggregation_function",
        "description": "polars sums, SQL averages.",
        "offline": {
            "language": "polars",
            "python_source": groupby_sum_src,
            "function": "offline",
            "input_schema": events,
            "source_name": "events",
        },
        "online": {
            "language": "sql",
            "sql": "SELECT uid, AVG(score) AS score FROM events GROUP BY uid\n",
            "schemas": {"events": events},
        },
        "expected_divergences": [{"category": "aggregation_function"}],
        "expected_localization": [_loc("aggregate")],
    }


# --- adversarial pairs ------------------------------------------------------


def _adversarial_predicate_pairs() -> Iterable[dict[str, Any]]:
    """Filter-predicate differences that exercise the classifier's
    fallback paths.

    Plain threshold changes do not map to any taxonomy category cleanly;
    the classifier emits ``schema_drift`` as a fallback so the divergence
    does not silently vanish. NULL-mentioning predicate differences route
    to ``null_handling``.
    """

    events = [("uid", "int64"), ("score", "float64")]

    yield {
        "name": "adversarial_filter_threshold_change",
        "category": "schema_drift",
        "description": (
            "Same shape, different filter threshold. No taxonomy category "
            "fits cleanly; classifier falls back to schema_drift."
        ),
        "offline": {
            "language": "sql",
            "sql": "SELECT uid FROM events WHERE score > 0\n",
            "schemas": {"events": events},
        },
        "online": {
            "language": "sql",
            "sql": "SELECT uid FROM events WHERE score > 1\n",
            "schemas": {"events": events},
        },
        "expected_divergences": [{"category": "schema_drift"}],
        "expected_localization": [_loc("filter")],
    }

    yield {
        "name": "adversarial_filter_null_handling",
        "category": "null_handling",
        "description": (
            "Offline filters by IS NOT NULL; online uses a numeric "
            "threshold. The predicate mentions NULL so the classifier "
            "routes to null_handling."
        ),
        "offline": {
            "language": "sql",
            "sql": "SELECT uid FROM events WHERE score IS NOT NULL\n",
            "schemas": {"events": events},
        },
        "online": {
            "language": "sql",
            "sql": "SELECT uid FROM events WHERE score > 0\n",
            "schemas": {"events": events},
        },
        "expected_divergences": [{"category": "null_handling"}],
        "expected_localization": [_loc("filter")],
    }


def _adversarial_structure_pairs() -> Iterable[dict[str, Any]]:
    """Pipelines that differ in OPERATION COUNT, not just operation
    content. Tests the engine's behavior on length mismatch."""

    events = [("uid", "int64"), ("score", "float64")]

    yield {
        "name": "adversarial_extra_filter_op",
        "category": "schema_drift",
        "description": (
            "Online has an extra WHERE filter that offline lacks. Length "
            "mismatch surfaces as schema_drift."
        ),
        "offline": {
            "language": "sql",
            "sql": "SELECT uid FROM events\n",
            "schemas": {"events": events},
        },
        "online": {
            "language": "sql",
            "sql": "SELECT uid FROM events WHERE score > 0\n",
            "schemas": {"events": events},
        },
        "expected_divergences": [{"category": "schema_drift"}],
        "expected_localization": [_loc("filter", side="online")],
    }


def _adversarial_cosmetic_pairs() -> Iterable[dict[str, Any]]:
    """Cosmetic differences that the canonicalizer must absorb.

    Identity-equivalent SQL that differs only in keyword case or
    whitespace must diff to ``()``. These pairs guard precision: a
    false-positive divergence here is the kind that erodes user trust.
    """

    events = [("uid", "int64"), ("score", "float64")]

    yield {
        "name": "adversarial_cosmetic_keyword_case",
        "category": "identity",
        "description": "Same query, different SQL keyword casing.",
        "offline": {
            "language": "sql",
            "sql": "SELECT uid FROM events WHERE score > 0\n",
            "schemas": {"events": events},
        },
        "online": {
            "language": "sql",
            "sql": "select uid from events where score > 0\n",
            "schemas": {"events": events},
        },
        "expected_divergences": [],
    }


def _adversarial_multi_divergence_pairs() -> Iterable[dict[str, Any]]:
    """A pair where both sides differ in MORE THAN ONE way at once. The
    engine must surface every category that applies."""

    events_left = [
        ("uid", "int64"),
        ("ts", "timestamp[ns, UTC]"),
        ("score", "float64"),
    ]
    events_right = [
        ("uid", "int64"),
        ("ts", "timestamp[ns, US/Pacific]"),
        ("score", "int64"),
    ]
    yield {
        "name": "adversarial_multi_divergence_tz_and_dtype",
        "category": "timezone_mismatch",
        "description": (
            "ts has different timezone AND score has different dtype. "
            "Engine must emit both timezone_mismatch and type_coercion."
        ),
        "offline": {
            "language": "sql",
            "sql": "SELECT uid, ts, score FROM events\n",
            "schemas": {"events": events_left},
        },
        "online": {
            "language": "sql",
            "sql": "SELECT uid, ts, score FROM events\n",
            "schemas": {"events": events_right},
        },
        "expected_divergences": [
            {"category": "timezone_mismatch"},
            {"category": "type_coercion"},
        ],
        "expected_localization": [_loc("source")],
    }


def regenerate(target_dir: Path = SYNTHETIC_DIR) -> int:
    """Wipe ``target_dir`` and write the full synthetic set. Returns the
    number of pairs written.
    """

    if target_dir.exists():
        shutil.rmtree(target_dir)

    count = 0
    for spec in _all_pair_specs():
        pair_dir = target_dir / spec["category"] / spec["name"]
        _write_pair(pair_dir, spec)
        count += 1
    return count


if __name__ == "__main__":  # pragma: no cover
    n = regenerate()
    print(f"Wrote {n} synthetic pairs to {SYNTHETIC_DIR}")
