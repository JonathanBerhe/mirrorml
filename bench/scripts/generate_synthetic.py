"""Programmatic generator for synthetic MirrorBench pairs.

Per ``CLAUDE.md``: synthetic pairs MUST be programmatically generated;
hand-authoring is disallowed. This module is the single entry point.
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
    yield from _cross_framework_identity_pairs()
    yield from _cross_framework_divergence_pairs()
    yield from _cross_framework_polars_pairs()
    yield from _cross_framework_sort_pairs()
    yield from _adversarial_predicate_pairs()
    yield from _adversarial_structure_pairs()
    yield from _adversarial_cosmetic_pairs()
    yield from _adversarial_multi_divergence_pairs()


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

    return {
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
        return {
            "language": language,
            "source": source,
            "function": side["function"],
            "input_schema": [list(col) for col in side["input_schema"]],
            "source_name": side.get("source_name", "input"),
        }
    raise ValueError(f"unsupported pair language: {language!r}")


def _write_pair(target: Path, spec: dict[str, Any]) -> None:
    spec = _normalize_spec(spec)
    target.mkdir(parents=True, exist_ok=True)

    meta_offline = _write_side(target, spec["offline"], side_label="offline")
    meta_online = _write_side(target, spec["online"], side_label="online")

    meta = {
        "name": spec["name"],
        "bucket": "synthetic",
        "category": spec["category"],
        "description": spec["description"],
        "expected_divergences": spec["expected_divergences"],
        "offline": meta_offline,
        "online": meta_online,
        "generator": {
            "module": "bench.scripts.generate_synthetic",
            "version": 1,
        },
    }
    with (target / "meta.yaml").open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


# --- cross-framework pairs ---------------------------------------------------


def _cross_framework_identity_pairs() -> Iterable[dict[str, Any]]:
    """Cross-framework pairs that MUST diff to ``()`` -- the empirical
    backbone of PAPER.md C4 (a pandas pipeline and the structurally
    equivalent SQL query produce equivalent fingerprints)."""

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
    }


def _cross_framework_polars_pairs() -> Iterable[dict[str, Any]]:
    """Cross-framework pairs exercising the third tracer (Polars), both
    against SQL and directly against pandas. Identity pairs MUST diff to
    ``()`` (PAPER.md C4, extended to three frameworks); the sum-vs-avg
    pair must still surface ``aggregation_function``."""

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
