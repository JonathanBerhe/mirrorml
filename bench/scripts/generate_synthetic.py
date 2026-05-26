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


def _meta_for(spec: dict[str, Any]) -> dict[str, Any]:
    """Build the ``meta.yaml`` dict from a pair spec."""

    return {
        "name": spec["name"],
        "bucket": "synthetic",
        "category": spec["category"],
        "description": spec["description"],
        "expected_divergences": spec["expected_divergences"],
        "offline": {
            "language": "sql",
            "source": "offline.sql",
            "schemas": {
                table: [list(col) for col in cols]
                for table, cols in spec["offline_schemas"].items()
            },
        },
        "online": {
            "language": "sql",
            "source": "online.sql",
            "schemas": {
                table: [list(col) for col in cols] for table, cols in spec["online_schemas"].items()
            },
        },
        "generator": {
            "module": "bench.scripts.generate_synthetic",
            "version": 1,
        },
    }


def _write_pair(target: Path, spec: dict[str, Any]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    meta = _meta_for(spec)
    with (target / "meta.yaml").open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    (target / "offline.sql").write_text(spec["offline_sql"])
    (target / "online.sql").write_text(spec["online_sql"])


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
