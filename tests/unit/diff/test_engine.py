"""Diff engine end-to-end tests against fingerprints produced by trace_sql.

Each category section pairs a positive case (a representative divergence
is detected and localized) with the identity property (equivalent
fingerprints diff to ``()``).
"""

from __future__ import annotations

from mirrorml import diff, trace_sql
from mirrorml.fingerprint import build_fingerprint
from mirrorml.fingerprint.operations import Source

EVENTS = (("uid", "int64"), ("score", "float64"))
EVENTS_TS = (
    ("uid", "int64"),
    ("score", "float64"),
    ("ts", "timestamp[ns, UTC]"),
)


# --- identity ----------------------------------------------------------------


def test_identical_fingerprints_diff_to_empty() -> None:
    fp = trace_sql("SELECT uid FROM events", schemas={"events": EVENTS})
    assert diff(fp, fp) == ()


def test_two_independent_traces_of_same_query_diff_to_empty() -> None:
    a = trace_sql("SELECT uid FROM events", schemas={"events": EVENTS})
    b = trace_sql("SELECT uid FROM events", schemas={"events": EVENTS})
    assert a.fingerprint_id == b.fingerprint_id
    assert diff(a, b) == ()


def test_cross_dialect_equivalent_sql_diffs_to_empty() -> None:
    a = trace_sql(
        "SELECT uid FROM events WHERE score > 0",
        schemas={"events": EVENTS},
        dialect="postgres",
    )
    b = trace_sql(
        "SELECT uid FROM events WHERE score > 0",
        schemas={"events": EVENTS},
        dialect="snowflake",
    )
    assert diff(a, b) == ()


# --- fast path ---------------------------------------------------------------


def test_fast_path_short_circuits_on_fingerprint_id_equality() -> None:
    """When the ids match, the engine returns ``()`` without walking. We
    cannot observe the walk being skipped from the outside, but we can
    confirm the return value."""

    fp = trace_sql("SELECT uid FROM events", schemas={"events": EVENTS})
    assert diff(fp, fp) == ()


# --- schema_drift ------------------------------------------------------------


def test_schema_drift_column_only_on_one_side() -> None:
    a = trace_sql("SELECT uid FROM events", schemas={"events": EVENTS})
    b = trace_sql("SELECT uid, score FROM events", schemas={"events": EVENTS})
    divs = diff(a, b)
    categories = [d.category for d in divs]
    assert "schema_drift" in categories
    assert any("score" in d.detail for d in divs)


def test_schema_drift_op_count_differs() -> None:
    a = trace_sql("SELECT uid FROM events", schemas={"events": EVENTS})
    b = trace_sql(
        "SELECT uid FROM events WHERE score > 0",
        schemas={"events": EVENTS},
    )
    divs = diff(a, b)
    assert any(d.category == "schema_drift" and "count" in d.detail for d in divs)


# --- type_coercion -----------------------------------------------------------


def test_type_coercion_int_vs_float() -> None:
    a = trace_sql("SELECT x FROM t", schemas={"t": (("x", "int64"),)})
    b = trace_sql("SELECT x FROM t", schemas={"t": (("x", "float64"),)})
    divs = diff(a, b)
    assert any(d.category == "type_coercion" for d in divs)


def test_type_coercion_int_widening() -> None:
    a = trace_sql("SELECT x FROM t", schemas={"t": (("x", "int32"),)})
    b = trace_sql("SELECT x FROM t", schemas={"t": (("x", "int64"),)})
    divs = diff(a, b)
    assert any(d.category == "type_coercion" for d in divs)


# --- timezone_mismatch -------------------------------------------------------


def test_timezone_mismatch_utc_vs_pacific() -> None:
    a = trace_sql(
        "SELECT ts FROM t",
        schemas={"t": (("ts", "timestamp[ns, UTC]"),)},
    )
    b = trace_sql(
        "SELECT ts FROM t",
        schemas={"t": (("ts", "timestamp[ns, US/Pacific]"),)},
    )
    divs = diff(a, b)
    assert any(d.category == "timezone_mismatch" for d in divs)


def test_timezone_mismatch_naive_vs_aware() -> None:
    a = trace_sql(
        "SELECT ts FROM t",
        schemas={"t": (("ts", "timestamp[ns]"),)},
    )
    b = trace_sql(
        "SELECT ts FROM t",
        schemas={"t": (("ts", "timestamp[ns, UTC]"),)},
    )
    divs = diff(a, b)
    assert any(d.category == "timezone_mismatch" for d in divs)


# --- rounding_precision ------------------------------------------------------


def test_rounding_precision_timestamp_unit() -> None:
    a = trace_sql(
        "SELECT ts FROM t",
        schemas={"t": (("ts", "timestamp[ns, UTC]"),)},
    )
    b = trace_sql(
        "SELECT ts FROM t",
        schemas={"t": (("ts", "timestamp[us, UTC]"),)},
    )
    divs = diff(a, b)
    # Same timezone, different unit -> rounding_precision (not timezone_mismatch)
    assert any(d.category == "rounding_precision" for d in divs)
    assert not any(d.category == "timezone_mismatch" for d in divs)


def test_rounding_precision_decimal_scale() -> None:
    a = trace_sql(
        "SELECT amount FROM t",
        schemas={"t": (("amount", "decimal[18, 2]"),)},
    )
    b = trace_sql(
        "SELECT amount FROM t",
        schemas={"t": (("amount", "decimal[18, 4]"),)},
    )
    divs = diff(a, b)
    assert any(d.category == "rounding_precision" for d in divs)


# --- aggregation_function ----------------------------------------------------


def test_aggregation_function_sum_vs_avg() -> None:
    a = trace_sql(
        "SELECT uid, SUM(score) AS s FROM events GROUP BY uid",
        schemas={"events": EVENTS},
    )
    b = trace_sql(
        "SELECT uid, AVG(score) AS s FROM events GROUP BY uid",
        schemas={"events": EVENTS},
    )
    divs = diff(a, b)
    assert any(d.category == "aggregation_function" for d in divs)


def test_aggregation_function_different_input_column() -> None:
    schemas = {"t": (("uid", "int64"), ("a", "float64"), ("b", "float64"))}
    fp_a = trace_sql("SELECT uid, SUM(a) AS x FROM t GROUP BY uid", schemas=schemas)
    fp_b = trace_sql("SELECT uid, SUM(b) AS x FROM t GROUP BY uid", schemas=schemas)
    divs = diff(fp_a, fp_b)
    assert any(d.category == "aggregation_function" for d in divs)


def test_aggregation_function_localizes_to_aggregate_op() -> None:
    a = trace_sql(
        "SELECT uid, SUM(score) AS s FROM events GROUP BY uid",
        schemas={"events": EVENTS},
    )
    b = trace_sql(
        "SELECT uid, AVG(score) AS s FROM events GROUP BY uid",
        schemas={"events": EVENTS},
    )
    divs = diff(a, b)
    agg_divs = [d for d in divs if d.category == "aggregation_function"]
    assert agg_divs
    # The aggregate op is the second op (source, aggregate).
    agg_op_id_a = a.operations[1].op_id
    agg_op_id_b = b.operations[1].op_id
    assert all(d.left_op_id == agg_op_id_a for d in agg_divs)
    assert all(d.right_op_id == agg_op_id_b for d in agg_divs)


# --- join_key_mismatch -------------------------------------------------------


def test_join_key_mismatch_different_keys() -> None:
    schemas = {
        "a": (("k1", "int64"), ("k2", "int64"), ("v", "float64")),
        "b": (("k1", "int64"), ("k2", "int64"), ("v", "float64")),
    }
    fp_a = trace_sql("SELECT a.v FROM a JOIN b ON a.k1 = b.k1", schemas=schemas)
    fp_b = trace_sql("SELECT a.v FROM a JOIN b ON a.k2 = b.k2", schemas=schemas)
    divs = diff(fp_a, fp_b)
    assert any(d.category == "join_key_mismatch" for d in divs)


def test_join_kind_difference_surfaces_a_divergence() -> None:
    schemas = {
        "a": (("k", "int64"), ("v", "float64")),
        "b": (("k", "int64"), ("v", "float64")),
    }
    fp_inner = trace_sql("SELECT a.v FROM a JOIN b ON a.k = b.k", schemas=schemas)
    fp_left = trace_sql("SELECT a.v FROM a LEFT JOIN b ON a.k = b.k", schemas=schemas)
    divs = diff(fp_inner, fp_left)
    # join kind diff is currently surfaced as schema_drift on the Join op.
    assert any(d.category == "schema_drift" and "join kind" in d.detail for d in divs)


# --- ordering_dependence -----------------------------------------------------


def test_ordering_dependence_different_sort_by() -> None:
    a = trace_sql(
        "SELECT uid, score FROM events ORDER BY uid",
        schemas={"events": EVENTS},
    )
    b = trace_sql(
        "SELECT uid, score FROM events ORDER BY score",
        schemas={"events": EVENTS},
    )
    divs = diff(a, b)
    assert any(d.category == "ordering_dependence" for d in divs)


def test_ordering_dependence_direction_change() -> None:
    a = trace_sql(
        "SELECT uid FROM events ORDER BY uid ASC",
        schemas={"events": EVENTS},
    )
    b = trace_sql(
        "SELECT uid FROM events ORDER BY uid DESC",
        schemas={"events": EVENTS},
    )
    divs = diff(a, b)
    assert any(d.category == "ordering_dependence" for d in divs)


# --- cross-framework ---------------------------------------------------------


def test_cross_framework_diff_does_not_flag_framework() -> None:
    """A pandas-traced fingerprint and a SQL-traced fingerprint differ in
    the ``framework`` field but that is not itself a divergence (PAPER.md
    C4 is the cross-framework equivalence claim)."""

    sql_fp = trace_sql("SELECT uid FROM events", schemas={"events": EVENTS})
    pandas_fp = build_fingerprint(
        framework="pandas",
        input_schema=(("uid", "int64"), ("score", "float64")),
        output_schema=(("uid", "int64"),),
        operations=[
            Source(op_id="s", name="events", columns=EVENTS),
        ],
    )
    # Same input + output, but the SQL fingerprint also has a Project op.
    divs = diff(sql_fp, pandas_fp)
    categories = {d.category for d in divs}
    # No category is "framework" -- the field is intentionally not a divergence.
    assert "framework" not in categories


# --- output stability --------------------------------------------------------


def test_diff_output_is_deterministic() -> None:
    a = trace_sql("SELECT uid FROM events", schemas={"events": EVENTS})
    b = trace_sql("SELECT uid, score FROM events", schemas={"events": EVENTS})
    first = diff(a, b)
    second = diff(a, b)
    assert first == second


def test_diff_returns_tuple_not_list() -> None:
    a = trace_sql("SELECT uid FROM events", schemas={"events": EVENTS})
    b = trace_sql("SELECT uid, score FROM events", schemas={"events": EVENTS})
    assert isinstance(diff(a, b), tuple)
