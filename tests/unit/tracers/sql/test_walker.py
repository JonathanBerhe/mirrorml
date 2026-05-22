"""M2 phase 1 SQL tracer surface: SELECT / FROM / WHERE / projection.

Tests cover the supported subset and assert that everything outside it
raises :class:`UnsupportedOperationError` with a message that names the
missing feature."""

from __future__ import annotations

import pytest

from mirrorml import trace_sql
from mirrorml.exceptions import UnsupportedOperationError

EVENTS_SCHEMA = (
    ("uid", "int64"),
    ("score", "float64"),
    ("ts", "timestamp[ns, UTC]"),
)


# --- supported surface -------------------------------------------------------


def test_select_star_emits_only_source() -> None:
    fp = trace_sql("SELECT * FROM events", schemas={"events": EVENTS_SCHEMA})
    assert fp.framework == "sql"
    assert [op.kind for op in fp.operations] == ["source"]
    assert fp.output_schema == EVENTS_SCHEMA


def test_select_columns_emits_source_then_project() -> None:
    fp = trace_sql(
        "SELECT uid, score FROM events",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert [op.kind for op in fp.operations] == ["source", "project"]
    assert fp.output_schema == (("uid", "int64"), ("score", "float64"))


def test_select_with_where_emits_source_filter_project() -> None:
    fp = trace_sql(
        "SELECT uid FROM events WHERE score > 0",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert [op.kind for op in fp.operations] == ["source", "filter", "project"]
    assert fp.output_schema == (("uid", "int64"),)


def test_select_star_with_where_emits_source_filter() -> None:
    fp = trace_sql(
        "SELECT * FROM events WHERE score > 0",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert [op.kind for op in fp.operations] == ["source", "filter"]
    assert fp.output_schema == EVENTS_SCHEMA


def test_qualified_column_names_drop_table_prefix() -> None:
    fp = trace_sql(
        "SELECT events.uid FROM events",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert fp.output_schema == (("uid", "int64"),)


def test_source_op_carries_full_table_schema() -> None:
    """Even with a narrow projection, the Source op records all columns of
    the table. Projection narrows the output, not the source."""

    fp = trace_sql(
        "SELECT uid FROM events",
        schemas={"events": EVENTS_SCHEMA},
    )
    source = fp.operations[0]
    assert source.kind == "source"
    assert source.columns == EVENTS_SCHEMA


def test_projection_order_is_preserved() -> None:
    fp_ab = trace_sql(
        "SELECT uid, score FROM events",
        schemas={"events": EVENTS_SCHEMA},
    )
    fp_ba = trace_sql(
        "SELECT score, uid FROM events",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert fp_ab.output_schema != fp_ba.output_schema
    assert fp_ab.fingerprint_id != fp_ba.fingerprint_id


# --- dialect handling --------------------------------------------------------


def test_dialect_does_not_change_fingerprint_for_equivalent_sql() -> None:
    """A query that parses identically under two dialects must produce the
    same fingerprint. This is the canonical-form claim from PAPER.md C4."""

    base = "SELECT uid FROM events WHERE score > 0"
    fp_a = trace_sql(base, schemas={"events": EVENTS_SCHEMA}, dialect="postgres")
    fp_b = trace_sql(base, schemas={"events": EVENTS_SCHEMA}, dialect="snowflake")
    assert fp_a.fingerprint_id == fp_b.fingerprint_id


# --- rejection of missing schemas / bad columns ------------------------------


def test_missing_schema_for_referenced_table_is_actionable() -> None:
    with pytest.raises(UnsupportedOperationError, match="missing_table"):
        trace_sql("SELECT a FROM missing_table", schemas={})


def test_invalid_column_in_projection_is_actionable() -> None:
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_sql(
            "SELECT bogus FROM events",
            schemas={"events": EVENTS_SCHEMA},
        )


def test_select_without_from_is_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="without FROM"):
        trace_sql("SELECT 1", schemas={})


def test_empty_query_is_rejected() -> None:
    with pytest.raises(UnsupportedOperationError):
        trace_sql("", schemas={})


def test_unparseable_query_is_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="parse"):
        trace_sql("SELECT FROM WHERE", schemas={})


# --- rejection of out-of-scope features --------------------------------------


@pytest.mark.parametrize(
    "query,marker",
    [
        ("SELECT a FROM t JOIN u ON t.x = u.x", "JOIN"),
        ("SELECT a FROM t GROUP BY a", "GROUP BY"),
        ("SELECT a FROM t ORDER BY a", "ORDER BY"),
        ("SELECT a FROM t GROUP BY a HAVING a > 0", "GROUP BY"),
        ("SELECT a FROM t LIMIT 5", "LIMIT"),
        ("SELECT DISTINCT a FROM t", "DISTINCT"),
        ("WITH c AS (SELECT 1) SELECT * FROM c", "WITH"),
        ("SELECT a FROM t UNION SELECT a FROM u", "UNION"),
    ],
)
def test_out_of_scope_features_raise_with_actionable_message(query: str, marker: str) -> None:
    schemas = {
        "t": (("a", "int64"), ("x", "int64")),
        "u": (("a", "int64"), ("x", "int64")),
    }
    with pytest.raises(UnsupportedOperationError, match=marker):
        trace_sql(query, schemas=schemas)


def test_projection_aliases_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="alias"):
        trace_sql(
            "SELECT uid AS user_id FROM events",
            schemas={"events": EVENTS_SCHEMA},
        )


def test_projection_expressions_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bare column"):
        trace_sql(
            "SELECT uid + 1 FROM events",
            schemas={"events": EVENTS_SCHEMA},
        )


def test_subquery_in_from_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="Subqueries"):
        trace_sql(
            "SELECT * FROM (SELECT uid FROM events)",
            schemas={"events": EVENTS_SCHEMA},
        )


# --- predicate capture -------------------------------------------------------


def test_filter_predicate_is_captured_as_string() -> None:
    fp = trace_sql(
        "SELECT uid FROM events WHERE score > 0",
        schemas={"events": EVENTS_SCHEMA},
    )
    flt = fp.operations[1]
    assert flt.kind == "filter"
    assert isinstance(flt.predicate, str)
    assert "score" in flt.predicate


def test_different_predicates_produce_different_fingerprints() -> None:
    fp_gt = trace_sql(
        "SELECT uid FROM events WHERE score > 0",
        schemas={"events": EVENTS_SCHEMA},
    )
    fp_lt = trace_sql(
        "SELECT uid FROM events WHERE score < 0",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert fp_gt.fingerprint_id != fp_lt.fingerprint_id


# --- structural-hash op_id property ------------------------------------------


def test_equivalent_pipelines_produce_identical_fingerprint_ids() -> None:
    """Two SQL queries with the same logical structure produce the same
    fingerprint_id regardless of cosmetic differences in the SQL text."""

    schemas = {"events": EVENTS_SCHEMA}
    fp_a = trace_sql(
        "SELECT uid, score FROM events WHERE score > 0",
        schemas=schemas,
    )
    fp_b = trace_sql(
        "select uid, score from events where score > 0",  # different case
        schemas=schemas,
    )
    assert fp_a.fingerprint_id == fp_b.fingerprint_id
