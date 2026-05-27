"""M2 phase 1 SQL tracer surface: SELECT / FROM / WHERE / projection.

Tests cover the supported subset and assert that everything outside it
raises :class:`UnsupportedOperationError` with a message that names the
missing feature."""

from __future__ import annotations

import pytest

from mirrorml import trace_sql
from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint.operations import Aggregate, Sort

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
    same fingerprint. This is the canonical-form claim."""

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
        ("SELECT a FROM t LIMIT 5", "LIMIT"),
        ("SELECT DISTINCT a FROM t", "DISTINCT"),
        ("WITH c AS (SELECT 1) SELECT * FROM c", "WITH"),
        ("SELECT a FROM t UNION SELECT a FROM u", "UNION"),
        ("SELECT a FROM t CROSS JOIN u", "CROSS"),
        ("SELECT a FROM t JOIN u USING (x)", "USING"),
    ],
)
def test_out_of_scope_features_raise_with_actionable_message(query: str, marker: str) -> None:
    schemas = {
        "t": (("a", "int64"), ("x", "int64")),
        "u": (("a", "int64"), ("x", "int64")),
    }
    with pytest.raises(UnsupportedOperationError, match=marker):
        trace_sql(query, schemas=schemas)


def test_alias_of_expression_rejected() -> None:
    """Aliasing a bare column is supported; aliasing an expression (function
    call, arithmetic) is not yet."""

    with pytest.raises(UnsupportedOperationError, match="non-column"):
        trace_sql(
            "SELECT uid + 1 AS bumped FROM events",
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


# --- column aliasing ---------------------------------------------------------


def test_alias_renames_column_in_output_schema() -> None:
    fp = trace_sql(
        "SELECT uid AS user_id FROM events",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert [op.kind for op in fp.operations] == ["source", "project"]
    assert fp.output_schema == (("user_id", "int64"),)


def test_alias_records_rename_in_project_schema_delta() -> None:
    fp = trace_sql(
        "SELECT uid AS user_id FROM events",
        schemas={"events": EVENTS_SCHEMA},
    )
    project = fp.operations[1]
    assert project.kind == "project"
    assert project.columns == ("user_id",)
    assert project.schema_delta.renamed == (("uid", "user_id"),)


def test_unaliased_columns_do_not_appear_in_renamed() -> None:
    fp = trace_sql(
        "SELECT uid, score AS s FROM events",
        schemas={"events": EVENTS_SCHEMA},
    )
    project = fp.operations[1]
    assert project.schema_delta.renamed == (("score", "s"),)


def test_alias_changes_fingerprint_id() -> None:
    """A rename is a real semantic change; two pipelines that differ only by
    an alias must produce different fingerprint_ids."""

    fp_no_alias = trace_sql(
        "SELECT uid FROM events",
        schemas={"events": EVENTS_SCHEMA},
    )
    fp_with_alias = trace_sql(
        "SELECT uid AS user_id FROM events",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert fp_no_alias.fingerprint_id != fp_with_alias.fingerprint_id


# --- ORDER BY ----------------------------------------------------------------


def test_order_by_emits_sort_as_last_op() -> None:
    fp = trace_sql(
        "SELECT uid, score FROM events ORDER BY score",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert [op.kind for op in fp.operations] == ["source", "project", "sort"]


def test_order_by_default_direction_is_asc() -> None:
    fp = trace_sql(
        "SELECT uid, score FROM events ORDER BY score",
        schemas={"events": EVENTS_SCHEMA},
    )
    sort = fp.operations[-1]
    assert sort.kind == "sort"
    assert sort.by == (("score", "asc"),)


def test_order_by_desc_is_captured() -> None:
    fp = trace_sql(
        "SELECT uid, score FROM events ORDER BY score DESC",
        schemas={"events": EVENTS_SCHEMA},
    )
    sort = fp.operations[-1]
    assert isinstance(sort, Sort)
    assert sort.by == (("score", "desc"),)


def test_order_by_multiple_columns_preserves_order_and_direction() -> None:
    fp = trace_sql(
        "SELECT uid, score FROM events ORDER BY score DESC, uid ASC",
        schemas={"events": EVENTS_SCHEMA},
    )
    sort = fp.operations[-1]
    assert isinstance(sort, Sort)
    assert sort.by == (("score", "desc"), ("uid", "asc"))


def test_order_by_on_select_star() -> None:
    """ORDER BY against a SELECT * pipeline references source columns."""

    fp = trace_sql(
        "SELECT * FROM events ORDER BY ts DESC",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert [op.kind for op in fp.operations] == ["source", "sort"]
    sort = fp.operations[-1]
    assert isinstance(sort, Sort)
    assert sort.by == (("ts", "desc"),)


def test_order_by_uses_output_alias_not_source_name() -> None:
    fp = trace_sql(
        "SELECT uid AS user_id FROM events ORDER BY user_id",
        schemas={"events": EVENTS_SCHEMA},
    )
    sort = fp.operations[-1]
    assert isinstance(sort, Sort)
    assert sort.by == (("user_id", "asc"),)


def test_order_by_references_unknown_column_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_sql(
            "SELECT uid FROM events ORDER BY bogus",
            schemas={"events": EVENTS_SCHEMA},
        )


def test_order_by_expression_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bare column"):
        trace_sql(
            "SELECT uid, score FROM events ORDER BY score * 2",
            schemas={"events": EVENTS_SCHEMA},
        )


def test_order_by_direction_changes_fingerprint() -> None:
    fp_asc = trace_sql(
        "SELECT uid FROM events ORDER BY uid ASC",
        schemas={"events": EVENTS_SCHEMA},
    )
    fp_desc = trace_sql(
        "SELECT uid FROM events ORDER BY uid DESC",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert fp_asc.fingerprint_id != fp_desc.fingerprint_id


# --- combined ---------------------------------------------------------------


def test_alias_with_where_and_order_by() -> None:
    fp = trace_sql(
        "SELECT uid AS user_id, score FROM events WHERE score > 0 ORDER BY score DESC",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert [op.kind for op in fp.operations] == [
        "source",
        "filter",
        "project",
        "sort",
    ]
    assert fp.output_schema == (("user_id", "int64"), ("score", "float64"))
    sort = fp.operations[-1]
    assert isinstance(sort, Sort)
    assert sort.by == (("score", "desc"),)


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


# --- GROUP BY + aggregations -------------------------------------------------


def test_count_star_emits_aggregate_with_none_input() -> None:
    fp = trace_sql(
        "SELECT uid, COUNT(*) FROM events GROUP BY uid",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert [op.kind for op in fp.operations] == ["source", "aggregate"]
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.by == ("uid",)
    assert agg.aggregations == (("count(*)", None, "count"),)
    assert fp.output_schema == (("uid", "int64"), ("count(*)", "int64"))


def test_count_column_emits_aggregate_with_input() -> None:
    fp = trace_sql(
        "SELECT uid, COUNT(score) FROM events GROUP BY uid",
        schemas={"events": EVENTS_SCHEMA},
    )
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.aggregations == (("count(score)", "score", "count"),)


def test_count_distinct_maps_to_count_distinct_function() -> None:
    fp = trace_sql(
        "SELECT COUNT(DISTINCT uid) AS distinct_users FROM events",
        schemas={"events": EVENTS_SCHEMA},
    )
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.by == ()
    assert agg.aggregations == (("distinct_users", "uid", "count_distinct"),)
    assert fp.output_schema == (("distinct_users", "int64"),)


@pytest.mark.parametrize(
    "sql_fn,canonical",
    [
        ("SUM", "sum"),
        ("AVG", "mean"),
        ("MIN", "min"),
        ("MAX", "max"),
    ],
)
def test_canonical_aggregate_function_names(sql_fn: str, canonical: str) -> None:
    fp = trace_sql(
        f"SELECT uid, {sql_fn}(score) FROM events GROUP BY uid",
        schemas={"events": EVENTS_SCHEMA},
    )
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.aggregations[0][2] == canonical


def test_sum_preserves_input_dtype() -> None:
    fp = trace_sql(
        "SELECT uid, SUM(score) AS total FROM events GROUP BY uid",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert fp.output_schema == (("uid", "int64"), ("total", "float64"))


def test_avg_always_returns_float64() -> None:
    fp = trace_sql(
        "SELECT uid, AVG(uid) AS avg_uid FROM events GROUP BY uid",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert fp.output_schema == (("uid", "int64"), ("avg_uid", "float64"))


def test_default_output_name_uses_canonical_function() -> None:
    """No alias on AVG(score): the output column is "mean(score)", the
    canonical-function-name form, not the SQL "AVG(score)" spelling."""

    fp = trace_sql(
        "SELECT uid, AVG(score) FROM events GROUP BY uid",
        schemas={"events": EVENTS_SCHEMA},
    )
    names = [c for c, _ in fp.output_schema]
    assert names == ["uid", "mean(score)"]


def test_multi_key_group_by() -> None:
    schemas = {
        "events": (("uid", "int64"), ("country", "utf8"), ("score", "float64")),
    }
    fp = trace_sql(
        "SELECT uid, country, SUM(score) AS total FROM events GROUP BY uid, country",
        schemas=schemas,
    )
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.by == ("uid", "country")
    assert fp.output_schema == (
        ("uid", "int64"),
        ("country", "utf8"),
        ("total", "float64"),
    )


def test_select_non_grouped_column_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="GROUP BY key"):
        trace_sql(
            "SELECT uid, score FROM events GROUP BY uid",
            schemas={"events": EVENTS_SCHEMA},
        )


def test_unsupported_aggregate_function_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="supported"):
        trace_sql(
            "SELECT uid, STDDEV(score) FROM events GROUP BY uid",
            schemas={"events": EVENTS_SCHEMA},
        )


def test_aggregate_of_expression_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bare column"):
        trace_sql(
            "SELECT uid, SUM(score * 2) FROM events GROUP BY uid",
            schemas={"events": EVENTS_SCHEMA},
        )


def test_select_star_with_group_by_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="SELECT \\*"):
        trace_sql(
            "SELECT * FROM events GROUP BY uid",
            schemas={"events": EVENTS_SCHEMA},
        )


def test_aggregate_without_group_by() -> None:
    """SELECT with aggregates but no GROUP BY aggregates over all rows
    (single implicit group). Aggregate.by is the empty tuple."""

    fp = trace_sql(
        "SELECT COUNT(*) AS total FROM events",
        schemas={"events": EVENTS_SCHEMA},
    )
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.by == ()
    assert fp.output_schema == (("total", "int64"),)


# --- HAVING ------------------------------------------------------------------


def test_having_emits_filter_after_aggregate() -> None:
    fp = trace_sql(
        "SELECT uid, COUNT(*) AS n FROM events GROUP BY uid HAVING COUNT(*) > 10",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert [op.kind for op in fp.operations] == [
        "source",
        "aggregate",
        "filter",
    ]


def test_having_predicate_is_captured() -> None:
    fp = trace_sql(
        "SELECT uid, AVG(score) AS m FROM events GROUP BY uid HAVING AVG(score) > 0.5",
        schemas={"events": EVENTS_SCHEMA},
    )
    flt = fp.operations[-1]
    assert flt.kind == "filter"
    assert isinstance(flt.predicate, str)
    assert "AVG(score)" in flt.predicate or "avg" in flt.predicate.lower()


def test_full_pipeline_where_groupby_having_orderby() -> None:
    fp = trace_sql(
        "SELECT uid, AVG(score) AS avg_score FROM events "
        "WHERE score > 0 GROUP BY uid "
        "HAVING AVG(score) > 0.5 ORDER BY avg_score DESC",
        schemas={"events": EVENTS_SCHEMA},
    )
    assert [op.kind for op in fp.operations] == [
        "source",
        "filter",
        "aggregate",
        "filter",
        "sort",
    ]
    assert fp.output_schema == (("uid", "int64"), ("avg_score", "float64"))


def test_group_by_unknown_column_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_sql(
            "SELECT bogus, COUNT(*) FROM events GROUP BY bogus",
            schemas={"events": EVENTS_SCHEMA},
        )
