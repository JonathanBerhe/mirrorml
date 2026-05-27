"""M2.polars phase 1: wrapper-object tracer for the Source / Filter /
Project / Aggregate surface, plus the cross-framework diff tests that
extend the PAPER.md C4 equivalence claim to a third framework.

The tracer never imports the real ``polars`` package, so these tests do
not require polars to be installed; the ``pl`` argument is the tracer's
own expression namespace.
"""

from __future__ import annotations

from typing import Any

import pytest

from mirrorml import diff, trace_pandas, trace_polars, trace_sql
from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint.operations import Aggregate, Filter, Project, Sort, Source, Window
from mirrorml.fingerprint.schema import Fingerprint

EVENTS = (("uid", "int64"), ("score", "float64"))
EVENTS3 = (("uid", "int64"), ("country", "utf8"), ("score", "float64"))


# --- Source -----------------------------------------------------------------


def test_passthrough_pipeline_emits_only_source() -> None:
    fp = trace_polars(lambda lf, pl: lf, input_schema=EVENTS)
    assert [op.kind for op in fp.operations] == ["source"]
    assert fp.output_schema == EVENTS


def test_source_name_can_be_overridden() -> None:
    fp = trace_polars(lambda lf, pl: lf, input_schema=EVENTS, source_name="events")
    source = fp.operations[0]
    assert isinstance(source, Source)
    assert source.name == "events"


# --- Filter -----------------------------------------------------------------


def test_simple_greater_than_filter() -> None:
    fp = trace_polars(
        lambda lf, pl: lf.filter(pl.col("score") > 0),
        input_schema=EVENTS,
    )
    assert [op.kind for op in fp.operations] == ["source", "filter"]
    flt = fp.operations[1]
    assert isinstance(flt, Filter)
    assert flt.predicate == "score > 0"


@pytest.mark.parametrize(
    "op_fn,expected",
    [
        (lambda c: c > 0, "score > 0"),
        (lambda c: c < 0, "score < 0"),
        (lambda c: c >= 0, "score >= 0"),
        (lambda c: c <= 0, "score <= 0"),
        (lambda c: c == 0, "score = 0"),
        (lambda c: c != 0, "score <> 0"),
    ],
)
def test_comparison_operators_render_in_sql_form(op_fn: Any, expected: str) -> None:
    fp = trace_polars(
        lambda lf, pl: lf.filter(op_fn(pl.col("score"))),
        input_schema=EVENTS,
    )
    flt = fp.operations[1]
    assert isinstance(flt, Filter)
    assert flt.predicate == expected


def test_string_literal_is_quoted() -> None:
    schemas = (("uid", "int64"), ("name", "utf8"))
    fp = trace_polars(
        lambda lf, pl: lf.filter(pl.col("name") == "alice"),
        input_schema=schemas,
    )
    flt = fp.operations[1]
    assert isinstance(flt, Filter)
    assert flt.predicate == "name = 'alice'"


def test_lit_wrapper_unwraps_to_same_predicate() -> None:
    fp = trace_polars(
        lambda lf, pl: lf.filter(pl.col("score") > pl.lit(0)),
        input_schema=EVENTS,
    )
    flt = fp.operations[1]
    assert isinstance(flt, Filter)
    assert flt.predicate == "score > 0"


def test_multiple_filter_predicates_compose_with_and() -> None:
    schemas = (("a", "int64"), ("b", "int64"))
    fp = trace_polars(
        lambda lf, pl: lf.filter(pl.col("a") > 0, pl.col("b") < 10),
        input_schema=schemas,
    )
    flt = fp.operations[1]
    assert isinstance(flt, Filter)
    assert flt.predicate == "a > 0 AND b < 10"


def test_explicit_and_predicate_composition() -> None:
    schemas = (("a", "int64"), ("b", "int64"))
    fp = trace_polars(
        lambda lf, pl: lf.filter((pl.col("a") > 0) & (pl.col("b") < 10)),
        input_schema=schemas,
    )
    flt = fp.operations[1]
    assert isinstance(flt, Filter)
    assert flt.predicate == "a > 0 AND b < 10"


# --- Project (select / rename) ----------------------------------------------


def test_select_by_name_projection() -> None:
    fp = trace_polars(
        lambda lf, pl: lf.select("uid"),
        input_schema=EVENTS,
    )
    assert [op.kind for op in fp.operations] == ["source", "project"]
    project = fp.operations[1]
    assert isinstance(project, Project)
    assert project.columns == ("uid",)
    assert fp.output_schema == (("uid", "int64"),)


def test_select_with_pl_col_preserves_order() -> None:
    fp = trace_polars(
        lambda lf, pl: lf.select(pl.col("score"), pl.col("uid")),
        input_schema=EVENTS,
    )
    project = fp.operations[1]
    assert isinstance(project, Project)
    assert project.columns == ("score", "uid")
    assert fp.output_schema == (("score", "float64"), ("uid", "int64"))


def test_select_list_form() -> None:
    fp = trace_polars(
        lambda lf, pl: lf.select(["uid", "score"]),
        input_schema=EVENTS,
    )
    project = fp.operations[1]
    assert isinstance(project, Project)
    assert project.columns == ("uid", "score")


def test_select_alias_records_rename() -> None:
    fp = trace_polars(
        lambda lf, pl: lf.select(pl.col("uid").alias("user_id"), pl.col("score")),
        input_schema=EVENTS,
    )
    project = fp.operations[1]
    assert isinstance(project, Project)
    assert project.columns == ("user_id", "score")
    assert project.schema_delta.renamed == (("uid", "user_id"),)
    assert fp.output_schema == (("user_id", "int64"), ("score", "float64"))


def test_rename_emits_project_with_renamed_schema_delta() -> None:
    fp = trace_polars(
        lambda lf, pl: lf.rename({"uid": "user_id"}),
        input_schema=EVENTS3,
    )
    assert [op.kind for op in fp.operations] == ["source", "project"]
    project = fp.operations[1]
    assert isinstance(project, Project)
    assert project.columns == ("user_id", "country", "score")
    assert project.schema_delta.renamed == (("uid", "user_id"),)


def test_select_unknown_column_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_polars(lambda lf, pl: lf.select("bogus"), input_schema=EVENTS)


def test_rename_unknown_source_column_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_polars(lambda lf, pl: lf.rename({"bogus": "x"}), input_schema=EVENTS)


# --- Aggregate --------------------------------------------------------------


def test_group_by_agg_emits_aggregate() -> None:
    fp = trace_polars(
        lambda lf, pl: lf.group_by("uid").agg(pl.col("score").sum()),
        input_schema=EVENTS,
    )
    assert [op.kind for op in fp.operations] == ["source", "aggregate"]
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.by == ("uid",)
    assert agg.aggregations == (("score", "score", "sum"),)
    assert fp.output_schema == (("uid", "int64"), ("score", "float64"))


@pytest.mark.parametrize(
    "method,canonical,output_dtype",
    [
        ("sum", "sum", "float64"),
        ("mean", "mean", "float64"),
        ("min", "min", "float64"),
        ("max", "max", "float64"),
        ("count", "count", "int64"),
        ("n_unique", "count_distinct", "int64"),
    ],
)
def test_aggregation_methods(method: str, canonical: str, output_dtype: str) -> None:
    fp = trace_polars(
        lambda lf, pl: lf.group_by("uid").agg(getattr(pl.col("score"), method)()),
        input_schema=EVENTS,
    )
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.aggregations == (("score", "score", canonical),)
    assert fp.output_schema == (("uid", "int64"), ("score", output_dtype))


def test_agg_alias_sets_output_name() -> None:
    fp = trace_polars(
        lambda lf, pl: lf.group_by("uid").agg(pl.col("score").mean().alias("avg_score")),
        input_schema=EVENTS,
    )
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.aggregations == (("avg_score", "score", "mean"),)
    assert fp.output_schema == (("uid", "int64"), ("avg_score", "float64"))


def test_multi_key_group_by() -> None:
    fp = trace_polars(
        lambda lf, pl: lf.group_by("uid", "country").agg(pl.col("score").mean()),
        input_schema=EVENTS3,
    )
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.by == ("uid", "country")


def test_multiple_aggregations() -> None:
    schemas = (("uid", "int64"), ("a", "float64"), ("b", "float64"))
    fp = trace_polars(
        lambda lf, pl: lf.group_by("uid").agg(
            pl.col("a").sum(),
            pl.col("b").mean(),
        ),
        input_schema=schemas,
    )
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.aggregations == (("a", "a", "sum"), ("b", "b", "mean"))


def test_agg_unknown_method_rejected() -> None:
    with pytest.raises(AttributeError):
        trace_polars(
            lambda lf, pl: lf.group_by("uid").agg(pl.col("score").stddev_pop()),
            input_schema=EVENTS,
        )


def test_group_by_unknown_key_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_polars(
            lambda lf, pl: lf.group_by("bogus").agg(pl.col("score").sum()),
            input_schema=EVENTS,
        )


def test_agg_target_is_key_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="group_by key"):
        trace_polars(
            lambda lf, pl: lf.group_by("uid").agg(pl.col("uid").count()),
            input_schema=EVENTS,
        )


# --- failure modes ----------------------------------------------------------


def test_pipeline_returning_scalar_is_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="LazyFrame-like"):
        trace_polars(lambda lf, pl: 42, input_schema=EVENTS)


def test_non_renderable_predicate_literal_rejected() -> None:
    """A comparison RHS that is not a renderable literal is rejected.

    (Column existence inside a filter predicate is not validated at trace
    time: like real Polars, ``pl.col(...)`` is frame-independent and
    resolved later. ``select`` / ``group_by`` / ``agg`` do validate, since
    those reference the frame directly.)
    """

    with pytest.raises(UnsupportedOperationError, match="cannot render"):
        trace_polars(
            lambda lf, pl: lf.filter(pl.col("score") > object()),
            input_schema=EVENTS,
        )


def test_pl_col_multi_column_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="multi-column"):
        trace_polars(
            lambda lf, pl: lf.select(pl.col("uid", "score")),
            input_schema=EVENTS,
        )


# --- sort -------------------------------------------------------------------


def test_sort_single_column_ascending() -> None:
    fp = trace_polars(lambda lf, pl: lf.sort("score"), input_schema=EVENTS)
    assert [op.kind for op in fp.operations] == ["source", "sort"]
    srt = fp.operations[1]
    assert isinstance(srt, Sort)
    assert srt.by == (("score", "asc"),)


def test_sort_descending() -> None:
    fp = trace_polars(lambda lf, pl: lf.sort("score", descending=True), input_schema=EVENTS)
    srt = fp.operations[1]
    assert isinstance(srt, Sort)
    assert srt.by == (("score", "desc"),)


def test_sort_multiple_columns_varargs() -> None:
    fp = trace_polars(lambda lf, pl: lf.sort("uid", "score"), input_schema=EVENTS3)
    srt = fp.operations[1]
    assert isinstance(srt, Sort)
    assert srt.by == (("uid", "asc"), ("score", "asc"))


def test_sort_multi_column_mixed_direction() -> None:
    fp = trace_polars(
        lambda lf, pl: lf.sort(["uid", "score"], descending=[False, True]),
        input_schema=EVENTS,
    )
    srt = fp.operations[1]
    assert isinstance(srt, Sort)
    assert srt.by == (("uid", "asc"), ("score", "desc"))


def test_sort_with_pl_col() -> None:
    fp = trace_polars(lambda lf, pl: lf.sort(pl.col("score"), descending=True), input_schema=EVENTS)
    srt = fp.operations[1]
    assert isinstance(srt, Sort)
    assert srt.by == (("score", "desc"),)


def test_sort_unknown_column_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_polars(lambda lf, pl: lf.sort("bogus"), input_schema=EVENTS)


# --- rolling (time windows -> Window op, window_boundary) -------------------

TS_EVENTS = (("uid", "int64"), ("ts", "timestamp[ns, UTC]"), ("score", "float64"))


def _rolling(closed: str = "right", period: str = "3d") -> Fingerprint:
    return trace_polars(
        lambda lf, pl: lf.rolling(
            index_column="ts", period=period, closed=closed, group_by="uid"
        ).agg(pl.col("score").mean()),
        input_schema=TS_EVENTS,
        source_name="events",
    )


def test_rolling_emits_window_op() -> None:
    fp = trace_polars(
        lambda lf, pl: lf.rolling(index_column="ts", period="3d", group_by="uid").agg(
            pl.col("score").mean()
        ),
        input_schema=TS_EVENTS,
    )
    assert [op.kind for op in fp.operations] == ["source", "window"]
    win = fp.operations[1]
    assert isinstance(win, Window)
    assert win.over == ("uid",)
    assert win.order_by == ("ts",)
    assert win.size == "3d"
    assert win.temporal.closed == "right"
    assert win.aggregations == (("score", "score", "mean"),)
    assert fp.output_schema == TS_EVENTS


def test_rolling_closed_none_maps_to_neither() -> None:
    fp = trace_polars(
        lambda lf, pl: lf.rolling(
            index_column="ts", period="3d", closed="none", group_by="uid"
        ).agg(pl.col("score").mean()),
        input_schema=TS_EVENTS,
    )
    win = fp.operations[1]
    assert isinstance(win, Window)
    assert win.temporal.closed == "neither"


def test_identical_rolling_windows_diff_to_empty() -> None:
    assert diff(_rolling("left"), _rolling("left")) == ()


def test_different_rolling_boundary_surfaces_window_boundary() -> None:
    divs = diff(_rolling("left"), _rolling("right"))
    assert [d.category for d in divs] == ["window_boundary"]


def test_different_rolling_period_surfaces_window_size_mismatch() -> None:
    divs = diff(_rolling(period="3d"), _rolling(period="7d"))
    assert [d.category for d in divs] == ["window_size_mismatch"]


def test_rolling_offset_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="offset"):
        trace_polars(
            lambda lf, pl: lf.rolling(
                index_column="ts", period="3d", offset="1d", group_by="uid"
            ).agg(pl.col("score").mean()),
            input_schema=TS_EVENTS,
        )


def test_rolling_bad_closed_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="closed"):
        trace_polars(
            lambda lf, pl: lf.rolling(
                index_column="ts", period="3d", closed="sideways", group_by="uid"
            ).agg(pl.col("score").mean()),
            input_schema=TS_EVENTS,
        )


def test_rolling_unknown_index_column_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_polars(
            lambda lf, pl: lf.rolling(index_column="bogus", period="3d").agg(
                pl.col("score").mean()
            ),
            input_schema=TS_EVENTS,
        )


# --- cross-framework equivalence (PAPER.md C4, third framework) -------------


def test_polars_and_sql_identity_diff_to_empty() -> None:
    polars_fp = trace_polars(
        lambda lf, pl: lf.filter(pl.col("score") > 0).select("uid", "score"),
        input_schema=EVENTS,
        source_name="events",
    )
    sql_fp = trace_sql(
        "SELECT uid, score FROM events WHERE score > 0",
        schemas={"events": EVENTS},
    )
    assert polars_fp.framework == "polars"
    assert polars_fp.fingerprint_id != sql_fp.fingerprint_id
    assert diff(polars_fp, sql_fp) == ()


def test_polars_and_sql_groupby_diff_to_empty() -> None:
    polars_fp = trace_polars(
        lambda lf, pl: lf.group_by("uid").agg(pl.col("score").mean()),
        input_schema=EVENTS,
        source_name="events",
    )
    sql_fp = trace_sql(
        "SELECT uid, AVG(score) AS score FROM events GROUP BY uid",
        schemas={"events": EVENTS},
    )
    assert diff(polars_fp, sql_fp) == ()


def test_polars_and_pandas_realistic_pipeline_diff_to_empty() -> None:
    """The three-framework headline: a filter+group+aggregate pipeline
    written in Polars and pandas produces identical structure."""

    def offline_pl(lf: Any, pl: Any) -> Any:
        return lf.filter(pl.col("score") > 0).group_by("uid").agg(pl.col("score").mean())

    def offline_pd(df: Any) -> Any:
        return df[df["score"] > 0].groupby("uid").agg({"score": "mean"})

    polars_fp = trace_polars(offline_pl, input_schema=EVENTS, source_name="events")
    pandas_fp = trace_pandas(offline_pd, input_schema=EVENTS, source_name="events")
    assert diff(polars_fp, pandas_fp) == ()


def test_polars_rename_matches_sql_alias() -> None:
    polars_fp = trace_polars(
        lambda lf, pl: lf.rename({"uid": "user_id"}),
        input_schema=EVENTS3,
        source_name="events",
    )
    sql_fp = trace_sql(
        "SELECT uid AS user_id, country, score FROM events",
        schemas={"events": EVENTS3},
    )
    assert diff(polars_fp, sql_fp) == ()


def test_polars_sum_vs_sql_avg_surfaces_divergence() -> None:
    """A genuine difference must still surface across frameworks: Polars
    sums while SQL averages -> aggregation_function divergence."""

    polars_fp = trace_polars(
        lambda lf, pl: lf.group_by("uid").agg(pl.col("score").sum()),
        input_schema=EVENTS,
        source_name="events",
    )
    sql_fp = trace_sql(
        "SELECT uid, AVG(score) AS score FROM events GROUP BY uid",
        schemas={"events": EVENTS},
    )
    divs = diff(polars_fp, sql_fp)
    assert any(d.category == "aggregation_function" for d in divs)
