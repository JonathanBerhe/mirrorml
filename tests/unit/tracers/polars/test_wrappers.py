"""M2.polars phase 1: wrapper-object tracer for the Source / Filter /
Project / Aggregate surface, plus the cross-framework diff tests that
extend the cross-framework equivalence claim to a third framework.

The tracer never imports the real ``polars`` package, so these tests do
not require polars to be installed; the ``pl`` argument is the tracer's
own expression namespace.
"""

from __future__ import annotations

from typing import Any

import pytest

from mirrorml import diff, trace_pandas, trace_polars, trace_sql
from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint.operations import (
    Aggregate,
    AsOfJoin,
    FillNa,
    Filter,
    Project,
    Sort,
    Source,
    Window,
)
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


# --- join_asof (point-in-time joins -> AsOfJoin op, as_of_join_direction) ---

ASOF_LEFT = (("uid", "int64"), ("ts", "timestamp[ns, UTC]"), ("score", "float64"))
ASOF_RIGHT = [("uid", "int64"), ("ts", "timestamp[ns, UTC]"), ("price", "float64")]


def _asof(strategy: str = "backward", tolerance: str | None = None) -> Fingerprint:
    def offline(lf: object, pl: object) -> object:
        prices = pl.source("prices", schema=ASOF_RIGHT)  # type: ignore[attr-defined]
        return lf.join_asof(  # type: ignore[attr-defined]
            prices, on="ts", by="uid", strategy=strategy, tolerance=tolerance
        )

    return trace_polars(offline, input_schema=ASOF_LEFT, source_name="events")


def test_join_asof_emits_as_of_join_op() -> None:
    fp = _asof("backward")
    assert [op.kind for op in fp.operations] == ["source", "source", "as_of_join"]
    aj = fp.operations[-1]
    assert isinstance(aj, AsOfJoin)
    assert aj.left_keys == ("uid",)
    assert aj.right_keys == ("uid",)
    assert aj.on_time == "ts"
    assert aj.temporal.direction == "backward"
    # left columns + right's non-shared column (price).
    assert fp.output_schema == (
        ("uid", "int64"),
        ("ts", "timestamp[ns, UTC]"),
        ("score", "float64"),
        ("price", "float64"),
    )


def test_identical_asof_joins_diff_to_empty() -> None:
    assert diff(_asof("backward"), _asof("backward")) == ()


def test_different_asof_direction_surfaces_divergence() -> None:
    divs = diff(_asof("backward"), _asof("forward"))
    assert [d.category for d in divs] == ["as_of_join_direction"]


def test_different_asof_tolerance_surfaces_divergence() -> None:
    divs = diff(_asof("backward", tolerance="1d"), _asof("backward", tolerance="2d"))
    assert any(d.category == "as_of_join_direction" for d in divs)


def test_join_asof_bad_strategy_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="strategy"):
        _asof("sideways")


def test_join_asof_non_frame_right_rejected() -> None:
    def offline(lf: object, pl: object) -> object:
        return lf.join_asof("not a frame", on="ts", by="uid")  # type: ignore[attr-defined]

    with pytest.raises(UnsupportedOperationError, match="must be a frame"):
        trace_polars(offline, input_schema=ASOF_LEFT)


def test_join_asof_on_column_missing_rejected() -> None:
    def offline(lf: object, pl: object) -> object:
        prices = pl.source("prices", schema=ASOF_RIGHT)  # type: ignore[attr-defined]
        return lf.join_asof(prices, on="nope", by="uid")  # type: ignore[attr-defined]

    with pytest.raises(UnsupportedOperationError, match="on="):
        trace_polars(offline, input_schema=ASOF_LEFT)


# --- fill_null --------------------------------------------------------------


def test_fill_null_scalar_fills_all_columns() -> None:
    fp = trace_polars(lambda lf, pl: lf.fill_null(0), input_schema=EVENTS)
    assert [op.kind for op in fp.operations] == ["source", "fill_na"]
    op = fp.operations[1]
    assert isinstance(op, FillNa)
    assert op.columns == ("uid", "score")
    assert op.value == "0"
    assert op.strategy == "constant"


def test_fill_null_via_lit() -> None:
    fp = trace_polars(lambda lf, pl: lf.fill_null(pl.lit(0)), input_schema=EVENTS)
    op = fp.operations[1]
    assert isinstance(op, FillNa)
    assert op.value == "0"


def test_fill_null_no_value_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="scalar fill value"):
        trace_polars(lambda lf, pl: lf.fill_null(), input_schema=EVENTS)


def test_fill_null_value_difference_surfaces_null_handling() -> None:
    a = trace_polars(lambda lf, pl: lf.fill_null(0), input_schema=EVENTS, source_name="e")
    b = trace_polars(lambda lf, pl: lf.fill_null(-1), input_schema=EVENTS, source_name="e")
    divs = diff(a, b)
    assert [d.category for d in divs] == ["null_handling"]


# --- map_batches (UDF) ------------------------------------------------------


def _identity_frame(df: Any) -> Any:
    return df


def _drop_score(df: Any) -> Any:
    return df.drop("score")


def test_map_batches_emits_udf_op_with_source_hash() -> None:
    from mirrorml.fingerprint.operations import Udf

    fp = trace_polars(
        lambda lf, pl: lf.map_batches(_identity_frame), input_schema=EVENTS, source_name="e"
    )
    udf_ops = [op for op in fp.operations if isinstance(op, Udf)]
    assert len(udf_ops) == 1
    assert udf_ops[0].ref.qualname.endswith("_identity_frame")
    assert len(udf_ops[0].ref.source_hash) == 64


def test_map_batches_same_callable_diffs_to_empty() -> None:
    a = trace_polars(
        lambda lf, pl: lf.map_batches(_identity_frame), input_schema=EVENTS, source_name="e"
    )
    b = trace_polars(
        lambda lf, pl: lf.map_batches(_identity_frame), input_schema=EVENTS, source_name="e"
    )
    assert diff(a, b) == ()


def test_map_batches_different_callables_surface_divergence() -> None:
    a = trace_polars(
        lambda lf, pl: lf.map_batches(_identity_frame), input_schema=EVENTS, source_name="e"
    )
    b = trace_polars(
        lambda lf, pl: lf.map_batches(_drop_score), input_schema=EVENTS, source_name="e"
    )
    divs = diff(a, b)
    assert divs
    assert any("udf body" in d.detail for d in divs)


def test_map_batches_non_callable_is_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="needs a callable"):
        trace_polars(
            lambda lf, pl: lf.map_batches("not a function"),
            input_schema=EVENTS,
            source_name="e",
        )


def test_cross_framework_pandas_apply_polars_map_batches_same_body_diffs_empty() -> None:
    """The killer cross-framework test: same callable body on pandas and
    polars produces the same source-hash, so the Udf ops fingerprint
    identically and diff is empty even though the two sides reach the UDF
    through different APIs."""

    pandas_fp = trace_pandas(
        lambda df: df.apply(_identity_frame), input_schema=EVENTS, source_name="e"
    )
    polars_fp = trace_polars(
        lambda lf, pl: lf.map_batches(_identity_frame), input_schema=EVENTS, source_name="e"
    )
    assert diff(pandas_fp, polars_fp) == ()


# --- sample (Sample) --------------------------------------------------------


def test_polars_sample_emits_sample_op_with_seed() -> None:
    from mirrorml.fingerprint.operations import Sample

    fp = trace_polars(lambda lf, pl: lf.sample(n=2, seed=42), input_schema=EVENTS, source_name="e")
    samples = [op for op in fp.operations if isinstance(op, Sample)]
    assert len(samples) == 1
    assert samples[0].n == 2
    assert samples[0].seed == 42


def test_polars_sample_different_seeds_surface_seed_mismatch() -> None:
    a = trace_polars(lambda lf, pl: lf.sample(n=2, seed=42), input_schema=EVENTS, source_name="e")
    b = trace_polars(lambda lf, pl: lf.sample(n=2, seed=7), input_schema=EVENTS, source_name="e")
    divs = diff(a, b)
    assert [d.category for d in divs] == ["seed_mismatch"]


def test_cross_framework_pandas_sample_polars_sample_same_seed_diffs_empty() -> None:
    """pandas df.sample(random_state=42) and polars lf.sample(seed=42) carry
    the same seed; the Sample ops fingerprint identically across
    frameworks."""

    pandas_fp = trace_pandas(
        lambda df: df.sample(n=2, random_state=42), input_schema=EVENTS, source_name="e"
    )
    polars_fp = trace_polars(
        lambda lf, pl: lf.sample(n=2, seed=42), input_schema=EVENTS, source_name="e"
    )
    assert diff(pandas_fp, polars_fp) == ()


def test_polars_sample_requires_n_or_fraction() -> None:
    with pytest.raises(UnsupportedOperationError, match="either n or fraction"):
        trace_polars(lambda lf, pl: lf.sample(), input_schema=EVENTS, source_name="e")


# --- to_dummies (Encode) ----------------------------------------------------


def test_to_dummies_emits_encode_op() -> None:
    from mirrorml.fingerprint.operations import Encode

    fp = trace_polars(
        lambda lf, pl: lf.to_dummies(columns=["country"]),
        input_schema=(("uid", "int64"), ("country", "utf8")),
    )
    encodes = [op for op in fp.operations if isinstance(op, Encode)]
    assert len(encodes) == 1
    assert encodes[0].columns == ("country",)
    assert encodes[0].method == "one_hot"
    assert encodes[0].categories is None  # runtime fit


def test_to_dummies_same_columns_diffs_to_empty() -> None:
    schema = (("uid", "int64"), ("country", "utf8"))
    a = trace_polars(lambda lf, pl: lf.to_dummies(columns=["country"]), input_schema=schema)
    b = trace_polars(lambda lf, pl: lf.to_dummies(columns=["country"]), input_schema=schema)
    assert diff(a, b) == ()


def test_to_dummies_different_columns_surfaces_categorical_encoding() -> None:
    schema = (("uid", "int64"), ("country", "utf8"), ("city", "utf8"))
    a = trace_polars(lambda lf, pl: lf.to_dummies(columns=["country"]), input_schema=schema)
    b = trace_polars(lambda lf, pl: lf.to_dummies(columns=["city"]), input_schema=schema)
    divs = diff(a, b)
    assert [d.category for d in divs] == ["categorical_encoding"]


def test_to_dummies_unknown_column_rejected() -> None:
    schema = (("uid", "int64"),)
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_polars(lambda lf, pl: lf.to_dummies(columns=["bogus"]), input_schema=schema)


def test_to_dummies_defaults_to_every_column() -> None:
    from mirrorml.fingerprint.operations import Encode

    schema = (("uid", "int64"), ("country", "utf8"))
    fp = trace_polars(lambda lf, pl: lf.to_dummies(), input_schema=schema)
    encode = next(op for op in fp.operations if isinstance(op, Encode))
    assert encode.columns == ("uid", "country")


# --- cross-framework equivalence (third framework) --------------------------


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
