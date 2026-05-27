"""M2.pandas phase 1a: wrapper-object tracer for the Source / Filter /
Project surface. Includes the cross-framework diff test that demonstrates
the cross-framework equivalence claim."""

from __future__ import annotations

import pytest

from mirrorml import diff, trace_pandas, trace_polars, trace_sql
from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint.operations import FillNa, Filter, Project, Sort, Source

EVENTS = (("uid", "int64"), ("score", "float64"))


# --- Source -----------------------------------------------------------------


def test_passthrough_pipeline_emits_only_source() -> None:
    fp = trace_pandas(lambda df: df, input_schema=EVENTS)
    assert [op.kind for op in fp.operations] == ["source"]
    assert fp.output_schema == EVENTS


def test_source_name_defaults_to_input() -> None:
    fp = trace_pandas(lambda df: df, input_schema=EVENTS)
    source = fp.operations[0]
    assert isinstance(source, Source)
    assert source.name == "input"


def test_source_name_can_be_overridden() -> None:
    fp = trace_pandas(lambda df: df, input_schema=EVENTS, source_name="events")
    source = fp.operations[0]
    assert isinstance(source, Source)
    assert source.name == "events"


# --- Filter -----------------------------------------------------------------


def test_simple_greater_than_filter() -> None:
    fp = trace_pandas(
        lambda df: df[df["score"] > 0],
        input_schema=EVENTS,
    )
    assert [op.kind for op in fp.operations] == ["source", "filter"]
    flt = fp.operations[1]
    assert isinstance(flt, Filter)
    assert flt.predicate == "score > 0"


@pytest.mark.parametrize(
    "op_fn,expected",
    [
        (lambda s: s > 0, "score > 0"),
        (lambda s: s < 0, "score < 0"),
        (lambda s: s >= 0, "score >= 0"),
        (lambda s: s <= 0, "score <= 0"),
        (lambda s: s == 0, "score = 0"),
        (lambda s: s != 0, "score <> 0"),
    ],
)
def test_comparison_operators_render_in_sql_form(op_fn: object, expected: str) -> None:
    fp = trace_pandas(
        lambda df: df[op_fn(df["score"])],  # type: ignore[operator]
        input_schema=EVENTS,
    )
    flt = fp.operations[1]
    assert isinstance(flt, Filter)
    assert flt.predicate == expected


def test_string_literal_is_quoted() -> None:
    schemas = (("uid", "int64"), ("name", "utf8"))
    fp = trace_pandas(
        lambda df: df[df["name"] == "alice"],
        input_schema=schemas,
    )
    flt = fp.operations[1]
    assert isinstance(flt, Filter)
    assert flt.predicate == "name = 'alice'"


def test_string_literal_with_apostrophe_is_escaped() -> None:
    schemas = (("uid", "int64"), ("name", "utf8"))
    fp = trace_pandas(
        lambda df: df[df["name"] == "O'Brien"],
        input_schema=schemas,
    )
    flt = fp.operations[1]
    assert isinstance(flt, Filter)
    assert flt.predicate == "name = 'O''Brien'"


def test_and_predicate_composition() -> None:
    schemas = (("a", "int64"), ("b", "int64"))
    fp = trace_pandas(
        lambda df: df[(df["a"] > 0) & (df["b"] < 10)],
        input_schema=schemas,
    )
    flt = fp.operations[1]
    assert isinstance(flt, Filter)
    assert flt.predicate == "a > 0 AND b < 10"


def test_or_predicate_composition() -> None:
    schemas = (("a", "int64"), ("b", "int64"))
    fp = trace_pandas(
        lambda df: df[(df["a"] > 0) | (df["b"] < 10)],
        input_schema=schemas,
    )
    flt = fp.operations[1]
    assert isinstance(flt, Filter)
    assert flt.predicate == "a > 0 OR b < 10"


def test_not_predicate() -> None:
    fp = trace_pandas(
        lambda df: df[~(df["score"] > 0)],
        input_schema=EVENTS,
    )
    flt = fp.operations[1]
    assert isinstance(flt, Filter)
    assert flt.predicate == "NOT (score > 0)"


# --- Project ----------------------------------------------------------------


def test_column_list_projection() -> None:
    fp = trace_pandas(
        lambda df: df[["uid"]],
        input_schema=EVENTS,
    )
    assert [op.kind for op in fp.operations] == ["source", "project"]
    project = fp.operations[1]
    assert isinstance(project, Project)
    assert project.columns == ("uid",)
    assert fp.output_schema == (("uid", "int64"),)


def test_multi_column_projection_preserves_order() -> None:
    fp = trace_pandas(
        lambda df: df[["score", "uid"]],
        input_schema=EVENTS,
    )
    project = fp.operations[1]
    assert isinstance(project, Project)
    assert project.columns == ("score", "uid")
    assert fp.output_schema == (("score", "float64"), ("uid", "int64"))


def test_projection_of_missing_column_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_pandas(
            lambda df: df[["bogus"]],
            input_schema=EVENTS,
        )


def test_projection_of_non_string_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="strings"):
        trace_pandas(
            lambda df: df[[1, 2]],
            input_schema=EVENTS,
        )


# --- Filter + Project combined ----------------------------------------------


def test_filter_then_project() -> None:
    fp = trace_pandas(
        lambda df: df[df["score"] > 0][["uid", "score"]],
        input_schema=EVENTS,
    )
    assert [op.kind for op in fp.operations] == ["source", "filter", "project"]
    assert fp.output_schema == (("uid", "int64"), ("score", "float64"))


# --- failure modes ----------------------------------------------------------


def test_pipeline_returning_scalar_is_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="DataFrame-like"):
        trace_pandas(lambda df: 42, input_schema=EVENTS)


def test_pipeline_returning_series_is_rejected() -> None:
    """``df['col']`` returns a Series wrapper, which is not a final
    pipeline output."""

    with pytest.raises(UnsupportedOperationError, match="DataFrame-like"):
        trace_pandas(lambda df: df["score"], input_schema=EVENTS)


def test_unsupported_getitem_key_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="__getitem__"):
        trace_pandas(
            lambda df: df[42],
            input_schema=EVENTS,
        )


def test_unknown_column_in_filter_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_pandas(
            lambda df: df[df["bogus"] > 0],
            input_schema=EVENTS,
        )


# --- sort_values ------------------------------------------------------------


def test_sort_values_single_column_ascending() -> None:
    fp = trace_pandas(lambda df: df.sort_values("score"), input_schema=EVENTS)
    assert [op.kind for op in fp.operations] == ["source", "sort"]
    srt = fp.operations[1]
    assert isinstance(srt, Sort)
    assert srt.by == (("score", "asc"),)
    assert fp.output_schema == EVENTS


def test_sort_values_descending() -> None:
    fp = trace_pandas(lambda df: df.sort_values("score", ascending=False), input_schema=EVENTS)
    srt = fp.operations[1]
    assert isinstance(srt, Sort)
    assert srt.by == (("score", "desc"),)


def test_sort_values_multi_column_mixed_direction() -> None:
    fp = trace_pandas(
        lambda df: df.sort_values(["uid", "score"], ascending=[True, False]),
        input_schema=EVENTS,
    )
    srt = fp.operations[1]
    assert isinstance(srt, Sort)
    assert srt.by == (("uid", "asc"), ("score", "desc"))


def test_sort_values_unknown_column_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_pandas(lambda df: df.sort_values("bogus"), input_schema=EVENTS)


def test_sort_values_ascending_length_mismatch_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="length"):
        trace_pandas(
            lambda df: df.sort_values(["uid", "score"], ascending=[True]),
            input_schema=EVENTS,
        )


def test_cross_framework_sort_pandas_vs_polars_diffs_to_empty() -> None:
    pandas_fp = trace_pandas(
        lambda df: df.sort_values("score", ascending=False),
        input_schema=EVENTS,
        source_name="events",
    )
    polars_fp = trace_polars(
        lambda lf, pl: lf.sort("score", descending=True),
        input_schema=EVENTS,
        source_name="events",
    )
    assert diff(pandas_fp, polars_fp) == ()


# --- fillna -----------------------------------------------------------------


def test_fillna_scalar_fills_all_columns() -> None:
    fp = trace_pandas(lambda df: df.fillna(0), input_schema=EVENTS)
    assert [op.kind for op in fp.operations] == ["source", "fill_na"]
    op = fp.operations[1]
    assert isinstance(op, FillNa)
    assert op.columns == ("uid", "score")
    assert op.value == "0"
    assert op.strategy == "constant"
    assert fp.output_schema == EVENTS


def test_fillna_dict_fills_named_columns() -> None:
    fp = trace_pandas(lambda df: df.fillna({"score": -1}), input_schema=EVENTS)
    op = fp.operations[1]
    assert isinstance(op, FillNa)
    assert op.columns == ("score",)
    assert op.value == "-1"


def test_fillna_differing_per_column_values_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="differing per-column"):
        trace_pandas(lambda df: df.fillna({"uid": 0, "score": 1}), input_schema=EVENTS)


def test_fillna_no_value_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="scalar value"):
        trace_pandas(lambda df: df.fillna(), input_schema=EVENTS)


def test_fillna_value_difference_surfaces_null_handling() -> None:
    a = trace_pandas(lambda df: df.fillna(0), input_schema=EVENTS, source_name="e")
    b = trace_pandas(lambda df: df.fillna(-1), input_schema=EVENTS, source_name="e")
    divs = diff(a, b)
    assert [d.category for d in divs] == ["null_handling"]


def test_cross_framework_fillna_pandas_vs_polars_diffs_to_empty() -> None:
    pandas_fp = trace_pandas(lambda df: df.fillna(0), input_schema=EVENTS, source_name="events")
    polars_fp = trace_polars(
        lambda lf, pl: lf.fill_null(0), input_schema=EVENTS, source_name="events"
    )
    assert diff(pandas_fp, polars_fp) == ()


# --- THE BIG ONE: cross-framework equivalence -------------------------------


def test_equivalent_pandas_and_sql_pipelines_diff_to_empty() -> None:
    """A pandas pipeline and the structurally equivalent SQL query
    produce fingerprints that diff to ``()``. The fingerprint_ids
    themselves differ (the framework field is part of the canonical
    body), but the diff engine sees identical structure across the two
    frameworks and emits no divergences.
    """

    def offline(df: object) -> object:
        return df[df["score"] > 0][["uid", "score"]]  # type: ignore[index]

    pandas_fp = trace_pandas(
        offline,
        input_schema=EVENTS,
        source_name="events",
    )
    sql_fp = trace_sql(
        "SELECT uid, score FROM events WHERE score > 0",
        schemas={"events": EVENTS},
    )

    # Sanity: different framework, hence different fingerprint_id.
    assert pandas_fp.framework == "pandas"
    assert sql_fp.framework == "sql"
    assert pandas_fp.fingerprint_id != sql_fp.fingerprint_id

    # The headline: structure matches across frameworks.
    assert diff(pandas_fp, sql_fp) == ()


def test_cross_framework_filter_only_diffs_to_empty() -> None:
    pandas_fp = trace_pandas(
        lambda df: df[df["score"] > 0],
        input_schema=EVENTS,
        source_name="events",
    )
    sql_fp = trace_sql(
        "SELECT * FROM events WHERE score > 0",
        schemas={"events": EVENTS},
    )
    assert diff(pandas_fp, sql_fp) == ()


def test_cross_framework_predicate_difference_is_surfaced() -> None:
    """A genuine pipeline difference (different filter threshold) must
    not silently vanish. The classifier falls back to ``schema_drift``
    when the predicate change does not map to a more specific category.
    """

    pandas_fp = trace_pandas(
        lambda df: df[df["score"] > 0],
        input_schema=EVENTS,
        source_name="events",
    )
    sql_fp = trace_sql(
        "SELECT * FROM events WHERE score > 1",
        schemas={"events": EVENTS},
    )
    divs = diff(pandas_fp, sql_fp)
    assert any(d.category == "schema_drift" and "predicate" in d.detail for d in divs)


def test_cross_framework_null_predicate_routes_to_null_handling() -> None:
    """A predicate that filters by NULL on one side and a non-null
    threshold on the other routes to the ``null_handling`` category."""

    pandas_fp = trace_pandas(
        lambda df: df[df["score"] > 0],
        input_schema=EVENTS,
        source_name="events",
    )
    sql_fp = trace_sql(
        "SELECT * FROM events WHERE score IS NOT NULL",
        schemas={"events": EVENTS},
    )
    divs = diff(pandas_fp, sql_fp)
    assert any(d.category == "null_handling" for d in divs)
