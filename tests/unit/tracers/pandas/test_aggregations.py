"""pandas tracer phase 1b: groupby aggregations + rename, with cross-framework
parity against the SQL tracer.

Each section pairs a positive case (the pandas trace produces the right
Aggregate / Project op) with a cross-framework equivalence case (the
pandas fingerprint and the equivalent SQL fingerprint diff to ``()``).
"""

from __future__ import annotations

from typing import Any

import pytest

from mirrorml import diff, trace_pandas, trace_sql
from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint.operations import Aggregate, Project

EVENTS = (
    ("uid", "int64"),
    ("country", "utf8"),
    ("score", "float64"),
)


# --- groupby.agg(dict) ------------------------------------------------------


def test_groupby_agg_dict_emits_aggregate() -> None:
    fp = trace_pandas(
        lambda df: df.groupby("uid").agg({"score": "sum"}),
        input_schema=EVENTS,
    )
    assert [op.kind for op in fp.operations] == ["source", "aggregate"]
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.by == ("uid",)
    assert agg.aggregations == (("score", "score", "sum"),)
    assert fp.output_schema == (("uid", "int64"), ("score", "float64"))


def test_multi_key_groupby_agg() -> None:
    fp = trace_pandas(
        lambda df: df.groupby(["uid", "country"]).agg({"score": "mean"}),
        input_schema=EVENTS,
    )
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.by == ("uid", "country")
    assert fp.output_schema == (
        ("uid", "int64"),
        ("country", "utf8"),
        ("score", "float64"),
    )


def test_agg_dict_with_multiple_columns() -> None:
    schemas = (
        ("uid", "int64"),
        ("a", "float64"),
        ("b", "float64"),
    )
    fp = trace_pandas(
        lambda df: df.groupby("uid").agg({"a": "sum", "b": "mean"}),
        input_schema=schemas,
    )
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.aggregations == (
        ("a", "a", "sum"),
        ("b", "b", "mean"),
    )


# --- groupby column shortcuts -----------------------------------------------


@pytest.mark.parametrize(
    "method,canonical,output_dtype",
    [
        ("sum", "sum", "float64"),
        ("mean", "mean", "float64"),
        ("min", "min", "float64"),
        ("max", "max", "float64"),
        ("count", "count", "int64"),
        ("nunique", "count_distinct", "int64"),
    ],
)
def test_single_column_shortcut(method: str, canonical: str, output_dtype: str) -> None:
    fn = lambda df: getattr(df.groupby("uid")["score"], method)()  # noqa: E731
    fp = trace_pandas(fn, input_schema=EVENTS)
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.aggregations == (("score", "score", canonical),)
    assert fp.output_schema == (("uid", "int64"), ("score", output_dtype))


def test_multi_column_selection_then_sum() -> None:
    schemas = (
        ("uid", "int64"),
        ("a", "float64"),
        ("b", "float64"),
    )
    fp = trace_pandas(
        lambda df: df.groupby("uid")[["a", "b"]].sum(),
        input_schema=schemas,
    )
    agg = fp.operations[1]
    assert isinstance(agg, Aggregate)
    assert agg.aggregations == (
        ("a", "a", "sum"),
        ("b", "b", "sum"),
    )


# --- rejections -------------------------------------------------------------


def test_groupby_unknown_key_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_pandas(
            lambda df: df.groupby("bogus").agg({"score": "sum"}),
            input_schema=EVENTS,
        )


def test_groupby_invalid_key_type_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="string"):
        trace_pandas(
            lambda df: df.groupby(42).agg({"score": "sum"}),
            input_schema=EVENTS,
        )


def test_agg_unknown_function_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="canonical"):
        trace_pandas(
            lambda df: df.groupby("uid").agg({"score": "stddev_pop"}),
            input_schema=EVENTS,
        )


def test_agg_callable_function_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="UDF"):
        trace_pandas(
            lambda df: df.groupby("uid")["score"].agg(lambda s: s.sum()),
            input_schema=EVENTS,
        )


def test_groupby_string_shortcut_without_selection_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="selection"):
        trace_pandas(
            lambda df: df.groupby("uid").agg("sum"),
            input_schema=EVENTS,
        )


def test_groupby_selecting_key_column_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="groupby key"):
        trace_pandas(
            lambda df: df.groupby("uid")["uid"].count(),
            input_schema=EVENTS,
        )


def test_agg_unknown_column_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_pandas(
            lambda df: df.groupby("uid").agg({"bogus": "sum"}),
            input_schema=EVENTS,
        )


# --- rename -----------------------------------------------------------------


def test_rename_emits_project_with_renamed_schema_delta() -> None:
    fp = trace_pandas(
        lambda df: df.rename(columns={"uid": "user_id"}),
        input_schema=EVENTS,
    )
    assert [op.kind for op in fp.operations] == ["source", "project"]
    project = fp.operations[1]
    assert isinstance(project, Project)
    assert project.columns == ("user_id", "country", "score")
    assert project.schema_delta.renamed == (("uid", "user_id"),)
    assert fp.output_schema == (
        ("user_id", "int64"),
        ("country", "utf8"),
        ("score", "float64"),
    )


def test_rename_with_multiple_mappings() -> None:
    fp = trace_pandas(
        lambda df: df.rename(columns={"uid": "u", "score": "s"}),
        input_schema=EVENTS,
    )
    project = fp.operations[1]
    assert isinstance(project, Project)
    assert project.columns == ("u", "country", "s")
    assert set(project.schema_delta.renamed) == {
        ("uid", "u"),
        ("score", "s"),
    }


def test_rename_unknown_source_column_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="bogus"):
        trace_pandas(
            lambda df: df.rename(columns={"bogus": "x"}),
            input_schema=EVENTS,
        )


def test_rename_without_columns_kwarg_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="columns"):
        trace_pandas(
            lambda df: df.rename(),
            input_schema=EVENTS,
        )


def test_rename_with_non_string_value_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="string"):
        trace_pandas(
            lambda df: df.rename(columns={"uid": 42}),
            input_schema=EVENTS,
        )


# --- cross-framework parity (the headline) ----------------------------------


def test_cross_framework_groupby_sum_diffs_to_empty() -> None:
    """pandas ``df.groupby('uid').agg({'score': 'sum'})`` and SQL
    ``SELECT uid, SUM(score) AS score FROM events GROUP BY uid`` produce
    fingerprints that diff to ``()``."""

    pandas_fp = trace_pandas(
        lambda df: df.groupby("uid").agg({"score": "sum"}),
        input_schema=EVENTS,
        source_name="events",
    )
    sql_fp = trace_sql(
        "SELECT uid, SUM(score) AS score FROM events GROUP BY uid",
        schemas={"events": EVENTS},
    )
    assert diff(pandas_fp, sql_fp) == ()


def test_cross_framework_groupby_single_column_shortcut_diffs_to_empty() -> None:
    pandas_fp = trace_pandas(
        lambda df: df.groupby("uid")["score"].sum(),
        input_schema=EVENTS,
        source_name="events",
    )
    sql_fp = trace_sql(
        "SELECT uid, SUM(score) AS score FROM events GROUP BY uid",
        schemas={"events": EVENTS},
    )
    assert diff(pandas_fp, sql_fp) == ()


def test_cross_framework_multi_key_groupby_diffs_to_empty() -> None:
    pandas_fp = trace_pandas(
        lambda df: df.groupby(["uid", "country"]).agg({"score": "mean"}),
        input_schema=EVENTS,
        source_name="events",
    )
    sql_fp = trace_sql(
        "SELECT uid, country, AVG(score) AS score FROM events GROUP BY uid, country",
        schemas={"events": EVENTS},
    )
    assert diff(pandas_fp, sql_fp) == ()


def test_cross_framework_rename_diffs_to_empty() -> None:
    """pandas ``df.rename(columns={'uid': 'user_id'})`` over the full
    schema matches SQL ``SELECT uid AS user_id, country, score FROM
    events``: same Project op, same schema_delta.renamed, same output."""

    pandas_fp = trace_pandas(
        lambda df: df.rename(columns={"uid": "user_id"}),
        input_schema=EVENTS,
        source_name="events",
    )
    sql_fp = trace_sql(
        "SELECT uid AS user_id, country, score FROM events",
        schemas={"events": EVENTS},
    )
    assert diff(pandas_fp, sql_fp) == ()


def test_cross_framework_realistic_feature_pipeline_diffs_to_empty() -> None:
    """A realistic feature-engineering pipeline: filter, then group + aggregate.
    This is the kind of pipeline the project claims to detect skew on."""

    def offline(df: Any) -> Any:
        return df[df["score"] > 0].groupby("uid").agg({"score": "mean"})

    pandas_fp = trace_pandas(
        offline,
        input_schema=EVENTS,
        source_name="events",
    )
    sql_fp = trace_sql(
        "SELECT uid, AVG(score) AS score FROM events WHERE score > 0 GROUP BY uid",
        schemas={"events": EVENTS},
    )
    assert diff(pandas_fp, sql_fp) == ()


def test_cross_framework_pipeline_with_different_agg_function_surfaces_divergence() -> None:
    """The cross-framework path must still surface genuine differences:
    pandas sums while SQL averages -> aggregation_function divergence."""

    pandas_fp = trace_pandas(
        lambda df: df.groupby("uid").agg({"score": "sum"}),
        input_schema=EVENTS,
        source_name="events",
    )
    sql_fp = trace_sql(
        "SELECT uid, AVG(score) AS score FROM events GROUP BY uid",
        schemas={"events": EVENTS},
    )
    divs = diff(pandas_fp, sql_fp)
    assert any(d.category == "aggregation_function" for d in divs)
