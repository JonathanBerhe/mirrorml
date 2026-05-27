"""Property-based coverage for the cross-framework equivalence claim (C4).

The hand-written tests demonstrate that *specific* equivalent pandas /
Polars / SQL pipelines diff to ``()``. These generate a space of feature
pipelines, render each in all three frameworks, and assert every pairwise
diff is empty. This turns C4 from "holds for the examples we wrote" into
"holds across a generated space of filter + group-by-aggregate pipelines."
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from mirrorml import diff, trace_pandas, trace_polars, trace_sql

EVENTS = (("uid", "int64"), ("country", "utf8"), ("score", "float64"))

# canonical reduction name -> (pandas/Polars method name, SQL function)
_AGG = {
    "sum": ("sum", "SUM"),
    "mean": ("mean", "AVG"),
    "min": ("min", "MIN"),
    "max": ("max", "MAX"),
}


@st.composite
def feature_specs(draw: st.DrawFn) -> tuple[int | None, str, tuple[str, ...]]:
    """Draw (filter_threshold_or_None, agg_name, group_keys)."""

    threshold = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=5)))
    agg = draw(st.sampled_from(sorted(_AGG)))
    group_keys = draw(st.sampled_from([("uid",), ("uid", "country")]))
    return threshold, agg, group_keys


@given(spec=feature_specs())
@settings(max_examples=80, deadline=None)
def test_filter_groupby_agg_is_cross_framework_equivalent(
    spec: tuple[int | None, str, tuple[str, ...]],
) -> None:
    threshold, agg, group_keys = spec
    pandas_method, sql_fn = _AGG[agg]
    keys = list(group_keys)

    def offline_pandas(df: Any) -> Any:
        frame = df if threshold is None else df[df["score"] > threshold]
        return frame.groupby(keys).agg({"score": pandas_method})

    def offline_polars(lf: Any, pl: Any) -> Any:
        frame = lf if threshold is None else lf.filter(pl.col("score") > threshold)
        return frame.group_by(*keys).agg(getattr(pl.col("score"), pandas_method)())

    where = "" if threshold is None else f" WHERE score > {threshold}"
    group_cols = ", ".join(keys)
    sql = f"SELECT {group_cols}, {sql_fn}(score) AS score FROM events{where} GROUP BY {group_cols}"

    pandas_fp = trace_pandas(offline_pandas, input_schema=EVENTS, source_name="events")
    polars_fp = trace_polars(offline_polars, input_schema=EVENTS, source_name="events")
    sql_fp = trace_sql(sql, schemas={"events": EVENTS})

    assert diff(pandas_fp, sql_fp) == ()
    assert diff(polars_fp, sql_fp) == ()
    assert diff(pandas_fp, polars_fp) == ()


@given(agg=st.sampled_from(sorted(_AGG)))
@settings(max_examples=20, deadline=None)
def test_agg_function_difference_is_detected_in_all_frameworks(agg: str) -> None:
    """Sanity counterpart: when the SQL side uses a *different* aggregation
    than pandas/Polars, every cross-framework diff surfaces it (so the
    equivalence above is not vacuous)."""

    pandas_method, _ = _AGG[agg]
    other = "sum" if agg != "sum" else "max"
    _, other_sql_fn = _AGG[other]

    pandas_fp = trace_pandas(
        lambda df: df.groupby("uid").agg({"score": pandas_method}),
        input_schema=EVENTS,
        source_name="events",
    )
    sql_fp = trace_sql(
        f"SELECT uid, {other_sql_fn}(score) AS score FROM events GROUP BY uid",
        schemas={"events": EVENTS},
    )
    assert any(d.category == "aggregation_function" for d in diff(pandas_fp, sql_fp))
