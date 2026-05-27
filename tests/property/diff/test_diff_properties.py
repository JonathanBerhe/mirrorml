"""Property-based tests for the diff engine's structural invariants.

Pipelines are drawn from a small menu of real SQL queries (so every
fingerprint is tracer-produced and valid) that spans the operations added
across the tracer work: filter/project, group-by aggregation, window
functions, and sort. The invariants asserted hold for *any* pair of
fingerprints:

- **Reflexivity**: a pipeline never diverges from itself.
- **Symmetric equivalence**: ``diff(a, b)`` is empty iff ``diff(b, a)`` is.
- **Category symmetry**: the set of divergence categories does not depend
  on argument order (only the per-op ``left``/``right`` labels do).
- **Determinism**: tracing the same query twice yields the same fingerprint.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from mirrorml import diff, trace_sql
from mirrorml.fingerprint.schema import Fingerprint

SCHEMAS = {
    "events": (
        ("uid", "int64"),
        ("ts", "timestamp[ns, UTC]"),
        ("score", "float64"),
    )
}


@st.composite
def sql_queries(draw: st.DrawFn) -> str:
    """Draw a valid SQL query over the ``events`` schema."""

    kind = draw(st.sampled_from(["agg", "filter_project", "window", "sort"]))
    if kind == "agg":
        fn = draw(st.sampled_from(["SUM", "AVG", "MIN", "MAX"]))
        return f"SELECT uid, {fn}(score) AS score FROM events GROUP BY uid"
    if kind == "filter_project":
        threshold = draw(st.integers(min_value=0, max_value=5))
        cols = draw(st.sampled_from(["uid", "uid, score", "uid, ts, score"]))
        return f"SELECT {cols} FROM events WHERE score > {threshold}"
    if kind == "window":
        fn = draw(st.sampled_from(["AVG", "SUM", "MIN", "MAX"]))
        lookback = draw(st.integers(min_value=1, max_value=5))
        return (
            f"SELECT uid, ts, {fn}(score) OVER ("
            f"PARTITION BY uid ORDER BY ts ROWS BETWEEN {lookback} PRECEDING AND CURRENT ROW"
            f") AS roll FROM events"
        )
    column = draw(st.sampled_from(["uid", "score", "ts"]))
    direction = draw(st.sampled_from(["ASC", "DESC"]))
    return f"SELECT * FROM events ORDER BY {column} {direction}"


def _trace(query: str) -> Fingerprint:
    return trace_sql(query, schemas=SCHEMAS)


@given(query=sql_queries())
@settings(max_examples=60, deadline=None)
def test_diff_is_reflexive(query: str) -> None:
    fp = _trace(query)
    assert diff(fp, fp) == ()


@given(query=sql_queries())
@settings(max_examples=40, deadline=None)
def test_tracing_is_deterministic(query: str) -> None:
    assert _trace(query).fingerprint_id == _trace(query).fingerprint_id


@given(left=sql_queries(), right=sql_queries())
@settings(max_examples=100, deadline=None)
def test_equivalence_is_symmetric(left: str, right: str) -> None:
    a = _trace(left)
    b = _trace(right)
    assert bool(diff(a, b)) == bool(diff(b, a))


@given(left=sql_queries(), right=sql_queries())
@settings(max_examples=100, deadline=None)
def test_diff_category_set_is_symmetric(left: str, right: str) -> None:
    a = _trace(left)
    b = _trace(right)
    forward = {d.category for d in diff(a, b)}
    backward = {d.category for d in diff(b, a)}
    assert forward == backward


@given(left=sql_queries(), right=sql_queries())
@settings(max_examples=60, deadline=None)
def test_equal_queries_never_diverge(left: str, right: str) -> None:
    """If two queries are byte-identical, their fingerprints are equal and
    the diff is empty (a stricter slice of reflexivity over the menu)."""

    if left != right:
        return
    assert _trace(left).fingerprint_id == _trace(right).fingerprint_id
    assert diff(_trace(left), _trace(right)) == ()
