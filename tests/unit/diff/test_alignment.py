"""LCS-based op alignment in the diff engine.

These tests exercise the alignment layer directly: ops inserted or
deleted in the middle of a pipeline should still align cleanly on the
matching kinds, and the orphan ops should each surface as a localized
``schema_drift`` (carrying the right ``op_id`` on the side it came from)
rather than the engine giving up at the first kind mismatch.
"""

from __future__ import annotations

from mirrorml import diff, trace_sql

EVENTS = (("uid", "int64"), ("score", "float64"))


def test_extra_filter_op_localizes_to_right_orphan() -> None:
    """Online has an extra WHERE filter. The Source and Project ops still
    align on both sides; the orphan Filter surfaces as a single
    schema_drift carrying the right's filter op_id."""

    a = trace_sql("SELECT uid FROM events", schemas={"events": EVENTS})
    b = trace_sql(
        "SELECT uid FROM events WHERE score > 0",
        schemas={"events": EVENTS},
    )
    divs = diff(a, b)
    drift = [d for d in divs if d.category == "schema_drift" and "filter" in d.detail]
    assert len(drift) == 1
    assert drift[0].right_op_id is not None
    assert drift[0].left_op_id is None


def test_extra_filter_op_localizes_to_left_orphan() -> None:
    """Symmetric: offline has the extra Filter. Should produce one
    schema_drift carrying the LEFT op_id."""

    a = trace_sql(
        "SELECT uid FROM events WHERE score > 0",
        schemas={"events": EVENTS},
    )
    b = trace_sql("SELECT uid FROM events", schemas={"events": EVENTS})
    divs = diff(a, b)
    drift = [d for d in divs if d.category == "schema_drift" and "filter" in d.detail]
    assert len(drift) == 1
    assert drift[0].left_op_id is not None
    assert drift[0].right_op_id is None


def test_two_extra_ops_produce_two_localized_drifts() -> None:
    """Online inserts both a WHERE and an ORDER BY. The diff should
    surface two orphan divergences (filter and sort), each localized to
    the right's op_id."""

    a = trace_sql("SELECT uid FROM events", schemas={"events": EVENTS})
    b = trace_sql(
        "SELECT uid FROM events WHERE score > 0 ORDER BY uid",
        schemas={"events": EVENTS},
    )
    divs = diff(a, b)
    extras = [d for d in divs if d.category == "schema_drift" and "extra" in d.detail]
    kinds = {extra.detail.split("'")[1] for extra in extras}
    assert kinds == {"filter", "sort"}
    for d in extras:
        assert d.right_op_id is not None
        assert d.left_op_id is None


def test_aligned_pairs_classify_via_classify_op_pair() -> None:
    """When two pipelines have identical kind sequences but a parameter
    difference on an aligned pair (e.g. different ORDER BY), the engine
    must classify via the existing per-kind rule (ordering_dependence),
    not as an orphan schema_drift."""

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
    assert not any(d.category == "schema_drift" and "extra" in d.detail for d in divs)


def test_alignment_does_not_match_across_different_kinds() -> None:
    """If one side is Source -> Filter and the other is Source -> Project,
    the second positions are different kinds. The LCS finds the Source
    match and reports Filter / Project as orphans (one LeftOnly, one
    RightOnly), not as an aligned pair to be classified as a parameter
    difference."""

    schemas = {"events": EVENTS}
    a = trace_sql("SELECT * FROM events WHERE score > 0", schemas=schemas)  # Source -> Filter
    b = trace_sql("SELECT uid FROM events", schemas=schemas)  # Source -> Project
    divs = diff(a, b)
    drift_details = [d.detail for d in divs if d.category == "schema_drift" and "extra" in d.detail]
    joined = " | ".join(drift_details)
    assert "filter" in joined
    assert "project" in joined
