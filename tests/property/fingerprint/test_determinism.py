"""Property-based tests for fingerprint determinism.

These exercise the critical invariant that two pipelines with the same
structural content yield the same :attr:`Fingerprint.fingerprint_id`,
regardless of:

- tracer-assigned ``op_id`` labels,
- the insertion order of operations that lack a data-flow dependency.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from mirrorml.fingerprint import build_fingerprint
from mirrorml.fingerprint.operations import Filter, Project, Source
from mirrorml.fingerprint.schema import Operation

# Restrict op_id labels to a small printable alphabet so shrunken
# counterexamples remain readable.
_op_id = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz",
    min_size=1,
    max_size=8,
)


@given(
    src_id=_op_id,
    filter_id=_op_id,
    project_id=_op_id,
)
@settings(max_examples=50, deadline=None)
def test_op_id_relabeling_preserves_fingerprint_id(
    src_id: str, filter_id: str, project_id: str
) -> None:
    """Three distinct labels for three ops must yield the same fingerprint_id
    as a canonical labeling."""

    if len({src_id, filter_id, project_id}) != 3:
        return  # uniqueness required; Hypothesis will explore other draws

    src = Source(op_id=src_id, name="t", columns=(("x", "int64"),))
    flt = Filter(op_id=filter_id, dependencies=(src_id,), predicate="x > 0")
    prj = Project(op_id=project_id, dependencies=(filter_id,), columns=("x",))

    fp_random = build_fingerprint(
        framework="pandas",
        input_schema=(("x", "int64"),),
        output_schema=(("x", "int64"),),
        operations=[src, flt, prj],
    )

    src_canon = Source(op_id="a", name="t", columns=(("x", "int64"),))
    flt_canon = Filter(op_id="b", dependencies=("a",), predicate="x > 0")
    prj_canon = Project(op_id="c", dependencies=("b",), columns=("x",))

    fp_canon = build_fingerprint(
        framework="pandas",
        input_schema=(("x", "int64"),),
        output_schema=(("x", "int64"),),
        operations=[src_canon, flt_canon, prj_canon],
    )

    assert fp_random.fingerprint_id == fp_canon.fingerprint_id


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=25, deadline=None)
def test_independent_filter_ordering_preserves_fingerprint_id(seed: int) -> None:
    """Two filters with the same dependency and different predicates are
    independent, emitting them in either order must yield the same
    fingerprint_id."""

    import random

    rng = random.Random(seed)
    src = Source(op_id="s", name="t", columns=(("x", "int64"), ("y", "int64")))
    f_x = Filter(op_id="fx", dependencies=("s",), predicate="x > 0")
    f_y = Filter(op_id="fy", dependencies=("s",), predicate="y > 0")

    ordering_a: list[Operation] = [src, f_x, f_y]
    ordering_b: list[Operation] = [src, f_y, f_x]
    rng.shuffle(ordering_a)
    rng.shuffle(ordering_b)

    fp_a = build_fingerprint(
        framework="pandas",
        input_schema=(("x", "int64"), ("y", "int64")),
        output_schema=(("x", "int64"), ("y", "int64")),
        operations=ordering_a,
    )
    fp_b = build_fingerprint(
        framework="pandas",
        input_schema=(("x", "int64"), ("y", "int64")),
        output_schema=(("x", "int64"), ("y", "int64")),
        operations=ordering_b,
    )

    assert fp_a.fingerprint_id == fp_b.fingerprint_id
