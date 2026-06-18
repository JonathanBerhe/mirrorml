"""End-to-end :func:`build_fingerprint` properties."""

from __future__ import annotations

from mirrorml.fingerprint import Fingerprint, build_fingerprint
from mirrorml.fingerprint.operations import Filter, Project, Source
from mirrorml.fingerprint.schema import Operation


def _three_op_pipeline(*, source_id: str, filter_id: str, project_id: str) -> list[Operation]:
    return [
        Source(op_id=source_id, name="t", columns=(("x", "int64"), ("y", "int64"))),
        Filter(op_id=filter_id, dependencies=(source_id,), predicate="x > 0"),
        Project(op_id=project_id, dependencies=(filter_id,), columns=("x", "y")),
    ]


def _build(*, source_id: str, filter_id: str, project_id: str) -> Fingerprint:
    return build_fingerprint(
        framework="pandas",
        input_schema=(("x", "int64"), ("y", "int64")),
        output_schema=(("x", "int64"), ("y", "int64")),
        operations=_three_op_pipeline(
            source_id=source_id, filter_id=filter_id, project_id=project_id
        ),
    )


def test_relabeling_op_ids_does_not_change_fingerprint() -> None:
    """The structural-hash rewrite during canonicalization erases tracer-
    assigned op_ids: equivalent pipelines hash identically regardless of
    label choices."""

    fp_a = _build(source_id="a", filter_id="b", project_id="c")
    fp_b = _build(source_id="src", filter_id="flt", project_id="prj")
    assert fp_a.fingerprint_id == fp_b.fingerprint_id


def test_reordering_independent_operations_does_not_change_fingerprint() -> None:
    src = Source(op_id="s", name="t", columns=(("x", "int64"), ("y", "int64")))
    f1 = Filter(op_id="f1", dependencies=("s",), predicate="x > 0")
    f2 = Filter(op_id="f2", dependencies=("s",), predicate="y > 0")

    fp_a = build_fingerprint(
        framework="pandas",
        input_schema=(("x", "int64"), ("y", "int64")),
        output_schema=(("x", "int64"), ("y", "int64")),
        operations=[src, f1, f2],
    )
    fp_b = build_fingerprint(
        framework="pandas",
        input_schema=(("x", "int64"), ("y", "int64")),
        output_schema=(("x", "int64"), ("y", "int64")),
        operations=[src, f2, f1],
    )
    assert fp_a.fingerprint_id == fp_b.fingerprint_id


def test_changing_a_param_changes_the_fingerprint() -> None:
    fp_a = _build(source_id="s", filter_id="f", project_id="p")

    src = Source(op_id="s", name="t", columns=(("x", "int64"), ("y", "int64")))
    f = Filter(op_id="f", dependencies=("s",), predicate="x > 99")  # changed
    p = Project(op_id="p", dependencies=("f",), columns=("x", "y"))
    fp_b = build_fingerprint(
        framework="pandas",
        input_schema=(("x", "int64"), ("y", "int64")),
        output_schema=(("x", "int64"), ("y", "int64")),
        operations=[src, f, p],
    )
    assert fp_a.fingerprint_id != fp_b.fingerprint_id


def test_changing_framework_changes_the_fingerprint() -> None:
    src = Source(op_id="s", name="t", columns=(("x", "int64"),))
    fp_pandas = build_fingerprint(
        framework="pandas",
        input_schema=(("x", "int64"),),
        output_schema=(("x", "int64"),),
        operations=[src],
    )
    fp_sql = build_fingerprint(
        framework="sql",
        input_schema=(("x", "int64"),),
        output_schema=(("x", "int64"),),
        operations=[src],
    )
    assert fp_pandas.fingerprint_id != fp_sql.fingerprint_id


def test_build_fingerprint_produces_topologically_sorted_ops() -> None:
    """Output operations must satisfy: all dependencies appear before the
    dependent op."""

    fp = _build(source_id="s", filter_id="f", project_id="p")
    positions = {op.op_id: i for i, op in enumerate(fp.operations)}
    for op in fp.operations:
        for dep in op.dependencies:
            assert positions[dep] < positions[op.op_id]
