"""Canonical-JSON encoding and operation canonicalization are
reproducibility-critical. These tests pin the behavior tightly."""

from __future__ import annotations

import json

import pytest

from mirrorml.exceptions import CanonicalizationError
from mirrorml.fingerprint.canonical import canonical_json, canonicalize_operations
from mirrorml.fingerprint.operations import Filter, Project, Source


def test_canonical_json_sorts_keys() -> None:
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_canonical_json_omits_none() -> None:
    assert canonical_json({"a": None, "b": 1}) == b'{"b":1}'
    assert canonical_json({"a": {"x": None, "y": 1}}) == b'{"a":{"y":1}}'


def test_canonical_json_omits_separators() -> None:
    assert b" " not in canonical_json({"a": [1, 2, 3], "b": "c"})


def test_canonical_json_rejects_nan_and_inf() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})


def test_canonical_json_is_byte_stable_across_dict_orderings() -> None:
    a = canonical_json({"b": 1, "a": 2, "c": [3, 1, 2]})
    b = canonical_json({"c": [3, 1, 2], "a": 2, "b": 1})
    assert a == b


def test_canonical_json_handles_tuples_as_arrays() -> None:
    out = canonical_json({"x": (1, 2, 3)})
    assert json.loads(out) == {"x": [1, 2, 3]}


def test_canonicalize_empty_returns_empty() -> None:
    assert canonicalize_operations(()) == ()


def test_canonicalize_rejects_duplicate_op_ids() -> None:
    s1 = Source(op_id="a", name="t", columns=(("x", "int64"),))
    s2 = Source(op_id="a", name="t", columns=(("x", "int64"),))
    with pytest.raises(CanonicalizationError, match="duplicate"):
        canonicalize_operations((s1, s2))


def test_canonicalize_rejects_orphan_dependencies() -> None:
    f = Filter(op_id="f", dependencies=("missing",), predicate="x > 0")
    with pytest.raises(CanonicalizationError, match="unknown"):
        canonicalize_operations((f,))


def test_canonicalize_rewrites_op_ids_to_structural_hashes() -> None:
    src = Source(op_id="tracer_chose_this", name="t", columns=(("x", "int64"),))
    out = canonicalize_operations((src,))
    assert len(out) == 1
    assert len(out[0].op_id) == 64
    assert all(c in "0123456789abcdef" for c in out[0].op_id)


def test_canonicalize_is_order_independent_for_independent_filters() -> None:
    """Two independent filters in either order canonicalize to the same tuple
    (and therefore the same fingerprint downstream)."""

    s = Source(op_id="s", name="t", columns=(("x", "int64"), ("y", "int64")))
    f1 = Filter(op_id="f1", dependencies=("s",), predicate="x > 0")
    f2 = Filter(op_id="f2", dependencies=("s",), predicate="y > 0")
    p_a = Project(op_id="p", dependencies=("f1",), columns=("x", "y"))

    s_b = Source(op_id="s", name="t", columns=(("x", "int64"), ("y", "int64")))
    f1_b = Filter(op_id="f1", dependencies=("s",), predicate="x > 0")
    f2_b = Filter(op_id="f2", dependencies=("s",), predicate="y > 0")
    p_b = Project(op_id="p", dependencies=("f1",), columns=("x", "y"))

    out_a = canonicalize_operations((s, f1, f2, p_a))
    out_b = canonicalize_operations((s_b, f2_b, f1_b, p_b))

    assert tuple(op.op_id for op in out_a) == tuple(op.op_id for op in out_b)


def test_canonicalize_detects_cycles() -> None:
    # Construct a two-op cycle by referencing each other. We have to
    # post-construct because Pydantic doesn't validate cross-op references.
    f1 = Filter(op_id="f1", dependencies=("f2",), predicate="x > 0")
    f2 = Filter(op_id="f2", dependencies=("f1",), predicate="y > 0")
    with pytest.raises(CanonicalizationError, match="cycle"):
        canonicalize_operations((f1, f2))
