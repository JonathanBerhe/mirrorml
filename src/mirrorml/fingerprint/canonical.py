"""Canonical encoding of fingerprint data.

Two responsibilities:

1. :func:`canonical_json` emits a deterministic JSON byte string from any
   JSON-compatible object. Output rules: sorted keys, no whitespace,
   ``None``-valued dict entries omitted, NaN/Inf forbidden, UTF-8.
2. :func:`canonicalize_operations` normalizes an operation list into a
   canonical form: validates the DAG, rewrites tracer-assigned ``op_id``
   values to deterministic structural hashes, and emits a topologically-
   sorted tuple.

The structural-hash rewrite is what gives two semantically-equivalent
pipelines (potentially produced by different tracers with different ``op_id``
schemes) the same :class:`~mirrorml.fingerprint.schema.Fingerprint`.

The set of commutativity rewrites is intentionally small in v1.0.0.
Adding a rewrite changes :data:`~mirrorml.fingerprint.schema.SCHEMA_VERSION`.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from mirrorml.exceptions import CanonicalizationError
from mirrorml.fingerprint._typing import OpId

if TYPE_CHECKING:
    from mirrorml.fingerprint.schema import Operation


def canonical_json(data: object) -> bytes:
    """Encode ``data`` as canonical JSON bytes.

    Rules:

    - Keys sorted at every level.
    - Separators ``(",", ":")``, with no whitespace.
    - ``None``-valued dict entries omitted (a missing key and an explicit
      ``None`` are equivalent in the schema).
    - ``ensure_ascii=False`` for byte-stable UTF-8.
    - ``allow_nan=False`` so NaN / Inf raise rather than serializing.
    - Floats rendered via Python's shortest-round-trip representation, which
      is stable across CPython versions.

    Examples:
        >>> canonical_json({"b": 1, "a": 2})
        b'{"a":2,"b":1}'
        >>> canonical_json({"a": None, "b": 1})
        b'{"b":1}'
        >>> canonical_json([3, 1, 2])
        b'[3,1,2]'
    """

    return json.dumps(
        _strip_none(data),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _strip_none(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list | tuple):
        return [_strip_none(x) for x in obj]
    return obj


def canonicalize_operations(ops: tuple[Operation, ...]) -> tuple[Operation, ...]:
    """Normalize an operation tuple into canonical form.

    Steps:

    1. Validate: no duplicate ``op_id``, no missing dependencies, no cycles.
    2. Compute structural hashes for each op in topological order. The hash
       captures the op's content and the structural hashes of its
       dependencies (in order, dependency order is significant for binary
       ops like joins). Excludes ``op_id`` itself so tracer-assigned ids do
       not leak into the hash.
    3. Rewrite each op: ``op_id`` and ``dependencies`` become structural
       hashes.
    4. Topologically sort the rewritten ops with lexicographic tie-break.
       Since the op_ids are now content hashes, the order is content-derived
       and stable across equivalent inputs.

    Raises :class:`~mirrorml.exceptions.CanonicalizationError` on malformed
    graphs.
    """

    if not ops:
        return ()

    by_id: dict[OpId, Operation] = {}
    for op in ops:
        if op.op_id in by_id:
            raise CanonicalizationError(
                f"duplicate op_id {op.op_id!r}; tracers must assign unique op_ids "
                f"within a fingerprint"
            )
        by_id[op.op_id] = op

    for op in ops:
        for dep in op.dependencies:
            if dep not in by_id:
                raise CanonicalizationError(
                    f"operation {op.op_id!r} depends on unknown op {dep!r}; "
                    f"the tracer emitted an orphaned dependency reference"
                )

    initial_order = _topological_order(by_id)

    structural: dict[OpId, str] = {}
    for oid in initial_order:
        op = by_id[oid]
        body = op.model_dump(exclude={"op_id", "dependencies"}, mode="json")
        body["__deps"] = [structural[d] for d in op.dependencies]
        structural[oid] = hashlib.sha256(canonical_json(body)).hexdigest()

    rewritten_by_id: dict[OpId, Operation] = {}
    for oid in initial_order:
        op = by_id[oid]
        new_op = op.model_copy(
            update={
                "op_id": structural[oid],
                "dependencies": tuple(structural[d] for d in op.dependencies),
            }
        )
        rewritten_by_id[structural[oid]] = new_op

    final_order = _topological_order(rewritten_by_id)
    return tuple(rewritten_by_id[oid] for oid in final_order)


def _topological_order(by_id: dict[OpId, Operation]) -> list[OpId]:
    """Kahn's algorithm with lexicographic tie-break on ``op_id``.

    Returns the operations in a deterministic topological order. Raises
    :class:`CanonicalizationError` if the graph contains a cycle.
    """

    indegree: dict[OpId, int] = {oid: len(op.dependencies) for oid, op in by_id.items()}
    successors: dict[OpId, list[OpId]] = {oid: [] for oid in by_id}
    for oid, op in by_id.items():
        for dep in op.dependencies:
            successors[dep].append(oid)

    ready = sorted(oid for oid, d in indegree.items() if d == 0)
    order: list[OpId] = []
    while ready:
        oid = ready.pop(0)
        order.append(oid)
        for succ in successors[oid]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                _insort(ready, succ)

    if len(order) != len(by_id):
        unprocessed = sorted(oid for oid, d in indegree.items() if d > 0)
        raise CanonicalizationError(
            f"operation graph contains a cycle; ops involved: {unprocessed}; "
            f"check the tracer for self-referential or cyclic dependencies"
        )
    return order


def _insort(seq: list[str], item: str) -> None:
    """Lexicographic in-place insort. Small enough to avoid an import."""

    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if seq[mid] < item:
            lo = mid + 1
        else:
            hi = mid
    seq.insert(lo, item)
