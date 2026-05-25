"""Diff engine. Compares two fingerprints and emits a tuple of classified
divergences.

The engine is the orchestrator. Each kind of comparison (schema, op-pair,
dtype) is delegated to a focused helper in :mod:`mirrorml.diff.classify`.

Alignment strategy in M3 phase 1: position-based. Both fingerprints'
operation tuples are canonically ordered by ``canonicalize_operations``,
so equivalent pipelines align trivially. When the two pipelines diverge
structurally (different op counts, different kinds at a position), a
``schema_drift`` divergence is emitted and the walk continues across the
common prefix so downstream localizable divergences still surface. A
smarter LCS-style alignment is on the roadmap for phase 2.

Cross-framework comparison (e.g. pandas fingerprint vs SQL fingerprint)
is the intended use case from PAPER.md C4. The ``framework`` field is
informational; differences there do not produce divergences.
"""

from __future__ import annotations

from mirrorml.diff.classify import (
    Divergence,
    classify_op_pair,
    compare_schemas,
)
from mirrorml.fingerprint.schema import Fingerprint

__all__ = ["diff"]


def diff(left: Fingerprint, right: Fingerprint, /) -> tuple[Divergence, ...]:
    """Return every classified divergence between ``left`` and ``right``.

    Returns an empty tuple iff the two fingerprints are MirrorML-equivalent.
    Equivalence is decided by ``fingerprint_id`` equality (fast-path); when
    the ids differ, the engine walks the structures and classifies each
    disagreement into one of the taxonomy categories.

    Cross-framework comparison is supported: a SQL-traced fingerprint can
    be diffed against a pandas-traced fingerprint and the ``framework``
    field is not itself a divergence.

    Order of the returned divergences:

    1. Input-schema divergences (added / dropped / type-coerced columns
       on the input side).
    2. Output-schema divergences (same, on the output side).
    3. Op-by-op divergences in pipeline position order.

    The order is deterministic so CI snapshots are stable.

    Examples:
        >>> from mirrorml.fingerprint import build_fingerprint
        >>> from mirrorml.fingerprint.operations import Source
        >>> a = build_fingerprint(
        ...     framework="pandas",
        ...     input_schema=(("x", "int64"),),
        ...     output_schema=(("x", "int64"),),
        ...     operations=[Source(op_id="s", name="t", columns=(("x", "int64"),))],
        ... )
        >>> diff(a, a)
        ()
    """

    if left.fingerprint_id == right.fingerprint_id:
        return ()

    divergences: list[Divergence] = []

    divergences.extend(compare_schemas(left.input_schema, right.input_schema, location="input"))
    divergences.extend(compare_schemas(left.output_schema, right.output_schema, location="output"))

    left_ops = left.operations
    right_ops = right.operations

    if len(left_ops) != len(right_ops):
        divergences.append(
            Divergence(
                category="schema_drift",
                detail=(
                    f"operation count differs: left has {len(left_ops)}, right has {len(right_ops)}"
                ),
            )
        )

    for left_op, right_op in zip(left_ops, right_ops, strict=False):
        divergences.extend(classify_op_pair(left_op, right_op))

    return tuple(divergences)
