"""Diff engine. Compares two fingerprints and emits a tuple of classified
divergences.

The engine is the orchestrator. Each kind of comparison (schema, op-pair,
dtype) is delegated to a focused helper in :mod:`mirrorml.diff.classify`.

Alignment strategy (M3 phase 2): longest-common-subsequence (LCS) over
op kinds. The position-walk used in phase 1 misaligned everything
downstream of an insertion or deletion in the middle of a pipeline; the
LCS finds the largest set of same-kind pairings between the two
operation tuples and reports unmatched ops on either side as orphans
localized to their own ``op_id``. Aligned same-kind pairs are then
classified by :func:`~mirrorml.diff.classify.classify_op_pair` exactly
as in phase 1, so the per-kind taxonomy logic is unchanged.

Cross-framework comparison (e.g. pandas fingerprint vs SQL fingerprint)
is the intended use case. The ``framework`` field is informational;
differences there do not produce divergences.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from mirrorml.diff.classify import Divergence, classify_op_pair, compare_schemas
from mirrorml.fingerprint.schema import Fingerprint, Operation

__all__ = ["diff"]


@dataclass(frozen=True)
class _Aligned:
    """A pair of ops the LCS aligned by kind."""

    left: Operation
    right: Operation


@dataclass(frozen=True)
class _LeftOnly:
    """An op on the left side with no match on the right (deleted in online)."""

    op: Operation


@dataclass(frozen=True)
class _RightOnly:
    """An op on the right side with no match on the left (added in online)."""

    op: Operation


_AlignmentStep = _Aligned | _LeftOnly | _RightOnly


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
    3. Op divergences in alignment order (LCS over op kinds, then the
       per-pair / orphan divergences for each step).

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

    for step in _align_operations(left.operations, right.operations):
        divergences.extend(_classify_alignment_step(step))

    return tuple(divergences)


def _classify_alignment_step(step: _AlignmentStep) -> Iterator[Divergence]:
    """Yield divergences for a single alignment step."""

    if isinstance(step, _Aligned):
        yield from classify_op_pair(step.left, step.right)
        return
    if isinstance(step, _LeftOnly):
        yield Divergence(
            category="schema_drift",
            left_op_id=step.op.op_id,
            detail=(f"left has an extra {step.op.kind!r} op with no matching op on the right"),
        )
        return
    # _RightOnly
    yield Divergence(
        category="schema_drift",
        right_op_id=step.op.op_id,
        detail=(f"right has an extra {step.op.kind!r} op with no matching op on the left"),
    )


def _align_operations(
    left: tuple[Operation, ...],
    right: tuple[Operation, ...],
) -> list[_AlignmentStep]:
    """Align two operation tuples by their kinds via standard LCS.

    Two ops match if they share a kind; aligned same-kind ops are then
    delegated to :func:`classify_op_pair` for content-level
    classification. Ops without a match are reported as :class:`_LeftOnly`
    or :class:`_RightOnly` so downstream localization can point at the
    orphan ``op_id``.

    The LCS is the same DP that ``difflib`` and ``diff(1)`` use; the
    backtrace prefers left-only steps before right-only ones when the DP
    table is tied, which gives a deterministic ordering.
    """

    n, m = len(left), len(right)
    if n == 0 and m == 0:
        return []
    if n == 0:
        return [_RightOnly(op) for op in right]
    if m == 0:
        return [_LeftOnly(op) for op in left]

    # lcs[i][j] = length of the longest common subsequence of kinds for
    # left[:i] and right[:j].
    lcs: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if left[i].kind == right[j].kind:
                lcs[i + 1][j + 1] = lcs[i][j] + 1
            else:
                lcs[i + 1][j + 1] = max(lcs[i + 1][j], lcs[i][j + 1])

    steps: list[_AlignmentStep] = []
    i, j = n, m
    while i > 0 and j > 0:
        if left[i - 1].kind == right[j - 1].kind:
            steps.append(_Aligned(left[i - 1], right[j - 1]))
            i -= 1
            j -= 1
        elif lcs[i - 1][j] >= lcs[i][j - 1]:
            steps.append(_LeftOnly(left[i - 1]))
            i -= 1
        else:
            steps.append(_RightOnly(right[j - 1]))
            j -= 1

    while i > 0:
        steps.append(_LeftOnly(left[i - 1]))
        i -= 1
    while j > 0:
        steps.append(_RightOnly(right[j - 1]))
        j -= 1

    steps.reverse()
    return steps
