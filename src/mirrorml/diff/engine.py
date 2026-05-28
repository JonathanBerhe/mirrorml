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
from mirrorml.fingerprint.operations import Source
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

    # The leakage rule is computed first because, when it fires, the
    # asymmetric Window-vs-Aggregate pair also surfaces as ``schema_drift``
    # at both the output-schema level and the LCS-orphan level. We collect
    # the op_ids the rule names so we can suppress those collateral
    # ``schema_drift`` divergences below.
    leakage_divs = list(_feature_leakage_check(left, right))
    leakage_op_ids = {d.left_op_id for d in leakage_divs} | {d.right_op_id for d in leakage_divs}
    leakage_op_ids.discard(None)

    def _accept(div: Divergence) -> bool:
        if div.category != "schema_drift":
            return True
        return div.left_op_id not in leakage_op_ids and div.right_op_id not in leakage_op_ids

    divergences: list[Divergence] = []

    left_input_op_id = _primary_source_op_id(left)
    right_input_op_id = _primary_source_op_id(right)
    divergences.extend(
        d
        for d in compare_schemas(
            left.input_schema,
            right.input_schema,
            location="input",
            left_op_id=left_input_op_id,
            right_op_id=right_input_op_id,
        )
        if _accept(d)
    )

    left_output_op_id = left.operations[-1].op_id if left.operations else None
    right_output_op_id = right.operations[-1].op_id if right.operations else None
    divergences.extend(
        d
        for d in compare_schemas(
            left.output_schema,
            right.output_schema,
            location="output",
            left_op_id=left_output_op_id,
            right_op_id=right_output_op_id,
        )
        if _accept(d)
    )

    for step in _align_operations(left.operations, right.operations):
        for div in _classify_alignment_step(step):
            if _accept(div):
                divergences.append(div)

    divergences.extend(leakage_divs)

    return tuple(divergences)


def _feature_leakage_check(left: Fingerprint, right: Fingerprint) -> Iterator[Divergence]:
    """Whole-graph rule: flag ``feature_leakage_temporal`` when both sides
    declare an ``event_time_column`` on their primary Source, but one side
    bounds its aggregation in a Window (point-in-time safe) and the other
    leaves it unbounded.

    The unbounded side's plain Aggregate can see rows from after the
    label time, which is the canonical look-ahead leakage pattern. We
    fire only when the divergence is *asymmetric* (one side guarded, the
    other not, with at least one aggregation on the unguarded side); same
    pattern on both sides is not a divergence even if both leak.
    """

    left_event_time = _primary_source_event_time(left)
    right_event_time = _primary_source_event_time(right)
    if left_event_time is None or right_event_time is None:
        return

    left_has_window = any(op.kind == "window" for op in left.operations)
    right_has_window = any(op.kind == "window" for op in right.operations)
    left_has_agg = any(op.kind == "aggregate" for op in left.operations)
    right_has_agg = any(op.kind == "aggregate" for op in right.operations)

    leaky_left = right_has_window and left_has_agg and not left_has_window
    leaky_right = left_has_window and right_has_agg and not right_has_window
    if not (leaky_left or leaky_right):
        return

    leaky_label = "offline (left)" if leaky_left else "online (right)"
    guarded_label = "online (right)" if leaky_left else "offline (left)"
    leaky_fp = left if leaky_left else right
    guarded_fp = right if leaky_left else left

    leaky_op = next((op for op in leaky_fp.operations if op.kind == "aggregate"), None)
    guarded_op = next((op for op in guarded_fp.operations if op.kind == "window"), None)
    if leaky_op is None or guarded_op is None:
        return

    yield Divergence(
        category="feature_leakage_temporal",
        left_op_id=(leaky_op.op_id if leaky_left else guarded_op.op_id),
        right_op_id=(guarded_op.op_id if leaky_left else leaky_op.op_id),
        detail=(
            f"{guarded_label} wraps its aggregation in a Window over the "
            f"event_time column (point-in-time safe); {leaky_label} uses a "
            f"plain Aggregate with no temporal bound, which may see rows "
            f"from after the label time. "
            f"event_time_column = {left_event_time!r} / {right_event_time!r}."
        ),
    )


def _primary_source_event_time(fp: Fingerprint) -> str | None:
    """Return the ``event_time_column`` declared on the primary Source op,
    or ``None`` if the pipeline did not declare one."""

    for op in fp.operations:
        if isinstance(op, Source) and op.columns == fp.input_schema:
            return op.event_time_column
    for op in fp.operations:
        if isinstance(op, Source):
            return op.event_time_column
    return None


def _primary_source_op_id(fp: Fingerprint) -> str | None:
    """Find the Source op whose columns match the fingerprint's
    ``input_schema``. Multi-source pipelines (e.g. joins) carry multiple
    Source ops; the one whose columns equal ``input_schema`` is the
    primary table the input-schema check is comparing against. Returns
    ``None`` for pipelines with no Source.
    """

    for op in fp.operations:
        if isinstance(op, Source) and op.columns == fp.input_schema:
            return op.op_id

    # HACK: every tracer shipped today constructs the primary Source so its
    # ``columns`` are exactly ``input_schema``, so the exact-match loop
    # above always returns. The fallback below covers a hypothetical future
    # tracer that diverges from that invariant: rather than crash the diff
    # engine, we attribute input-schema divergences to *some* Source so the
    # engine output stays well-formed. In a multi-source pipeline this can
    # mislocalize. The right long-term fix is to either enforce the
    # invariant at fingerprint-construction time or extend the schema to
    # name a primary-source op_id explicitly.
    for op in fp.operations:
        if isinstance(op, Source):
            return op.op_id
    return None


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
