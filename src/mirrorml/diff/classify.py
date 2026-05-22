"""Divergence classifier — map structural fingerprint differences to the
closed taxonomy in ``docs/concepts/divergence_taxonomy.md``.

In v0.0.1 only the :class:`Divergence` data model is implemented; the
classifier rules land in M3 together with the diff engine.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mirrorml._taxonomy import DivergenceCategory
from mirrorml.fingerprint._typing import OpId


class Divergence(BaseModel):
    """A single classified disagreement between two fingerprints.

    ``category`` is drawn from the closed taxonomy of fifteen labels — see
    ``docs/concepts/divergence_taxonomy.md``. ``left_op_id`` and
    ``right_op_id`` locate the responsible operation on each side; either
    may be ``None`` for divergences where the responsible op exists on only
    one side (e.g. a missing operation, an extra filter).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: DivergenceCategory
    left_op_id: OpId | None = None
    right_op_id: OpId | None = None
    detail: str = Field(
        default="",
        description="Human-readable explanation suitable for CLI output.",
    )
