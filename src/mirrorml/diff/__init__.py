"""Diff engine — compare two fingerprints; classify and localize divergences.

The :class:`Divergence` data model is implemented in v0.0.1 so downstream
code can lock the classifier interface. The diff function itself lands in M3.
"""

from __future__ import annotations

from mirrorml.diff.classify import Divergence
from mirrorml.diff.engine import diff

__all__ = ["Divergence", "diff"]
