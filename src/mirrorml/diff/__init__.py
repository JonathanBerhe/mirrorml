"""Diff engine — compare two fingerprints; classify and localize divergences.

:func:`diff` aligns two fingerprints, classifies each disagreement into the
taxonomy, and localizes it to the responsible operation; :class:`Divergence`
is the data model it returns.
"""

from __future__ import annotations

from mirrorml.diff.classify import Divergence
from mirrorml.diff.engine import diff

__all__ = ["Divergence", "diff"]
