"""Diff engine — top-level comparison of two fingerprints.

**Not implemented in v0.0.1.** The engine and the rules it dispatches to
land together in M3.

The signature is fixed: ``diff`` always returns a tuple of
:class:`~mirrorml.diff.classify.Divergence` objects (empty when the
fingerprints are equivalent). Callers in CI compare ``len(diff(...))`` to
zero; callers in tools render the tuple via :mod:`rich`.
"""

from __future__ import annotations

from mirrorml.diff.classify import Divergence
from mirrorml.fingerprint.schema import Fingerprint

__all__ = ["diff"]


def diff(left: Fingerprint, right: Fingerprint, /) -> tuple[Divergence, ...]:
    """Return the divergences between two fingerprints.

    Returns an empty tuple iff the two fingerprints are MirrorML-equivalent.

    Not implemented in v0.0.1. The diff engine lands in M3.

    Raises:
        NotImplementedError: Always.
    """

    raise NotImplementedError(
        "diff: not yet implemented in v0.0.1 (lands in M3). "
        "Track progress in the project's issue tracker."
    )
