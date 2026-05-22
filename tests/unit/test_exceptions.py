"""All MirrorML exceptions descend from a single base; messages are
actionable per CLAUDE.md § Error handling."""

from __future__ import annotations

from mirrorml.exceptions import (
    CanonicalizationError,
    FingerprintVersionError,
    MirrorMLError,
    UnsupportedOperationError,
)


def test_all_subclass_the_base() -> None:
    for cls in (CanonicalizationError, FingerprintVersionError, UnsupportedOperationError):
        assert issubclass(cls, MirrorMLError)


def test_base_is_an_exception() -> None:
    assert issubclass(MirrorMLError, Exception)
