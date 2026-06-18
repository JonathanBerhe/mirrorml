"""Fingerprint identifier: SHA-256 over canonical bytes.

This module is intentionally tiny. Keeping it isolated from the rest of the
fingerprint package makes the algorithm easy to audit and gives downstream
work (e.g. comparing fingerprint_ids across schema versions) a single
function to reach for.
"""

from __future__ import annotations

import hashlib

from mirrorml.fingerprint._typing import FingerprintId


def fingerprint_id(canonical_bytes: bytes) -> FingerprintId:
    """Return the SHA-256 hex digest of ``canonical_bytes``.

    The input must be the canonical-JSON encoding of the fingerprint body
    *excluding* its own ``fingerprint_id`` field (which would introduce a
    recursive dependency).

    Examples:
        >>> fingerprint_id(b"")
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        >>> fingerprint_id(b"hello") == hashlib.sha256(b"hello").hexdigest()
        True
    """

    return hashlib.sha256(canonical_bytes).hexdigest()
