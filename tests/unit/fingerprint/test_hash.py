"""The fingerprint_id hash function — locked behavior."""

from __future__ import annotations

import hashlib

from mirrorml.fingerprint.hash import fingerprint_id


def test_empty_input_matches_sha256_empty() -> None:
    assert fingerprint_id(b"") == hashlib.sha256(b"").hexdigest()


def test_output_is_64_hex_chars() -> None:
    h = fingerprint_id(b"hello world")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_distinct_inputs_produce_distinct_hashes() -> None:
    assert fingerprint_id(b"hello") != fingerprint_id(b"world")
    assert fingerprint_id(b"a") != fingerprint_id(b"A")


def test_repeated_invocation_is_stable() -> None:
    h1 = fingerprint_id(b"some canonical bytes")
    h2 = fingerprint_id(b"some canonical bytes")
    assert h1 == h2
