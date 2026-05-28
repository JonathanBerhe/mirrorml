"""Tests for the UDF source-hash normalizer (libcst-norm-v1).

The normalizer's job is "two sources that compute the same value hash to
the same string regardless of formatting / comments / docstrings, and two
sources that compute different values hash differently." Tests assert both
directions and exercise both the string and callable entrypoints.
"""

from __future__ import annotations

import hashlib

import pytest

from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint.udf_hash import (
    SOURCE_HASH_ALGORITHM,
    normalize_callable_source,
    normalize_source,
    source_hash,
)


def _hash(source: str) -> str:
    return hashlib.sha256(normalize_source(source).encode("utf-8")).hexdigest()


# --- equivalence (different formatting / comments / docstrings) -------------


def test_whitespace_only_difference_collapses_to_same_hash() -> None:
    a = "def f(x):\n    return x + 1\n"
    b = "def f(x):\n    return  x  +  1\n"
    assert _hash(a) == _hash(b)


def test_blank_lines_inside_body_do_not_change_hash() -> None:
    a = "def f(x):\n    y = x + 1\n    return y\n"
    b = "def f(x):\n    y = x + 1\n\n    return y\n"
    assert _hash(a) == _hash(b)


def test_end_of_line_comments_do_not_change_hash() -> None:
    a = "def f(x):\n    return x + 1\n"
    b = "def f(x):\n    return x + 1  # trailing\n"
    assert _hash(a) == _hash(b)


def test_whole_line_comments_do_not_change_hash() -> None:
    a = "def f(x):\n    return x + 1\n"
    b = "def f(x):\n    # increment by one\n    return x + 1\n"
    assert _hash(a) == _hash(b)


def test_single_line_docstring_does_not_change_hash() -> None:
    a = "def f(x):\n    return x + 1\n"
    b = 'def f(x):\n    """Add one."""\n    return x + 1\n'
    assert _hash(a) == _hash(b)


def test_triple_quoted_docstring_does_not_change_hash() -> None:
    a = "def f(x):\n    return x + 1\n"
    b = 'def f(x):\n    """\n    Multi-line.\n\n    Docstring.\n    """\n    return x + 1\n'
    assert _hash(a) == _hash(b)


def test_class_docstring_does_not_change_hash() -> None:
    a = "class C:\n    x = 1\n"
    b = 'class C:\n    """Class doc."""\n    x = 1\n'
    assert _hash(a) == _hash(b)


def test_leading_indentation_does_not_change_hash() -> None:
    """A nested function's source carries the enclosing indent; dedent in
    the normalizer must absorb it."""

    a = "def f(x):\n    return x\n"
    b = "    def f(x):\n        return x\n"
    assert _hash(a) == _hash(b)


# --- non-equivalence (different bodies) --------------------------------------


def test_different_constant_changes_hash() -> None:
    assert _hash("def f(x): return x + 1") != _hash("def f(x): return x + 2")


def test_different_operator_changes_hash() -> None:
    assert _hash("def f(x): return x + 1") != _hash("def f(x): return x - 1")


def test_different_function_name_changes_hash() -> None:
    """We hash the full source including the def name. The diff layer is
    free to ignore qualname separately when classifying a UdfRef
    mismatch."""

    assert _hash("def a(x): return x + 1") != _hash("def b(x): return x + 1")


# --- output shape ------------------------------------------------------------


def test_source_hash_is_64_hex_chars() -> None:
    def f(x: int) -> int:
        return x

    h = source_hash(f)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_normalize_source_returns_a_compact_string() -> None:
    out = normalize_source("def f(x):\n    # comment\n    return x + 1\n")
    assert isinstance(out, str)
    assert "\n" not in out  # whitespace collapsed
    assert "#" not in out  # comments stripped
    assert "def" in out


def test_algorithm_id_is_versioned() -> None:
    """The algorithm string lives in every UdfRef; it must be stable so a
    bump is a deliberate decision."""

    assert SOURCE_HASH_ALGORITHM == "libcst-norm-v1"


# --- callable entrypoint -----------------------------------------------------


def test_normalize_callable_source_round_trips_to_normalize_source() -> None:
    """The callable entrypoint is a thin wrapper over the string version;
    they must agree on a file-defined function."""

    import inspect

    def f(x: int) -> int:
        return x + 1

    assert normalize_callable_source(f) == normalize_source(inspect.getsource(f))


# --- failure modes -----------------------------------------------------------


def test_unparseable_source_raises_unsupported() -> None:
    with pytest.raises(UnsupportedOperationError, match="could not parse"):
        normalize_source("def f(:")


def test_builtin_callable_raises_unsupported() -> None:
    """C extensions don't have Python source; the normalizer must surface
    an actionable error rather than silently hash nothing."""

    with pytest.raises(UnsupportedOperationError, match="cannot retrieve source"):
        source_hash(len)
