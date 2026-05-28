"""Source-hash for UDF bodies.

A :class:`~mirrorml.fingerprint.schema.UdfRef` carries a SHA-256 over a
normalized rendering of a user-defined callable's source so two callables
that compute the same thing (differently formatted, differently commented)
hash to the same value. This module is the normalizer.

The algorithm name is :data:`SOURCE_HASH_ALGORITHM`; it is versioned (and
embedded in every :class:`UdfRef`) so a libcst upgrade that changes the
rendering does not silently invalidate user fingerprints. Bumping the
version requires a corresponding migration entry.

Normalization steps:

1. ``inspect.getsource(fn)`` (the function's textual definition).
2. ``textwrap.dedent`` so a nested-function source does not carry leading
   indentation that would change its hash.
3. Parse with ``libcst.parse_module``; any unparseable input raises
   :class:`~mirrorml.exceptions.UnsupportedOperationError`.
4. Strip every comment (the ``comment`` field on ``TrailingWhitespace`` and
   ``EmptyLine`` nodes).
5. Drop the docstring statement on each ``FunctionDef`` / ``ClassDef``.
6. Re-render via ``module.code`` and collapse all whitespace runs to a
   single space. This is intentionally aggressive: any whitespace the
   transformer missed is normalized here.
7. SHA-256 the resulting bytes.
"""

from __future__ import annotations

import hashlib
import inspect
import textwrap
from collections.abc import Callable
from typing import Any, Final, TypeVar

import libcst as cst

from mirrorml.exceptions import UnsupportedOperationError

SOURCE_HASH_ALGORITHM: Final[str] = "libcst-norm-v1"
"""Algorithm identifier embedded in every :class:`UdfRef`. Bump when the
normalization output changes for any input (including libcst upgrades that
alter the CST shape) so old fingerprints can be routed through a migration
rather than silently mismatched."""


def normalize_source(source: str) -> str:
    """Render a Python source string to a deterministic canonical form.

    Comments, blank lines, and docstrings are stripped; every run of
    whitespace collapses to a single space. Two sources that compute the
    same value (formatted differently) collapse to the same string.

    Raises:
        UnsupportedOperationError: if ``source`` is not parseable as
            Python.

    Examples:
        >>> normalize_source("def f(x):\\n    return x + 1\\n") == \\
        ...     normalize_source("def f(x):\\n    # add one\\n    return x + 1\\n")
        True
    """

    dedented = textwrap.dedent(source)
    try:
        module = cst.parse_module(dedented)
    except cst.ParserSyntaxError as exc:
        raise UnsupportedOperationError(
            f"udf source-hash: libcst could not parse the source: {exc}"
        ) from exc

    stripped = module.visit(_Normalizer())
    rendered = stripped.code
    return " ".join(rendered.split())


def normalize_callable_source(fn: Callable[..., Any]) -> str:
    """Render a callable's source through :func:`normalize_source`.

    Raises:
        UnsupportedOperationError: if the callable's source cannot be
            retrieved (lambdas defined in the REPL, C extensions) or is
            not parseable as Python.
    """

    try:
        raw = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise UnsupportedOperationError(
            f"udf source-hash: cannot retrieve source for {fn!r}; "
            f"define the callable in a file (not the REPL) and avoid "
            f"lambdas / C extensions: {exc}"
        ) from exc
    return normalize_source(raw)


def source_hash(fn: Callable[..., Any]) -> str:
    """Return the hex SHA-256 of the normalized source of ``fn``."""

    return source_hash_for_source(normalize_callable_source(fn))


def source_hash_for_source(normalized: str) -> str:
    """SHA-256 the already-normalized source string. Internal helper for
    callers that already have a normalized string in hand."""

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class _Normalizer(cst.CSTTransformer):
    """Strip comments and docstrings; whitespace is handled by the caller's
    ``" ".join(rendered.split())`` pass.

    Comments in libcst are not standalone statements; they live on the
    ``comment`` slot of ``TrailingWhitespace`` (end-of-line comments) and
    ``EmptyLine`` (whole-line comments). Both are visited here.
    """

    def leave_TrailingWhitespace(
        self,
        original_node: cst.TrailingWhitespace,
        updated_node: cst.TrailingWhitespace,
    ) -> cst.TrailingWhitespace:
        if updated_node.comment is not None:
            return updated_node.with_changes(comment=None)
        return updated_node

    def leave_EmptyLine(
        self,
        original_node: cst.EmptyLine,
        updated_node: cst.EmptyLine,
    ) -> cst.EmptyLine:
        if updated_node.comment is not None:
            return updated_node.with_changes(comment=None)
        return updated_node

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        return _drop_docstring(updated_node)

    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        return _drop_docstring(updated_node)


_DefT = TypeVar("_DefT", cst.FunctionDef, cst.ClassDef)


def _drop_docstring(node: _DefT) -> _DefT:
    """If the first statement of ``node`` is a bare string expression
    (the docstring convention), drop it."""

    body = node.body
    if not isinstance(body, cst.IndentedBlock):
        return node
    if not body.body:
        return node
    first = body.body[0]
    if not isinstance(first, cst.SimpleStatementLine):
        return node
    if len(first.body) != 1:
        return node
    expr = first.body[0]
    if not isinstance(expr, cst.Expr):
        return node
    if not isinstance(expr.value, cst.SimpleString | cst.ConcatenatedString):
        return node
    new_body = body.with_changes(body=tuple(body.body[1:]))
    return node.with_changes(body=new_body)
