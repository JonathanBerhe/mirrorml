"""Polars tracer entry point.

Like the pandas tracer, this is a wrapper-object tracer: the user's
pipeline runs against a proxy ``LazyFrame`` and a tracing expression
namespace defined in :mod:`mirrorml.tracers._polars_wrappers`. The tracer
never imports the real ``polars`` package; the proxies expose enough of
the Polars API to support ``filter`` / ``select`` / ``group_by().agg()``
/ ``rename`` over ``pl.col(...)`` expressions.

Phase 1 scope mirrors the pandas tracer's phase 1a/1b: ``Source``,
``Filter``, ``Project``, ``Aggregate``. Predicate and aggregation
rendering is shared with the pandas tracer (see
:mod:`mirrorml.tracers._trace_common`) so a Polars pipeline and the
equivalent pandas or SQL pipeline produce fingerprints that
``diff() == ()``.

Unlike :func:`~mirrorml.tracers.trace_pandas`, the pipeline takes *two*
arguments: the proxy frame and the ``pl`` namespace. Polars expressions
are namespace-level (``pl.col("x")``), not frame-attached, so the
namespace is passed in rather than imported globally.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint import build_fingerprint
from mirrorml.fingerprint.schema import ColumnSpec, Fingerprint

__all__ = ["trace_polars"]


def trace_polars(
    pipeline: Callable[..., object],
    /,
    *,
    input_schema: Iterable[ColumnSpec],
    source_name: str = "input",
) -> Fingerprint:
    """Trace a Polars feature pipeline; return its canonical fingerprint.

    The ``pipeline`` callable is invoked exactly once as
    ``pipeline(frame, pl)`` where ``frame`` is a proxy ``LazyFrame`` and
    ``pl`` is a tracing namespace exposing ``col`` and ``lit``. Whatever
    the callable returns must be a wrapped frame (the result of one or
    more supported operations).

    Args:
        pipeline: A callable ``(frame, pl) -> frame``.
        input_schema: The pipeline's input columns and canonical dtypes.
            Required; the tracer does not introspect runtime frames.
        source_name: Name to record on the Source operation. Defaults to
            ``"input"``. Set this to the SQL table name (e.g. ``"events"``)
            when you want the Polars fingerprint to ``diff() == ()``
            against an equivalent SQL fingerprint.

    Returns:
        A canonical :class:`~mirrorml.fingerprint.schema.Fingerprint`.

    Raises:
        UnsupportedOperationError: When the pipeline uses operations
            outside the phase 1 surface, references unknown columns, or
            returns a value the tracer cannot interpret.

    Examples:
        >>> def offline(lf, pl):
        ...     return lf.filter(pl.col("score") > 0).group_by("uid").agg(pl.col("score").mean())
        >>> fp = trace_polars(
        ...     offline,
        ...     input_schema=(("uid", "int64"), ("score", "float64")),
        ...     source_name="events",
        ... )
        >>> fp.framework
        'polars'
        >>> [op.kind for op in fp.operations]
        ['source', 'filter', 'aggregate']
        >>> fp.output_schema
        (('uid', 'int64'), ('score', 'float64'))
    """

    from mirrorml.tracers._polars_wrappers import (
        _TraceExprNamespace,
        _TraceLazyFrame,
        build_initial_frame,
    )

    input_schema_tuple = tuple(input_schema)
    frame, operations = build_initial_frame(
        source_name=source_name,
        input_schema=input_schema_tuple,
    )

    result = pipeline(frame, _TraceExprNamespace(operations))

    if not isinstance(result, _TraceLazyFrame):
        raise UnsupportedOperationError(
            f"polars tracer: pipeline returned {type(result).__name__!r}; "
            f"expected a LazyFrame-like wrapper. Did the pipeline reduce to "
            f"a scalar or forget to return the frame? The pipeline must take "
            f"two arguments: (frame, pl)."
        )

    return build_fingerprint(
        framework="polars",
        input_schema=input_schema_tuple,
        output_schema=tuple(result.dtypes.items()),
        operations=operations,
    )
