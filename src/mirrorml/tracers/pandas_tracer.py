"""pandas tracer entry point.

The current surface is wrapper-object tracing: the user's pipeline runs
against proxy DataFrame / Series / Predicate objects defined in
:mod:`mirrorml.tracers._pandas_wrappers`. The tracer never imports
pandas in phase 1a; the wrappers expose enough of the DataFrame API to
support ``df[col_list]`` projection and ``df[bool_mask]`` filtering with
the usual comparison and boolean operators.

Phase 1a scope mirrors the SQL tracer's phase 1: ``Source``, ``Filter``,
``Project``. Aggregations, joins, sorts, column writes, and dtype
inference from a runtime example DataFrame land in later phases.

Predicate rendering follows SQL form (``=``, ``<>``, ``AND``, ``OR``,
``NOT (...)``) so a pandas Filter and the equivalent SQL Filter produce
byte-identical predicate strings. This is what makes the cross-framework
``diff(pandas_fp, sql_fp) == ()`` claim testable end to end.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint import build_fingerprint
from mirrorml.fingerprint.schema import ColumnSpec, Fingerprint

__all__ = ["trace_pandas"]


def trace_pandas(
    pipeline: Callable[..., object],
    /,
    *,
    input_schema: Iterable[ColumnSpec],
    source_name: str = "input",
) -> Fingerprint:
    """Trace a pandas feature pipeline; return its canonical fingerprint.

    The ``pipeline`` callable is invoked exactly once with a wrapper
    object that quacks like a ``pd.DataFrame`` for the operations the
    tracer supports. Whatever the callable returns must be a wrapped
    frame (i.e., the result of one or more supported operations).

    Args:
        pipeline: A callable taking a DataFrame-like and returning a
            DataFrame-like.
        input_schema: The pipeline's input columns and canonical dtypes.
            Required; the tracer does not introspect runtime DataFrames
            in phase 1a.
        source_name: Name to record on the Source operation. Defaults to
            ``"input"``. Set this to the SQL table name (e.g.,
            ``"events"``) when you want the pandas fingerprint to
            ``diff() == ()`` against an equivalent SQL fingerprint.

    Returns:
        A canonical :class:`~mirrorml.fingerprint.schema.Fingerprint`.

    Raises:
        UnsupportedOperationError: When the pipeline uses operations
            outside the phase 1a surface, references unknown columns,
            or returns a value the tracer cannot interpret.

    Examples:
        >>> def offline(df):
        ...     return df[df["score"] > 0][["uid", "score"]]
        >>> fp = trace_pandas(
        ...     offline,
        ...     input_schema=(("uid", "int64"), ("score", "float64")),
        ...     source_name="events",
        ... )
        >>> fp.framework
        'pandas'
        >>> [op.kind for op in fp.operations]
        ['source', 'filter', 'project']
        >>> fp.output_schema
        (('uid', 'int64'), ('score', 'float64'))
    """

    from mirrorml.tracers._pandas_wrappers import (
        _TraceFrame,
        build_initial_frame,
    )

    input_schema_tuple = tuple(input_schema)
    frame, operations = build_initial_frame(
        source_name=source_name,
        input_schema=input_schema_tuple,
    )

    result = pipeline(frame)

    if not isinstance(result, _TraceFrame):
        raise UnsupportedOperationError(
            f"pandas tracer: pipeline returned {type(result).__name__!r}; "
            f"expected a DataFrame-like wrapper. Did the pipeline reduce "
            f"to a scalar or a Series?"
        )

    return build_fingerprint(
        framework="pandas",
        input_schema=input_schema_tuple,
        output_schema=tuple(result.dtypes.items()),
        operations=operations,
    )
