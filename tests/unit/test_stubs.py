"""Stub callables raise NotImplementedError with milestone-tagged messages.
Verifying the tag prevents the message from drifting to a less informative
form. ``trace_sql`` is no longer a stub as of M2 phase 1; see
``tests/unit/tracers/sql/`` for its tests."""

from __future__ import annotations

import pytest

from mirrorml import diff, trace_pandas, trace_polars


def test_trace_pandas_stub_carries_m2_message() -> None:
    with pytest.raises(NotImplementedError, match="M2"):
        trace_pandas(lambda x: x)


def test_trace_polars_stub_carries_m2_message() -> None:
    with pytest.raises(NotImplementedError, match="M2"):
        trace_polars(lambda x: x)


def test_diff_stub_carries_m3_message() -> None:
    """``diff`` needs two real fingerprints to exercise its signature; we
    construct a minimal one to invoke it."""

    from mirrorml import fingerprint
    from mirrorml.fingerprint.operations import Source

    fp = fingerprint(
        framework="pandas",
        input_schema=(("x", "int64"),),
        output_schema=(("x", "int64"),),
        operations=[Source(op_id="s", name="src", columns=(("x", "int64"),))],
    )
    with pytest.raises(NotImplementedError, match="M3"):
        diff(fp, fp)
