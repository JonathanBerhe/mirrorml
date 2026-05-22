"""The four stub callables raise NotImplementedError with milestone-tagged
messages. Verifying the tag prevents the message from drifting to a less
informative form."""

from __future__ import annotations

import pytest

from mirrorml import diff, trace_pandas, trace_polars, trace_sql


def test_trace_pandas_stub_carries_m2_message() -> None:
    with pytest.raises(NotImplementedError, match="M2"):
        trace_pandas(lambda x: x)


def test_trace_polars_stub_carries_m2_message() -> None:
    with pytest.raises(NotImplementedError, match="M2"):
        trace_polars(lambda x: x)


def test_trace_sql_stub_carries_m2_message() -> None:
    with pytest.raises(NotImplementedError, match="M2"):
        trace_sql("SELECT 1")


def test_diff_stub_carries_m3_message() -> None:
    """``diff`` cannot be invoked without two real fingerprints. We only need
    to verify the stub message, so pass dummy positional args via ``pytest.raises``
    after constructing a minimal fingerprint."""

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
