"""Remaining stub callables raise NotImplementedError with milestone-tagged
messages. Verifying the tag prevents the message from drifting to a less
informative form. ``trace_sql`` (M2 phase 1), ``trace_pandas`` (M2 phase
1a), and ``diff`` (M3 phase 1) are now real; see their dedicated test
directories. Only the Polars tracer remains a stub."""

from __future__ import annotations

import pytest

from mirrorml import trace_polars


def test_trace_polars_stub_carries_m2_message() -> None:
    with pytest.raises(NotImplementedError, match="M2"):
        trace_polars(lambda x: x)
