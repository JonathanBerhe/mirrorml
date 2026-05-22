"""The public API surface is a binding contract — exactly seven names."""

from __future__ import annotations

import mirrorml


def test_all_lists_exactly_the_documented_seven_names() -> None:
    expected = {
        "Divergence",
        "Fingerprint",
        "diff",
        "fingerprint",
        "trace_pandas",
        "trace_polars",
        "trace_sql",
    }
    assert set(mirrorml.__all__) == expected
    assert len(mirrorml.__all__) == 7


def test_every_public_name_is_importable() -> None:
    for name in mirrorml.__all__:
        assert hasattr(mirrorml, name), f"public name {name!r} missing from mirrorml"


def test_version_is_set() -> None:
    assert mirrorml.__version__ == "0.0.1"


def test_pandas_and_polars_are_not_imported_eagerly() -> None:
    """Tracers must stay lazy w.r.t. their target frameworks (CLAUDE.md
    < 200ms import-time budget)."""

    import sys

    assert "pandas" not in sys.modules, (
        "importing mirrorml must not import pandas; tracer must be lazy"
    )
    assert "polars" not in sys.modules, (
        "importing mirrorml must not import polars; tracer must be lazy"
    )
