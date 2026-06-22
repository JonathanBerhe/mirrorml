"""The public API surface is a binding contract: exactly seven names."""

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
    assert mirrorml.__version__ == "0.1.1"


def test_pandas_and_polars_are_not_imported_eagerly() -> None:
    """Tracers must stay lazy w.r.t. their target frameworks (the < 200ms
    import-time budget).

    Checked in a fresh subprocess so that other tests which legitimately
    import pandas / polars (e.g. the statistical companion check) cannot
    pollute this process's ``sys.modules`` and mask an eager import.
    """

    import subprocess
    import sys

    code = (
        "import sys; import mirrorml; "
        "assert 'pandas' not in sys.modules, 'mirrorml eagerly imported pandas'; "
        "assert 'polars' not in sys.modules, 'mirrorml eagerly imported polars'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
