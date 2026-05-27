"""Shared pytest configuration."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom command-line flags."""

    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help=(
            "Regenerate golden fixture files in tests/golden/ instead of "
            "comparing against them. Use sparingly: golden updates require an "
            "explicit PR rationale."
        ),
    )


@pytest.fixture
def update_golden(request: pytest.FixtureRequest) -> bool:
    """Whether the suite was invoked with ``--update-golden``."""

    return bool(request.config.getoption("--update-golden"))
