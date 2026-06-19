"""End-to-end test for the demo project.

Runs ``demo/check.py`` the way a user would (as a subprocess) and confirms it
exits cleanly, that both the static and statistical layers ran, and that the
statistical layer was not skipped (pandas is present in the test environment).
The script self-asserts internally, so a zero exit code means every static
verdict and every statistical comparison matched expectations.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = REPO_ROOT / "demo" / "check.py"


def test_demo_runs_end_to_end() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "All checks passed" in result.stdout
    # Both layers must have run.
    assert "STATIC" in result.stdout
    assert "STATISTICAL" in result.stdout
    # The statistical layer must not have been skipped (pandas is installed in
    # the dev/test environment via the project extras).
    assert "skipped: pandas is not installed" not in result.stdout
