"""Tests for the ``mirrorml`` CLI subcommands.

Uses :class:`typer.testing.CliRunner` to invoke the typer app the same
way an end-user invocation does, without going through a subprocess.
The bench's committed synthetic pairs are reused as fixtures where the
test only needs a real cross-framework pair; bespoke pairs are written
into ``tmp_path`` when the test needs a tailored ``expected_divergences``
list.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from mirrorml.cli.app import app
from mirrorml.fingerprint.schema import Fingerprint

REPO_ROOT = Path(__file__).resolve().parents[3]
IDENTITY_PAIR = (
    REPO_ROOT
    / "bench"
    / "pairs"
    / "synthetic"
    / "identity"
    / "cross_framework_identity_filter_project"
)
AGG_DIVERGENT_PAIR = (
    REPO_ROOT
    / "bench"
    / "pairs"
    / "synthetic"
    / "aggregation_function"
    / "cross_framework_aggregation_function_sum_vs_avg"
)


def _runner() -> CliRunner:
    return CliRunner()


def _write_sql_pair(
    target: Path,
    *,
    name: str,
    category: str,
    offline_sql: str,
    online_sql: str,
    schemas: dict[str, list[tuple[str, str]]],
    expected_divergences: list[dict[str, str]],
) -> None:
    target.mkdir(parents=True)
    (target / "offline.sql").write_text(offline_sql)
    (target / "online.sql").write_text(online_sql)
    schemas_dump = {t: [list(c) for c in cols] for t, cols in schemas.items()}
    meta = {
        "name": name,
        "bucket": "synthetic",
        "category": category,
        "expected_divergences": expected_divergences,
        "offline": {"language": "sql", "source": "offline.sql", "schemas": schemas_dump},
        "online": {"language": "sql", "source": "online.sql", "schemas": schemas_dump},
    }
    with (target / "meta.yaml").open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


# --- trace -------------------------------------------------------------------


def test_trace_emits_fingerprint_json_for_offline_side(tmp_path: Path) -> None:
    runner = _runner()
    result = runner.invoke(app, ["trace", str(IDENTITY_PAIR), "--side", "offline"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    fp = Fingerprint.model_validate(payload)
    assert fp.framework == "pandas"
    assert len(fp.fingerprint_id) == 64


def test_trace_emits_fingerprint_json_for_online_side() -> None:
    runner = _runner()
    result = runner.invoke(app, ["trace", str(IDENTITY_PAIR), "--side", "online"])
    assert result.exit_code == 0, result.stdout
    fp = Fingerprint.model_validate_json(result.stdout)
    assert fp.framework == "sql"


def test_trace_round_trips_to_an_equivalent_fingerprint(tmp_path: Path) -> None:
    """The JSON form must be lossless: serialize a fingerprint, parse it
    back, and the fingerprint_id must match."""

    runner = _runner()
    out = tmp_path / "offline.json"
    result = runner.invoke(app, ["trace", str(IDENTITY_PAIR), "--side", "offline", "-o", str(out)])
    assert result.exit_code == 0
    fp = Fingerprint.model_validate_json(out.read_text())
    # The fingerprint_id is content-derived; if dump/load loses any
    # canonical field the id would not survive a second build.
    redumped = Fingerprint.model_validate_json(fp.model_dump_json())
    assert redumped.fingerprint_id == fp.fingerprint_id


def test_trace_rejects_missing_pair_dir(tmp_path: Path) -> None:
    runner = _runner()
    result = runner.invoke(app, ["trace", str(tmp_path / "nope"), "--side", "offline"])
    assert result.exit_code != 0


def test_trace_writes_to_output_path(tmp_path: Path) -> None:
    runner = _runner()
    out = tmp_path / "fp.json"
    result = runner.invoke(
        app, ["trace", str(IDENTITY_PAIR), "--side", "offline", "--output", str(out)]
    )
    assert result.exit_code == 0
    assert out.is_file()
    # Output path being used means stdout stays empty.
    assert result.stdout.strip() == ""
    Fingerprint.model_validate_json(out.read_text())


# --- diff --------------------------------------------------------------------


def test_diff_exits_zero_for_equivalent_fingerprints(tmp_path: Path) -> None:
    runner = _runner()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    fp_result = runner.invoke(app, ["trace", str(IDENTITY_PAIR), "--side", "offline", "-o", str(a)])
    assert fp_result.exit_code == 0
    b.write_text(a.read_text())

    result = runner.invoke(app, ["diff", str(a), str(b)])
    assert result.exit_code == 0
    assert "equivalent" in result.stdout


def test_diff_exits_nonzero_when_divergent(tmp_path: Path) -> None:
    runner = _runner()
    offline = tmp_path / "offline.json"
    online = tmp_path / "online.json"
    assert (
        runner.invoke(
            app, ["trace", str(AGG_DIVERGENT_PAIR), "--side", "offline", "-o", str(offline)]
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app, ["trace", str(AGG_DIVERGENT_PAIR), "--side", "online", "-o", str(online)]
        ).exit_code
        == 0
    )

    result = runner.invoke(app, ["diff", str(offline), str(online)])
    assert result.exit_code == 1
    assert "aggregation_function" in result.stdout


def test_diff_rejects_invalid_json(tmp_path: Path) -> None:
    runner = _runner()
    bad = tmp_path / "bad.json"
    bad.write_text("{not even valid json")
    good = tmp_path / "good.json"
    assert (
        runner.invoke(
            app, ["trace", str(IDENTITY_PAIR), "--side", "offline", "-o", str(good)]
        ).exit_code
        == 0
    )

    result = runner.invoke(app, ["diff", str(bad), str(good)])
    assert result.exit_code == 2


# --- verify ------------------------------------------------------------------


def test_verify_passes_on_identity_pair() -> None:
    runner = _runner()
    result = runner.invoke(app, ["verify", str(IDENTITY_PAIR)])
    assert result.exit_code == 0, result.stdout
    assert "PASS" in result.stdout


def test_verify_passes_when_expected_matches_predicted() -> None:
    runner = _runner()
    result = runner.invoke(app, ["verify", str(AGG_DIVERGENT_PAIR)])
    assert result.exit_code == 0, result.stdout
    assert "PASS" in result.stdout


def test_verify_fails_on_unexpected_divergence(tmp_path: Path) -> None:
    """A pair that diverges but declares no expected divergences should
    surface as a FAIL (predicted has extras over the empty expected set).
    """

    pair_dir = tmp_path / "bogus_identity"
    _write_sql_pair(
        pair_dir,
        name="bogus_identity",
        category="identity",
        offline_sql="SELECT uid, SUM(score) AS score FROM t GROUP BY uid\n",
        online_sql="SELECT uid, AVG(score) AS score FROM t GROUP BY uid\n",
        schemas={"t": [("uid", "int64"), ("score", "float64")]},
        expected_divergences=[],
    )

    runner = _runner()
    result = runner.invoke(app, ["verify", str(pair_dir)])
    assert result.exit_code == 1
    assert "FAIL" in result.stdout
    assert "extra:" in result.stdout
    assert "aggregation_function" in result.stdout


def test_verify_fails_on_missing_expected(tmp_path: Path) -> None:
    """A pair that does NOT diverge but declares an expected divergence
    should fail with a 'missing' annotation.
    """

    pair_dir = tmp_path / "wrong_expectation"
    _write_sql_pair(
        pair_dir,
        name="wrong_expectation",
        category="aggregation_function",
        offline_sql="SELECT uid FROM t\n",
        online_sql="SELECT uid FROM t\n",
        schemas={"t": [("uid", "int64")]},
        expected_divergences=[{"category": "aggregation_function"}],
    )

    runner = _runner()
    result = runner.invoke(app, ["verify", str(pair_dir)])
    assert result.exit_code == 1
    assert "FAIL" in result.stdout
    assert "missing:" in result.stdout


def test_verify_rejects_malformed_pair(tmp_path: Path) -> None:
    pair_dir = tmp_path / "broken"
    pair_dir.mkdir()
    runner = _runner()
    result = runner.invoke(app, ["verify", str(pair_dir)])
    assert result.exit_code == 2


# --- top-level ---------------------------------------------------------------


def test_version_flag_prints_version_and_exits() -> None:
    runner = _runner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "mirrorml" in result.stdout


def test_no_args_shows_help() -> None:
    """Bare ``mirrorml`` invocation: help text is printed and the process
    exits non-zero (typer's ``no_args_is_help`` convention).
    """

    runner = _runner()
    result = runner.invoke(app, [])
    # typer routes ``no_args_is_help`` through the standard usage-error
    # path; the exact non-zero code is a typer/click implementation
    # detail, so we only assert it is non-zero and the help is rendered.
    assert result.exit_code != 0
    assert "trace" in result.stdout
    assert "diff" in result.stdout
    assert "verify" in result.stdout
