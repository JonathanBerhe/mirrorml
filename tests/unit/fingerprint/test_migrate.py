"""Migrate behavior in v1.0.0: validates the current version; rejects others
with an actionable error."""

from __future__ import annotations

import pytest

from mirrorml.exceptions import FingerprintVersionError
from mirrorml.fingerprint import SCHEMA_VERSION, build_fingerprint
from mirrorml.fingerprint.migrate import migrate
from mirrorml.fingerprint.operations import Source


def _round_tripable_dict() -> dict[str, object]:
    fp = build_fingerprint(
        framework="pandas",
        input_schema=(("x", "int64"),),
        output_schema=(("x", "int64"),),
        operations=[Source(op_id="s", name="t", columns=(("x", "int64"),))],
    )
    return fp.model_dump(mode="json")


def test_migrate_accepts_current_version() -> None:
    raw = _round_tripable_dict()
    fp = migrate(raw)
    assert fp.schema_version == SCHEMA_VERSION


def test_migrate_rejects_missing_version_field() -> None:
    raw = _round_tripable_dict()
    raw.pop("schema_version")
    with pytest.raises(FingerprintVersionError, match="missing"):
        migrate(raw)


def test_migrate_rejects_unsupported_version() -> None:
    raw = _round_tripable_dict()
    raw["schema_version"] = "0.9.0"
    with pytest.raises(FingerprintVersionError, match="older"):
        migrate(raw)


def test_migrate_rejects_future_version_when_no_path_registered() -> None:
    raw = _round_tripable_dict()
    raw["schema_version"] = "2.0.0"
    with pytest.raises(FingerprintVersionError, match="cannot be migrated"):
        migrate(raw)
