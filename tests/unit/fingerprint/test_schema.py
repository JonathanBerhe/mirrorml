"""The :class:`Fingerprint` schema is the public contract; these tests lock
construction, immutability, and validation invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mirrorml.fingerprint import SCHEMA_VERSION, Fingerprint, build_fingerprint
from mirrorml.fingerprint.operations import Source
from mirrorml.fingerprint.schema import SchemaDelta, TemporalSemantics, UdfRef


def _minimal_fp() -> Fingerprint:
    return build_fingerprint(
        framework="pandas",
        input_schema=(("uid", "int64"),),
        output_schema=(("uid", "int64"),),
        operations=[Source(op_id="s", name="events", columns=(("uid", "int64"),))],
    )


def test_minimal_construction_succeeds() -> None:
    fp = _minimal_fp()
    assert fp.schema_version == SCHEMA_VERSION
    assert fp.framework == "pandas"
    assert fp.input_schema == (("uid", "int64"),)
    assert fp.output_schema == (("uid", "int64"),)
    assert len(fp.operations) == 1
    assert len(fp.fingerprint_id) == 64


def test_fingerprint_is_frozen() -> None:
    fp = _minimal_fp()
    with pytest.raises(ValidationError):
        fp.framework = "polars"  # type: ignore[misc]


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Fingerprint.model_validate(
            {
                "schema_version": SCHEMA_VERSION,
                "framework": "pandas",
                "input_schema": [["x", "int64"]],
                "output_schema": [["x", "int64"]],
                "operations": [],
                "fingerprint_id": "0" * 64,
                "extra_field": "nope",
            }
        )


def test_schema_version_rejects_non_semver() -> None:
    with pytest.raises(ValidationError):
        Fingerprint(
            schema_version="1.0",  # missing patch
            framework="pandas",
            input_schema=(),
            output_schema=(),
            operations=(),
            fingerprint_id="0" * 64,
        )


def test_fingerprint_id_must_be_64_hex_chars() -> None:
    with pytest.raises(ValidationError):
        Fingerprint(
            schema_version=SCHEMA_VERSION,
            framework="pandas",
            input_schema=(),
            output_schema=(),
            operations=(),
            fingerprint_id="not-a-hash",
        )


def test_unknown_framework_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_fingerprint(
            framework="spark",  # type: ignore[arg-type]
            input_schema=(),
            output_schema=(),
            operations=[],
        )


def test_schema_delta_defaults_to_empty() -> None:
    sd = SchemaDelta()
    assert sd.added == ()
    assert sd.dropped == ()
    assert sd.renamed == ()
    assert sd.coerced == ()


def test_temporal_semantics_all_optional() -> None:
    ts = TemporalSemantics()
    assert ts.closed is None
    assert ts.direction is None
    assert ts.timezone is None
    assert ts.is_point_in_time_safe is None
    assert ts.tolerance is None


def test_udfref_requires_64_hex_source_hash() -> None:
    with pytest.raises(ValidationError):
        UdfRef(qualname="m.f", source_hash="abc", signature="(x)")
    ok = UdfRef(qualname="m.f", source_hash="a" * 64, signature="(x)")
    assert ok.source_hash_algorithm == "libcst-norm-v1"
