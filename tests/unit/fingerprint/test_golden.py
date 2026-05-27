"""Golden regression for a minimal fingerprint.

The checked-in golden file is the canonical-JSON encoding of the simplest
possible fingerprint (single :class:`~mirrorml.fingerprint.operations.Source`
op). If this byte string ever changes (because the schema changed, the
canonicalizer changed, or the hash algorithm changed), this test fails
loudly. Updates require an explicit PR rationale.

Regenerate locally with ``pytest --update-golden``.
"""

from __future__ import annotations

from pathlib import Path

from mirrorml.fingerprint import Fingerprint, build_fingerprint
from mirrorml.fingerprint.canonical import canonical_json
from mirrorml.fingerprint.operations import Source

GOLDEN_PATH = (
    Path(__file__).resolve().parents[2] / "golden" / "fingerprint" / "minimal_source_only.json"
)


def _build_minimal() -> Fingerprint:
    return build_fingerprint(
        framework="pandas",
        input_schema=(("uid", "int64"), ("ts", "timestamp[ns, UTC]")),
        output_schema=(("uid", "int64"), ("ts", "timestamp[ns, UTC]")),
        operations=[
            Source(
                op_id="src",
                name="events",
                columns=(("uid", "int64"), ("ts", "timestamp[ns, UTC]")),
                event_time_column="ts",
                default_timezone="UTC",
            )
        ],
    )


def test_minimal_fingerprint_matches_golden(update_golden: bool) -> None:
    fp = _build_minimal()
    body = fp.model_dump(mode="json")
    canonical = canonical_json(body)

    if update_golden or not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_bytes(canonical + b"\n")

    expected = GOLDEN_PATH.read_bytes().rstrip(b"\n")
    assert canonical == expected, (
        "Canonical fingerprint diverged from "
        f"{GOLDEN_PATH.name}. If intentional, regenerate with "
        "`pytest --update-golden` and document the rationale in the PR."
    )


def test_minimal_fingerprint_round_trips_through_pydantic() -> None:
    fp = _build_minimal()
    dumped = fp.model_dump(mode="json")
    reloaded = Fingerprint.model_validate(dumped)
    assert reloaded == fp
    assert reloaded.fingerprint_id == fp.fingerprint_id
