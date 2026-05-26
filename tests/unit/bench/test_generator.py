"""Tests for the synthetic-pair generator. The generator MUST be
deterministic and reproducible (CLAUDE.md: synthetic pairs cannot be
hand-authored; regenerating must produce identical output)."""

from __future__ import annotations

from pathlib import Path

from bench.scripts.generate_synthetic import regenerate
from bench.scripts.pair import load_pair


def test_generator_is_idempotent(tmp_path: Path) -> None:
    """Running the generator twice on the same directory must produce
    byte-identical output."""

    first = tmp_path / "first"
    second = tmp_path / "second"
    regenerate(first)
    regenerate(second)

    first_files = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    second_files = sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
    assert first_files == second_files

    for rel in first_files:
        assert (first / rel).read_bytes() == (second / rel).read_bytes(), (
            f"generator non-determinism on {rel}"
        )


def test_generator_emits_at_least_one_identity_pair(tmp_path: Path) -> None:
    """Identity pairs (no expected divergences) are essential for the
    precision metric; without them the bench only measures recall."""

    regenerate(tmp_path)
    identity_dir = tmp_path / "identity"
    assert identity_dir.is_dir()
    pairs = [load_pair(p.parent) for p in identity_dir.rglob("meta.yaml")]
    assert len(pairs) >= 1
    for p in pairs:
        assert p.expected == ()


def test_generated_pairs_cover_each_targeted_category(tmp_path: Path) -> None:
    """Every category the diff engine detects should have at least one
    generated pair so a regression in any one of them surfaces in the
    evaluator output."""

    regenerate(tmp_path)
    categories_with_pairs = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    targeted = {
        "schema_drift",
        "type_coercion",
        "timezone_mismatch",
        "rounding_precision",
        "aggregation_function",
        "join_key_mismatch",
        "ordering_dependence",
    }
    missing = targeted - categories_with_pairs
    assert not missing, f"generator missing pairs for: {sorted(missing)}"


def test_every_generated_pair_loads_cleanly(tmp_path: Path) -> None:
    """Smoke test: the generator's output must be a valid bench corpus
    that the loader can parse without errors."""

    regenerate(tmp_path)
    for meta_path in tmp_path.rglob("meta.yaml"):
        pair = load_pair(meta_path.parent)
        assert pair.offline.fingerprint_id
        assert pair.online.fingerprint_id
