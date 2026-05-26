"""Tests for the synthetic-pair generator. The generator MUST be
deterministic and reproducible (CLAUDE.md: synthetic pairs cannot be
hand-authored; regenerating must produce identical output)."""

from __future__ import annotations

import filecmp
from pathlib import Path

from bench.scripts.generate_synthetic import SYNTHETIC_DIR, regenerate
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


def test_committed_synthetic_corpus_matches_fresh_regeneration(tmp_path: Path) -> None:
    """The committed ``bench/pairs/synthetic/`` tree must be byte-identical
    to what the current generator produces. If a contributor changes the
    generator without rerunning it, this test catches the stale-corpus
    drift before CI runs the evaluator on outdated pairs."""

    fresh = tmp_path / "fresh"
    regenerate(fresh)

    committed = SYNTHETIC_DIR
    assert committed.exists(), (
        "bench/pairs/synthetic/ is missing; run `uv run python -m bench.scripts.generate_synthetic`"
    )

    def _interesting_files(root: Path) -> set[Path]:
        # __pycache__ entries appear as a side effect of loading pandas
        # pair modules; they are not part of the corpus and are
        # gitignored. Skip them here so the drift guard is not noise.
        return {
            p.relative_to(root)
            for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }

    fresh_files = _interesting_files(fresh)
    committed_files = _interesting_files(committed)

    only_in_fresh = sorted(map(str, fresh_files - committed_files))
    only_in_committed = sorted(map(str, committed_files - fresh_files))
    assert not only_in_fresh and not only_in_committed, (
        f"committed synthetic corpus is out of sync with the generator. "
        f"only_in_fresh={only_in_fresh!r}, only_in_committed={only_in_committed!r}. "
        f"Run `uv run python -m bench.scripts.generate_synthetic` and commit."
    )

    differing: list[str] = []
    for rel in sorted(map(str, fresh_files)):
        if not filecmp.cmp(fresh / rel, committed / rel, shallow=False):
            differing.append(rel)
    assert not differing, (
        f"committed synthetic corpus differs from the generator output in: "
        f"{differing!r}. Run `uv run python -m bench.scripts.generate_synthetic`."
    )
