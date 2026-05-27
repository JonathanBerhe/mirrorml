"""Tests for the bench pair format + loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from bench.scripts.pair import discover_pairs, load_pair


def _write_pair(
    target: Path,
    *,
    name: str,
    category: str,
    offline_sql: str,
    online_sql: str,
    offline_schemas: dict[str, list[tuple[str, str]]],
    online_schemas: dict[str, list[tuple[str, str]]],
    expected_divergences: list[dict[str, str]],
    bucket: str = "synthetic",
    extra_meta: dict[str, object] | None = None,
) -> None:
    target.mkdir(parents=True)
    (target / "offline.sql").write_text(offline_sql)
    (target / "online.sql").write_text(online_sql)
    meta: dict[str, object] = {
        "name": name,
        "bucket": bucket,
        "category": category,
        "description": "test pair",
        "expected_divergences": expected_divergences,
        "offline": {
            "language": "sql",
            "source": "offline.sql",
            "schemas": {t: [list(c) for c in cols] for t, cols in offline_schemas.items()},
        },
        "online": {
            "language": "sql",
            "source": "online.sql",
            "schemas": {t: [list(c) for c in cols] for t, cols in online_schemas.items()},
        },
    }
    if extra_meta:
        meta.update(extra_meta)
    with (target / "meta.yaml").open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def _kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "p",
        "category": "identity",
        "offline_sql": "SELECT uid FROM t\n",
        "online_sql": "SELECT uid FROM t\n",
        "offline_schemas": {"t": [("uid", "int64")]},
        "online_schemas": {"t": [("uid", "int64")]},
        "expected_divergences": [],
    }
    base.update(overrides)
    return base


def test_real_world_pair_without_source_url_is_rejected(tmp_path: Path) -> None:
    pair_dir = tmp_path / "rw"
    _write_pair(pair_dir, bucket="real_world", **_kwargs())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source_url"):
        load_pair(pair_dir)


def test_real_world_pair_with_source_url_loads(tmp_path: Path) -> None:
    pair_dir = tmp_path / "rw"
    url = "https://github.com/example/repo/blob/main/features.py"
    _write_pair(
        pair_dir,
        bucket="real_world",
        extra_meta={"source_url": url},
        **_kwargs(),  # type: ignore[arg-type]
    )
    pair = load_pair(pair_dir)
    assert pair.bucket == "real_world"
    assert pair.source_url == url


def test_replayed_bug_without_postmortem_url_is_rejected(tmp_path: Path) -> None:
    pair_dir = tmp_path / "rb"
    _write_pair(pair_dir, bucket="replayed_bugs", **_kwargs())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="postmortem_url"):
        load_pair(pair_dir)


def test_replayed_bug_with_postmortem_url_loads(tmp_path: Path) -> None:
    pair_dir = tmp_path / "rb"
    url = "https://engineering.example.com/postmortem/skew-2021"
    _write_pair(
        pair_dir,
        bucket="replayed_bugs",
        extra_meta={"postmortem_url": url},
        **_kwargs(),  # type: ignore[arg-type]
    )
    pair = load_pair(pair_dir)
    assert pair.postmortem_url == url


def test_synthetic_pair_needs_no_provenance(tmp_path: Path) -> None:
    pair_dir = tmp_path / "syn"
    _write_pair(pair_dir, bucket="synthetic", **_kwargs())  # type: ignore[arg-type]
    pair = load_pair(pair_dir)
    assert pair.source_url is None
    assert pair.postmortem_url is None


def test_load_pair_round_trips_simple_sql_pair(tmp_path: Path) -> None:
    pair_dir = tmp_path / "sample"
    _write_pair(
        pair_dir,
        name="sample",
        category="identity",
        offline_sql="SELECT uid FROM t\n",
        online_sql="SELECT uid FROM t\n",
        offline_schemas={"t": [("uid", "int64")]},
        online_schemas={"t": [("uid", "int64")]},
        expected_divergences=[],
    )

    pair = load_pair(pair_dir)
    assert pair.name == "sample"
    assert pair.category == "identity"
    assert pair.offline.framework == "sql"
    assert pair.online.framework == "sql"
    assert pair.offline.fingerprint_id == pair.online.fingerprint_id
    assert pair.expected == ()


def test_load_pair_carries_expected_divergence_categories(tmp_path: Path) -> None:
    pair_dir = tmp_path / "sample"
    _write_pair(
        pair_dir,
        name="sample",
        category="timezone_mismatch",
        offline_sql="SELECT ts FROM t\n",
        online_sql="SELECT ts FROM t\n",
        offline_schemas={"t": [("ts", "timestamp[ns, UTC]")]},
        online_schemas={"t": [("ts", "timestamp[ns, US/Pacific]")]},
        expected_divergences=[{"category": "timezone_mismatch"}],
    )

    pair = load_pair(pair_dir)
    assert len(pair.expected) == 1
    assert pair.expected[0].category == "timezone_mismatch"


def test_load_pair_rejects_missing_meta(tmp_path: Path) -> None:
    pair_dir = tmp_path / "broken"
    pair_dir.mkdir()
    with pytest.raises(ValueError, match=r"meta\.yaml"):
        load_pair(pair_dir)


def test_load_pair_rejects_missing_required_field(tmp_path: Path) -> None:
    pair_dir = tmp_path / "broken"
    pair_dir.mkdir()
    (pair_dir / "meta.yaml").write_text("name: x\nbucket: synthetic\ncategory: identity\n")
    with pytest.raises(ValueError, match="offline"):
        load_pair(pair_dir)


def test_load_pair_rejects_unknown_language(tmp_path: Path) -> None:
    pair_dir = tmp_path / "broken"
    pair_dir.mkdir()
    (pair_dir / "meta.yaml").write_text(
        """
name: x
bucket: synthetic
category: identity
offline: { language: cobol, source: o.cobol, schemas: {} }
online: { language: cobol, source: o.cobol, schemas: {} }
"""
    )
    with pytest.raises(ValueError, match="unknown language"):
        load_pair(pair_dir)


def test_load_pair_rejects_missing_source_file(tmp_path: Path) -> None:
    pair_dir = tmp_path / "broken"
    pair_dir.mkdir()
    (pair_dir / "meta.yaml").write_text(
        """
name: x
bucket: synthetic
category: identity
offline:
  language: sql
  source: missing.sql
  schemas: {t: [[x, int64]]}
online:
  language: sql
  source: missing.sql
  schemas: {t: [[x, int64]]}
"""
    )
    with pytest.raises(ValueError, match="not found"):
        load_pair(pair_dir)


def test_discover_pairs_finds_nested_pairs(tmp_path: Path) -> None:
    """A bucket root contains category subdirectories, each containing
    pair directories. ``discover_pairs`` should find every leaf
    directory that has a ``meta.yaml``, regardless of depth.
    """

    _write_pair(
        tmp_path / "category_a" / "pair_001",
        name="pair_001",
        category="category_a",
        offline_sql="SELECT uid FROM t\n",
        online_sql="SELECT uid FROM t\n",
        offline_schemas={"t": [("uid", "int64")]},
        online_schemas={"t": [("uid", "int64")]},
        expected_divergences=[],
    )
    _write_pair(
        tmp_path / "category_b" / "pair_002",
        name="pair_002",
        category="category_b",
        offline_sql="SELECT uid FROM t\n",
        online_sql="SELECT uid FROM t\n",
        offline_schemas={"t": [("uid", "int64")]},
        online_schemas={"t": [("uid", "int64")]},
        expected_divergences=[],
    )

    found = discover_pairs(tmp_path)
    assert len(found) == 2
    assert tmp_path / "category_a" / "pair_001" in found
    assert tmp_path / "category_b" / "pair_002" in found


def test_discover_pairs_returns_empty_for_missing_root(tmp_path: Path) -> None:
    assert discover_pairs(tmp_path / "does_not_exist") == []


def test_discover_pairs_is_sorted_for_determinism(tmp_path: Path) -> None:
    for cat in ("z_last", "a_first", "m_middle"):
        _write_pair(
            tmp_path / cat / "p",
            name=f"{cat}_pair",
            category=cat,
            offline_sql="SELECT uid FROM t\n",
            online_sql="SELECT uid FROM t\n",
            offline_schemas={"t": [("uid", "int64")]},
            online_schemas={"t": [("uid", "int64")]},
            expected_divergences=[],
        )

    found = discover_pairs(tmp_path)
    cat_order = [p.parent.name for p in found]
    assert cat_order == sorted(cat_order)
