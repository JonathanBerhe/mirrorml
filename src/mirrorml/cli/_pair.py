"""Pair format + loader. INTERNAL.

A pair is a directory with a ``meta.yaml`` and the source files for the
offline and online pipelines. The loader reads the metadata, traces both
sides, and returns a :class:`Pair` that the CLI (and the bench harness)
can diff.

Lives under ``mirrorml.cli`` because the CLI subcommands (``trace``,
``verify``) are the in-tree consumers; ``bench.scripts.pair`` re-exports
the names so the bench harness keeps working unchanged. Kept private
(underscore prefix) because the pair format is still a developing
contract, not a stable public surface.

``meta.yaml`` schema:

.. code-block:: yaml

    name: timezone_mismatch_001        # unique within the bucket
    bucket: synthetic                  # synthetic | real_world | replayed_bugs
    category: timezone_mismatch        # one of the 15 taxonomy labels
    description: >
      Offline reads events with UTC timestamps; online reads with US/Pacific.
    expected_divergences:
      - category: timezone_mismatch
    offline:
      language: sql                    # sql | pandas | polars
      source: offline.sql              # filename relative to pair dir
      schemas:                         # required for SQL pairs
        events:
          - [ts, "timestamp[ns, UTC]"]
    online:
      language: polars                 # pandas / polars trace a Python function
      source: online.py                # Python module relative to pair dir
      function: online                 # function name to look up + trace
      input_schema:                    # required for pandas / polars pairs
        - [ts, "timestamp[ns, US/Pacific]"]
      source_name: events              # optional; matches FROM table for cross-framework parity
    generator:                         # synthetic only
      module: bench.scripts.generate_synthetic
      version: 1
    source_url: ...                    # real_world only
    postmortem_url: ...                # replayed_bugs only
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mirrorml.fingerprint.schema import Fingerprint
from mirrorml.tracers import trace_pandas, trace_polars, trace_sql


@dataclass(frozen=True)
class ExpectedDivergence:
    """One row of the expected-divergences list in ``meta.yaml``.

    Currently only ``category`` is matched; ``detail_pattern`` is a
    forward-compatible field for substring or regex matching when the
    evaluator gets more discriminating.
    """

    category: str
    detail_pattern: str | None = None


@dataclass(frozen=True)
class Pair:
    """A loaded pair, ready for diffing."""

    name: str
    bucket: str
    category: str
    description: str
    offline: Fingerprint
    online: Fingerprint
    expected: tuple[ExpectedDivergence, ...]
    path: Path
    source_url: str | None = None
    postmortem_url: str | None = None


# Provenance required per bucket: a real_world pair must cite the public
# source it was mined from, a replayed_bugs pair the public postmortem it
# reconstructs. This guards the benchmark's credibility (an unsourced
# "real" pair is worse than none).
_REQUIRED_PROVENANCE: dict[str, str] = {
    "real_world": "source_url",
    "replayed_bugs": "postmortem_url",
}


def load_pair(pair_dir: Path) -> Pair:
    """Read a pair directory and return its loaded :class:`Pair`.

    Raises :class:`ValueError` for missing required fields, unknown
    languages, or other malformed metadata. The error message names the
    pair so reasonable batch processing can continue past one bad pair.
    """

    meta_path = pair_dir / "meta.yaml"
    if not meta_path.is_file():
        raise ValueError(f"pair {pair_dir}: missing meta.yaml")

    with meta_path.open() as f:
        meta: dict[str, Any] = yaml.safe_load(f)

    for required in ("name", "bucket", "category", "offline", "online"):
        if required not in meta:
            raise ValueError(f"pair {pair_dir}: meta.yaml missing required field {required!r}")

    bucket = str(meta["bucket"])
    provenance_field = _REQUIRED_PROVENANCE.get(bucket)
    if provenance_field is not None and not meta.get(provenance_field):
        raise ValueError(
            f"pair {pair_dir}: {bucket!r} pairs must declare a non-empty "
            f"{provenance_field!r} in meta.yaml"
        )

    offline_fp = _trace_side(pair_dir, meta["offline"], side_label="offline")
    online_fp = _trace_side(pair_dir, meta["online"], side_label="online")

    expected_raw = meta.get("expected_divergences", []) or []
    expected = tuple(
        ExpectedDivergence(
            category=e["category"],
            detail_pattern=e.get("detail_pattern"),
        )
        for e in expected_raw
    )

    source_url = meta.get("source_url")
    postmortem_url = meta.get("postmortem_url")

    return Pair(
        name=str(meta["name"]),
        bucket=bucket,
        category=str(meta["category"]),
        description=str(meta.get("description", "")).strip(),
        offline=offline_fp,
        online=online_fp,
        expected=expected,
        path=pair_dir,
        source_url=str(source_url) if source_url else None,
        postmortem_url=str(postmortem_url) if postmortem_url else None,
    )


def _trace_side(pair_dir: Path, side: dict[str, Any], *, side_label: str = "?") -> Fingerprint:
    language = side.get("language")
    if language == "sql":
        source_file = pair_dir / side["source"]
        if not source_file.is_file():
            raise ValueError(
                f"pair {pair_dir}: {side_label} source file {source_file.name!r} not found"
            )
        query = source_file.read_text()
        raw_schemas = side.get("schemas") or {}
        schemas = {table: tuple(tuple(col) for col in cols) for table, cols in raw_schemas.items()}
        dialect = side.get("dialect")
        return trace_sql(query, schemas=schemas, dialect=dialect)

    if language in ("pandas", "polars"):
        source_file = pair_dir / side["source"]
        if not source_file.is_file():
            raise ValueError(
                f"pair {pair_dir}: {side_label} source file {source_file.name!r} not found"
            )
        function_name = side.get("function", side_label)
        function = _load_python_function(source_file, function_name, side_label=side_label)
        raw_schema = side.get("input_schema")
        if not raw_schema:
            raise ValueError(
                f"pair {pair_dir}: {side_label} ({language}) meta.yaml must declare an "
                f"input_schema list"
            )
        input_schema = tuple(tuple(col) for col in raw_schema)
        source_name = side.get("source_name", "input")
        if language == "pandas":
            return trace_pandas(function, input_schema=input_schema, source_name=source_name)
        return trace_polars(function, input_schema=input_schema, source_name=source_name)

    raise ValueError(
        f"pair {pair_dir}: {side_label} has unknown language {language!r}; "
        f"expected 'sql', 'pandas', or 'polars'"
    )


def _load_python_function(
    source_file: Path, function_name: str, *, side_label: str
) -> Callable[..., object]:
    """Load ``function_name`` from a Python file via importlib.

    Pair pipelines are loaded as modules so each pair file is a real
    Python module (with its own namespace, imports, etc.). The module is
    named with the pair-side label so import errors point back at the
    offending file.
    """

    spec = importlib.util.spec_from_file_location(f"_mirrorml_bench_pair_{side_label}", source_file)
    if spec is None or spec.loader is None:
        raise ValueError(
            f"pair source file {source_file!r} ({side_label}): "
            f"importlib could not build a module spec"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, function_name):
        raise ValueError(
            f"pair source file {source_file.name!r} ({side_label}): "
            f"no function named {function_name!r} found"
        )
    function = getattr(module, function_name)
    if not callable(function):
        raise ValueError(
            f"pair source file {source_file.name!r} ({side_label}): "
            f"{function_name!r} is not callable"
        )
    return function  # type: ignore[no-any-return]


def discover_pairs(root: Path) -> list[Path]:
    """Yield every pair directory under ``root``.

    A directory counts as a pair iff it contains a ``meta.yaml``. The
    walk is deterministic (sorted) so evaluator output is reproducible.
    """

    pairs: list[Path] = []
    if not root.exists():
        return pairs
    for path in sorted(root.rglob("meta.yaml")):
        pairs.append(path.parent)
    return pairs
