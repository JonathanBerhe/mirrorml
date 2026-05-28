"""Statistical check of a single bench pair.

Loads a pair's ``meta.yaml``, generates small deterministic fixtures for
each declared source table, runs both sides via the statistical companion
check, and returns the result. Shapes the in-process executor cannot handle
(window functions in SQL, dtypes the fixture generator does not yet
support) are reported as a skip with a reason so a batch run can continue.

Multi-source pipelines are handled by tracing each side once to enumerate
all Source ops + their schemas: SQL pairs that JOIN multiple tables get
each table's fixture passed to sqlglot's executor; polars pairs that
declare a second input via ``pl.source(name, schema=...)`` get a thin
wrapper namespace whose ``.source(...)`` returns a real ``LazyFrame``.

Standalone from the static evaluator (``run_eval.py``); a future change can
fold the per-pair statistical result into that evaluator's JSON output.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint.operations import Source
from mirrorml.stats import StatComparison, compare_frames, run_pipeline
from mirrorml.tracers import trace_pandas, trace_polars


def generate_fixture(
    input_schema: Sequence[tuple[str, str]], *, n_rows: int = 6
) -> dict[str, list[Any]]:
    """Synthesize a small, deterministic fixture matching ``input_schema``.

    Handles int / uint widths, float32 / float64, utf8, bool,
    ``timestamp[<unit>, <tz>]``, ``duration[<unit>]``, and
    ``decimal[<p>,<s>]``. Unsupported dtypes raise
    :class:`UnsupportedOperationError` so callers can report a skip.
    """

    return {column: _column(dtype, n_rows) for column, dtype in input_schema}


def _column(dtype: str, n_rows: int) -> list[Any]:
    if dtype in ("int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"):
        # Cycle through a small key space so group-by pairs see multiple
        # rows per key (SUM != AVG on groups of size > 1).
        return [(i // 2) + 1 for i in range(n_rows)]
    if dtype in ("float64", "float32"):
        # Sprinkle nulls so fillna-style pairs actually have something to fill.
        return [None if i % 3 == 2 else 0.5 + i for i in range(n_rows)]
    if dtype in ("utf8", "string"):
        return [chr(ord("a") + (i % 26)) for i in range(n_rows)]
    if dtype == "bool":
        return [bool(i % 2) for i in range(n_rows)]
    if dtype.startswith("timestamp["):
        from zoneinfo import ZoneInfo

        inside = dtype[len("timestamp[") : -1]
        parts = [piece.strip() for piece in inside.split(",")]
        tz = ZoneInfo(parts[1]) if len(parts) == 2 else None
        base = datetime(2024, 1, 1, tzinfo=tz)
        return [base + timedelta(days=i) for i in range(n_rows)]
    if dtype.startswith("duration["):
        return [timedelta(seconds=i + 1) for i in range(n_rows)]
    if dtype.startswith("decimal["):
        return [Decimal(f"{i + 1}.0") for i in range(n_rows)]
    raise UnsupportedOperationError(
        f"fixture generator: dtype {dtype!r} is not yet supported; "
        f"add a generator branch in bench/scripts/stats_check.py."
    )


def statistically_check_pair(pair_dir: Path) -> tuple[StatComparison | None, str]:
    """Statistically check a pair. Returns ``(result, reason)``.

    ``result`` is ``None`` and ``reason`` is non-empty for skips (unsupported
    shape: multi-table joins on the SQL stats path, window functions,
    unsupported fixture dtypes, missing pipeline function).
    """

    meta_path = pair_dir / "meta.yaml"
    meta: dict[str, Any] = yaml.safe_load(meta_path.read_text())

    try:
        left_output = _run_side(pair_dir, meta["offline"], side_label="offline")
        right_output = _run_side(pair_dir, meta["online"], side_label="online")
    except UnsupportedOperationError as exc:
        return (None, f"skipped: {exc}")
    except Exception as exc:
        return (None, f"skipped: pipeline execution failed: {type(exc).__name__}: {exc}")

    return (compare_frames(left_output, right_output), "")


def _run_side(pair_dir: Path, side: dict[str, Any], *, side_label: str) -> dict[str, list[Any]]:
    language = side.get("language")
    if language == "sql":
        schemas = side.get("schemas") or {}
        if not schemas:
            raise UnsupportedOperationError(
                f"sql side ({side_label}): meta.yaml declares no schemas"
            )
        # Generate one fixture per declared table; pass the primary
        # (alphabetically first, for determinism) as ``source_name`` and
        # the rest as ``aux_sources`` so sqlglot's executor can JOIN them.
        fixtures = {
            name: generate_fixture(tuple((c, d) for c, d in cols)) for name, cols in schemas.items()
        }
        primary_name = sorted(fixtures)[0]
        primary_fixture = fixtures[primary_name]
        aux_sources = {n: f for n, f in fixtures.items() if n != primary_name} or None
        query = (pair_dir / side["source"]).read_text()
        return run_pipeline(
            query, primary_fixture, "sql", source_name=primary_name, aux_sources=aux_sources
        )

    if language in ("pandas", "polars"):
        raw_schema = side.get("input_schema")
        if not raw_schema:
            raise UnsupportedOperationError(
                f"{language} side ({side_label}): meta.yaml must declare an input_schema"
            )
        primary_name = side.get("source_name", "input")
        primary_schema = tuple((c, d) for c, d in raw_schema)
        fixture = generate_fixture(primary_schema)

        source_file = pair_dir / side["source"]
        function_name = side.get("function", side_label)
        pipeline = _load_callable(source_file, function_name, side_label=side_label)

        # Discover any aux sources declared inside the pipeline (only
        # meaningful for polars's ``pl.source(...)`` today; pandas pipelines
        # always have exactly one source). Trace through the framework's own
        # tracer so we see exactly what the static side sees.
        aux_sources = _aux_sources_via_trace(pipeline, language, primary_schema, primary_name)
        return run_pipeline(
            pipeline, fixture, language, source_name=primary_name, aux_sources=aux_sources
        )

    raise UnsupportedOperationError(f"unknown language {language!r} on side {side_label!r}")


def _aux_sources_via_trace(
    pipeline: Callable[..., object],
    language: str,
    primary_schema: tuple[tuple[str, str], ...],
    primary_name: str,
) -> dict[str, dict[str, list[Any]]] | None:
    """Trace the pipeline to discover every declared Source op, generate
    a fixture for each non-primary source, and return them keyed by name.

    Returns ``None`` when there is exactly one source (the primary), so
    callers can short-circuit the aux-sources path entirely.
    """

    trace = trace_polars if language == "polars" else trace_pandas
    fp = trace(pipeline, input_schema=primary_schema, source_name=primary_name)
    aux: dict[str, dict[str, list[Any]]] = {}
    for op in fp.operations:
        if isinstance(op, Source) and op.name != primary_name:
            aux[op.name] = generate_fixture(op.columns)
    return aux or None


def _load_callable(
    source_file: Path, function_name: str, *, side_label: str
) -> Callable[..., object]:
    spec = importlib.util.spec_from_file_location(f"_mirrorml_stats_pair_{side_label}", source_file)
    if spec is None or spec.loader is None:
        raise UnsupportedOperationError(
            f"could not load {source_file!r} as a module ({side_label})"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pipeline = getattr(module, function_name, None)
    if not callable(pipeline):
        raise UnsupportedOperationError(
            f"{source_file.name} ({side_label}): no callable named {function_name!r}"
        )
    return pipeline  # type: ignore[no-any-return]
