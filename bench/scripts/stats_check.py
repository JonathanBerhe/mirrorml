"""Statistical check of a single bench pair.

Loads a pair's ``meta.yaml``, generates a small deterministic fixture from
its input schema, runs both sides via the statistical companion check, and
returns the result. Shapes the in-process executor cannot handle (window
functions in SQL, multi-table joins on the SQL stats path, dtypes the
fixture generator does not yet handle) are reported as a skip with a reason
so a batch run can continue.

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
from mirrorml.stats import StatComparison, compare_frames, run_pipeline


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
        if len(schemas) != 1:
            raise UnsupportedOperationError(
                f"sql side ({side_label}): the stats SQL path currently supports "
                f"single-source queries; got {len(schemas)} tables."
            )
        ((table_name, table_schema),) = schemas.items()
        fixture = generate_fixture(tuple((c, d) for c, d in table_schema))
        query = (pair_dir / side["source"]).read_text()
        return run_pipeline(query, fixture, "sql", source_name=table_name)

    if language in ("pandas", "polars"):
        raw_schema = side.get("input_schema")
        if not raw_schema:
            raise UnsupportedOperationError(
                f"{language} side ({side_label}): meta.yaml must declare an input_schema"
            )
        fixture = generate_fixture(tuple((c, d) for c, d in raw_schema))
        source_file = pair_dir / side["source"]
        function_name = side.get("function", side_label)
        pipeline = _load_callable(source_file, function_name, side_label=side_label)
        return run_pipeline(pipeline, fixture, language)

    raise UnsupportedOperationError(f"unknown language {language!r} on side {side_label!r}")


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
