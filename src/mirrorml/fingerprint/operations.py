"""Concrete operation models: one per family.

Operations are grouped by family (e.g. :class:`Aggregate` covers
``groupby.mean``, ``groupby.sum``, ``groupby.agg``) rather than one-per-
pandas-method, so the schema balloon is bounded at ~12 classes. Tracers
normalize their framework-specific operations into these models.

Adding a new family is a breaking schema change (bump
:data:`~mirrorml.fingerprint.schema.SCHEMA_VERSION`); adding fields to an
existing family is additive and minor-version-only.
"""

from __future__ import annotations

from typing import Literal

from mirrorml.fingerprint._typing import ColumnName, Dtype
from mirrorml.fingerprint.schema import ColumnSpec, TemporalSemantics, UdfRef, _OpBase


class Source(_OpBase):
    """A pipeline's input table.

    ``event_time_column`` and ``default_timezone`` are optional; tracers
    populate them when the source has a known time semantics so downstream
    temporal-correctness checks can reason about ``feature_leakage_temporal``.
    """

    kind: Literal["source"] = "source"
    name: str
    columns: tuple[ColumnSpec, ...]
    event_time_column: ColumnName | None = None
    default_timezone: str | None = None


class Filter(_OpBase):
    """Row-wise selection.

    ``predicate`` is a stable string representation produced by the tracer
    (e.g. a normalized SQL WHERE clause or a libcst-rendered pandas
    expression). UDF-based predicates use :class:`UdfRef`.
    """

    kind: Literal["filter"] = "filter"
    predicate: str | UdfRef


class Project(_OpBase):
    """Column subset / reordering. Order is significant."""

    kind: Literal["project"] = "project"
    columns: tuple[ColumnName, ...]


class Aggregate(_OpBase):
    """Group-by aggregation.

    ``aggregations`` is a tuple of ``(output_column, input_column, function)``
    triples. ``input_column`` is ``None`` for row-counting aggregations like
    SQL's ``COUNT(*)`` that do not target a specific column. The function is
    one of the canonical reduction strings (``"count"``, ``"count_distinct"``,
    ``"sum"``, ``"mean"``, ``"min"``, ``"max"``, ``"std"``, ``"var"``,
    ``"median"``, ``"first"``, ``"last"``) or a :class:`UdfRef` for
    user-defined reductions. Time-keyed aggregations populate ``temporal``.
    """

    kind: Literal["aggregate"] = "aggregate"
    by: tuple[ColumnName, ...]
    aggregations: tuple[tuple[ColumnName, ColumnName | None, str | UdfRef], ...]
    temporal: TemporalSemantics | None = None


class Join(_OpBase):
    """Equi-join. The left side dependency precedes the right in
    :attr:`_OpBase.dependencies`.

    For temporal joins, use :class:`AsOfJoin` instead.
    """

    kind: Literal["join"] = "join"
    how: Literal["inner", "left", "right", "outer"]
    left_keys: tuple[ColumnName, ...]
    right_keys: tuple[ColumnName, ...]
    suffix_left: str = ""
    suffix_right: str = "_right"


class AsOfJoin(_OpBase):
    """As-of join (point-in-time correct join on a time column).

    The left side dependency precedes the right in
    :attr:`_OpBase.dependencies`. The ``temporal`` field is required and
    must set ``direction``: the most common source of silent skew in
    feature pipelines.
    """

    kind: Literal["as_of_join"] = "as_of_join"
    left_keys: tuple[ColumnName, ...]
    right_keys: tuple[ColumnName, ...]
    on_time: ColumnName
    suffix_left: str = ""
    suffix_right: str = "_right"
    temporal: TemporalSemantics


class Window(_OpBase):
    """Time-windowed or row-windowed aggregation.

    ``size`` is a string like ``"5min"``, ``"3d"``, or ``"10rows"``,
    parsed by the diff classifier when comparing window sizes across
    framework conventions.
    """

    kind: Literal["window"] = "window"
    over: tuple[ColumnName, ...]
    order_by: tuple[ColumnName, ...]
    size: str
    aggregations: tuple[tuple[ColumnName, ColumnName | None, str | UdfRef], ...]
    temporal: TemporalSemantics


class Sort(_OpBase):
    """Stable sort on a list of ``(column, direction)`` tuples."""

    kind: Literal["sort"] = "sort"
    by: tuple[tuple[ColumnName, Literal["asc", "desc"]], ...]


class FillNa(_OpBase):
    """Null replacement.

    ``value`` is serialized as a string (the tracer renders integers, floats,
    booleans, and strings into a stable form) to avoid Pydantic union-
    discrimination ambiguity in the schema.
    """

    kind: Literal["fill_na"] = "fill_na"
    columns: tuple[ColumnName, ...]
    value: str | None = None
    strategy: Literal["constant", "ffill", "bfill"] | None = None


class Cast(_OpBase):
    """Type coercion on one or more columns."""

    kind: Literal["cast"] = "cast"
    columns: tuple[tuple[ColumnName, Dtype], ...]


class Encode(_OpBase):
    """Categorical encoding.

    Capturing ``categories`` is what lets diff classify
    ``categorical_encoding`` divergences, if the offline pipeline encodes
    against ``{a, b, c}`` and the online pipeline against ``{a, b}``, the
    fingerprints differ at this op.
    """

    kind: Literal["encode"] = "encode"
    columns: tuple[ColumnName, ...]
    method: Literal["one_hot", "label", "target", "hashing"]
    categories: tuple[str, ...] | None = None


class Udf(_OpBase):
    """Opaque user-defined transformation.

    Use this when a tracer cannot reduce a transformation to one of the
    typed op families. The :class:`UdfRef` body lets diff still report
    "UDF changed" without claiming to understand the semantics.
    """

    kind: Literal["udf"] = "udf"
    ref: UdfRef
    input_columns: tuple[ColumnName, ...]
    output_columns: tuple[ColumnName, ...]


class Sample(_OpBase):
    """Random subsampling. Capturing the seed is what lets diff classify
    ``seed_mismatch`` divergences.

    Either ``n`` (absolute row count) or ``fraction`` (0 < f <= 1) is
    populated; the unused field is ``None``. ``seed`` is the random-state
    integer passed by the user; ``None`` means the pipeline did not pin
    a seed, which is itself a divergence vs. a seed-pinned counterpart
    (a non-reproducible sample on one side and a reproducible one on the
    other is a real training-serving skew).
    """

    kind: Literal["sample"] = "sample"
    n: int | None = None
    fraction: float | None = None
    seed: int | None = None
    with_replacement: bool = False
