"""Top-level fingerprint schema — the public contract.

A :class:`Fingerprint` captures four dimensions along which two pipelines
could disagree (operation graph, parameters, schema effects, temporal
semantics) and exposes :attr:`Fingerprint.fingerprint_id` — a SHA-256 of the
canonical encoding — as a single value safe for equality comparison in CI.

This module is intentionally the project's stable spine. Pydantic models are
``frozen=True, extra="forbid"`` so fingerprints are hashable and forward-
incompatible inputs fail loudly. Sequence fields are :class:`tuple` rather
than :class:`list` to keep models hashable (diff alignment tables key on
them).

Schema changes are breaking by default per ``CLAUDE.md`` § The Fingerprint
Schema. Every change must bump :data:`SCHEMA_VERSION` and add a registered
path in :mod:`mirrorml.fingerprint.migrate`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from mirrorml.fingerprint._typing import ColumnName, Dtype, FingerprintId, OpId
from mirrorml.fingerprint.canonical import canonical_json, canonicalize_operations
from mirrorml.fingerprint.hash import fingerprint_id as compute_fingerprint_id

SCHEMA_VERSION: Final[str] = "1.0.0"
"""Current fingerprint schema version. Bumped on every breaking change to
:class:`Fingerprint`, :class:`Operation`, or canonicalization rules."""

MIN_SUPPORTED_SCHEMA_VERSION: Final[str] = "1.0.0"
"""Oldest fingerprint version this build can load directly. Older documents
must be routed through :func:`mirrorml.fingerprint.migrate.migrate`."""

_PLACEHOLDER_FINGERPRINT_ID: Final[str] = "0" * 64
"""Sentinel id used during the two-phase construction of a fingerprint; the
real id is computed from the canonical body and stamped in via model_copy."""

Framework: TypeAlias = Literal["pandas", "polars", "sql"]

ColumnSpec: TypeAlias = tuple[ColumnName, Dtype]
"""A column name paired with its canonical dtype string."""


class SchemaDelta(BaseModel):
    """Per-operation effect on the table's schema.

    Empty tuples (the default) mean the operation did not change that aspect
    of the schema. Coerced columns are listed separately from renamed
    columns; a rename does not imply a coercion.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    added: tuple[ColumnSpec, ...] = ()
    dropped: tuple[ColumnName, ...] = ()
    renamed: tuple[tuple[ColumnName, ColumnName], ...] = ()
    coerced: tuple[tuple[ColumnName, Dtype, Dtype], ...] = ()


class TemporalSemantics(BaseModel):
    """Temporal semantics attached to operations that have them.

    Fields map directly to divergence-taxonomy categories: ``closed`` →
    ``window_boundary``, ``direction`` → ``as_of_join_direction``,
    ``timezone`` → ``timezone_mismatch``. Diff reads these by name; a free-
    form ``dict`` would not work because the classifier could not reach
    them without string-keyed lookups.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    closed: Literal["left", "right", "both", "neither"] | None = None
    direction: Literal["backward", "forward", "nearest"] | None = None
    timezone: str | None = None
    is_point_in_time_safe: bool | None = None
    tolerance: str | None = None


class UdfRef(BaseModel):
    """Opaque reference to a user-defined function.

    UDF bodies are treated as opaque per ``CLAUDE.md`` § Limits. The fields
    here let diff classify "UDF body differs" or "UDF signature differs"
    without claiming to understand semantics.

    ``source_hash`` is SHA-256 over a normalized libcst rendering of the
    callable (whitespace, comments, and docstrings stripped). The
    normalization algorithm is itself versioned via
    ``source_hash_algorithm`` so a libcst upgrade does not invalidate user
    fingerprints.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    qualname: str
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    signature: str
    is_pure: bool | None = None
    source_hash_algorithm: str = "libcst-norm-v1"


class _OpBase(BaseModel):
    """Shared base for every operation model. INTERNAL — do not instantiate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op_id: OpId
    dependencies: tuple[OpId, ...] = ()
    schema_delta: SchemaDelta = SchemaDelta()


# Import the concrete operation classes *after* the shared building blocks
# are defined, so the operations module can import them at module-load time.
# Python's import machinery resolves this circular pattern because every name
# operations.py needs from this module is defined above this import.
from mirrorml.fingerprint.operations import (  # noqa: E402
    Aggregate,
    AsOfJoin,
    Cast,
    Encode,
    FillNa,
    Filter,
    Join,
    Project,
    Sort,
    Source,
    Udf,
    Window,
)

Operation: TypeAlias = Annotated[
    Source
    | Filter
    | Project
    | Aggregate
    | Join
    | AsOfJoin
    | Window
    | Sort
    | FillNa
    | Cast
    | Encode
    | Udf,
    Field(discriminator="kind"),
]
"""Discriminated union over every supported operation kind. The ``kind``
field on each subclass is the discriminator."""


class Fingerprint(BaseModel):
    """Canonical semantic representation of a feature pipeline.

    Always construct via :func:`build_fingerprint`. The direct constructor
    exists for deserialization (the canonical fingerprint round-trips
    through Pydantic) and trusts the caller's ``fingerprint_id`` — pass a
    wrong value and downstream comparisons silently lie.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$")
    framework: Framework
    input_schema: tuple[ColumnSpec, ...]
    output_schema: tuple[ColumnSpec, ...]
    operations: tuple[Operation, ...]
    fingerprint_id: FingerprintId = Field(pattern=r"^[a-f0-9]{64}$")


def build_fingerprint(
    *,
    framework: Framework,
    input_schema: Iterable[ColumnSpec],
    output_schema: Iterable[ColumnSpec],
    operations: Iterable[Operation],
) -> Fingerprint:
    """Construct a canonical, hashed :class:`Fingerprint`.

    Keyword-only. The function:

    1. Canonicalizes the operation list (validation, structural-hash
       ``op_id`` rewrite, topological order).
    2. Builds a canonical JSON encoding of the body (excluding the
       ``fingerprint_id`` itself).
    3. Computes the SHA-256 ``fingerprint_id`` and stamps it in.
    4. Returns a frozen :class:`Fingerprint`.

    Two callers passing semantically equivalent pipelines get the same
    ``fingerprint_id`` regardless of operation insertion order or tracer-
    assigned ``op_id`` values.

    Examples:
        >>> from mirrorml.fingerprint.operations import Source
        >>> fp = build_fingerprint(
        ...     framework="pandas",
        ...     input_schema=(("uid", "int64"), ("ts", "timestamp[ns, UTC]")),
        ...     output_schema=(("uid", "int64"), ("ts", "timestamp[ns, UTC]")),
        ...     operations=[
        ...         Source(
        ...             op_id="s",
        ...             name="events",
        ...             columns=(("uid", "int64"), ("ts", "timestamp[ns, UTC]")),
        ...         )
        ...     ],
        ... )
        >>> len(fp.fingerprint_id)
        64
        >>> fp.schema_version
        '1.0.0'
    """

    input_schema_t = tuple(input_schema)
    output_schema_t = tuple(output_schema)
    operations_t = tuple(operations)

    canonical_ops = canonicalize_operations(operations_t)

    draft = Fingerprint(
        schema_version=SCHEMA_VERSION,
        framework=framework,
        input_schema=input_schema_t,
        output_schema=output_schema_t,
        operations=canonical_ops,
        fingerprint_id=_PLACEHOLDER_FINGERPRINT_ID,
    )
    body = draft.model_dump(exclude={"fingerprint_id"}, mode="json")
    fid = compute_fingerprint_id(canonical_json(body))

    return draft.model_copy(update={"fingerprint_id": fid})
