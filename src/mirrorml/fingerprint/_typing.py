"""Type aliases shared across the fingerprint package.

:data:`Dtype` is re-exported from :mod:`mirrorml.fingerprint.dtypes` so all
the canonical-vocabulary validation runs transparently when other modules
reference ``Dtype`` here.
"""

from __future__ import annotations

from typing import TypeAlias

from mirrorml.fingerprint.dtypes import Dtype

__all__ = ["ColumnName", "Dtype", "FingerprintId", "OpId"]

OpId: TypeAlias = str
"""Opaque operation identifier. Tracer-assigned initially; replaced by a
deterministic structural hash during canonicalization."""

ColumnName: TypeAlias = str
"""A column name as it appears in the pipeline."""

FingerprintId: TypeAlias = str
"""SHA-256 hex digest of the canonical fingerprint body. 64 lower-case hex characters."""
