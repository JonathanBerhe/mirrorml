"""Divergence taxonomy: the closed set of fifteen categories. INTERNAL.

Definitive prose definitions live in ``docs/concepts/divergence_taxonomy.md``.
Adding a category requires: a doc update, a classifier rule in
``diff/classify.py``, and at least five MirrorBench examples. Removing or
renaming a category is a breaking change.

The :data:`DIVERGENCE_CATEGORIES` tuple is the runtime source of truth; the
:data:`DivergenceCategory` ``TypeAlias`` is the static-typing mirror. A unit
test asserts they remain in lockstep.
"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

DivergenceCategory: TypeAlias = Literal[
    "window_boundary",
    "window_size_mismatch",
    "timezone_mismatch",
    "null_handling",
    "categorical_encoding",
    "join_key_mismatch",
    "as_of_join_direction",
    "aggregation_function",
    "type_coercion",
    "ordering_dependence",
    "seed_mismatch",
    "schema_drift",
    "rounding_precision",
    "feature_leakage_temporal",
    "unit_mismatch",
]

DIVERGENCE_CATEGORIES: Final[tuple[DivergenceCategory, ...]] = (
    "window_boundary",
    "window_size_mismatch",
    "timezone_mismatch",
    "null_handling",
    "categorical_encoding",
    "join_key_mismatch",
    "as_of_join_direction",
    "aggregation_function",
    "type_coercion",
    "ordering_dependence",
    "seed_mismatch",
    "schema_drift",
    "rounding_precision",
    "feature_leakage_temporal",
    "unit_mismatch",
)
