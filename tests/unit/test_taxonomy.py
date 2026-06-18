"""The 15-category divergence taxonomy is closed; the static type and the
runtime tuple must stay in lockstep."""

from __future__ import annotations

from typing import get_args

from mirrorml._taxonomy import DIVERGENCE_CATEGORIES, DivergenceCategory


def test_taxonomy_has_exactly_fifteen_categories() -> None:
    assert len(DIVERGENCE_CATEGORIES) == 15
    assert len(set(DIVERGENCE_CATEGORIES)) == 15  # no duplicates


def test_taxonomy_tuple_matches_literal() -> None:
    """DIVERGENCE_CATEGORIES and DivergenceCategory must agree.

    They are two views of the same closed set: one runtime-iterable, one
    static-typeable. Drift between them is a silent landmine; this test is
    the trip-wire.
    """

    assert set(get_args(DivergenceCategory)) == set(DIVERGENCE_CATEGORIES)


def test_taxonomy_order_is_documented() -> None:
    """The tuple's ordering is the documented order, so reviewers can scan
    it linearly."""

    expected = (
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
    assert expected == DIVERGENCE_CATEGORIES
