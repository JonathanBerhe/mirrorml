"""SQL JOIN: INNER / LEFT / RIGHT / FULL OUTER with ON, multi-way chaining,
table aliases, suffix-on-collision schema combination, and sided ON-key
resolution."""

from __future__ import annotations

import pytest

from mirrorml import trace_sql
from mirrorml.exceptions import UnsupportedOperationError
from mirrorml.fingerprint.operations import Join

EVENTS = (("uid", "int64"), ("score", "float64"))
USERS = (("uid", "int64"), ("country", "utf8"))
PROFILES = (("uid", "int64"), ("score", "float64"))
ORDERS = (("oid", "int64"), ("uid", "int64"), ("amount", "float64"))


# --- basic kinds -------------------------------------------------------------


@pytest.mark.parametrize(
    "sql_kind,canonical",
    [
        ("JOIN", "inner"),
        ("INNER JOIN", "inner"),
        ("LEFT JOIN", "left"),
        ("LEFT OUTER JOIN", "left"),
        ("RIGHT JOIN", "right"),
        ("RIGHT OUTER JOIN", "right"),
        ("FULL OUTER JOIN", "outer"),
        ("FULL JOIN", "outer"),
    ],
)
def test_join_kind_mapping(sql_kind: str, canonical: str) -> None:
    fp = trace_sql(
        f"SELECT events.uid FROM events {sql_kind} users ON events.uid = users.uid",
        schemas={"events": EVENTS, "users": USERS},
    )
    join = next(op for op in fp.operations if op.kind == "join")
    assert isinstance(join, Join)
    assert join.how == canonical


def test_join_emits_two_sources_and_one_join() -> None:
    fp = trace_sql(
        "SELECT events.uid FROM events JOIN users ON events.uid = users.uid",
        schemas={"events": EVENTS, "users": USERS},
    )
    kinds = [op.kind for op in fp.operations]
    assert kinds.count("source") == 2
    assert kinds.count("join") == 1


# --- ON keys: single and multi -----------------------------------------------


def test_single_equi_join_keys() -> None:
    fp = trace_sql(
        "SELECT events.uid FROM events JOIN users ON events.uid = users.uid",
        schemas={"events": EVENTS, "users": USERS},
    )
    join = next(op for op in fp.operations if op.kind == "join")
    assert isinstance(join, Join)
    assert join.left_keys == ("uid",)
    assert join.right_keys == ("uid",)


def test_multi_key_equi_join() -> None:
    fp = trace_sql(
        "SELECT a.k FROM a JOIN b ON a.k = b.k AND a.t = b.t",
        schemas={
            "a": (("k", "int64"), ("t", "int64"), ("v", "float64")),
            "b": (("k", "int64"), ("t", "int64"), ("v", "float64")),
        },
    )
    join = next(op for op in fp.operations if op.kind == "join")
    assert isinstance(join, Join)
    assert join.left_keys == ("k", "t")
    assert join.right_keys == ("k", "t")


def test_reversed_on_order_resolves_to_same_keys() -> None:
    """``a.x = b.x`` and ``b.x = a.x`` are semantically equivalent and must
    fingerprint identically."""

    fp_a = trace_sql(
        "SELECT events.uid FROM events JOIN users ON events.uid = users.uid",
        schemas={"events": EVENTS, "users": USERS},
    )
    fp_b = trace_sql(
        "SELECT events.uid FROM events JOIN users ON users.uid = events.uid",
        schemas={"events": EVENTS, "users": USERS},
    )
    assert fp_a.fingerprint_id == fp_b.fingerprint_id


def test_non_equi_join_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="equi"):
        trace_sql(
            "SELECT a.x FROM a JOIN b ON a.x > b.x",
            schemas={"a": (("x", "int64"),), "b": (("x", "int64"),)},
        )


def test_unqualified_ambiguous_join_key_rejected() -> None:
    """An unqualified column that exists on both sides cannot be resolved."""

    with pytest.raises(UnsupportedOperationError, match="ambiguous"):
        trace_sql(
            "SELECT events.uid FROM events JOIN users ON uid = uid",
            schemas={"events": EVENTS, "users": USERS},
        )


def test_unqualified_unambiguous_join_keys_resolve() -> None:
    """When each side has a uniquely-named column referenced in ON, the
    walker can resolve without qualifiers."""

    fp = trace_sql(
        "SELECT oid FROM orders JOIN users ON owner_id = country_uid",
        schemas={
            "orders": (("oid", "int64"), ("owner_id", "int64")),
            "users": (("country_uid", "int64"), ("country", "utf8")),
        },
    )
    join = next(op for op in fp.operations if op.kind == "join")
    assert isinstance(join, Join)
    assert join.left_keys == ("owner_id",)
    assert join.right_keys == ("country_uid",)


def test_unknown_qualifier_in_on_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="qualifier"):
        trace_sql(
            "SELECT events.uid FROM events JOIN users ON unknown.uid = users.uid",
            schemas={"events": EVENTS, "users": USERS},
        )


def test_on_clause_required() -> None:
    """Plain ``JOIN`` without ``ON`` should be rejected, separately from
    CROSS / USING which have their own messages."""

    with pytest.raises(UnsupportedOperationError):
        trace_sql(
            "SELECT * FROM events JOIN users",
            schemas={"events": EVENTS, "users": USERS},
        )


# --- table aliases -----------------------------------------------------------


def test_table_alias_in_from_and_join() -> None:
    fp = trace_sql(
        "SELECT e.uid FROM events AS e JOIN users AS u ON e.uid = u.uid",
        schemas={"events": EVENTS, "users": USERS},
    )
    sources = [op for op in fp.operations if op.kind == "source"]
    names = sorted(source.name for source in sources)
    assert names == ["events", "users"]


def test_alias_resolution_in_on_clause() -> None:
    fp = trace_sql(
        "SELECT e.uid FROM events AS e LEFT JOIN users AS u ON e.uid = u.uid",
        schemas={"events": EVENTS, "users": USERS},
    )
    join = next(op for op in fp.operations if op.kind == "join")
    assert isinstance(join, Join)
    assert join.how == "left"
    assert join.left_keys == ("uid",)
    assert join.right_keys == ("uid",)


def test_alias_or_base_name_in_on_both_work() -> None:
    fp_alias = trace_sql(
        "SELECT e.uid FROM events AS e JOIN users AS u ON e.uid = u.uid",
        schemas={"events": EVENTS, "users": USERS},
    )
    fp_base = trace_sql(
        "SELECT events.uid FROM events JOIN users ON events.uid = users.uid",
        schemas={"events": EVENTS, "users": USERS},
    )
    assert fp_alias.fingerprint_id == fp_base.fingerprint_id


# --- schema combination + suffix on collision --------------------------------


def test_select_star_after_inner_join_combines_schemas() -> None:
    fp = trace_sql(
        "SELECT * FROM events JOIN users ON events.uid = users.uid",
        schemas={"events": EVENTS, "users": USERS},
    )
    assert fp.output_schema == (
        ("uid", "int64"),
        ("score", "float64"),
        ("uid_right", "int64"),
        ("country", "utf8"),
    )


def test_collision_columns_get_right_suffix() -> None:
    """``events`` and ``profiles`` both have ``uid`` and ``score`` -- after
    join the right's collisions get ``_right`` appended."""

    fp = trace_sql(
        "SELECT * FROM events JOIN profiles ON events.uid = profiles.uid",
        schemas={"events": EVENTS, "profiles": PROFILES},
    )
    assert fp.output_schema == (
        ("uid", "int64"),
        ("score", "float64"),
        ("uid_right", "int64"),
        ("score_right", "float64"),
    )


def test_non_colliding_right_columns_keep_original_names() -> None:
    fp = trace_sql(
        "SELECT * FROM events JOIN orders ON events.uid = orders.uid",
        schemas={"events": EVENTS, "orders": ORDERS},
    )
    names = [name for name, _ in fp.output_schema]
    # left first: uid, score; then right (uid -> uid_right collision, oid, amount)
    assert names == ["uid", "score", "oid", "uid_right", "amount"]


# --- multi-way joins ---------------------------------------------------------


def test_three_way_join_produces_chained_join_ops() -> None:
    fp = trace_sql(
        "SELECT events.uid FROM events "
        "JOIN users ON events.uid = users.uid "
        "JOIN orders ON events.uid = orders.uid",
        schemas={"events": EVENTS, "users": USERS, "orders": ORDERS},
    )
    kinds = [op.kind for op in fp.operations]
    assert kinds.count("source") == 3
    assert kinds.count("join") == 2


def test_three_way_join_output_schema_combines_all_sides() -> None:
    fp = trace_sql(
        "SELECT * FROM a JOIN b ON a.k = b.k JOIN c ON a.k = c.k",
        schemas={
            "a": (("k", "int64"), ("av", "int64")),
            "b": (("k", "int64"), ("bv", "int64")),
            "c": (("k", "int64"), ("cv", "int64")),
        },
    )
    names = [name for name, _ in fp.output_schema]
    assert names == ["k", "av", "k_right", "bv", "k_right_right", "cv"]


# --- missing schemas / unknown tables ---------------------------------------


def test_missing_right_schema_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="users"):
        trace_sql(
            "SELECT * FROM events JOIN users ON events.uid = users.uid",
            schemas={"events": EVENTS},
        )


def test_cross_join_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="CROSS"):
        trace_sql(
            "SELECT * FROM a CROSS JOIN b",
            schemas={
                "a": (("x", "int64"),),
                "b": (("x", "int64"),),
            },
        )


def test_using_clause_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="USING"):
        trace_sql(
            "SELECT * FROM events JOIN users USING (uid)",
            schemas={"events": EVENTS, "users": USERS},
        )


# --- combined with other features --------------------------------------------


def test_join_with_where_filter_uses_post_join_schema() -> None:
    fp = trace_sql(
        "SELECT events.uid FROM events JOIN users ON events.uid = users.uid WHERE country = 'US'",
        schemas={"events": EVENTS, "users": USERS},
    )
    kinds = [op.kind for op in fp.operations]
    # Source, Source, Join, Filter, Project
    assert kinds == ["source", "source", "join", "filter", "project"]


def test_join_with_group_by_aggregation() -> None:
    fp = trace_sql(
        "SELECT country, COUNT(*) AS n "
        "FROM events JOIN users ON events.uid = users.uid "
        "GROUP BY country",
        schemas={"events": EVENTS, "users": USERS},
    )
    kinds = [op.kind for op in fp.operations]
    assert kinds == ["source", "source", "join", "aggregate"]
    assert fp.output_schema == (("country", "utf8"), ("n", "int64"))


def test_join_with_full_pipeline() -> None:
    """Full canonical pipeline order: Source * 2 -> Join -> Filter (WHERE) ->
    Aggregate (GROUP BY) -> Filter (HAVING) -> Sort (ORDER BY)."""

    fp = trace_sql(
        "SELECT country, AVG(score) AS avg_score "
        "FROM events JOIN users ON events.uid = users.uid "
        "WHERE score > 0 "
        "GROUP BY country "
        "HAVING AVG(score) > 0.5 "
        "ORDER BY avg_score DESC",
        schemas={"events": EVENTS, "users": USERS},
    )
    assert [op.kind for op in fp.operations] == [
        "source",
        "source",
        "join",
        "filter",
        "aggregate",
        "filter",
        "sort",
    ]
    assert fp.output_schema == (("country", "utf8"), ("avg_score", "float64"))
