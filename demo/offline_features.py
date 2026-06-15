"""Training-time feature pipeline for the demo churn model.

This is the "offline" side. It runs over a historical export of the orders
table (a pandas DataFrame) while the churn model is being trained.

The feature is simple on purpose: the average value of each customer's valid
orders. The online side (online_features.py) recomputes the same feature in
SQL at serving time, and check.py uses MirrorML to confirm the two agree.
"""

# The orders table's columns and their canonical dtypes. Both the offline and
# online pipelines read from this same schema, so the comparison is apples to
# apples. ts is carried on the table even though this feature does not
# aggregate on it, which is realistic: the column is there, and a timezone
# mismatch on it is exactly the kind of subtle skew MirrorML can catch.
ORDERS_SCHEMA = (
    ("customer_id", "int64"),
    ("amount", "float64"),
    ("ts", "timestamp[ns, UTC]"),
)


def average_order_value(df):
    """Average value of each customer's valid (positive-amount) orders."""
    valid_orders = df[df["amount"] > 0]
    return valid_orders.groupby("customer_id").agg({"amount": "mean"})
