"""Serving-time feature pipeline for the demo churn model.

This is the "online" side: SQL that runs against the warehouse to produce the
same feature at request time. ``CORRECT`` mirrors the offline pandas pipeline
exactly. The ``SKEW_*`` queries are realistic mistakes a developer might
introduce on the serving side; check.py uses them to show MirrorML catching
each one.
"""

# Mirrors offline_features.average_order_value exactly. MirrorML should report
# these two pipelines as equivalent.
CORRECT = (
    "SELECT customer_id, AVG(amount) AS amount FROM orders WHERE amount > 0 GROUP BY customer_id"
)

# Realistic bug 1: the serving query sums where training averages. The model
# would be trained on mean order value and served total order value.
SKEW_AGGREGATION = (
    "SELECT customer_id, SUM(amount) AS amount FROM orders WHERE amount > 0 GROUP BY customer_id"
)

# Realistic bug 2: the serving query forgot the validity filter the training
# side applies, so refunds and zero-amount rows leak into the average.
SKEW_MISSING_FILTER = "SELECT customer_id, AVG(amount) AS amount FROM orders GROUP BY customer_id"
