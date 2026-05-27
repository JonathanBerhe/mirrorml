def online(lf, pl):
    # Serving: a backward as-of join uses the last value at-or-before the
    # event, the point-in-time-correct direction.
    features = pl.source(
        "features",
        schema=[("uid", "int64"), ("ts", "timestamp[ns, UTC]"), ("price", "float64")],
    )
    return lf.join_asof(features, on="ts", by="uid", strategy="backward")
