def offline(lf, pl):
    # Training: a forward as-of join pulls the feature value from AFTER the
    # label event, leaking future information into the training set.
    features = pl.source(
        "features",
        schema=[("uid", "int64"), ("ts", "timestamp[ns, UTC]"), ("price", "float64")],
    )
    return lf.join_asof(features, on="ts", by="uid", strategy="forward")
