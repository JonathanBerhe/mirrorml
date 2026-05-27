def online(df):
    # Serving: the failed RPC propagates the -1 error sentinel as the value.
    return df.fillna({"score": -1})
