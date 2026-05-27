def offline(df):
    # Training: missing scores fall back to the legitimate default 0.
    return df.fillna({"score": 0})
