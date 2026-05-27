def offline(df):
    return df.fillna({'score': 0})
