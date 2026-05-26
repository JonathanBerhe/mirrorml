def offline(df):
    return df.groupby('uid').agg({'score': 'sum'})
