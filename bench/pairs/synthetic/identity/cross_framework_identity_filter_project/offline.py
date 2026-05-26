def offline(df):
    return df[df['score'] > 0][['uid', 'score']]
