def online(df):
    return df[df['score'] > 0][['uid', 'score']]
