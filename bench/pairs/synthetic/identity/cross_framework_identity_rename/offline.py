def offline(df):
    return df.rename(columns={'uid': 'user_id'})
