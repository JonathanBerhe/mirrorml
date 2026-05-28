def _identity(df):
    return df

def offline(df):
    return df.apply(_identity)
