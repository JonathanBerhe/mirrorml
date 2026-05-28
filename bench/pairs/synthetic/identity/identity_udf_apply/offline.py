def _double(col):
    return col * 2

def offline(df):
    return df.apply(_double)
