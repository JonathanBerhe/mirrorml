def _double(col):
    return col * 2

def online(df):
    return df.apply(_double)
