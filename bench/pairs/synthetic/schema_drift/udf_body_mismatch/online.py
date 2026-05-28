def _triple(col):
    return col * 3

def online(df):
    return df.apply(_triple)
