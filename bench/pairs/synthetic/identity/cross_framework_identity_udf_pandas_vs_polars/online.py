def _identity(df):
    return df

def online(lf, pl):
    return lf.map_batches(_identity)
