def _double(col):
    # multiply by two
    """Double the column."""
    return col * 2

def online(df):
    return df.apply(_double)
