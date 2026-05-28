def offline(df):
    return df.sample(frac=0.5, random_state=42)
