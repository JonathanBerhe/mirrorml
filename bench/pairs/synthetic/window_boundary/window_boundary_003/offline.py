def offline(lf, pl):
    return lf.rolling(index_column='ts', period='3d', closed='right', group_by='uid').agg(pl.col('score').mean())
