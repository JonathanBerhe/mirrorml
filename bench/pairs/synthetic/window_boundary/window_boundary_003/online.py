def online(lf, pl):
    return lf.rolling(index_column='ts', period='3d', closed='none', group_by='uid').agg(pl.col('score').mean())
