def offline(lf, pl):
    return lf.rolling(index_column='ts', period='7d', closed='left', group_by='uid').agg(pl.col('score').mean())
