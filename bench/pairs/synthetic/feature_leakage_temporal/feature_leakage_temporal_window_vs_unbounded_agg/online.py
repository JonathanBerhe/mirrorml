def online(lf, pl):
    return lf.group_by('uid').agg(pl.col('score').mean())
