def offline(lf, pl):
    return lf.filter(pl.col('score') > 0).select('uid', 'score')
