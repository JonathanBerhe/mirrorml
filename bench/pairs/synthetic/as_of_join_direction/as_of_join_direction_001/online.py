def online(lf, pl):
    prices = pl.source('prices', schema=[('uid', 'int64'), ('ts', 'timestamp[ns, UTC]'), ('price', 'float64')])
    return lf.join_asof(prices, on='ts', by='uid', strategy='nearest')
