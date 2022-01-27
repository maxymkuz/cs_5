import numpy as np

# WMAPE metric
def wmape(y, y_hat):
    return np.abs((y - y_hat)).sum() / y.sum()
# RMSE metric
def rmse(y, y_hat):
    return np.sqrt(((y - y_hat) ** 2).mean())