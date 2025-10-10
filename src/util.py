import numpy as np

def check_invalid_value(x: np.ndarray):
    is_nan = np.isnan(x).any()
    is_posinf = np.isposinf(x).any()
    is_neginf = np.isneginf(x).any()
    is_inf = np.isinf(x).any()
    is_nonfinite = not np.isfinite(x).all()
    is_overflow = np.abs(x).max() > 1e4

    if is_nan or is_posinf or is_neginf or is_inf or is_nonfinite or is_overflow:
        return True
    else:
        return False

def temperal_average(Y: np.array, k: int):
    maxlen = len(Y)
    Y_mean = np.zeros(maxlen)

    for i in range(maxlen):

        idx_start = i - k // 2
        idx_end = i + k // 2

        if idx_end >= maxlen:
            idx_end = maxlen - 1

        if idx_start < 0:
            idx_start = 0

        Y_mean[i] = np.mean(Y[idx_start:idx_end])

    return Y_mean
