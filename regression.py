import numpy as np
from sklearn.linear_model import LinearRegression
from functools import reduce
import math

from generator import generate_partitions


# avoid recomputing regression on the same interval
interval_cache = {}


def regression_on_interval(data, interval):
    cache_entry = interval_cache.get(interval, None)
    if cache_entry is not None:
        return cache_entry
    begin, end = interval
    y = np.array(data[begin: end], dtype=np.float)
    X = np.array(range(begin, end), dtype=np.float)
    X = X.reshape(-1, 1)  # needed because sklearn expects a 2d array
    assert X.ndim == 2
    reg = LinearRegression().fit(X, y)
    # compute residual
    yhat = reg.predict(X)
    residual = np.sum((yhat - y) ** 2)  # sum of square error
    # write into cache
    interval_cache[interval] = residual, reg
    return residual, reg


def piecewise_linear_regression(data, num_intervals, min_interval_len):
    print("number of data points: ", len(data))
    assert len(data) >= num_intervals * min_interval_len
    partition_generator = generate_partitions(len(data), num_intervals, min_interval_len)

    lowest_sum_square_error = math.inf
    optimal_partition = None
    optimal_param = None

    for partition in partition_generator:  # a brute-force search for the optimal partition
        interval_reg = [regression_on_interval(data, interval) for interval in partition]  # [(res, reg)]
        summary = reduce(
            lambda acc, x: (acc[0] + x[0], acc[1] + [x[1]]),
            interval_reg, (0, [])
        )
        if summary[0] < lowest_sum_square_error:
            lowest_sum_square_error = summary[0]
            optimal_param = summary[1]
            optimal_partition = partition
    assert optimal_partition is not None
    return optimal_partition, optimal_param
