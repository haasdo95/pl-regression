import numpy as np
from sklearn.linear_model import LinearRegression
from functools import reduce
import math

from generator import generate_partitions


def regression_on_interval(data, interval, interval_cache):
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


def piecewise_linear_regression(data, num_intervals, min_interval_len, op_partition=None):
    print("number of data points: ", len(data))
    assert len(data) >= num_intervals * min_interval_len
    # avoid recomputing regression on the same interval
    interval_cache = {}

    if op_partition is None:
        partition_generator = generate_partitions(len(data), num_intervals, min_interval_len)
    else:
        partition_generator = [op_partition]

    lowest_sum_square_error = math.inf
    optimal_partition = None
    optimal_param = None

    for partition in partition_generator:  # a brute-force search for the optimal partition
        interval_reg = [regression_on_interval(data, interval, interval_cache) for interval in partition]  # [(res, reg)]
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


def get_residual(data, optimal_partition, optimal_param):
    residual = []
    for par, reg in zip(optimal_partition, optimal_param):
        start, end = par
        for x in range(start, end):
            y_hat = predict_one_point(x, reg)
            residual.append(data[x] - y_hat)
    return residual


def predict_one_point(x, reg):
    x = np.reshape(x, (-1, 1))
    y = reg.predict(x)
    return y.item()


def draw_regression_lines(ax, series, optimal_partition, optimal_param):
    param = [(reg.coef_, reg.intercept_) for reg in optimal_param]
    nodes = [(0, param[0][1])]  # put in first intercept
    # inner nodes
    for i in range(len(param) - 1):
        node_x = (optimal_partition[i][1] + optimal_partition[i+1][0]) / 2
        node_y = predict_one_point(node_x, optimal_param[i])
        nodes.append((node_x, node_y))
    # fill in last node
    node_x = len(series) - 1
    node_y = predict_one_point(node_x, optimal_param[-1])
    nodes.append((node_x, node_y))
    assert len(nodes) == len(optimal_param) + 1
    # draw lines, actually
    for i in range(len(param)):
        x1, y1 = nodes[i]
        x2, y2 = nodes[i+1]
        ax.plot([x1, x2], [y1, y2], marker='o', color='g')
