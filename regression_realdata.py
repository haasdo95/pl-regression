import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path

from regression import piecewise_linear_regression
from regression import draw_regression_lines
from regression import get_residual


def get_data_loader(use_log):
    filename = "daily_data.csv"
    df = pd.read_csv(filename)
    for header in df.keys():
        column_vals = df[header]
        column_vals = column_vals.dropna()
        vals = list(column_vals.values)
        # vals = vals[:133]
        if use_log:
            vals = np.array(vals)
            vals = np.log(vals)
            vals = list(vals)
        yield header, vals


def plot_series(data):
    fig = plt.figure(figsize=(12, 24))
    num_subplots = len(data)
    axes = []
    for i, (header, series) in enumerate(data):
        ax = plt.subplot(num_subplots, 1, i + 1)
        x, y = list(zip(*enumerate(series)))
        ax.set_title(header, x=-0.2, y=0.5)
        ax.scatter(x, y)
        axes.append(ax)
    return axes


def plot_all():
    dl = get_data_loader(use_log=False)
    data = list(dl)
    axes = plot_series(data)
    for i, (header, series) in enumerate(data):
        # plot stuff
        print(header)
        op_partition, op_param = piecewise_linear_regression(series, 3, 100)
        print("OPTIMAL PARTITION: ", op_partition)

        with open("{}_partition.pkl".format(header), "wb") as f:
            pickle.dump(op_partition, f)

        ax = plt.subplot(len(data), 1, i + 1)
        draw_regression_lines(ax, series, op_partition, op_param)
    plt.show()


def plot_residual():
    dl = get_data_loader(use_log=False)
    data = list(dl)
    fig = plt.figure(figsize=(12, 24))
    for i, (header, series) in enumerate(data):
        print(header)
        # no need to recompute if we've done it
        op_partition = None
        fname = "{}_partition.pkl".format(header)
        if os.path.isfile(fname):
            with open(fname, "rb") as f:
                op_partition = pickle.load(f)

        op_partition, op_param = piecewise_linear_regression(series, 3, 10, op_partition)
        print("OPTIMAL PARTITION: ", op_partition)
        # compute residual
        res = get_residual(series, op_partition, op_param)
        print(res)
        ax = plt.subplot(len(data), 1, i + 1)
        ax.set_title(header, x=-0.2, y=0.5)
        ax.scatter(range(len(series)), res)
    plt.show()


if __name__ == '__main__':
    plot_all()
    # plot_residual()
