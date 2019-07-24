import pandas as pd
import matplotlib.pyplot as plt

from regression import piecewise_linear_regression
from regression import draw_regression_lines


def get_data_loader():
    filename = "dataset.csv"
    df = pd.read_csv(filename)
    for header in df.keys():
        column_vals = df[header]
        column_vals = column_vals.dropna()
        yield header, list(column_vals.values)


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


if __name__ == '__main__':
    dl = get_data_loader()
    data = list(dl)
    axes = plot_series(data)
    for i, (header, series) in enumerate(data):
        print(header)
        op_partition, op_param = piecewise_linear_regression(series, 4, 10)
        print("OPTIMAL PARTITION: ", op_partition)
        ax = plt.subplot(len(data), 1, i + 1)
        draw_regression_lines(ax, series, op_partition, op_param)
    plt.show()
