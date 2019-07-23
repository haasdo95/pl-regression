import unittest
from regression import piecewise_linear_regression
import matplotlib.pyplot as plt
import numpy as np


def generate_points(ks, partition):
    assert len(ks) == len(partition)
    anchor_point = 0
    points = []
    for interval_idx, (k, interval) in enumerate(zip(ks, partition)):
        if interval_idx == 0:
            linear_model = lambda x, anchor: k * x
        else:
            linear_model = lambda x, anchor: k * (x - (interval[0] - 0.5)) + anchor_point
        for i in range(interval[0], interval[1]):
            points.append(linear_model(i, anchor_point))
        # CAVEAT: lambda expr always refers to nonlocal variables
        anchor_point = linear_model(interval[1] - 0.5, anchor_point)
    return points


def add_white_noise(points, sd=2):
    points = np.array(points)
    noise = np.random.normal(0, sd, size=points.shape)
    return list(points + noise)


class TestRegression(unittest.TestCase):
    def test_plot(self):
        partition = [(0, 30), (30, 50), (50, 70), (70, 100)]
        ks = [1, 3, -1, -3]
        points = generate_points(ks, partition)
        plt.plot(range(100), points, color='b')
        plt.plot(range(100), add_white_noise(points), color='r')
        plt.show()

    def test_fit_perfect(self):
        partition = [(0, 30), (30, 50), (50, 70), (70, 100)]
        ks = [1, 3, -1, -3]
        points = generate_points(ks, partition)
        optimal_partition, optimal_param = piecewise_linear_regression(points, 4, 15)
        print(optimal_partition)
        self.assertTrue(all([a == b for a, b in zip(optimal_partition, partition)]))
        # a new set of ks
        ks = [0, 2, -3, 0]
        points = generate_points(ks, partition)
        optimal_partition, optimal_param = piecewise_linear_regression(points, 4, 15)
        self.assertTrue(all([a == b for a, b in zip(optimal_partition, partition)]))

    def test_noisy(self):
        partition = [(0, 30), (30, 50), (50, 70), (70, 100)]
        ks = [1, 3, -1, -3]
        points = generate_points(ks, partition)
        plt.plot(range(100), points, color='b')
        noisy_points = add_white_noise(points)
        plt.plot(range(100), noisy_points, color='r')
        optimal_partition, optimal_param = piecewise_linear_regression(noisy_points, 4, 15)
        print(optimal_partition)
        self.assertTrue(all([abs(a[0]-b[0]) <= 5 for a, b in zip(optimal_partition, partition)]))
        plt.show()

    def test_another_noisy(self):
        partition = [(0, 30), (30, 50), (50, 70), (70, 100)]
        ks = [0, 2, -3, 0]
        points = generate_points(ks, partition)
        plt.plot(range(100), points, color='b')
        noisy_points = add_white_noise(points)
        plt.plot(range(100), noisy_points, color='r')
        optimal_partition, optimal_param = piecewise_linear_regression(noisy_points, 4, 15)
        print(optimal_partition)
        self.assertTrue(all([abs(a[0]-b[0]) <= 5 for a, b in zip(optimal_partition, partition)]))
        plt.show()
