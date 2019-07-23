import unittest
from generator import generate_partitions


class TestGenerator(unittest.TestCase):
    def test_base(self):
        intervals = list(generate_partitions(5, 1, 3))
        print("trivial intervals: ", intervals)
        self.assertTrue(len(intervals) == 1)
        self.assertTrue(intervals[0][0] == (0, 5))

    def test_two_partitions(self):
        intervals = list(generate_partitions(7, 2, 3))
        print(intervals)
        self.assertTrue(len(intervals) == 2)
        intervals = list(generate_partitions(5, 2, 2))
        print(intervals)
        self.assertTrue(len(intervals) == 2)

    def test_more_partitions(self):
        intervals = list(generate_partitions(7, 3, 2))
        print(intervals)
        self.assertTrue(len(intervals) == 3)
        intervals = list(generate_partitions(9, 4, 2))
        print(intervals)
        self.assertTrue(len(intervals) == 4)


if __name__ == '__main__':
    unittest.main()
