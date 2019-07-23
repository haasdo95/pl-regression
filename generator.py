"""
This file implements a generator that will generate all possibilities
of partitioning a sequence of length $data_len$ into $num_intervals$ left-closed right-open intervals,
each with length at least $min_interval_len$
"""


def generate_partitions(data_len, num_intervals, min_interval_len):
    """
    a generator of left-closed right-open intervals between 0 and data_len
    """
    yield from _generate_partitions_helper(0, data_len, num_intervals, min_interval_len)


def _generate_partitions_helper(start_idx, data_len, num_intervals, min_interval_len):
    """
    :param start_idx: beginning of the interval
    :param data_len: length of remaining data
    :param num_intervals: number of intervals wanted
    :param min_interval_len: shortest an interval can be
    :return: a generator of intervals starting from start_idx
    """
    if num_intervals == 1 and data_len >= min_interval_len:
        yield [(start_idx, start_idx + data_len)]  # base case: simply return everything remaining
    else:
        # beyond this point there's no need to recurse any further since there's no room
        free_space = data_len - (num_intervals - 1) * min_interval_len
        for i in range(start_idx + min_interval_len, start_idx + free_space + 1):
            new_interval = (start_idx, i)
            # recurse with a short sequence
            for possibility in _generate_partitions_helper(i, data_len - (i - start_idx),
                                                           num_intervals - 1, min_interval_len):
                yield [new_interval] + possibility
