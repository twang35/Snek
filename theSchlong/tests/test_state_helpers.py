from theSchlong.state_helpers import *
import numpy as np


def test_hwt_no_touching():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 1, 1, 2, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 1), (1, 1)) == [1, 1, 1, 1]
