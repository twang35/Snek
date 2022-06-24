from state_helpers import *


# food: 1, body: 2, head: 3, wall: 4, tail: 5 in these tests
def test_hwt_no_touching():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 5, 2, 3, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 1), (1, 1)) == [1, 1, 1, 1]


def test_hwt_following_right_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 2, 3, 5, 4],
                     [4, 0, 0, 2, 2, 2, 4],
                     [4, 0, 0, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # don't move tail because it could potentially not move if the head moves into food
    assert head_with_tail(grid, (3, 0), (4, 0)) == [0, 1, 0, 0]


def test_hwt_following_tail_straight():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 2, 2, 2, 4],
                     [4, 0, 0, 2, 3, 5, 4],
                     [4, 0, 0, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # don't move tail because it could potentially not move if the head moves into food
    assert head_with_tail(grid, (3, 1), (4, 1)) == [0, 1, 0, 0]


def test_hwt_following_down_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 2, 3, 2, 4],
                     [4, 0, 0, 2, 5, 2, 4],
                     [4, 0, 0, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 0), (3, 1)) == [0, 0, 0, 1]


def test_hwt_no_left():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 2, 0, 0, 4],
                     [4, 5, 2, 2, 3, 0, 4],
                     [4, 0, 0, 2, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 1), (0, 1)) == [0, 1, 1, 1]


def test_hwt_no_right():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 2, 4],
                     [4, 5, 2, 2, 3, 0, 4],
                     [4, 0, 0, 0, 0, 2, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 1), (0, 1)) == [1, 0, 1, 1]


def test_hwt_no_up():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 2, 0, 2, 4],
                     [4, 0, 5, 2, 3, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 1), (1, 1)) == [1, 1, 0, 1]


def test_hwt_follow_tail_and_empty_up():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 2, 2, 0, 0, 4],
                     [4, 0, 2, 2, 3, 5, 4],
                     [4, 0, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 1), (4, 1)) == [0, 1, 1, 0]


def test_hwt_follow_tail_and_empty_left():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 2, 2, 0, 0, 4],
                     [4, 0, 0, 0, 3, 5, 4],
                     [4, 0, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 1), (4, 1)) == [0, 1, 1, 0]
