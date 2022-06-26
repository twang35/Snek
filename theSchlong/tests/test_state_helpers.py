from state_helpers import *

# food: 1, head: 2, body: 3, wall: 4, tail: 5 in these tests


# =============================== head_with_tail tests ===============================
def test_hwt_no_touching():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 1), (1, 1), 'right') == [1, 1, 1]


def test_hwt_following_forward_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 2, 5, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # don't move tail because it could potentially not move if the head moves into food
    assert head_with_tail(grid, (3, 0), (4, 0), 'right') == [0, 0, 1]


def test_hwt_following_right_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 2, 5, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # don't move tail because it could potentially not move if the head moves into food
    assert head_with_tail(grid, (3, 0), (4, 0), 'up') == [0, 1, 0]


def test_hwt_following_left_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 2, 3, 4],
                     [4, 0, 0, 3, 5, 3, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 0), (3, 1), 'left') == [1, 0, 0]


def test_hwt_no_forward():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 3, 4],
                     [4, 5, 3, 3, 2, 0, 4],
                     [4, 0, 0, 0, 0, 3, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 1), (0, 1), 'right') == [1, 1, 0]


def test_hwt_no_left():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 0, 3, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 1), (1, 1), 'right') == [0, 1, 1]


def test_hwt_no_right():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 0, 3, 0, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 1), (1, 1), 'right') == [1, 0, 1]


def test_hwt_follow_tail_and_empty_forward():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 2, 2, 0, 0, 4],
                     [4, 0, 2, 2, 3, 5, 4],
                     [4, 0, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 1), (4, 1), 'up') == [0, 1, 1]


def test_hwt_multiple_open_groups():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 2, 2, 0, 0, 4],
                     [4, 0, 0, 0, 3, 5, 4],
                     [4, 0, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert head_with_tail(grid, (3, 1), (4, 1), 'left') == [0, 1, 0]


# =============================== body_and_wall_collisions tests ===============================
def test_bw_left_wall():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 1, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert body_and_wall_collisions(grid, (3, 0), (1, 0), 'right') == [0, 1, 1]


def test_bw_right_wall():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 1, 0, 0, 0, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert body_and_wall_collisions(grid, (3, 1), (1, 1), 'right') == [1, 0, 1]


def test_bw_forward_wall():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 1, 0, 0, 0, 4],
                     [4, 0, 5, 3, 3, 2, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert body_and_wall_collisions(grid, (4, 1), (1, 1), 'right') == [1, 1, 0]


def test_bw_left_food():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 1, 0, 4],
                     [4, 0, 5, 3, 2, 3, 4],
                     [4, 0, 0, 0, 3, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert body_and_wall_collisions(grid, (3, 1), (1, 1), 'right') == [1, 0, 0]


def test_bw_right_food():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 3, 0, 4],
                     [4, 0, 5, 3, 2, 3, 4],
                     [4, 0, 0, 0, 1, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert body_and_wall_collisions(grid, (3, 1), (1, 1), 'right') == [0, 1, 0]


def test_bw_forward_food():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 3, 0, 4],
                     [4, 0, 5, 3, 2, 1, 4],
                     [4, 0, 0, 0, 3, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert body_and_wall_collisions(grid, (3, 1), (1, 1), 'right') == [0, 0, 1]


def test_bw_left_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 3, 3, 5, 0, 4],
                     [4, 0, 3, 3, 2, 3, 4],
                     [4, 0, 0, 0, 3, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert body_and_wall_collisions(grid, (3, 1), (3, 0), 'right') == [1, 0, 0]


def test_bw_right_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 3, 0, 4],
                     [4, 0, 3, 3, 2, 3, 4],
                     [4, 0, 0, 0, 5, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert body_and_wall_collisions(grid, (3, 1), (3, 2), 'right') == [0, 1, 0]


def test_bw_forward_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 3, 0, 4],
                     [4, 0, 3, 3, 2, 5, 4],
                     [4, 0, 0, 0, 3, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert body_and_wall_collisions(grid, (3, 1), (4, 1), 'right') == [0, 0, 1]

