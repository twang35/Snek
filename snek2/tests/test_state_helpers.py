from state_helpers import *

# food: 1, head: 2, body: 3, wall: 4, tail: 5 in these tests
#
# 5 is not a value the real grid ever holds — Snake._rebuild_grid() writes the tail as body
# (3). It is used here only to make the tail visible in the fixtures, and is_open() treats it
# exactly like body, so it behaves as an occupied cell either way.
#
# These tests were stale for two signature generations. group_obs used to take a food position
# and return [head_with_tail, head_with_food, num_groups] per action; head_with_food was
# dropped from the observation (snake_environment.observation_spec sets head_with_food_obs = 0)
# and with it the argument. Signature now:
#
#     group_obs(grid, head_pos, tail_pos, head_move_dir) -> [head_with_tail, num_groups] * 3
#
# ordered by ACTIONS = ['left', 'right', 'forward'].
#
# The expectations also moved, because update_grid() now frees the tail cell: the snake
# advances as a whole, so the tail vacates as the head takes a new cell. Two consequences show
# up repeatedly below. Where the tail was the only link between two open regions, freeing it
# merges them and num_groups *drops*. Where the tail sat in a dead end, freeing it leaves a
# one-cell region and num_groups *rises*. The exception is a move that eats: add_segment()
# refills the tile the tail came from, so it stays occupied — see
# test_eating_move_does_not_free_the_tail.


# =============================== group_obs tests ===============================
def test_hwt_no_touching():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 1, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # Open board: every action leaves one region, and the freed tail keeps head and tail
    # sharing it.
    assert group_obs(grid, (3, 1), (1, 1), 'right') == [1, log2plus1(1),
                                                       1, log2plus1(1),
                                                       1, log2plus1(1)]


def test_hwt_no_touching_eats_food():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 5, 3, 2, 1, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # 'forward' eats the food at (4, 1), so the tail stays put. The board is open enough that
    # it makes no difference to the counts here.
    assert group_obs(grid, (3, 1), (1, 1), 'right') == [1, log2plus1(1),
                                                       1, log2plus1(1),
                                                       1, log2plus1(1)]


def test_hwt_following_forward_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 2, 5, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # The tail at (4, 0) is walled in by its own body, so vacating it leaves a one-cell region
    # and the count goes to 2 — except on 'forward', which moves into that very cell.
    assert group_obs(grid, (3, 0), (4, 0), 'right') == [0, log2plus1(2),
                                                       0, log2plus1(2),
                                                       1, log2plus1(1)]


def test_hwt_following_right_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 2, 5, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # Same board, facing up, so 'right' is the move onto the tail.
    assert group_obs(grid, (3, 0), (4, 0), 'up') == [0, log2plus1(2),
                                                    1, log2plus1(1),
                                                    0, log2plus1(2)]


def test_hwt_following_left_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 2, 3, 4],
                     [4, 0, 0, 3, 5, 3, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 0), (3, 1), 'left') == [1, log2plus1(1),
                                                      0, log2plus1(2),
                                                      0, log2plus1(2)]


def test_hwt_no_forward():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 3, 4],
                     [4, 5, 3, 3, 2, 0, 4],
                     [4, 0, 0, 0, 0, 3, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # The tail at (0, 1) is the only gap in the body wall, so freeing it merges the strip above
    # with everything below: 3 regions become 2, and on 'forward' 2 become 1. This is the case
    # the old behaviour got wrong most often in real play.
    assert group_obs(grid, (3, 1), (0, 1), 'right') == [1, log2plus1(2),
                                                       1, log2plus1(2),
                                                       0, log2plus1(1)]


def test_hwt_no_left():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 0, 3, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 1, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 1), (1, 1), 'right') == [0, log2plus1(1),
                                                       1, log2plus1(3),
                                                       1, log2plus1(2)]


def test_hwt_no_right():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 0, 3, 0, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 1), (1, 1), 'right') == [1, log2plus1(3),
                                                       0, log2plus1(1),
                                                       1, log2plus1(2)]


def test_hwt_follow_tail_and_empty_forward_no_food():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 2, 2, 0, 0, 4],
                     [4, 1, 2, 2, 3, 5, 4],
                     [4, 0, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 1), (4, 1), 'up') == [0, log2plus1(2),
                                                    1, log2plus1(2),
                                                    1, log2plus1(2)]


def test_hwt_multiple_open_groups_separate_food_and_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 2, 2, 0, 0, 4],
                     [4, 0, 0, 0, 3, 5, 4],
                     [4, 1, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 1), (4, 1), 'left') == [0, log2plus1(2),
                                                      1, log2plus1(2),
                                                      0, log2plus1(2)]


def test_groups_new_group_left():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 2, 2, 3, 5, 4],
                     [4, 1, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 1), (4, 1), 'right') == [1, log2plus1(2),
                                                       0, log2plus1(1),
                                                       1, log2plus1(1)]


def test_food_cell_counts_as_open_space():
    """Same board as test_groups_new_group_left with the food removed, for identical output.

    Replaces a test that passed 'no food' as the dropped food-position argument. Its point
    now is that is_open() accepts both 0 and 1, so a food cell joins the region around it
    rather than blocking like a body cell.
    """
    with_food = np.array([[4, 4, 4, 4, 4, 4, 4],
                          [4, 0, 0, 0, 0, 0, 4],
                          [4, 0, 2, 2, 3, 5, 4],
                          [4, 1, 2, 2, 2, 2, 4],
                          [4, 4, 4, 4, 4, 4, 4]])
    without_food = np.array([[4, 4, 4, 4, 4, 4, 4],
                             [4, 0, 0, 0, 0, 0, 4],
                             [4, 0, 2, 2, 3, 5, 4],
                             [4, 0, 2, 2, 2, 2, 4],
                             [4, 4, 4, 4, 4, 4, 4]])
    expected = [1, log2plus1(2),
                0, log2plus1(1),
                1, log2plus1(1)]
    assert group_obs(with_food, (3, 1), (4, 1), 'right') == expected
    assert group_obs(without_food, (3, 1), (4, 1), 'right') == expected


def test_groups_new_group_forward():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 0, 0, 3, 5, 4],
                     [4, 0, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 1), (4, 1), 'up') == [1, log2plus1(1),
                                                    1, log2plus1(1),
                                                    1, log2plus1(2)]


def test_eating_move_does_not_free_the_tail():
    """The one case where the tail must stay occupied.

    Eating grows the snake: add_segment() puts a new segment on the tile the tail came from,
    so that tile is still body after the move. Here the tail at (0, 1) is the only gap in the
    body wall and the food at (4, 1) is the only other link between the strip above and the
    strip below. Moving onto the food blocks that link, so 'forward' must report two regions.
    Freeing the tail unconditionally would merge them and report one.
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 5, 3, 3, 2, 1, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 1), (0, 1), 'right') == [1, log2plus1(1),
                                                       1, log2plus1(1),
                                                       1, log2plus1(2)]


def test_tail_vacating_merges_two_regions():
    """The tail is the only gap in a body wall, so vacating it joins both halves.

    Under the old behaviour the tail counted as a wall and every action here reported two
    regions.
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 2, 0, 0, 0, 0, 4],
                     [4, 3, 3, 5, 3, 3, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (0, 0), (2, 1), 'right') == [0, log2plus1(1),
                                                       1, log2plus1(1),
                                                       1, log2plus1(1)]


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
