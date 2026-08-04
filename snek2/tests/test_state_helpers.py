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
#     group_obs(grid, head_pos, tail_pos, next_tail_pos, head_move_dir, current_food)
#         -> ([head_with_tail, num_groups] * 3, [safe_to_chase_food] * 3)
#
# ordered by ACTIONS = ['left', 'right', 'forward'].
#
# The two return values sit in different places in the observation vector, so they come back as
# separate lists. The helpers below pick one half each, which also keeps the assertions that
# predate the food flag readable — none of them depend on `current_food`, because the first list
# genuinely does not: the food only enters the second one.
#
# `next_tail_pos` is the cell the tail lands on when the move does not eat, which in the game is
# the position of the segment ahead of it. In these fixtures it has to be supplied by hand, and
# it changes the head_with_tail answers, so each call below names the body cell that the tail's
# chain actually runs through. Several of the older fixtures are not strict snake chains — a
# body cell or two is orphaned — so where more than one neighbour would do, the choice is the
# one that makes the longest coherent chain, and the new fixtures at the end of this section
# spell their chains out.
#
# The expectations also moved, because update_grid() now frees the tail cell: the snake
# advances as a whole, so the tail vacates as the head takes a new cell. Two consequences show
# up repeatedly below. Where the tail was the only link between two open regions, freeing it
# merges them and num_groups *drops*. Where the tail sat in a dead end, freeing it leaves a
# one-cell region and num_groups *rises*. The exception is a move that eats: add_segment()
# refills the tile the tail came from, so it stays occupied — see
# test_eating_move_does_not_free_the_tail.


class FakeFood:
    """Stands in for Snake.Food, of which group_obs only ever reads `.position`."""

    def __init__(self, position):
        self.position = position


def group_values(grid, head_pos, tail_pos, next_tail_pos, head_move_dir, food='no food'):
    """The [head_with_tail, lg(regions)] half, which does not depend on where the food is."""
    return group_obs(grid, head_pos, tail_pos, next_tail_pos, head_move_dir, food)[0]


def food_chase_values(grid, head_pos, tail_pos, next_tail_pos, head_move_dir, food):
    """The [safe to chase the food] half, one value per action."""
    return group_obs(grid, head_pos, tail_pos, next_tail_pos, head_move_dir, food)[1]


def wall_hug_values(grid, head_pos, tail_pos, next_tail_pos, head_move_dir, food='no food'):
    """The [hugging a wall or body] third, one value per action."""
    return group_obs(grid, head_pos, tail_pos, next_tail_pos, head_move_dir, food)[2]


# =============================== group_obs tests ===============================
def test_hwt_no_touching():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 1, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # Open board: every action leaves one region, and the freed tail keeps head and tail
    # sharing it.
    assert group_values(grid, (3, 1), (1, 1), (2, 1), 'right') == [1, (log2plus1(1) / GROUPS_OBS_SCALE),
                                                       1, (log2plus1(1) / GROUPS_OBS_SCALE),
                                                       1, (log2plus1(1) / GROUPS_OBS_SCALE)]


def test_hwt_no_touching_eats_food():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 5, 3, 2, 1, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # 'forward' eats the food at (4, 1), so the tail stays put. The board is open enough that
    # it makes no difference to the counts here.
    assert group_values(grid, (3, 1), (1, 1), (2, 1), 'right') == [1, (log2plus1(1) / GROUPS_OBS_SCALE),
                                                       1, (log2plus1(1) / GROUPS_OBS_SCALE),
                                                       1, (log2plus1(1) / GROUPS_OBS_SCALE)]


def test_hwt_following_forward_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 2, 5, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # 'left' runs off the top of the board and 'right' into the body at (3, 1); both fatal, so
    # both are zeroed outright. Only 'forward', onto the tail, is legal here.
    assert group_values(grid, (3, 0), (4, 0), (4, 1), 'right') == [0, 0,
                                                       0, 0,
                                                       1, (log2plus1(1) / GROUPS_OBS_SCALE)]


def test_hwt_following_right_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 2, 5, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # Same board, facing up, so 'right' is the move onto the tail. 'left' walks into the body at
    # (2, 0) and 'forward' off the top of the board; both fatal, both zeroed.
    assert group_values(grid, (3, 0), (4, 0), (4, 1), 'up') == [0, 0,
                                                    1, (log2plus1(1) / GROUPS_OBS_SCALE),
                                                    0, 0]


def test_hwt_following_left_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 2, 3, 4],
                     [4, 0, 0, 3, 5, 3, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # 'left' is the move onto the tail; 'right' runs off the top of the board and 'forward' into
    # the body at (2, 0), both fatal.
    assert group_values(grid, (3, 0), (3, 1), (4, 1), 'left') == [1, (log2plus1(1) / GROUPS_OBS_SCALE),
                                                      0, 0,
                                                      0, 0]


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
    assert group_values(grid, (3, 1), (0, 1), (1, 1), 'right') == [1, (log2plus1(2) / GROUPS_OBS_SCALE),
                                                       1, (log2plus1(2) / GROUPS_OBS_SCALE),
                                                       0, (log2plus1(1) / GROUPS_OBS_SCALE)]


def test_hwt_no_left():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 0, 3, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 1, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_values(grid, (3, 1), (1, 1), (2, 1), 'right') == [0, (log2plus1(1) / GROUPS_OBS_SCALE),
                                                       1, (log2plus1(3) / GROUPS_OBS_SCALE),
                                                       1, (log2plus1(2) / GROUPS_OBS_SCALE)]


def test_hwt_no_right():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 0, 3, 0, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_values(grid, (3, 1), (1, 1), (2, 1), 'right') == [1, (log2plus1(3) / GROUPS_OBS_SCALE),
                                                       0, (log2plus1(1) / GROUPS_OBS_SCALE),
                                                       1, (log2plus1(2) / GROUPS_OBS_SCALE)]


def test_hwt_follow_tail_and_empty_forward_no_food():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 2, 2, 0, 0, 4],
                     [4, 1, 2, 2, 3, 5, 4],
                     [4, 0, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # 'left' walks into the blocked cell at (2, 1); fatal, zeroed. 'right' (onto the tail) and
    # 'forward' are both legal.
    assert group_values(grid, (3, 1), (4, 1), (4, 2), 'up') == [0, 0,
                                                    1, (log2plus1(2) / GROUPS_OBS_SCALE),
                                                    1, (log2plus1(2) / GROUPS_OBS_SCALE)]


def test_hwt_multiple_open_groups_separate_food_and_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 2, 2, 0, 0, 4],
                     [4, 0, 0, 0, 3, 5, 4],
                     [4, 1, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # 'left' walks into the blocked cell at (3, 2); fatal, zeroed. 'right' and 'forward' are
    # both legal.
    assert group_values(grid, (3, 1), (4, 1), (4, 2), 'left') == [0, 0,
                                                      1, (log2plus1(2) / GROUPS_OBS_SCALE),
                                                      0, (log2plus1(2) / GROUPS_OBS_SCALE)]


def test_groups_new_group_left():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 2, 2, 3, 5, 4],
                     [4, 1, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # 'right' walks into the blocked cell at (3, 2); fatal, zeroed. 'left' and 'forward' (onto
    # the tail) are both legal.
    assert group_values(grid, (3, 1), (4, 1), (4, 2), 'right') == [1, (log2plus1(2) / GROUPS_OBS_SCALE),
                                                       0, 0,
                                                       1, (log2plus1(1) / GROUPS_OBS_SCALE)]


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
    # 'right' walks into the blocked cell at (3, 2) in both variants; fatal, zeroed.
    expected = [1, (log2plus1(2) / GROUPS_OBS_SCALE),
                0, 0,
                1, (log2plus1(1) / GROUPS_OBS_SCALE)]
    assert group_values(with_food, (3, 1), (4, 1), (4, 2), 'right') == expected
    assert group_values(without_food, (3, 1), (4, 1), (4, 2), 'right') == expected


def test_groups_new_group_forward():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 0, 0, 3, 5, 4],
                     [4, 0, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_values(grid, (3, 1), (4, 1), (4, 2), 'up') == [1, (log2plus1(1) / GROUPS_OBS_SCALE),
                                                    1, (log2plus1(1) / GROUPS_OBS_SCALE),
                                                    1, (log2plus1(2) / GROUPS_OBS_SCALE)]


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
    assert group_values(grid, (3, 1), (0, 1), (1, 1), 'right') == [1, (log2plus1(1) / GROUPS_OBS_SCALE),
                                                       1, (log2plus1(1) / GROUPS_OBS_SCALE),
                                                       1, (log2plus1(2) / GROUPS_OBS_SCALE)]


def test_tail_vacating_merges_two_regions():
    """The tail is the only gap in a body wall, so vacating it joins both halves.

    Under the old behaviour the tail counted as a wall and every action here reported two
    regions. Only 'forward' is legal now - 'left' runs off the top of the board and 'right'
    into the body at (0, 1) - so it is the only one left to make that point.
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 2, 0, 0, 0, 0, 4],
                     [4, 3, 3, 5, 3, 3, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_values(grid, (0, 0), (2, 1), (1, 1), 'right') == [0, 0,
                                                       0, 0,
                                                       1, (log2plus1(1) / GROUPS_OBS_SCALE)]


def test_hwt_enclosed_vacated_tail_still_reaches_the_tail():
    """The endgame case the old code got wrong on 78% of the decisions that lost games.

    The snake is coiled so that the tail at (1, 2) is walled in by its own body on all four
    sides. update_grid() frees that cell, which makes it a region with *no open neighbours*, so
    asking which regions are adjacent to it returns nothing and the flag reads 0 whatever the
    head does. The question it should ask is about the cell the tail is moving *to* — (2, 2),
    the segment ahead of it — which touches the open right-hand side at (3, 2).

    Body chain, head first: (2,4) (2,3) (1,3) (0,3) (0,2) (0,1) (1,1) (2,1) (2,2) (1,2)=tail.

        y=0   . . . . .
        y=1   # # # . .
        y=2   # t # . .        t = tail, a = the segment ahead of it
        y=3   # # # . .
        y=4   . . H . .

    Under the old behaviour every action here reported 0. Under the current one, 'left' and
    'right' are also both fatal - 'left' walks into the body at (2, 3), 'right' into the wall
    row below the board - so 'forward' is the only action either version had a chance to get
    right, and both of its zeroed siblings read 0 by construction rather than by this bug.
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 3, 3, 3, 0, 0, 4],
                     [4, 3, 5, 3, 0, 0, 4],
                     [4, 3, 3, 3, 0, 0, 4],
                     [4, 0, 0, 2, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_values(grid, (2, 4), (1, 2), (2, 2), 'right') == [0, 0,
                                                               0, 0,
                                                               1, (log2plus1(3) / GROUPS_OBS_SCALE)]


def test_hwt_eating_move_does_not_advance_the_tail():
    """The other half of the fix: on a step that eats, the tail does not move.

    add_segment() refills the tile the tail came from, so the snake grows from the back and the
    tail stays put. Here that is the difference between 1 and 0. The tail at (0, 1) touches the
    open cell (0, 2), which is the region the head lands in when it eats the food at (0, 3).
    The segment ahead of it at (0, 0) is boxed in by the wall, its own body at (1, 0), and the
    tail itself — so advancing the tail on this step would report 0.

    Body chain, head first: (0,4) (1,4) (1,3) (1,2) (1,1) (1,0) (0,0) (0,1)=tail.

        y=0   a # . . .        a = the segment ahead of the tail
        y=1   t # . . .        t = tail
        y=2   . # . . .
        y=3   F # . . .
        y=4   H # . . .
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 3, 3, 0, 0, 0, 4],
                     [4, 5, 3, 0, 0, 0, 4],
                     [4, 0, 3, 0, 0, 0, 4],
                     [4, 1, 3, 0, 0, 0, 4],
                     [4, 2, 3, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # Facing up, so 'forward' is the move onto the food. 'left' leaves the board and 'right'
    # walks into the body at (1, 4); both fatal, both zeroed.
    assert group_values(grid, (0, 4), (0, 1), (0, 0), 'up') == [0, 0,
                                                            0, 0,
                                                            1, (log2plus1(2) / GROUPS_OBS_SCALE)]


def test_groups_obs_scale_matches_the_starve_pattern():
    """GROUPS_OBS_SCALE has to be log2(MAX_GROUPS_FOR_SCALE + 1), the same relationship
    STARVE_OBS_SCALE has to MAX_STARVE_BUDGET - not merely close to it.

    Pins the constant down directly rather than only through group_values() assertions, so a
    refactor that recomputes the scale a different way (off by one, wrong log base, forgetting
    the +1) fails here even if it happens to still satisfy every hand-picked fixture.
    """
    assert GROUPS_OBS_SCALE == math.log2(MAX_GROUPS_FOR_SCALE + 1)


def test_num_groups_reaches_exactly_one_at_the_design_cap():
    """The design cap is meant to be reached, not merely approached.

    At group_count == MAX_GROUPS_FOR_SCALE the scaled value is exactly 1.0 - the same relationship
    the starve budget has to its own scale (`test_starve_and_length_obs_stay_within_zero_and_one`
    reaches exactly 0.0 and 1.0, not just close to them). One cell past it - a board more
    fragmented than anything measured in the sweep that picked MAX_GROUPS_FOR_SCALE - runs past
    1.0 on purpose rather than losing information the way clamping the raw count would: there is
    no game rule capping the true count the way there is for the starve budget, so this is a
    design choice, not a guarantee.
    """
    assert log2plus1(MAX_GROUPS_FOR_SCALE) / GROUPS_OBS_SCALE == 1.0
    assert log2plus1(MAX_GROUPS_FOR_SCALE - 1) / GROUPS_OBS_SCALE < 1.0
    assert log2plus1(MAX_GROUPS_FOR_SCALE + 1) / GROUPS_OBS_SCALE > 1.0


def test_num_groups_stays_in_zero_to_one_for_measured_region_counts():
    """Every group_count actually measured in a 422,608-action sweep of real play, replayed
    through the scale directly. 13 was the highest seen; this checks well past it for margin.
    """
    values = [log2plus1(count) / GROUPS_OBS_SCALE for count in range(0, MAX_GROUPS_FOR_SCALE + 1)]
    assert min(values) == 0.0
    assert max(values) == 1.0
    assert all(0.0 <= value <= 1.0 for value in values)
    # The measured ceiling (13) still leaves real headroom under the design cap (16).
    assert log2plus1(13) / GROUPS_OBS_SCALE < 0.95


# =============================== safe-to-chase-food tests ===============================
def test_chase_food_open_board_is_safe_from_every_action():
    """Nothing is split, so head, food and tail share the one region whatever the snake does."""
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 1, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert food_chase_values(grid, (3, 1), (1, 1), (2, 1), 'right',
                             FakeFood((1, 2))) == [1, 1, 1]


def test_chase_food_is_zero_when_the_food_is_sealed_off():
    """The tail is reachable and the food is not, which is the case the flag exists for.

    The food at (0, 0) is boxed in by two body cells that are nowhere near the tail, so freeing
    the tail cannot open it. head_with_tail stays 1 for the survivable moves — that is the point:
    the existing flag says "you have an escape route" and says nothing about whether the meal on
    offer is reachable.

    Body chain, head first: (0,3) (0,2) (0,1) (1,1) (1,0) (2,0) (3,0)=tail.

        y=0   F # # t .        F = food, t = tail
        y=1   # # . . .
        y=2   # . . . .
        y=3   H . . . .
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 1, 3, 3, 5, 0, 4],
                     [4, 3, 3, 0, 0, 0, 4],
                     [4, 3, 0, 0, 0, 0, 4],
                     [4, 2, 0, 0, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    food = FakeFood((0, 0))
    # 'right' walks off the left edge, so it reaches nothing at all - both values zeroed, not
    # merely head_with_tail, since a fatal move gets nothing rather than a hypothetical count.
    assert group_values(grid, (0, 3), (3, 0), (2, 0), 'down', food) == [1, (log2plus1(2) / GROUPS_OBS_SCALE),
                                                                       0, 0,
                                                                       1, (log2plus1(2) / GROUPS_OBS_SCALE)]
    assert food_chase_values(grid, (0, 3), (3, 0), (2, 0), 'down', food) == [0, 0, 0]


def test_chase_food_counts_an_eating_move_by_whether_the_tail_survives_it():
    """Eating is the special case, and this is the outcome where it costs the snake its escape.

    The food at (1, 0) sits in a pocket boxed in by walls on two sides and the head on the third,
    so its only opening is the head's own cell. `forward` eats it and lands in that pocket with
    nowhere else to go: the tail is unreachable, 0, even though the move reached the food.

    `left` and `right` are both legal and both keep the tail reachable (`head_with_tail` is 1 for
    both), but neither can see the food either - its only opening is the cell the head started
    in, and that cell is never open in the single-move lookahead this function computes,
    occupied or not. So the food reads unreachable from every direction except eating it, and
    eating it is what seals the snake in. Paired with
    `test_chase_food_is_one_for_an_eating_move_that_keeps_the_tail`, which is the same eating
    branch coming out the other way.

    Body chain, head first: (1,1) (1,2) (1,3) (1,4)=tail.

        y=0   . F . . .        F = food, boxed except from below
        y=1   . H . . .
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 4, 1, 4, 0, 0, 4],
                     [4, 0, 2, 0, 0, 0, 4],
                     [4, 0, 3, 0, 0, 0, 4],
                     [4, 0, 3, 0, 0, 0, 4],
                     [4, 0, 5, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert food_chase_values(grid, (1, 1), (1, 4), (1, 3), 'up',
                             FakeFood((1, 0))) == [0, 0, 0]
    # head_with_tail confirms left and right are alive and well; the zeros above are the food
    # being unreachable, not the tail.
    assert group_values(grid, (1, 1), (1, 4), (1, 3), 'up', FakeFood((1, 0))) == [
        1, (log2plus1(2) / GROUPS_OBS_SCALE), 1, (log2plus1(2) / GROUPS_OBS_SCALE), 0, (log2plus1(2) / GROUPS_OBS_SCALE)]


def test_chase_food_is_one_for_an_eating_move_that_keeps_the_tail():
    """The move the flag exists to encourage, and the reason eating needs its own branch.

    'forward' takes the food at (4, 1) and the tail at (1, 1) stays reachable, so this is 1.
    Without the eating branch it would read 0: the food's cell is occupied by the head after the
    move, so it belongs to no region and a containment test finds nothing.
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 5, 3, 2, 1, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert food_chase_values(grid, (3, 1), (1, 1), (2, 1), 'right',
                             FakeFood((4, 1))) == [1, 1, 1]


def test_chase_food_needs_one_region_holding_food_and_tail_together():
    """All three in the *same* region, not merely all three reachable.

    The head can neighbour two regions at once, and `forward` is that case. It puts the head on
    (1, 0), which plugs the food's only opening and leaves the food at (0, 0) as a region of one —
    a region the head is standing next to but which does not touch the tail. So the head can reach
    the food and can reach its tail, through *different* regions: eating would seal it in with the
    meal. The flag is 0, and it would be 1 if the food were tested against everything the head can
    see instead of against the intersection with the tail.

    `right` leaves (1, 0) open, so food, head and tail all share the board's one big region and
    chasing is genuinely safe: 1. `left` walks into the body at (0, 1) - fatal, so 0 regardless
    of any of this.

    Body chain, head first: (1,1) (0,1) (0,2) (0,3)=tail.

        y=0   F . . . .        F = food
        y=1   # H . . .
        y=2   # . . . .
        y=3   t . . . .        t = tail
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 1, 0, 0, 0, 0, 4],
                     [4, 3, 2, 0, 0, 0, 4],
                     [4, 3, 0, 0, 0, 0, 4],
                     [4, 5, 0, 0, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    food = FakeFood((0, 0))
    assert food_chase_values(grid, (1, 1), (0, 3), (0, 2), 'up', food) == [0, 1, 0]
    # head_with_tail is 1 for the two legal moves, so those 0/1 values are the food half
    # talking, not the tail - and 0 for 'left', which is fatal.
    assert group_values(grid, (1, 1), (0, 3), (0, 2), 'up', food) == [0, 0,
                                                                     1, (log2plus1(1) / GROUPS_OBS_SCALE),
                                                                     1, (log2plus1(2) / GROUPS_OBS_SCALE)]


def test_chase_food_finds_a_food_cell_that_is_its_own_region():
    """Asks which region *holds* the food, not which regions neighbour it.

    `forward` moves the head from (0, 1) to (0, 0), which does not land on the food but does sit
    next to it. update_grid() marks (0, 0) occupied once the head arrives there, so the food at
    (1, 0) ends up with every neighbour blocked — the head's new cell on one side, the body at
    (1, 1) and the not-yet-vacated segment at (2, 0) on the others, the board edge on the last.
    It is a genuine region of one.

    Asking what neighbours the food cell directly finds nothing, since all four are blocked, and
    would read 0 for a move that is actually safe. The right question is which region *contains*
    the food — region 0, size 1 cell — and that region is what `new_head_pos` is adjacent to
    (through the food cell itself) and what `next_tail_pos` is *also* adjacent to, through the
    old tail cell (3, 0) freeing up on the same move. Both paths land on the same single-cell
    region, so the intersection holds it and the answer is 1.

    'left' leaves the board; 'right' walks into the body at (1, 1). Both are fatal, so both read
    0 regardless of the food.

    Body chain, head first: (0,1) (1,1) (2,1) (2,0) (3,0)=tail.

        y=0   . F # t .        F = food, t = tail
        y=1   H # # . .
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 1, 3, 5, 0, 4],
                     [4, 2, 3, 3, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    food = FakeFood((1, 0))
    assert food_chase_values(grid, (0, 1), (3, 0), (2, 0), 'up', food) == [0, 0, 1]


def test_chase_food_when_following_the_tail_into_open_space():
    """Following the tail needs its own case, the same way head_with_tail does.

    `forward` steps onto (2, 0), the cell the tail is leaving. No region test can see the tail from
    there — the segment ahead of it at (3, 0) is boxed in by the body at (4, 0) and (3, 1) and by
    the head's new cell, so the set of regions touching the post-move tail is empty. Intersecting
    with that empty set would say "not safe" about a move that is safe by construction: the snake
    is walking in its own wake, with the whole open board and the food at (0, 0) in front of it.

    `left` stays legal and reaches everything the ordinary way; `right` walks into the body at
    (3, 1) and is fatal, so 0 regardless of the food.

    Body chain, head first: (2,1) (3,1) (4,1) (4,0) (3,0) (2,0)=tail.

        y=0   F . t # #        F = food, t = tail
        y=1   . . H # #
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 1, 0, 5, 3, 3, 4],
                     [4, 0, 0, 2, 3, 3, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert food_chase_values(grid, (2, 1), (2, 0), (3, 0), 'up',
                             FakeFood((0, 0))) == [1, 0, 1]


def test_chase_food_is_zero_when_there_is_no_food():
    """The winning step: the last food is eaten and no replacement is placed."""
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert food_chase_values(grid, (3, 1), (1, 1), (2, 1), 'right', 'no food') == [0, 0, 0]


def test_chase_food_never_claims_safe_where_the_tail_is_lost():
    """The flag is a conjunction, so it can never be 1 where head_with_tail is 0.

    Sweeps the fixtures in this file rather than asserting on one board, since the invariant is
    what matters: a route to the food is worthless without a route back out.
    """
    boards = [
        (np.array([[4, 4, 4, 4, 4, 4, 4],
                   [4, 0, 0, 3, 2, 5, 4],
                   [4, 0, 0, 3, 3, 3, 4],
                   [4, 0, 0, 3, 3, 3, 4],
                   [4, 4, 4, 4, 4, 4, 4]]), (3, 0), (4, 0), (4, 1), 'right', (1, 0)),
        (np.array([[4, 4, 4, 4, 4, 4, 4],
                   [4, 0, 0, 0, 0, 0, 4],
                   [4, 3, 3, 3, 0, 0, 4],
                   [4, 3, 5, 3, 0, 0, 4],
                   [4, 3, 3, 3, 0, 0, 4],
                   [4, 0, 0, 2, 1, 0, 4],
                   [4, 4, 4, 4, 4, 4, 4]]), (2, 4), (1, 2), (2, 2), 'right', (3, 4)),
    ]
    for grid, head, tail, next_tail, move_dir, food_pos in boards:
        food = FakeFood(food_pos)
        pairs = group_values(grid, head, tail, next_tail, move_dir, food)
        chase = food_chase_values(grid, head, tail, next_tail, move_dir, food)
        for index in range(3):
            if not pairs[2 * index]:
                assert chase[index] == 0, (index, pairs, chase)


# =============================== wall/body hugging tests ===============================
def test_hug_open_board_hugs_nothing():
    """Nothing nearby in any direction, so every action reads 0."""
    grid = np.array([[4, 4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 0, 4],
                     [4, 0, 3, 3, 2, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4, 4]])
    assert wall_hug_values(grid, (3, 1), (1, 1), (2, 1), 'right') == [0, 0, 0]


def test_hug_wall_on_forward_fatal_on_left_open_on_right():
    """The head runs along the left edge of the board (x = 0), facing up.

    'forward' stays at x = 0: its left side, west, is off the board - a wall - so it hugs.
    'left' would step off the board entirely (x = -1): fatal, not a hugging question at all.
    'right' turns away from the edge into open space: nothing on either side.
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 2, 0, 0, 0, 0, 4],
                     [4, 3, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert wall_hug_values(grid, (0, 1), (0, 2), (0, 1), 'up') == [0, 0, 1]


def test_hug_body_on_one_side_of_forward_and_coincidentally_left_too():
    """A body segment at (4, 0) sits beside two different candidate landing cells.

    Facing right, 'forward' lands at (4, 1); (4, 0) is its left side (north), so it hugs.
    'left' turns to face up and lands at (3, 0); (4, 0) happens to be *its* right side (east)
    too, from a different angle - also hugs, which is correct, not a coincidence to explain
    away. 'right' turns to face down, landing at (3, 2), nowhere near (4, 0): open both sides.
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 3, 0, 4],
                     [4, 0, 3, 3, 2, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4, 4]])
    assert wall_hug_values(grid, (3, 1), (1, 1), (2, 1), 'right') == [1, 0, 1]


def test_hug_reads_one_when_both_sides_are_blocked_not_two():
    """Body segments on both sides of the forward landing cell still read a single 1.

    This is a flag, not a count - group_obs' own num_groups already carries "how fragmented",
    so hugging only has to say whether the move runs along an edge at all.
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 3, 0, 4],
                     [4, 0, 3, 3, 2, 0, 0, 4],
                     [4, 0, 0, 0, 0, 3, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4, 4]])
    assert wall_hug_values(grid, (3, 1), (1, 1), (2, 1), 'right') == [1, 1, 1]


def test_hug_is_zero_for_a_fatal_move_regardless_of_what_is_beside_it():
    """A move that kills the snake reads 0 here even where the geometry would say 1.

    'forward' now runs straight into a body segment at (4, 1) - fatal - despite the *same*
    obstacle at (4, 0) from the previous fixture sitting right where its left side would be if
    it survived the move. There is no "afterwards" to hug anything in.
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 3, 0, 4],
                     [4, 0, 3, 3, 2, 3, 0, 4],
                     [4, 0, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4, 4]])
    assert wall_hug_values(grid, (3, 1), (1, 1), (2, 1), 'right') == [1, 0, 0]


def test_hug_reads_the_board_after_the_move_not_before_it():
    """The cell beside the new head is the tail's *current* position, which vacates this step.

    Facing right from (1, 1), 'forward' lands at (2, 1); its left side (north) is (2, 0), where
    the tail sits right now. Checked against old_grid that cell is still body - `test_hug`
    would read 1, wrongly, for a move that is actually running along open space the tail is
    about to leave. Checked against the grid *after* the move, as group_obs does, it is open: 0.
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 0, 0, 4],
                     [4, 0, 2, 0, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert wall_hug_values(grid, (1, 1), (2, 0), (3, 1), 'right') == [0, 0, 0]
    # The wrong answer old_grid would have given, spelled out rather than left implicit.
    assert get_grid_value((2, 0), grid) not in (0, 1), 'fixture no longer exercises the case'


def test_hug_an_eating_move_does_not_free_the_tail_either():
    """Same board as the previous test, except 'forward' now lands on food instead of open space.

    Eating means add_segment() refills the tile the tail came from, so (2, 0) stays occupied
    after the move rather than vacating - and 'forward' hugs it: 1, the opposite of the
    otherwise-identical non-eating fixture above.
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 0, 0, 4],
                     [4, 0, 2, 1, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert wall_hug_values(grid, (1, 1), (2, 0), (3, 1), 'right') == [0, 0, 1]


def test_hug_never_fires_for_a_move_that_is_not_legal():
    """Sweeps every fixture above: wherever the chosen move is fatal, hug is 0 too.

    A cheap invariant check on top of the fixture-by-fixture assertions - the fatal case is
    covered directly above, but this confirms nothing else in this section accidentally relies
    on it going the other way.
    """
    boards = [
        (np.array([[4, 4, 4, 4, 4, 4, 4],
                   [4, 0, 0, 0, 0, 0, 4],
                   [4, 2, 0, 0, 0, 0, 4],
                   [4, 3, 0, 0, 0, 0, 4],
                   [4, 4, 4, 4, 4, 4, 4]]), (0, 1), (0, 2), (0, 1), 'up'),
        (np.array([[4, 4, 4, 4, 4, 4, 4, 4],
                   [4, 0, 0, 0, 0, 3, 0, 4],
                   [4, 0, 3, 3, 2, 3, 0, 4],
                   [4, 0, 0, 0, 0, 0, 0, 4],
                   [4, 4, 4, 4, 4, 4, 4, 4]]), (3, 1), (1, 1), (2, 1), 'right'),
    ]
    for grid, head, tail, next_tail, move_dir in boards:
        hug = wall_hug_values(grid, head, tail, next_tail, move_dir)
        for index, action in enumerate(ACTIONS):
            new_head = get_relative_pos(action, head, move_dir)
            grid_value = get_grid_value(new_head, grid)
            legal = grid_value in (0, 1) or tuple(new_head) == tuple(tail)
            if not legal:
                assert hug[index] == 0, (action, hug)


# =============================== starve and length tests ===============================
def test_starve_budget_is_ten_per_segment_between_its_limits():
    assert starve_budget(20) == 200
    assert starve_budget(37) == 370
    # Floored, so a short snake gets time to cross the board.
    assert starve_budget(1) == MIN_STARVE_BUDGET
    assert starve_budget(5) == MIN_STARVE_BUDGET
    assert starve_budget(10) == MIN_STARVE_BUDGET
    # Capped, so a long one cannot stall forever.
    assert starve_budget(50) == MAX_STARVE_BUDGET
    assert starve_budget(99) == MAX_STARVE_BUDGET


def test_steps_until_starve_counts_down_to_zero_at_the_budget():
    """The game rule reads this directly, so the boundary is the behaviour that matters."""
    assert steps_until_starve(0, 0, 20) == 200
    assert steps_until_starve(50, 0, 20) == 150
    assert steps_until_starve(199, 0, 20) == 1     # alive
    assert steps_until_starve(200, 0, 20) == 0     # Snake.step() starves at <= 0
    assert steps_until_starve(201, 0, 20) == -1
    # Counted from the last food, not the start of the episode.
    assert steps_until_starve(1000, 950, 20) == 150


def test_starve_rule_boundary_is_unchanged_by_the_rescale():
    """Guards the split: the observation was rescaled, the rule must not move.

    The old code returned `[log2plus1(budget - elapsed)]` and Snake.step() starved on `[0] <= 0`.
    log2plus1(x) <= 0 exactly when x <= 0, so an int return has to starve on the same step.
    """
    for snake_len in (5, 10, 23, 50, 99):
        budget = starve_budget(snake_len)
        for elapsed in range(budget - 2, budget + 2):
            new_rule = steps_until_starve(elapsed, 0, snake_len) <= 0
            old_rule = log2plus1(max(0, budget - elapsed)) <= 0
            assert new_rule == old_rule, (snake_len, elapsed)


def test_length_is_visible_past_fifty_segments():
    """The bug this replaced: every length from 50 up produced one identical value.

    The starve budget caps at MAX_STARVE_BUDGET, so `log2plus1(budget - elapsed)` was the same
    number for lengths 50 through 99 at equal elapsed steps - no length information at all for
    the whole second half of a game, which is where games are decided.
    """
    late = [tuple(starve_and_length_obs(0, 0, n)) for n in range(50, 100)]
    assert len(set(late)) == len(late)
    # The starve half is still identical up there, by design: the budget really is capped. It is
    # the length half that separates them.
    assert len({obs[0] for obs in late}) == 1
    assert len({obs[1] for obs in late}) == len(late)
    # And it is monotone, so "longer" is a direction the network can follow.
    assert [obs[1] for obs in late] == sorted(obs[1] for obs in late)


def test_starve_and_length_obs_stay_within_zero_and_one():
    """Scale was the other half of the problem: this input used to reach 8.97."""
    values = []
    for snake_len in range(1, PERFECT_SCORE + 1):
        budget = starve_budget(snake_len)
        for elapsed in (0, 1, budget // 2, budget - 1, budget, budget + 5):
            values.extend(starve_and_length_obs(elapsed, 0, snake_len))
    assert min(values) >= 0.0
    assert max(values) <= 1.0
    # Both ends are actually reached, so the range is used rather than merely respected.
    assert max(values) == 1.0
    assert min(values) == 0.0


def test_starve_obs_is_the_same_value_for_the_same_steps_remaining():
    """Scaled by the maximum budget, not by each snake's own, so the units stay comparable.

    100 steps left means the same number whether the snake is short or long; that is what lets
    the network read urgency off one input instead of combining two.
    """
    short = starve_and_length_obs(starve_budget(15) - 100, 0, 15)[0]
    long_ = starve_and_length_obs(starve_budget(80) - 100, 0, 80)[0]
    assert short == long_


def test_a_move_into_a_wall_reaches_nothing():
    """None of the three group_obs values may fire for a move off the board.

    `group_obs` checks legality itself now - the same test `body_and_wall_collisions` uses,
    food, open, or the tail-follow special case - and short-circuits a fatal move to zero before
    computing anything, rather than running the flood fill and reporting what a move that kills
    the snake would have found. Verified over 4,035 wall moves and 14,642 body-collision moves
    in a sweep of real games: zero non-zero flags either way, where the pre-fix code reported
    `head_with_tail = 1` on 5,289 of the body-collision moves - a hypothetical answer for a
    snake that no longer exists after that move.

    Before the legality gate, walls happened to read zero by accident rather than by design: an
    off-board cell's only on-board neighbour is the vacated head cell, which `update_grid` never
    clears, so no region test could find anything past it - and a version that "tidied up" by
    clearing that cell would have broken silently. The gate below makes that accident load-bearing
    no longer.

    Built on a full-size board rather than a hand-typed fixture, since the point is the real
    board's edges.
    """
    for head, move_dir, spine in (((0, 4), 'left', [(1, 4), (2, 4), (3, 4)]),
                                  ((SCREENTILES[0], 4), 'right',
                                   [(SCREENTILES[0] - 1, 4), (SCREENTILES[0] - 2, 4),
                                    (SCREENTILES[0] - 3, 4)]),
                                  ((4, 0), 'up', [(4, 1), (4, 2), (4, 3)]),
                                  ((4, SCREENTILES[1]), 'down',
                                   [(4, SCREENTILES[1] - 1), (4, SCREENTILES[1] - 2),
                                    (4, SCREENTILES[1] - 3)])):
        grid = np.zeros((SCREENTILES[1] + 3, SCREENTILES[0] + 3))
        grid[[0, -1], :] = 4
        grid[:, [0, -1]] = 4
        for cell in spine:
            grid[cell[1] + 1][cell[0] + 1] = 3
        grid[head[1] + 1][head[0] + 1] = 2
        tail, next_tail = spine[-1], spine[-2]
        # Food sitting in the wide-open middle, so it is reachable from everywhere on the board
        # and cannot be the reason a flag reads 0.
        food = FakeFood((6, 6))

        pairs = group_values(grid, head, tail, next_tail, move_dir, food)
        chase = food_chase_values(grid, head, tail, next_tail, move_dir, food)
        forward = ACTIONS.index('forward')
        assert pairs[2 * forward] == 0, (move_dir, pairs)
        assert pairs[2 * forward + 1] == 0, (move_dir, pairs)
        assert chase[forward] == 0, (move_dir, chase)
        # Turning away from the wall stays on the board and does reach both, so the zeros above
        # are the wall talking and not a broken fixture.
        assert any(pairs[2 * index] for index in range(3) if index != forward), (move_dir, pairs)
        assert any(chase[index] for index in range(3) if index != forward), (move_dir, chase)


# =============================== observation vector tests ===============================
def test_observation_spec_matches_what_the_game_emits():
    """A spec that disagrees with reality has bitten this project before.

    `head_with_food_obs` sat at 0 in the spec for a whole signature generation while the argument
    it needed was still being passed, and the count is maintained by hand as a sum of named
    parts, so it drifts silently. This asserts the two agree rather than trusting either.

    The literal below is a deliberate tripwire, not redundancy with the equality: adding a block
    is supposed to fail here so the count, the layout docstring in get_observations and
    hallOfFame/README.md's era markers all get updated in the same pass. 26 -> 29 on 2026-08-03
    when the following-tail block landed, then 29 -> 30 the same day for food-space.
    """
    import os
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    from snake_environment import SnakeEnvironment

    env = SnakeEnvironment(display=False)
    env.reset()
    assert env.observation_spec().shape == (30,)
    assert env._game.get_observation().shape == (30,)


def test_terminal_steps_carry_zero_discount():
    """The zero is the only thing that stops the loss bootstrapping off a terminal state.

    dqn_agent._loss computes `discounts = gamma * next_time_steps.discount` and then
    `rewards + discounts * next_q_values`, and DdqnAgent here is built without `gamma`, so gamma
    is 1.0 and this field carries all of it. A terminal step with a non-zero discount trains the
    final transition of every episode toward `reward + 0.9975 * V(terminal)`.
    """
    import os
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    import numpy
    from tf_agents.trajectories.time_step import StepType
    from snake_environment import SnakeEnvironment

    env = SnakeEnvironment(discount=0.9975, display=False)
    observations = env.reset().observation

    mid = env.to_tensor_time_step(StepType.MID, numpy.asarray(1.0), observations)
    last = env.to_tensor_time_step(StepType.LAST, numpy.asarray(-5.0), observations)
    first = env.to_tensor_time_step(StepType.FIRST, numpy.asarray(0.0), observations)

    assert float(last.discount) == 0.0
    # Every other step type keeps the tuned discount, which is where the horizon comes from.
    # Tolerance is float32-sized on purpose: the field is float32, so 0.9975 comes back as
    # 0.9975000023841858 and a 1e-9 comparison fails on the storage format rather than the logic.
    assert abs(float(mid.discount) - 0.9975) < 1e-6
    assert abs(float(first.discount) - 0.9975) < 1e-6


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


# ========================== following_tail_obs tests ==========================
#
# Pure position arithmetic: no grid, because the answer is "is this destination the tail's
# current cell" and nothing about the board can change that. Ordered by ACTIONS — left, right,
# forward — as relative turns from the heading.
#
# Coordinates are (x, y) with y increasing downward, so for a head at (5, 5) heading 'up' the
# three action destinations are (4, 5) left, (6, 5) right, (5, 4) forward. (5, 6) is the cell
# behind the head, which no action can reach.
#
# 1 is good, so the tail-chasing action is the 0 and everything else is 1.

def test_ft_tail_to_the_left():
    assert following_tail_obs((5, 5), (4, 5), 'up') == [0, 1, 1]


def test_ft_tail_to_the_right():
    assert following_tail_obs((5, 5), (6, 5), 'up') == [1, 0, 1]


def test_ft_tail_straight_ahead():
    assert following_tail_obs((5, 5), (5, 4), 'up') == [1, 1, 0]


def test_ft_tail_directly_behind_is_unreachable():
    # The straight-snake case, and the reason a fresh episode reads all ones: the tail sits
    # behind the head, and reversing is not one of the three actions.
    assert following_tail_obs((5, 5), (5, 6), 'up') == [1, 1, 1]


def test_ft_tail_far_away():
    assert following_tail_obs((5, 5), (1, 1), 'up') == [1, 1, 1]


def test_ft_tail_diagonally_adjacent_does_not_count():
    # One move never reaches a diagonal, so a tail one step off the diagonal is not "behind" it.
    assert following_tail_obs((5, 5), (4, 4), 'up') == [1, 1, 1]


def test_ft_relative_turns_follow_the_heading():
    # Same geometry, four headings: the cell at (4, 5) is to the left of a head facing up, to the
    # right of one facing down, behind one facing right, and ahead of one facing left. This is the
    # test that catches CURRENT_DIRECTION_MAPS being consulted with the wrong base direction.
    assert following_tail_obs((5, 5), (4, 5), 'up') == [0, 1, 1]
    assert following_tail_obs((5, 5), (4, 5), 'down') == [1, 0, 1]
    assert following_tail_obs((5, 5), (4, 5), 'right') == [1, 1, 1]
    assert following_tail_obs((5, 5), (4, 5), 'left') == [1, 1, 0]


def test_ft_ignores_the_board_entirely():
    # "Otherwise, even if wall or body, it is 0" — and conversely, a 1 does not depend on the
    # grid. following_tail_obs takes no grid at all, which is what makes that unconditional.
    import inspect
    assert 'grid' not in inspect.signature(following_tail_obs).parameters


def test_ft_accepts_a_list_tail_position():
    # Snake.tail.tile_pos is not guaranteed to be a tuple, and == between a list and a tuple is
    # False however equal the contents are. body_and_wall_collisions' own tail check needed the
    # same coercion. Without it the tail would never match and this would read [1, 1, 1].
    assert following_tail_obs((5, 5), [4, 5], 'up') == [0, 1, 1]


def test_ft_marks_at_most_one_action_as_a_tail_chase():
    # Three distinct destinations, so no two can be the same cell. Two zeros would mean
    # get_relative_pos had collapsed two actions onto one cell.
    for heading in DIRECTIONS:
        for tail in [(4, 5), (6, 5), (5, 4), (5, 6), (7, 7)]:
            values = following_tail_obs((5, 5), tail, heading)
            assert values.count(0) <= 1, (heading, tail, values)
            assert set(values) <= {0, 1}, (heading, tail, values)


# ============================ food_space_obs tests ============================
#
# One value, not per action, and 1 is safe: 0 when the food is sealed alone, 0.5 when its open
# region is two cells, 1 for anything roomier or no food. Grid cells are grid[y + 1][x + 1]:
# 0 open, 1 food, 2 head, 3 body, 4 wall.

def food_space(grid, position):
    return food_space_obs(grid, FakeFood(position))


def test_fs_no_food_is_safe():
    grid = np.array([[4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 4],
                     [4, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4]])
    # Nothing is stuck when there is no food. Unreachable in practice — no food means the board
    # is full, a terminal state — but the polarity should still be honest.
    assert food_space_obs(grid, 'no food') == [1]


def test_fs_open_board_is_safe():
    grid = np.array([[4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 4],
                     [4, 1, 0, 0, 4],
                     [4, 4, 4, 4, 4]])
    assert food_space(grid, (0, 1)) == [1]


def test_fs_food_sealed_by_body_is_zero():
    # Food at (1, 1) boxed in by body on all four sides.
    grid = np.array([[4, 4, 4, 4, 4, 4],
                     [4, 0, 3, 0, 0, 4],
                     [4, 3, 1, 3, 0, 4],
                     [4, 0, 3, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4]])
    assert food_space(grid, (1, 1)) == [0]


def test_fs_food_sealed_in_a_corner_by_walls_and_body_is_zero():
    # Two of the four neighbours are wall, the other two body. Still a one-cell region.
    grid = np.array([[4, 4, 4, 4, 4],
                     [4, 1, 3, 0, 4],
                     [4, 3, 0, 0, 4],
                     [4, 4, 4, 4, 4]])
    assert food_space(grid, (0, 0)) == [0]


def test_fs_food_against_the_head_counts_as_sealed():
    # The head (2) is not open space, and unlike the tail it does not vacate. Food with body on
    # three sides and the head on the fourth is stuck.
    grid = np.array([[4, 4, 4, 4, 4],
                     [4, 3, 3, 0, 4],
                     [4, 1, 2, 0, 4],
                     [4, 3, 0, 0, 4],
                     [4, 4, 4, 4, 4]])
    assert food_space(grid, (0, 1)) == [0]


def test_fs_food_with_exactly_one_open_neighbour_is_a_half():
    # Food at (1, 1), open cell at (1, 2) below it, and that cell's other neighbours are all
    # closed — so the region is exactly {food, (1, 2)}.
    grid = np.array([[4, 4, 4, 4, 4, 4],
                     [4, 0, 3, 0, 0, 4],
                     [4, 3, 1, 3, 0, 4],
                     [4, 3, 0, 3, 0, 4],
                     [4, 0, 3, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4]])
    assert food_space(grid, (1, 1)) == [0.5]


def test_fs_two_cell_pocket_against_a_wall_is_a_half():
    grid = np.array([[4, 4, 4, 4, 4],
                     [4, 1, 3, 0, 4],
                     [4, 0, 3, 0, 4],
                     [4, 3, 3, 0, 4],
                     [4, 4, 4, 4, 4]])
    assert food_space(grid, (0, 0)) == [0.5]


def test_fs_three_cell_pocket_is_safe():
    # The one-open-neighbour test passes, but that neighbour opens onto a third cell, so this is
    # a region of three and must read 1. This is the case a naive "count the food's neighbours"
    # implementation gets wrong.
    grid = np.array([[4, 4, 4, 4, 4],
                     [4, 1, 3, 0, 4],
                     [4, 0, 3, 0, 4],
                     [4, 0, 3, 0, 4],
                     [4, 4, 4, 4, 4]])
    assert food_space(grid, (0, 0)) == [1]


def test_fs_food_with_two_open_neighbours_is_safe():
    grid = np.array([[4, 4, 4, 4, 4],
                     [4, 0, 3, 0, 4],
                     [4, 1, 3, 0, 4],
                     [4, 0, 3, 0, 4],
                     [4, 4, 4, 4, 4]])
    assert food_space(grid, (0, 1)) == [1]


def test_fs_only_ever_returns_one_value():
    grid = np.array([[4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 4],
                     [4, 1, 0, 0, 4],
                     [4, 4, 4, 4, 4]])
    assert len(food_space(grid, (0, 1))) == 1


def test_food_space_matches_a_real_flood_fill():
    """Cross-checks the local test against count_groups over random boards.

    food_space_obs decides the 1 / 0.5 / 0 trichotomy from at most eight grid lookups instead of
    running a fourth flood fill per step. That is only worth doing if it is exactly equivalent, so
    this derives the answer the expensive way — find the region holding the food, count its bits —
    and asserts the two agree on every board.
    """
    rng = np.random.default_rng(11)
    size = 7                       # 5x5 interior plus the wall ring
    checked = {1: 0, 0.5: 0, 0: 0}
    for _ in range(3000):
        grid = np.zeros((size, size), dtype=int)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 4
        interior = grid[1:-1, 1:-1]
        # Dense body so sealed and two-cell pockets actually occur.
        interior[:] = np.where(rng.random(interior.shape) < 0.55, 3, 0)
        fx, fy = int(rng.integers(0, size - 2)), int(rng.integers(0, size - 2))
        grid[fy + 1][fx + 1] = 1

        regions, _ = count_groups(grid)
        food_bit = 1 << ((fy + 1) * grid.shape[1] + (fx + 1))
        holding = [r for r in regions if r & food_bit]
        assert len(holding) == 1, 'the food cell is open, so exactly one region holds it'
        actual_size = bin(holding[0]).count('1')
        expected = 0 if actual_size == 1 else (0.5 if actual_size == 2 else 1)

        got = food_space(grid, (fx, fy))[0]
        assert got == expected, (
            'region of %d should read %s, got %s\n%s' % (actual_size, expected, got, grid))
        checked[expected] += 1
    # A vacuous pass would be the real failure here, so require every branch to have been hit.
    assert min(checked.values()) > 20, checked
