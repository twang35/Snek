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
#     group_obs(grid, head_pos, tail_pos, next_tail_pos, head_move_dir)
#         -> [head_with_tail, num_groups] * 3
#
# ordered by ACTIONS = ['left', 'right', 'forward'].
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


# =============================== group_obs tests ===============================
def test_hwt_no_touching():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 1, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # Open board: every action leaves one region, and the freed tail keeps head and tail
    # sharing it.
    assert group_obs(grid, (3, 1), (1, 1), (2, 1), 'right') == [1, log2plus1(1),
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
    assert group_obs(grid, (3, 1), (1, 1), (2, 1), 'right') == [1, log2plus1(1),
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
    assert group_obs(grid, (3, 0), (4, 0), (4, 1), 'right') == [0, log2plus1(2),
                                                       0, log2plus1(2),
                                                       1, log2plus1(1)]


def test_hwt_following_right_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 2, 5, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # Same board, facing up, so 'right' is the move onto the tail.
    assert group_obs(grid, (3, 0), (4, 0), (4, 1), 'up') == [0, log2plus1(2),
                                                    1, log2plus1(1),
                                                    0, log2plus1(2)]


def test_hwt_following_left_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 2, 3, 4],
                     [4, 0, 0, 3, 5, 3, 4],
                     [4, 0, 0, 3, 3, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 0), (3, 1), (4, 1), 'left') == [1, log2plus1(1),
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
    assert group_obs(grid, (3, 1), (0, 1), (1, 1), 'right') == [1, log2plus1(2),
                                                       1, log2plus1(2),
                                                       0, log2plus1(1)]


def test_hwt_no_left():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 3, 0, 3, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 1, 0, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 1), (1, 1), (2, 1), 'right') == [0, log2plus1(1),
                                                       1, log2plus1(3),
                                                       1, log2plus1(2)]


def test_hwt_no_right():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 5, 3, 2, 0, 4],
                     [4, 0, 0, 3, 0, 3, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 1), (1, 1), (2, 1), 'right') == [1, log2plus1(3),
                                                       0, log2plus1(1),
                                                       1, log2plus1(2)]


def test_hwt_follow_tail_and_empty_forward_no_food():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 2, 2, 0, 0, 4],
                     [4, 1, 2, 2, 3, 5, 4],
                     [4, 0, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 1), (4, 1), (4, 2), 'up') == [0, log2plus1(2),
                                                    1, log2plus1(2),
                                                    1, log2plus1(2)]


def test_hwt_multiple_open_groups_separate_food_and_tail():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 2, 2, 0, 0, 4],
                     [4, 0, 0, 0, 3, 5, 4],
                     [4, 1, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 1), (4, 1), (4, 2), 'left') == [0, log2plus1(2),
                                                      1, log2plus1(2),
                                                      0, log2plus1(2)]


def test_groups_new_group_left():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 2, 2, 3, 5, 4],
                     [4, 1, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 1), (4, 1), (4, 2), 'right') == [1, log2plus1(2),
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
    assert group_obs(with_food, (3, 1), (4, 1), (4, 2), 'right') == expected
    assert group_obs(without_food, (3, 1), (4, 1), (4, 2), 'right') == expected


def test_groups_new_group_forward():
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 0, 0, 0, 3, 5, 4],
                     [4, 0, 2, 2, 2, 2, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    assert group_obs(grid, (3, 1), (4, 1), (4, 2), 'up') == [1, log2plus1(1),
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
    assert group_obs(grid, (3, 1), (0, 1), (1, 1), 'right') == [1, log2plus1(1),
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
    assert group_obs(grid, (0, 0), (2, 1), (1, 1), 'right') == [0, log2plus1(1),
                                                       1, log2plus1(1),
                                                       1, log2plus1(1)]


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

    Under the old behaviour every action here reported 0.
    """
    grid = np.array([[4, 4, 4, 4, 4, 4, 4],
                     [4, 0, 0, 0, 0, 0, 4],
                     [4, 3, 3, 3, 0, 0, 4],
                     [4, 3, 5, 3, 0, 0, 4],
                     [4, 3, 3, 3, 0, 0, 4],
                     [4, 0, 0, 2, 0, 0, 4],
                     [4, 4, 4, 4, 4, 4, 4]])
    # 'right' turns into the wall row below the board and reaches nothing, so it stays 0.
    assert group_obs(grid, (2, 4), (1, 2), (2, 2), 'right') == [1, log2plus1(3),
                                                               0, log2plus1(3),
                                                               1, log2plus1(3)]


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
    # Facing up, so 'forward' is the move onto the food. 'left' leaves the board.
    assert group_obs(grid, (0, 4), (0, 1), (0, 0), 'up') == [0, log2plus1(2),
                                                            0, log2plus1(2),
                                                            1, log2plus1(2)]


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


# =============================== observation vector tests ===============================
def test_observation_spec_matches_what_the_game_emits():
    """A spec that disagrees with reality has bitten this project before.

    `head_with_food_obs` sat at 0 in the spec for a whole signature generation while the argument
    it needed was still being passed, and the count is maintained by hand as a sum of named
    parts, so it drifts silently. This asserts the two agree rather than trusting either.
    """
    import os
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    from snake_environment import SnakeEnvironment

    env = SnakeEnvironment(display=False)
    env.reset()
    assert env.observation_spec().shape == (20,)
    assert env._game.get_observation().shape == (20,)


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
