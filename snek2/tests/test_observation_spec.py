"""Guards the one invariant that breaks silently: observation_spec's length must equal the
vector get_observations actually builds.

Nothing raises when those two disagree. `SnakeEnvironment.observation_spec()` sizes the network's
input layer, `eval_checkpoints.py` and `watch.py` restore weights with `expect_partial()`, and a
tf_agents network happily accepts a differently-shaped batch by broadcasting or truncating
depending on where the mismatch is. The symptom is a policy that trains or evaluates against the
wrong inputs and merely performs badly — which on this project is indistinguishable from a
hyperparameter that did not work.

Kept out of test_state_helpers.py because it needs the environment, which pulls in tf_agents and
pygame; state_helpers alone needs neither.
"""
import numpy as np

from snake_constants import GRID_LENGTH
from state_helpers import get_observations


class FakeFood:
    def __init__(self, position):
        self.position = position


def open_board():
    """A real-sized padded grid: walls (4) around a GRID_LENGTH square of open cells (0).

    Full size rather than the small hand-written grids in test_state_helpers, because
    get_grid_value bounds-checks against SCREENTILES + 1 and so assumes this shape — a smaller
    array indexes out of range once anything looks near the far edge.
    """
    grid = np.zeros((GRID_LENGTH + 2, GRID_LENGTH + 2), dtype=int)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 4
    return grid


def straight_snake():
    """A three-segment snake heading right, mid-board, with food well clear of it.

    Grid cells are grid[y + 1][x + 1] for the padding ring. 2 is the head, 3 body and tail.
    """
    grid = open_board()
    grid[5][5] = 2   # head  (4, 4)
    grid[5][4] = 3   # body  (3, 4)
    grid[5][3] = 3   # tail  (2, 4)
    return {'old_grid': grid, 'head_pos': (4, 4), 'tail_pos': (2, 4), 'next_tail_pos': (3, 4),
            'head_move_dir': 'right', 'current_food': FakeFood((7, 7)),
            'current_step': 10, 'last_food_step': 0, 'snake_len': 3}


def coiled_snake():
    """A snake whose tail sits immediately to the left of its head, so a tail-follow is available.

    Head (4, 4) faces up, tail (3, 4) is its left-hand neighbour, and the body runs between them
    the long way round so the shape is a real chain rather than two adjacent cells.
    """
    grid = open_board()
    for x, y in [(3, 4), (3, 3), (4, 3)]:
        grid[y + 1][x + 1] = 3
    grid[5][5] = 2
    return {'old_grid': grid, 'head_pos': (4, 4), 'tail_pos': (3, 4), 'next_tail_pos': (3, 3),
            'head_move_dir': 'up', 'current_food': FakeFood((7, 7)),
            'current_step': 10, 'last_food_step': 0, 'snake_len': 4}


def wall_runner():
    """A snake travelling along the top wall, chosen so hugging-wall and following-tail differ.

    That is the whole point of this fixture. In an open-board or coiled position the two blocks
    can happen to hold the same three values, which lets a swap of the last two blocks pass a
    positional test on a coincidence. Here the tail is far behind, so following-tail is all ones,
    while the forward move runs beside the wall, so hugging-wall is [0, 0, 1].

    Row y=0 is the topmost *interior* row: the padding ring is at y=-1, which is grid[0]. So the
    snake has to sit at y=0 to be beside the wall, not y=1.
    """
    grid = open_board()
    grid[1][5] = 2   # head  (4, 0)
    grid[1][4] = 3   # body  (3, 0)
    grid[1][3] = 3   # tail  (2, 0)
    return {'old_grid': grid, 'head_pos': (4, 0), 'tail_pos': (2, 0), 'next_tail_pos': (3, 0),
            'head_move_dir': 'right', 'current_food': FakeFood((7, 7)),
            'current_step': 10, 'last_food_step': 0, 'snake_len': 3}


def spec_length():
    # Imported lazily so a failure to build the environment reads as this test's failure rather
    # than a collection error in every test in the file.
    from snake_environment import SnakeEnvironment
    return SnakeEnvironment(discount=0.99, display=False,
                            policy_name='smoke').observation_spec().shape[0]


def test_observation_vector_length_matches_the_spec():
    assert len(get_observations(**straight_snake())) == spec_length()


def test_observation_vector_length_is_the_same_in_a_coiled_position():
    # Length must not depend on the board: group_obs short-circuits fatal moves down a different
    # branch, and a block that returned early would shorten the vector for some states only.
    assert len(get_observations(**coiled_snake())) == spec_length()


def test_following_tail_block_sits_at_26_to_28():
    # Documents the layout the docstring promises, and pins the block so a future insertion in the
    # middle fails here rather than silently repointing the frozen hyperparamTuning/diagnostics/
    # scripts that index this vector by position.
    # 1 is good, so the tail-chasing action is the 0.
    assert get_observations(**coiled_snake())[26:29] == [0, 1, 1], 'tail left of the head'
    assert get_observations(**straight_snake())[26:29] == [1, 1, 1], 'tail is directly behind'


def test_each_tail_block_sits_where_the_layout_says():
    # Compared against the producing functions rather than against literals, because literals let
    # an ordering bug pass on a coincidence — hugging-wall and following-tail can hold the same
    # three values in an open-board position. wall_runner() exists to break that coincidence.
    from state_helpers import following_tail_obs, food_space_obs, group_obs

    for fixture in (straight_snake(), coiled_snake(), wall_runner()):
        values = get_observations(**fixture)
        _, _, wall_hug = group_obs(fixture['old_grid'], fixture['head_pos'], fixture['tail_pos'],
                                   fixture['next_tail_pos'], fixture['head_move_dir'],
                                   fixture['current_food'])
        following = following_tail_obs(fixture['head_pos'], fixture['tail_pos'],
                                       fixture['head_move_dir'])
        food_space = food_space_obs(fixture['old_grid'], fixture['current_food'])
        assert values[23:26] == wall_hug, ('hugging-wall block moved', values, wall_hug)
        assert values[26:29] == following, ('following-tail block moved', values, following)
        assert values[29:30] == food_space, ('food-space value moved', values, food_space)


def test_observation_values_stay_in_range():
    # The network sees these raw. Two observations have been rescaled in this project's history
    # after reaching 4.4 and 8.97 on a vector of otherwise 0-1 inputs.
    for fixture in (straight_snake(), coiled_snake(), wall_runner()):
        values = get_observations(**fixture)
        assert all(0.0 <= float(v) <= 1.0 for v in values), \
            [round(float(v), 3) for v in values if not 0.0 <= float(v) <= 1.0]


def test_food_space_is_a_single_value_not_a_per_action_triple():
    # The only single-value observation about the food, and the only block since the starve/length
    # pair that is not one value per action. A future edit that made it per-action would take the
    # vector to 32 and break every checkpoint again, so it is worth pinning explicitly.
    from state_helpers import food_space_obs

    assert len(food_space_obs(open_board(), FakeFood((7, 7)))) == 1
    for fixture in (straight_snake(), coiled_snake(), wall_runner()):
        assert len(get_observations(**fixture)) == 30


def test_food_space_reads_low_when_the_food_is_sealed_in():
    # End to end through get_observations rather than the helper alone, so this also covers the
    # food position being handed through from the fixture.
    # Food at (7, 5) with room below it, so the three-cell case has somewhere to extend into.
    # (7, 8) is the last interior row and (7, 9) is already wall, which is why a pocket built
    # against the bottom edge cannot be opened up into a third cell.
    def board_with(body, food=(7, 5)):
        fixture = straight_snake()
        grid = fixture['old_grid']
        for x, y in body:
            grid[y + 1][x + 1] = 3
        grid[food[1] + 1][food[0] + 1] = 1
        fixture['current_food'] = FakeFood(food)
        return fixture

    # Sealed alone: all four neighbours are body. 1 is safe, so this is the 0.
    sealed = board_with([(6, 5), (8, 5), (7, 4), (7, 6)])
    assert get_observations(**sealed)[29] == 0

    # A two-cell pocket. Opening one of those four is not enough on its own — that cell connects
    # onward to the rest of the board — so the pocket only exists once (7, 6)'s own other
    # neighbours are closed too. That is exactly the distinction a naive "count the food's open
    # neighbours" implementation misses.
    pocket = board_with([(6, 5), (8, 5), (7, 4), (6, 6), (8, 6), (7, 7)])
    assert get_observations(**pocket)[29] == 0.5

    # Let the pocket open onto a third cell and it stops counting as cramped.
    roomy = board_with([(6, 5), (8, 5), (7, 4), (6, 6), (8, 6)])
    assert get_observations(**roomy)[29] == 1
