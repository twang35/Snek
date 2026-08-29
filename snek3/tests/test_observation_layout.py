"""The observation's declared length and block layout, against the vector actually built.

**Nothing raises when the spec and the builder disagree.** The spec sizes the network's input
layer; a torch `Linear` accepts any batch whose last dimension matches, and a policy fed the wrong
inputs merely performs badly — which on this project is indistinguishable from a hyperparameter
that did not work. `arch.json` catches a *checkpoint* meeting a changed env, but it records
`obs_len` from the same spec, so it cannot catch a spec that was wrong from the start.

The boards here are hand-built rather than played, because the blocks that need pinning only become
distinguishable in specific positions — see `wall_runner`. `test_observations.py` covers the
producing functions themselves and needs no env.
"""

import numpy as np

from env.constants import GRID_LENGTH, PERFECT_SCORE
from env.observations import get_observations


class FakeFood:
    def __init__(self, position):
        self.position = position


def open_board():
    """A real-sized padded grid: walls (4) around a `GRID_LENGTH` square of open cells (0).

    Full size rather than a small hand-written grid, because `get_grid_value` bounds-checks against
    `SCREENTILES + 1` and so assumes this shape — a smaller array indexes out of range once
    anything looks near the far edge.
    """
    grid = np.zeros((GRID_LENGTH + 2, GRID_LENGTH + 2), dtype=int)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 4
    return grid


def straight_snake():
    """A three-segment snake heading right, mid-board, with food well clear of it.

    Grid cells are `grid[y + 1][x + 1]` for the padding ring. 2 is the head, 3 body and tail.
    """
    grid = open_board()
    grid[5][5] = 2   # head  (4, 4)
    grid[5][4] = 3   # body  (3, 4)
    grid[5][3] = 3   # tail  (2, 4)
    return {'old_grid': grid, 'head_pos': (4, 4), 'tail_pos': (2, 4), 'next_tail_pos': (3, 4),
            'head_move_dir': 'right', 'current_food': FakeFood((7, 7)),
            'current_step': 10, 'last_food_step': 0, 'snake_len': 3}


def coiled_snake():
    """A snake whose tail sits immediately left of its head, so a tail-follow is available.

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

    That is the whole point of this fixture. In an open-board or coiled position the two blocks can
    happen to hold the same three values, which lets a swap of the last blocks pass a positional
    test on a coincidence. Here the tail is far behind, so following-tail is all ones, while the
    forward move runs beside the wall, so hugging-wall is [0, 0, 1].

    Row y=0 is the topmost *interior* row — the padding ring is at y=-1, which is `grid[0]` — so the
    snake sits at y=0 to be beside the wall, not y=1.
    """
    grid = open_board()
    grid[1][5] = 2   # head  (4, 0)
    grid[1][4] = 3   # body  (3, 0)
    grid[1][3] = 3   # tail  (2, 0)
    return {'old_grid': grid, 'head_pos': (4, 0), 'tail_pos': (2, 0), 'next_tail_pos': (3, 0),
            'head_move_dir': 'right', 'current_food': FakeFood((7, 7)),
            'current_step': 10, 'last_food_step': 0, 'snake_len': 3}


BOARDS = (('straight', straight_snake), ('coiled', coiled_snake), ('wall_runner', wall_runner))


def spec_length():
    """The length three independent places have to agree on: the block table, the env's spec, and
    `OBS_LEN`, which is what `arch.json` records and every checkpoint is validated against.

    Imported lazily so a failure to build the env reads as this test failing rather than as a
    collection error in every test in the file.
    """
    from env import constants, scalar_env

    env = scalar_env.SnakeEnv(discount=0.99, policy_name='smoke')
    lengths = {scalar_env.observation_length(), env.observation_spec()['shape'][0],
               constants.OBS_LEN}
    assert len(lengths) == 1, 'the block table, the spec and OBS_LEN disagree: {0}'.format(lengths)
    return lengths.pop()


def test_observation_vector_length_matches_the_spec():
    assert len(get_observations(**straight_snake())) == spec_length()


def test_observation_vector_length_does_not_depend_on_the_board():
    # `group_obs` short-circuits fatal moves down a different branch, so a block that returned early
    # would shorten the vector for some states only.
    for name, board in BOARDS:
        assert len(get_observations(**board())) == spec_length(), name


def test_the_vector_is_thirty_values():
    # A deliberate tripwire, not redundancy with the equalities above: adding a block is supposed to
    # fail here so `OBS_LEN`, `OBS_BLOCKS`, `OBS_ERA` and the layout table in docs/environment.md
    # all get updated in the same pass.
    assert spec_length() == 30


def test_following_tail_block_sits_at_26_to_28():
    # The literal indices, pinned: an insertion in the middle of the vector must fail here rather
    # than silently repointing anything that reads this vector by position.
    # 1 is good, so the tail-chasing action is the 0.
    assert get_observations(**coiled_snake())[26:29] == [0, 1, 1], 'tail left of the head'
    assert get_observations(**straight_snake())[26:29] == [1, 1, 1], 'tail directly behind'


def test_each_block_sits_where_the_layout_table_says():
    """`OBS_BLOCKS` is compared against the functions that fill it, on boards that make it matter.

    Comparing against literals is what let an ordering bug through in snek2: hugging-wall and
    following-tail can hold the same three values in an open-board position, so swapping them
    passed. `wall_runner()` exists to break that coincidence.
    """
    from env import scalar_env
    from env.observations import following_tail_obs, food_space_obs, group_obs

    ranges = scalar_env.block_ranges()
    for name, board in BOARDS:
        fixture = board()
        values = get_observations(**fixture)
        _, _, wall_hug = group_obs(fixture['old_grid'], fixture['head_pos'], fixture['tail_pos'],
                                   fixture['next_tail_pos'], fixture['head_move_dir'],
                                   fixture['current_food'])
        following = following_tail_obs(fixture['head_pos'], fixture['tail_pos'],
                                       fixture['head_move_dir'])
        food_space = food_space_obs(fixture['old_grid'], fixture['current_food'])

        def block(key):
            first, stop = ranges[key]
            return values[first:stop]

        assert block('hugging_wall') == wall_hug, (name, 'hugging-wall moved', values)
        assert block('not_following_tail') == following, (name, 'following-tail moved', values)
        assert block('food_space') == food_space, (name, 'food-space moved', values)
        assert block('board_fill') == [fixture['snake_len'] / PERFECT_SCORE], (
            name, 'board-fill moved', values)

    # The fixture is only worth anything if some board separates the two three-wide blocks.
    separating = [name for name, board in BOARDS
                  if get_observations(**board())[23:26] != get_observations(**board())[26:29]]
    assert separating, 'no board here distinguishes hugging-wall from following-tail'


def test_observation_values_stay_in_range():
    # The network sees these raw. Two observations were rescaled in snek2's history after reaching
    # 4.4 and 8.97 on a vector of otherwise 0-1 inputs.
    for name, board in BOARDS:
        values = get_observations(**board())
        assert all(0.0 <= float(v) <= 1.0 for v in values), (
            name, [round(float(v), 3) for v in values if not 0.0 <= float(v) <= 1.0])


def test_food_space_is_a_single_value_not_a_per_action_triple():
    # The only single-value observation about the food, and the only block since the starve/length
    # pair that is not one value per action. A future edit making it per-action would take the
    # vector to 32 and invalidate every checkpoint, so it is pinned explicitly.
    from env.observations import food_space_obs

    assert len(food_space_obs(open_board(), FakeFood((7, 7)))) == 1


def test_food_space_reads_low_when_the_food_is_sealed_in():
    """End to end through `get_observations`, so this also covers the food being handed through.

    Food at (7, 5) with room below it, so the three-cell case has somewhere to extend into. (7, 8)
    is the last interior row and (7, 9) is already wall, which is why a pocket built against the
    bottom edge cannot be opened into a third cell.
    """
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
