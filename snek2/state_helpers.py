import copy
import math

import numpy as np

from snake_constants import *

# The four neighbour offsets, in DIRECTIONS order so the flood fill visits cells in the same
# order it always did. Precomputed because the inner loop used to re-read MOVE_VECTORS by
# direction name on every cell.
NEIGHBOUR_OFFSETS = tuple(MOVE_VECTORS[direction] for direction in DIRECTIONS)


def get_observations(old_grid,
                     head_pos,
                     tail_pos,
                     head_move_dir,
                     current_food,
                     current_step,
                     last_food_step,
                     snake_len,
                     game_finished):
    observations = []
    observations.extend(food_observations(old_grid, head_pos, current_food, head_move_dir))
    observations.extend(body_and_wall_collisions(old_grid, head_pos, tail_pos, head_move_dir))
    observations.extend(group_obs(old_grid,
                                  head_pos,
                                  tail_pos,
                                  head_move_dir))
    observations.extend(perfect_game_obs(old_grid, head_pos, head_move_dir, snake_len))
    observations.extend(steps_until_starve(current_step, last_food_step, snake_len))
    # remaining spaces
    # observations.extend([((SCREENTILES[0] + 1) * (SCREENTILES[1] + 1)) - (snake_len + START_SEGMENTS + 1)])
    # end of game
    observations.extend([1] if game_finished else [0])
    return observations


# Returns moving closer and on food for each action.
# First number is 1 or 0 for closer or not
# Second number is log2 distance to food
def food_observations(grid, head_pos, current_food, head_move_dir):
    if current_food == 'no food':
        return [0, 0, 0, 0, 0, 0]
    food_pos = current_food.position
    starting_distance = distance_to_food(head_pos, food_pos)
    observations = []

    for action in ACTIONS:
        new_head_pos = get_relative_pos(action, head_pos, head_move_dir)
        grid_value = get_grid_value(new_head_pos, grid)
        if grid_value == 1:
            # on top of food
            observations.extend([1, 1])  # log2plus1(0) = 1
        else:
            to_food_steps = distance_to_food(new_head_pos, food_pos)
            reversed_distance_obs = 1 / (to_food_steps + 1)
            if to_food_steps < starting_distance:
                # closer to food
                observations.extend([1, reversed_distance_obs])
            else:
                # further away from food
                observations.extend([0, reversed_distance_obs])

    return observations


# Returns 1 for perfect game in each action
def perfect_game_obs(old_grid, head_pos, head_move_dir, snek_len):
    # if not one away from perfect game return 0s
    if snek_len != PERFECT_SCORE - 1:
        return [0, 0, 0]

    observations = []
    for action in ACTIONS:
        new_head_pos = get_relative_pos(action, head_pos, head_move_dir)
        grid_value = get_grid_value(new_head_pos, old_grid)
        if grid_value == 1:
            # on top of food
            observations.extend([1])
        else:
            observations.extend([0])
    return observations


# Returns 1 for no collision, 0 for collision in each action
# Reverse to help snek learn what is safe
def body_and_wall_collisions(grid, head_pos, tail_pos, head_move_dir):
    observations = []
    for action in ACTIONS:
        new_head_pos = get_relative_pos(action, head_pos, head_move_dir)
        grid_value = get_grid_value(new_head_pos, grid)
        if grid_value == 1 or grid_value == 0 or new_head_pos == tail_pos:
            observations.extend([1])
        else:
            observations.extend([0])

    return observations


# head_with_tail: returns 1 for with tail or 0 for no tail groups in each action
# total_group_obs: returns number of groups
def group_obs(old_grid, head_pos, tail_pos, head_move_dir):
    observations = []
    for action in ACTIONS:
        # .copy() rather than copy.deepcopy(): an ndarray copy is already independent, and
        # deepcopy walks the object graph to reach the same result. Worth almost nothing on
        # its own (0.3% of the profile) but there is no reason to pay it.
        grid = update_grid(action, head_pos, tail_pos, old_grid.copy(), head_move_dir)
        new_head_pos = get_relative_pos(action, head_pos, head_move_dir)

        groups = count_groups(grid)

        head_with_tail = 0
        num_groups = log2plus1(len(groups))

        head_groups = get_adjacent_groups(grid, groups, new_head_pos)
        tail_groups = get_adjacent_groups(grid, groups, tail_pos)

        if len(head_groups & tail_groups) > 0 or tuple(new_head_pos) == tail_pos:
            head_with_tail = 1

        observations.extend([head_with_tail, num_groups])

    return observations


# Returns log2 number of remaining steps until starving to death
def steps_until_starve(current_step, last_food_step, snake_len):
    # cap max at 500
    max_steps_until_starve = min(snake_len * MAX_STEPS_BEFORE_STARVE_SIZE_MULTIPLIER, 500)
    return [log2plus1(max(100, max_steps_until_starve) - (current_step - last_food_step))]


def log2plus1(num):
    return math.log2(num + 1)


def count_groups(grid):
    remaining_spaces = set()
    # maybe groups is not needed? or big groups are ok?
    groups = []

    # don't need to check the boundaries which are always value 4
    # Read the whole array out once. Indexing a numpy array cell by cell costs far more per
    # lookup than indexing nested lists, and this scan covers the full board on every call.
    cells = grid.tolist()
    for i in range(grid.shape[1] - 2):
        for j in range(grid.shape[0] - 2):
            # if is 0 or 1, add to remaining_spaces.
            value = cells[j + 1][i + 1]
            if value == 0 or value == 1:
                remaining_spaces.add((i, j))

    # for each grid element, recurse and try to connect
    while len(remaining_spaces) > 0:
        groups.append(set())
        populate_group(groups[-1], remaining_spaces.pop(), grid, remaining_spaces)

    return groups


def populate_group(group_set, tile_pos, grid, remaining_spaces):
    """Flood-fills the connected group containing tile_pos, consuming remaining_spaces.

    Iterative rather than recursive: an open board is one ~100-cell region, so the recursive
    form spent a Python call frame per cell and came within sight of the recursion limit.

    The `is_open()` re-check it used to do was redundant — remaining_spaces is built from
    exactly the open cells, so membership already implies openness. That check alone was
    ~2M calls per 8k steps, each one a numpy lookup.
    """
    stack = [tile_pos]
    group_set.add((tile_pos[0], tile_pos[1]))

    while stack:
        pos = stack.pop()
        x = pos[0]
        y = pos[1]
        for offset in NEIGHBOUR_OFFSETS:
            neighbour = (x + offset[0], y + offset[1])
            if neighbour in remaining_spaces:
                remaining_spaces.remove(neighbour)
                group_set.add(neighbour)
                stack.append(neighbour)


def is_open(tile_pos, grid):
    grid_num = get_grid_number(tile_pos, grid)
    return grid_num == 0 or grid_num == 1


def get_adjacent_groups(grid, groups, tile_pos):
    group_set = set()
    for direction in DIRECTIONS:
        direction_pos = get_pos(direction, tile_pos)
        grid_number = get_grid_number(direction_pos, grid)
        if grid_number == 0 or grid_number == 1:
            for i in range(len(groups)):
                if tuple(direction_pos) in groups[i]:
                    group_set.add(i)

    return group_set


def get_grid_number(coord, grid):
    if coord[0] > SCREENTILES[0] + 1 or coord[1] > SCREENTILES[1] + 1:
        return 4
    if coord[0] < 0 or coord[1] < 0:
        return 4
    return grid[coord[1] + 1][coord[0] + 1]


def distance_to_food(start_pos, food_pos):
    return abs(food_pos[0] - start_pos[0]) + abs(food_pos[1] - start_pos[1])


def update_grid(action, head_pos, tail_pos, grid, head_move_dir):
    """Applies a candidate move to a copy of the grid, for the connectivity observations.

    The tail tile is freed, because the snake advances as a whole and the tail vacates its
    tile on the same step the head takes a new one. Leaving it occupied — which is what this
    did before — computes connectivity as though the tail were a wall, so a region reachable
    only through the tail reads as sealed off. Measured against the 92% policy, that made
    num_groups wrong on 12.1% of steps overall and 40.0% of steps past score 80, and turned
    head_with_tail from 1 to 0 on 258 occasions, always in the pessimistic direction.

    The exception is a move that eats: add_segment() then refills the tile the tail came
    from, so it stays occupied and must not be cleared. That case is why the original line
    was commented out rather than simply absent.
    """
    new_head_pos = get_relative_pos(action, head_pos, head_move_dir)
    eats_food = get_grid_value(new_head_pos, grid) == 1   # read before mutating
    if not eats_food:
        set_number(grid, tail_pos, 0)
    set_number(grid, new_head_pos, 3)
    return grid


def set_number(grid, tile_pos, number):
    if out_of_bounds(tile_pos):
        return
    grid[tile_pos[1] + 1][tile_pos[0] + 1] = number


def out_of_bounds(tile_pos):
    if tile_pos[0] < 0 or tile_pos[0] >= SCREENTILES[0] + 1 or tile_pos[1] < 0 or tile_pos[1] >= SCREENTILES[1] + 1:
        return True
    return False


def get_grid_value(tile_pos, grid):
    if tile_pos[0] > SCREENTILES[0] + 1 or tile_pos[1] > SCREENTILES[1] + 1:
        return 4
    if tile_pos[0] < 0 or tile_pos[1] < 0:
        return 4
    return grid[tile_pos[1] + 1][tile_pos[0] + 1]


def get_pos(direction, tile_pos):
    # Plain tuple arithmetic. This was `tuple(np.add(tile_pos, MOVE_VECTORS[direction]))`,
    # which dispatched into numpy to add two pairs of small ints and then rebuilt a tuple
    # from the result. The flood fill calls this four times per cell visited, so it ran
    # 8.3M times per 8k game steps and was 68% of the entire observation cost — a bigger
    # share than everything else in state_helpers put together.
    vector = MOVE_VECTORS[direction]
    return tile_pos[0] + vector[0], tile_pos[1] + vector[1]


def get_relative_pos(action, tile_pos, move_dir):
    vector = MOVE_VECTORS[CURRENT_DIRECTION_MAPS[move_dir][action]]
    return tile_pos[0] + vector[0], tile_pos[1] + vector[1]
