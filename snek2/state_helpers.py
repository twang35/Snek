import math

import numpy as np

from snake_constants import *

# The four neighbour offsets, derived from MOVE_VECTORS so they cannot drift from it. Used by
# get_adjacent_groups; the flood fill in count_groups works on flat bit positions and shifts
# instead, so it needs no coordinate offsets at all.
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
    cols = old_grid.shape[1]
    for action in ACTIONS:
        # .copy() rather than copy.deepcopy(): an ndarray copy is already independent, and
        # deepcopy walks the object graph to reach the same result. Worth almost nothing on
        # its own (0.3% of the profile) but there is no reason to pay it.
        grid = update_grid(action, head_pos, tail_pos, old_grid.copy(), head_move_dir)
        new_head_pos = get_relative_pos(action, head_pos, head_move_dir)

        regions, group_count = count_groups(grid)

        head_with_tail = 0
        num_groups = log2plus1(group_count)

        head_groups = get_adjacent_groups(regions, cols, new_head_pos)
        tail_groups = get_adjacent_groups(regions, cols, tail_pos)

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
    """Groups the open cells into connected regions, as bitmasks over the padded grid.

    Returns `(regions, count)`. Each region is a Python int whose set bits are the cells it
    contains, tile (x, y) being bit `(y + 1) * cols + (x + 1)`. `count` is `len(regions)`.

    The fill is a bitwise dilation rather than a per-cell walk: repeatedly smear the region
    one cell in each direction and mask back down to open cells, until it stops growing. Each
    round is a handful of operations on a ~144-bit int regardless of how many cells the region
    holds, where a cell-at-a-time flood fill pays interpreter overhead per cell per neighbour.

    Two things make this safe, both courtesy of the wall ring Snake._rebuild_grid() pads with.
    Shifting by one crosses a row boundary — bit (r, cols-1) shifts into (r+1, 0) — but the
    first and last column are always wall, so no open bit ever sits where it could wrap, and
    `& open_mask` discards it regardless. The same ring keeps the vertical shifts in range.

    Region order differs from earlier versions. Nothing reads it: callers want the region
    count and whether two cells share a region.
    """
    cols = grid.shape[1]
    # Bit i set iff cell i is open (empty or food). packbits with little bit order matches
    # int.from_bytes little endian, so bit i lands on element i - verified against a
    # shift-and-or loop over 500 random cases.
    open_flags = ((grid == 0) | (grid == 1)).ravel()
    open_mask = int.from_bytes(np.packbits(open_flags, bitorder='little').tobytes(), 'little')

    regions = []
    remaining = open_mask
    while remaining:
        region = remaining & -remaining          # lowest set bit: an arbitrary unvisited cell
        while True:
            grown = (region | (region << 1) | (region >> 1)
                     | (region << cols) | (region >> cols)) & open_mask
            if grown == region:
                break
            region = grown
        regions.append(region)
        remaining &= ~region

    return regions, len(regions)


def get_adjacent_groups(regions, cols, tile_pos):
    """Indices of the regions holding an open cell orthogonally adjacent to tile_pos.

    `tile_pos` is allowed to be off the board — callers pass the prospective head position,
    which may be past a wall — so each neighbour gets the same guard get_grid_value() applies,
    reading anything off-board as wall rather than indexing for it.
    """
    found = set()
    x = tile_pos[0]
    y = tile_pos[1]
    for offset in NEIGHBOUR_OFFSETS:
        neighbour_x = x + offset[0]
        neighbour_y = y + offset[1]
        if neighbour_x < 0 or neighbour_y < 0:
            continue
        if neighbour_x > SCREENTILES[0] + 1 or neighbour_y > SCREENTILES[1] + 1:
            continue
        bit = 1 << ((neighbour_y + 1) * cols + (neighbour_x + 1))
        for index in range(len(regions)):
            if regions[index] & bit:
                found.add(index)
                break

    return found


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


def get_relative_pos(action, tile_pos, move_dir):
    # Plain tuple arithmetic. Its sibling get_pos() was `tuple(np.add(tile_pos, vector))`,
    # dispatching into numpy to add two pairs of small ints and rebuilding a tuple from the
    # result; the flood fill called it four times per cell visited, 8.3M times per 8k game
    # steps, and it was 68% of the entire observation cost. The rewritten flood fill works on
    # flat integer indices and needs no coordinate helper at all, so get_pos() is gone.
    vector = MOVE_VECTORS[CURRENT_DIRECTION_MAPS[move_dir][action]]
    return tile_pos[0] + vector[0], tile_pos[1] + vector[1]
