import copy
import math

import numpy as np

from snake_constants import *


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
    observations.extend(head_with_tail(old_grid, head_pos, tail_pos, head_move_dir))
    observations.extend(steps_until_starve(current_step, last_food_step, snake_len))
    observations.extend([1] if game_finished else [0])
    return observations


# Returns moving closer and on food for each action.
# First number is 1 or 0 for closer or not
# Second number is 1 or 0 for on top of food or not
# Third number is log2 distance to food
def food_observations(grid, head_pos, current_food, head_move_dir):
    if current_food == 'no food':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]
    food_pos = current_food.position
    starting_distance = distance_to_food(head_pos, food_pos)
    observations = []

    for action in ACTIONS:
        new_head_pos = get_relative_pos(action, head_pos, head_move_dir)
        grid_value = get_grid_value(new_head_pos, grid)
        if grid_value == 1:
            # on top of food
            observations.extend([1, 1, 1])  # log2plus1(0) = 1
        else:
            to_food_steps = distance_to_food(new_head_pos, food_pos)
            if to_food_steps < starting_distance:
                # closer to food
                observations.extend([1, 0, log2plus1(to_food_steps)])
            else:
                # further away from food
                observations.extend([0, 0, log2plus1(to_food_steps)])

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


# Returns 1 for with tail or 0 for no tail groups in each action
def head_with_tail(old_grid, head_pos, tail_pos, head_move_dir):
    observations = []
    for action in ACTIONS:
        grid = update_grid(action, head_pos, tail_pos, copy.deepcopy(old_grid), head_move_dir)
        new_head_pos = get_relative_pos(action, head_pos, head_move_dir)

        groups = count_groups(grid)

        head_groups = get_adjacent_groups(grid, groups, new_head_pos)
        tail_groups = get_adjacent_groups(grid, groups, tail_pos)

        # move second check above count_groups
        if len(head_groups & tail_groups) > 0 or tuple(new_head_pos) == tail_pos:
            observations.extend([1])
        else:
            observations.extend([0])

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
    for i in range(grid.shape[1] - 2):
        for j in range(grid.shape[0] - 2):
            # if is 0 or 1, add to remaining_spaces.
            if is_open((i, j), grid):
                remaining_spaces.add((i, j))

    # for each grid element, recurse and try to connect
    while len(remaining_spaces) > 0:
        groups.append(set())
        populate_group(groups[-1], remaining_spaces.pop(), grid, remaining_spaces)

    return groups


def populate_group(group_set, tile_pos, grid, remaining_spaces):
    group_set.add((tile_pos[0], tile_pos[1]))

    for direction in DIRECTIONS:
        new_tile_pos = get_pos(direction, tile_pos)
        pos_tuple = (new_tile_pos[0], new_tile_pos[1])

        if pos_tuple in remaining_spaces and is_open(new_tile_pos, grid):
            remaining_spaces.remove(pos_tuple)
            populate_group(group_set, new_tile_pos, grid, remaining_spaces)


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
    set_number(grid, get_relative_pos(action, head_pos, head_move_dir), 2)
    set_number(grid, tail_pos, 0)
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
    return tuple(np.add(tile_pos, MOVE_VECTORS[direction]))


def get_relative_pos(action, tile_pos, move_dir):
    return tuple(np.add(tile_pos, MOVE_VECTORS[CURRENT_DIRECTION_MAPS[move_dir][action]]))
