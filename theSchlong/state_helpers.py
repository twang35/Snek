import copy

import numpy as np

from snake_constants import *


# Returns moving closer and on food for each direction.
# First number is 1 or 0 for closer or not. Second number is 1 or 0 for on top of food or not.
def food_observations(grid, head_pos, tail_pos, current_food):
    if current_food == 'no food':
        return [0, 0, 0, 0, 0, 0, 0, 0]
    food_pos = current_food.position
    starting_distance = distance_to_food(head_pos, food_pos)
    observations = []

    for action in DIRECTIONS:
        new_head_pos = get_pos(action, head_pos)
        grid_value = get_grid_value(new_head_pos, grid)
        if grid_value == 1:
            # on top of food
            observations.extend([1, 1])
        elif grid_value == 0 or new_head_pos == tail_pos:
            if distance_to_food(new_head_pos, food_pos) < starting_distance:
                # closer to food
                observations.extend([1, 0])
            else:
                # further away from food
                observations.extend([0, 0])
        else:
            # hit a wall or body
            observations.extend([0, 0])

    return observations


# Returns 0 for no collision, 1 for collision in each direction
def body_and_wall_collisions(grid, head_pos, tail_pos):
    observations = []
    for action in DIRECTIONS:
        new_head_pos = get_pos(action, head_pos)
        grid_value = get_grid_value(new_head_pos, grid)
        if grid_value == 1 or grid_value == 0 or new_head_pos == tail_pos:
            observations.extend([0])
        else:
            observations.extend([1])

    return observations


def distance_to_food(start_pos, food_pos):
    return abs(food_pos[0] - start_pos[0]) + abs(food_pos[1] - start_pos[1])


def update_grid(action, head_pos, tail_pos, grid):
    set_number(grid, get_pos(action, head_pos), 2)
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


def get_pos(action, tile_pos):
    return tuple(np.add(tile_pos, MOVE_VECTORS[action]))
