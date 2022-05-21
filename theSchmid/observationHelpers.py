import copy

from Snake import *

DEATH_REWARD = -200000
TRAPPED_REWARD = -100000
GROUP_REWARD = -50000
DOUBLE_PATH_REWARD = -3000
FOOD_REWARD = 9000
# 10: 39.1
# 15: 48.8
# 15: 73.1 * with groups (-1000) 59.4, 66.8
# 15: 54.4 * with groups (-2000) 66.0
# 15: 67.5 * with lanes  (-2000) 83.0, 95.4, 86.7
# 15: 90.4 even food reward 90.6, 90.6 *136PR
MAX_SEARCH_DEPTH = 15

DIRECTIONS = ["up", "down", "left", "right"]
TRANSFORM_MAP = {'left': {'across': (1, 0), 'diag': {(1, -1), (1, 1)}},
                 'right': {'across': (-1, 0), 'diag': {(-1, -1), (-1, 1)}},
                 'up': {'across': (0, 1), 'diag': {(-1, 1), (1, 1)}},
                 'down': {'across': (0, -1), 'diag': {(-1, -1), (1, -1)}},
                 }
FAR_DIAGONALS = [[(-1, -1), (1, 1)], [(-1, 1), (1, -1)]]


def calculate_score(action, kitchen_sink):
    head = get_pos(action, kitchen_sink.snake.tilepos)

    # collides with walls
    if head[0] < 0 or head[0] >= SCREENTILES[0] + 1 or head[1] < 0 or head[1] >= SCREENTILES[1] + 1:
        return DEATH_REWARD, False

    grid_number = get_grid_number(head, kitchen_sink.game_grid)

    # collides with body
    if grid_number == 3:
        return DEATH_REWARD, False

    # move on grid
    updated_grid = update_grid(action, kitchen_sink.head.tilepos, kitchen_sink.tail.tilepos,
                               copy.deepcopy(kitchen_sink.game_grid))

    group_score = get_open_groups(updated_grid)

    double_path_score = get_double_path_score(action, updated_grid)

    # going to be trapped?
    trapped_score = get_steps_til_trapped(action, kitchen_sink)

    # food!
    food_score = get_food_score(grid_number, head, kitchen_sink.current_food)

    return trapped_score + food_score + group_score + double_path_score, double_path_score < 0


def get_grid_number(coord, grid):
    if coord[0] > SCREENTILES[0] + 1 or coord[1] > SCREENTILES[1] + 1:
        return 4
    if coord[0] < 0 or coord[1] < 0:
        return 4
    return grid[coord[1] + 1][coord[0] + 1]


def get_pos(action, tile_pos):
    return np.add(tile_pos, MOVE_VECTORS[action])


def get_open_groups(grid):
    # calculate groups
    groups = count_groups(grid)

    # return score based on # of groups
    return GROUP_REWARD * (groups - 1)


def update_grid(action, head_pos, tail_pos, grid):
    set_number(grid, get_pos(action, head_pos), 2)
    set_number(grid, tail_pos, 0)
    return grid


def set_number(grid, tile_pos, number):
    grid[tile_pos[1] + 1][tile_pos[0] + 1] = number


def count_groups(grid):
    remaining_spaces = set()
    # maybe groups is not needed? or big groups are ok?
    groups = []

    for i in range(grid.shape[0] - 1):
        for j in range(grid.shape[1] - 1):
            # if is 0, add to remaining_spaces.
            if is_open((i, j), grid):
                remaining_spaces.add((i, j))

    # for each grid element, recurse and try to connect
    while len(remaining_spaces) > 0:
        groups.append(set())
        populate_group(groups[-1], remaining_spaces.pop(), grid, remaining_spaces)

    return len(groups)


def populate_group(group_set, tile_pos, grid, remaining_spaces):
    group_set.add((tile_pos[0], tile_pos[1]))

    for action in DIRECTIONS:
        new_tile_pos = get_pos(action, tile_pos)
        pos_tuple = (new_tile_pos[0], new_tile_pos[1])

        if pos_tuple in remaining_spaces and is_open(new_tile_pos, grid):
            remaining_spaces.remove(pos_tuple)
            populate_group(group_set, new_tile_pos, grid, remaining_spaces)


# based on how many open tiles don't have perpendicular openings
def get_double_path_score(action, grid):
    not_double_path_positions = 0

    # for each open block, check if it has a path to get in and out
    for i in range(grid.shape[0] - 1):
        for j in range(grid.shape[1] - 1):
            if is_open((i, j), grid):
                if not has_double_path_opening((i, j), grid):
                    not_double_path_positions += 1

    # something is still wrong with corners
    # if not_double_path_positions > 0:
    #     print("not_double_path_positions: ", not_double_path_positions, " ", action)

    return DOUBLE_PATH_REWARD * not_double_path_positions


def has_double_path_opening(tile_pos, grid):
    check_diagonals = set()

    for direction in DIRECTIONS:
        if not is_open(get_pos(direction, tile_pos), grid):
            # other needs to be clear
            if not is_open(np.add(tile_pos, TRANSFORM_MAP[direction]['across']), grid):
                return False
            if len(check_diagonals) == 0:
                check_diagonals = TRANSFORM_MAP[direction]['diag']
            else:
                check_diagonals = check_diagonals.intersection(TRANSFORM_MAP[direction]['diag'])

    # check near diagonals
    for diagonals in check_diagonals:
        if not is_open(np.add(tile_pos, diagonals), grid):
            return False

    # check far diagonals
    for transform_set in FAR_DIAGONALS:
        if not is_open(np.add(tile_pos, transform_set[0]), grid) and not is_open(np.add(tile_pos, transform_set[1]),
                                                                                 grid):
            return False

    return True


def is_open(tile_pos, grid):
    grid_num = get_grid_number(tile_pos, grid)
    return grid_num == 0 or grid_num == 1


def get_food_score(head_grid_number, head_pos, current_food):
    distance_to_food = 0
    food_incentive = ((DOUBLE_PATH_REWARD * -3) * (5 - distance_to_food))
    if head_grid_number == 1:
        return FOOD_REWARD + food_incentive

    # calculate distance to food
    if current_food == 'no food':
        return 0
    distance_to_food = get_distance(head_pos, current_food.position)
    if distance_to_food < 5:
        return FOOD_REWARD + ((DOUBLE_PATH_REWARD * -3) * (5 - distance_to_food)) - distance_to_food * 10
    return FOOD_REWARD - distance_to_food * 10


# Manhattan distance
def get_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_steps_til_trapped(action, kitchen_sink):
    smol_snek = SmolSnek(kitchen_sink.snake, kitchen_sink.game_grid)
    steps = calc_trapped(action, copy.deepcopy(smol_snek), 0)
    if steps == MAX_SEARCH_DEPTH:
        # staying alive
        return 0
    return TRAPPED_REWARD + (steps * 10)


# crazy memoization idea:
#   save stack between moves

# multiprocess this could be up to 6x faster?
def calc_trapped(action, snake, total_steps):
    if total_steps == MAX_SEARCH_DEPTH:
        return total_steps

    # check collisions
    head = get_pos(action, snake.head.tile_pos)

    # collides with walls
    if head[0] < 0 or head[0] >= SCREENTILES[0] + 1 or head[1] < 0 or head[1] >= SCREENTILES[1] + 1:
        return total_steps

    grid_number = get_grid_number(head, snake.grid)

    # collides with body
    if grid_number == 3 or grid_number == 2:
        return total_steps

    snake.move(action)

    total_steps += 1

    most_steps = 0

    directions = [0, 1, 2, 3]
    # random.shuffle(random_directions)

    for choice in directions:
        copy_snek = copy.deepcopy(snake)
        steps = calc_trapped(DIRECTIONS[choice], copy_snek, total_steps)
        if most_steps < steps:
            most_steps = steps
            if most_steps == MAX_SEARCH_DEPTH:
                return most_steps

    return most_steps


class SmolSnek:
    def __init__(self, big_snake, grid):
        self.head = None
        self.tail = None
        self.copy_big_snake(big_snake)
        self.grid = copy.deepcopy(grid)

    class SnekSegment:
        def __init__(self, tile_pos):
            self.tile_pos = tile_pos
            self.front = None
            self.behind = None

    def copy_big_snake(self, big_snake):
        self.head = self.SnekSegment(big_snake.tilepos)
        current_segment = self.head
        big_segment = big_snake
        while big_segment.behind_segment is not None:
            current_segment.behind = self.SnekSegment(big_segment.behind_segment.tilepos)
            current_segment.behind.front = current_segment
            current_segment = current_segment.behind
            big_segment = big_segment.behind_segment
        self.tail = current_segment

    def move(self, action):
        # tail
        self.tail.front = None
        self.set_number(self.tail.tile_pos, 0)

        # head
        new_head = self.SnekSegment(get_pos(action, self.head.tile_pos))
        new_head.behind = self.head
        self.head.front = new_head
        self.head = new_head
        self.set_number(self.head.tile_pos, 3)

    def set_number(self, tile_pos, number):
        self.grid[tile_pos[1] + 1][tile_pos[0] + 1] = number