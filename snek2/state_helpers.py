import math

import numpy as np

from snake_constants import *

# The four neighbour offsets, derived from MOVE_VECTORS so they cannot drift from it. Used by
# get_adjacent_groups; the flood fill in count_groups works on flat bit positions and shifts
# instead, so it needs no coordinate offsets at all.
NEIGHBOUR_OFFSETS = tuple(MOVE_VECTORS[direction] for direction in DIRECTIONS)

# Divisor that puts the starve observation in [0, 1]: the largest value log2plus1 can return for
# a budget that tops out at MAX_STARVE_BUDGET. Spelled with math.log2 rather than log2plus1 so it
# does not depend on where in this file that function is defined.
STARVE_OBS_SCALE = math.log2(MAX_STARVE_BUDGET + 1)

# Same idea for lg(open regions): puts it in [0, 1] against the measured/designed ceiling in
# MAX_GROUPS_FOR_SCALE, rather than the unbounded raw value - which reached ~4.4 on a vector of
# otherwise 0-1 inputs, the same scale mismatch the starve observation had before it was split.
GROUPS_OBS_SCALE = math.log2(MAX_GROUPS_FOR_SCALE + 1)


def get_observations(old_grid,
                     head_pos,
                     tail_pos,
                     next_tail_pos,
                     head_move_dir,
                     current_food,
                     current_step,
                     last_food_step,
                     snake_len):
    """Builds the 26-value observation vector. Layout, in order:

    idx      values  what
    0-5      6       food: [is closer, 1/(distance+1)] per action
    6-8      3       is the move safe (not body or wall)
    9-14     6       [can still reach tail, lg(open regions) scaled to [0, 1]] per action
    15-17    3       is it safe to chase the food (head, food and tail in one region)
    18-20    3       does the move win the game
    21       1       starve budget left, lg-compressed to [0, 1]
    22       1       fraction of the board the snake fills
    23-25    3       is the post-move head hugging a wall or body on its left or right

    Anything "per action" is ordered by ACTIONS — left, right, forward — as relative turns
    from the current heading, not compass directions. Keep this in step with
    SnakeEnvironment.observation_spec(), which sums the same counts.
    """
    observations = []

    # 6 values, [is closer, 1/(distance+1)] for each action. The flag is 1 when the move cuts
    # the Manhattan distance to the food; the reciprocal is 1 on the food and falls towards 0
    # far away. All six are 0 only when there is no food, which happens on a winning step.
    observations.extend(food_observations(old_grid, head_pos, current_food, head_move_dir))

    # 3 values, one per action: 1 if the move is survivable, 0 if it hits body or wall. Note
    # the polarity - 1 means safe. The tail's own cell counts as safe, because it vacates as
    # the snake advances.
    observations.extend(body_and_wall_collisions(old_grid, head_pos, tail_pos, head_move_dir))

    # 6 values, [can still reach tail, lg(open regions) scaled to [0, 1]] for each action, then
    # 3 more below.
    # Together these are the "am I about to trap myself" signal: reaching the tail means an
    # escape route exists, and a rising region count means the move is cutting the free space
    # into pieces. Needs both tail positions, because the tail moves too - see group_obs.
    group_values, food_chase_values, wall_hug_values = group_obs(old_grid,
                                                head_pos,
                                                tail_pos,
                                                next_tail_pos,
                                                head_move_dir,
                                                current_food)
    observations.extend(group_values)

    # 3 values, one per action: 1 when the head, the food and the tail all end up in the same
    # region, so the snake can reach the food and still get out afterwards. This is the "is it
    # safe to chase" signal - the food values above say which way the food is and
    # head_with_tail says an escape exists, but nothing tied the two together, so a policy had
    # no way to tell a reachable meal from one that seals it in. Computed inside group_obs to
    # share the flood fill.
    observations.extend(food_chase_values)

    # 3 values, one per action: 1 if the move eats the last food and fills the board. All zero
    # unless the snake is exactly one food short, so this fires on the final move of a game.
    observations.extend(perfect_game_obs(old_grid, head_pos, head_move_dir, snake_len))

    # 2 values: how much of the starve budget is left, lg-compressed and scaled to [0, 1], and
    # how much of the board the snake fills, linear. The second is the only signal for how far
    # through the game this is. These used to be one entangled value that went flat from length
    # 50 up and reached 8.97 on a vector of otherwise sub-3 inputs - see starve_and_length_obs.
    # The board-fill value also supersedes the "remaining spaces" observation that
    # observation_spec() used to reserve a disabled slot for: open cells are the complement of
    # snake length, so it is the same signal already normalised.
    observations.extend(starve_and_length_obs(current_step, last_food_step, snake_len))

    # 3 values, one per action: 1 when the post-move head has a wall or body immediately to its
    # left or right, 0 when both sides are open (or the move is fatal). The intent is to let a
    # policy learn to travel along a wall or an existing pocket boundary rather than through the
    # middle of open space next to one, which can split one large pocket into two smaller ones -
    # harder to use later than the single pocket was. Unvalidated: this is a hypothesis about
    # what the feature enables, not yet a measured effect. Computed inside group_obs to reuse
    # its legality gate and its post-move grid - see group_obs.
    observations.extend(wall_hug_values)

    # There used to be a final "is the episode over" value here, and it is gone. It was 1 only in
    # a terminal observation, which no policy ever acts on, so it was a constant 0 for every state
    # the network is asked about — but it could not simply be deleted, because the environment was
    # handing terminal steps a non-zero discount and the loss therefore bootstrapped off terminal
    # Q-values. This flag was the only signal the network could use to learn that those states are
    # worth nothing. With to_tensor_time_step() now zeroing that discount, nothing reads a
    # terminal state's value and the input has no job left.
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
# total_group_obs: returns lg(number of groups), scaled to [0, 1]
# safe_to_chase_food: returns 1 when head, food and tail all end up in one group
# hugging_wall: returns 1 when the post-move head has a wall or body immediately to its left
#   or right, 0 if both sides are open or the move is fatal
def group_obs(old_grid, head_pos, tail_pos, next_tail_pos, head_move_dir, current_food):
    """Returns three lists, each one value per action: [can reach tail, lg(open regions) scaled
    to [0, 1]], [safe to chase the food], and [is the post-move head hugging a wall or body].

    Three lists rather than three functions because they share all of their expensive work.
    `count_groups` is roughly 46% of the cost of building an observation and runs three times
    here, once per action; computing the other two elsewhere would either run it three more
    times for nothing (safe-to-chase, which reads the regions it produces) or duplicate the
    legality gate and the post-move grid (hugging-wall, which needs neither the flood fill nor
    the regions, but does need to know a move is legal before asking what its neighbours are,
    and needs the grid *after* the move for the tail's vacated cell to read as open).

    **Safe to chase the food** is 1 when the head, the food and the tail all land in *one* region
    after the move - so there is a route to the food and a route back out afterwards. One region,
    not merely all three reachable: the head can neighbour two regions at once, and a meal in one
    of them with the only escape through the other is the trap the flag exists to name.
    Reaching the food is useless if it seals the snake in with it, and the two halves of that
    question were previously only answerable separately: `head_with_tail` says an escape exists
    and the food values say which way the food is, with nothing tying them together.
    `observation_spec` had a `head_with_food_obs` slot reserved and disabled for the weaker
    version of this, food reachability alone.

    A move that eats is the special case: the food is gone, so the flag is just whether the
    tail is still reachable from the cell the food was on. Without that branch the food's cell
    is occupied by the head and belongs to no region, so an eating move would always read 0 -
    precisely the move the flag exists to encourage.

    A fatal move - wall or body, from `body_and_wall_collisions`' own test - reports zero for
    every value here rather than "what if this move were legal". A move into a wall happens to
    read zero already: the only on-board neighbour of an off-board cell is the vacated head
    cell, which `update_grid` never clears, so no region test can find anything past it. A move
    into the snake's own body has no such accident protecting it - the new head cell is still
    on-board and can see whatever real regions sit beside it - so it was reporting `1` on 5,289
    of 14,642 body-collision actions measured in play, `head_with_tail` describing a hypothetical
    survivor of a move that kills the snake. Harmless as long as indices 6-8 mark the same move
    fatal and nothing downstream trusts one flag without the other, but there is no upside to
    leaving a fatal move free to say anything at all, so it is short-circuited before the flood
    fill runs - which also skips `count_groups` for it entirely.

    **Hugging a wall** is 1 when the cell immediately to the head's left *or* right, after the
    move and relative to the heading the move leaves it facing, is a wall or body segment; 0 if
    both sides are open, and 0 for a fatal move regardless of the geometry (there is no
    "afterwards" to hug anything in). "Left" and "right" are found by looking up
    `CURRENT_DIRECTION_MAPS` a second time, treating the head's *new* facing as the base
    direction rather than its old one, which is exactly the question "what is 90 degrees off my
    new heading" - the same table already answers "what is my new heading" for the action
    itself, one level up.

    The intent is to let a policy learn to travel *along* a wall or an existing pocket boundary
    rather than through open air next to one: cutting through the middle of free space can leave
    two smaller pockets where there was one big one, and a big single pocket is easier to use
    later than two small ones. This does not exist yet as a validated finding - it is a
    hypothesis about what the feature should let the network learn, not a measured effect.

    Checked against the grid *after* the move (the same `grid = update_grid(...)` used for the
    regions below), not the grid before it, so that a cell the tail is vacating this step reads
    as open rather than as body. The one place this matters is narrow: the left or right cell
    would have to be exactly the tail's current position for the two grids to disagree, which
    only arises in a tight coil. Whether a cell is a wall is unaffected either way, since
    `update_grid` never touches the padded wall ring.

    Takes two tail positions, because the tail moves on the same step the head does.
    `tail_pos` is where it is now; `next_tail_pos` is the cell ahead of it, which is where it
    lands on any move that does not eat. Both are needed: the vacated cell decides what
    `update_grid` frees, and the *post-move* tail is what "can the head reach the tail" is
    actually about.

    Using `tail_pos` for both was a defect, and a quiet one. Measured over 360 episodes of the
    92%-era champion, the flag separated the move that lost the game from a surviving move at
    only 22.1% of the 68 decisions that lost one, against 94.1% once the tail advances - it
    never claimed the wrong thing, it just went silent. The reason is that `update_grid` frees
    the vacated cell, and in a coiled endgame that cell is walled in by the snake's own body,
    so it becomes a region with no open neighbours and `get_adjacent_groups` returns nothing
    for it whatever the head does. See ../claudeFeatureRecommendations.md.
    """
    observations = []
    food_observations = []
    wall_hug_observations = []
    cols = old_grid.shape[1]
    # Bit for the food's cell, matching count_groups' layout, or None when there is no food -
    # which happens on the step that wins the game.
    food_bit = None
    if current_food != 'no food':
        food_bit = 1 << ((current_food.position[1] + 1) * cols + (current_food.position[0] + 1))

    for action in ACTIONS:
        new_head_pos = get_relative_pos(action, head_pos, head_move_dir)
        # Read from old_grid: it is untouched by update_grid, so this is one lookup against the
        # board as it stands before the move, same as body_and_wall_collisions' own test - grid
        # value 1 (food) or 0 (open), or the tail-follow special case, is what "legal" means. A
        # wall (4) or the snake's own body (3) is anything else.
        grid_value = get_grid_value(new_head_pos, old_grid)
        eats_food = grid_value == 1
        if not (eats_food or grid_value == 0 or tuple(new_head_pos) == tuple(tail_pos)):
            # Fatal move: no group means anything for a head that does not survive to occupy
            # one. Skips update_grid and count_groups for this action entirely.
            observations.extend([0, 0])
            food_observations.append(0)
            wall_hug_observations.append(0)
            continue

        # .copy() rather than copy.deepcopy(): an ndarray copy is already independent, and
        # deepcopy walks the object graph to reach the same result. Worth almost nothing on
        # its own (0.3% of the profile) but there is no reason to pay it.
        grid = update_grid(action, head_pos, tail_pos, old_grid.copy(), head_move_dir)

        # The heading this move leaves the head facing, and the cardinal directions 90 degrees
        # either side of it - CURRENT_DIRECTION_MAPS answers "what heading does turning left (or
        # right) from here produce", which is exactly "what is to my left (or right)" when asked
        # of the *new* heading rather than the old one used to get new_head_pos itself.
        new_facing = CURRENT_DIRECTION_MAPS[head_move_dir][action]
        left_vector = MOVE_VECTORS[CURRENT_DIRECTION_MAPS[new_facing]['left']]
        right_vector = MOVE_VECTORS[CURRENT_DIRECTION_MAPS[new_facing]['right']]
        left_pos = (new_head_pos[0] + left_vector[0], new_head_pos[1] + left_vector[1])
        right_pos = (new_head_pos[0] + right_vector[0], new_head_pos[1] + right_vector[1])
        left_blocked = get_grid_value(left_pos, grid) not in (0, 1)
        right_blocked = get_grid_value(right_pos, grid) not in (0, 1)
        wall_hug_observations.append(1 if (left_blocked or right_blocked) else 0)

        regions, group_count = count_groups(grid)

        head_with_tail = 0
        # Scaled to [0, 1] against the design cap in GROUPS_OBS_SCALE - see its comment and
        # MAX_GROUPS_FOR_SCALE. The raw log2plus1(group_count) reached ~4.4 in measured play,
        # the same kind of scale mismatch the starve observation had before it was split.
        num_groups = log2plus1(group_count) / GROUPS_OBS_SCALE

        # A move that eats does not move the tail at all: add_segment() refills the tile it
        # came from, so the snake grows from the back and the tail stays where it is.
        post_move_tail = tail_pos if eats_food else next_tail_pos

        head_groups = get_adjacent_groups(regions, cols, new_head_pos)
        tail_groups = get_adjacent_groups(regions, cols, post_move_tail)

        # The second clause stays, and it compares against the *current* tail: stepping onto
        # the cell the tail is vacating is always safe, and no region test can see that move,
        # because the cell ends up holding the head rather than being open. Dropping it while
        # advancing the tail looked catastrophic in measurement - 1481 spurious disagreements
        # in 15700 actions - and that single move was the whole difference.
        if len(head_groups & tail_groups) > 0 or tuple(new_head_pos) == tuple(tail_pos):
            head_with_tail = 1

        # Safe to chase: one region has to hold the food while touching both the head and the
        # tail. `head_groups & tail_groups` is that set, and testing the food against the
        # intersection rather than against head_groups alone is the whole point - the head can
        # sit next to two regions at once, and reaching the food through one while the tail is
        # only reachable through the other is exactly the trap this flag is meant to name.
        #
        # The food test is containment, not adjacency: a food cell is open and belongs to a
        # region of its own, unlike the head and tail cells, which are occupied and can only be
        # asked what they neighbour.
        if tuple(new_head_pos) == tuple(tail_pos):
            # Following the tail. Nothing reaches the vacated cell by a region test, so take the
            # regions the head can see from there; the tail is one step ahead by construction.
            escape_regions = head_groups
        else:
            escape_regions = head_groups & tail_groups

        safe_to_chase_food = 0
        if food_bit is not None:
            if eats_food:
                # The food is being taken this step, so there is no food cell left to reach and
                # the only question left is whether the tail survives the move.
                safe_to_chase_food = head_with_tail
            elif any(regions[index] & food_bit for index in escape_regions):
                safe_to_chase_food = 1

        observations.extend([head_with_tail, num_groups])
        food_observations.append(safe_to_chase_food)

    return observations, food_observations, wall_hug_observations


def starve_budget(snake_len):
    """Steps this snake may go without food. A game rule, not an observation."""
    return max(MIN_STARVE_BUDGET,
               min(snake_len * MAX_STEPS_BEFORE_STARVE_SIZE_MULTIPLIER, MAX_STARVE_BUDGET))


def steps_until_starve(current_step, last_food_step, snake_len):
    """Steps left before starving, as a plain count. Snake.step() ends the episode at 0 or less.

    Returns an int, where this used to return `[log2plus1(budget - elapsed)]`. The rule fires at
    exactly the same moment either way, since `log2plus1(x) <= 0` precisely when `x <= 0`, but
    the game rule and the observation were reading the same function - so rescaling the
    observation would silently have moved the starvation threshold. They are separate now.
    """
    return starve_budget(snake_len) - (current_step - last_food_step)


def starve_and_length_obs(current_step, last_food_step, snake_len):
    """[starve budget left, how much of the board the snake fills], both in [0, 1].

    Replaces a single value, `log2plus1(budget - elapsed)`, which had two problems.

    It was the only input carrying anything about the snake's length, and it carried nothing
    past length 50. The budget is `min(10 * len, 500)`, so from 50 segments up every length
    produces the identical value at equal elapsed steps - one number, 8.9687, for the whole
    second half of every game, while the fatal decisions sit at a median length of 83. Length
    now has its own input. Linear, not log: the difference between 80 and 90 segments matters at
    least as much as 20 to 30, so there is nothing to compress.

    It also reached 8.97 while every other input sat at or below 3.17. With one shared weight
    initialisation that lets a single input dominate the first layer's gradients for no reason
    beyond its units. Dividing by STARVE_OBS_SCALE puts it in [0, 1] and keeps the log
    compression, which is worth keeping on its own merits: the difference between 10 and 20
    steps of budget is worth reacting to, the difference between 400 and 410 is not.

    The budget is scaled by its own maximum rather than by this snake's budget, so a given value
    always means the same number of steps left whatever the length. The network can work out the
    fraction from the two together, now that it can see length at all.
    """
    remaining = steps_until_starve(current_step, last_food_step, snake_len)
    return [log2plus1(max(0, remaining)) / STARVE_OBS_SCALE,
            snake_len / PERFECT_SCORE]


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
