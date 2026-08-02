"""Dumps concrete states where the `holding` and `newtail` tail tests disagree."""
import os
import random
import sys

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Snake import Game
from snake_constants import ACTIONS
from state_helpers import (count_groups, get_adjacent_groups, get_grid_value,
                           get_relative_pos, update_grid)
from diag import branch_facts, sim_step

OPEN = (0, 1)
shown = 0
random.seed(5)
game = Game(display=False, policy_name='probe')

for episode in range(40):
    if shown >= 3:
        break
    game.reset()
    body = [tuple(p) for p in game.snake.get_positions()]
    food = tuple(game.current_food.position)
    move_dir = game.head.move_dir
    while shown < 3:
        grid = game.grid
        cols = grid.shape[1]
        head = game.head.tile_pos
        tail = tuple(game.tail.tile_pos)
        positions = [tuple(p) for p in game.snake.get_positions()]
        ahead = positions[-2]
        for action in ACTIONS:
            new_head = get_relative_pos(action, head, move_dir)
            value = get_grid_value(new_head, grid)
            if not (value in OPEN or tuple(new_head) == tail):
                continue
            eats = value == 1
            moved = update_grid(action, head, tail, grid.copy(), move_dir)
            regions, _ = count_groups(moved)
            head_regions = get_adjacent_groups(regions, cols, new_head)
            adjacent = get_adjacent_groups(regions, cols, tail)
            tail_bit = 1 << ((tail[1] + 1) * cols + (tail[0] + 1))
            contains = {i for i, r in enumerate(regions) if r & tail_bit}
            holding = bool(head_regions & (contains | adjacent)) or tuple(new_head) == tail
            post = tail if eats else ahead
            newtail = bool(get_adjacent_groups(regions, cols, post) & head_regions) \
                or tuple(new_head) == post
            if holding == newtail:
                continue
            shown += 1
            print('=' * 72)
            print('len {0} action {1} eats {2}  holding={3} newtail={4}'.format(
                len(positions), action, eats, holding, newtail))
            print('head {0} -> {1};  tail {2};  segment ahead of tail {3}'.format(
                head, new_head, tail, ahead))
            print('tail cell value after the move: {0}'.format(get_grid_value(tail, moved)))
            print('tail adjacent to segment-ahead? {0}'.format(
                abs(tail[0] - ahead[0]) + abs(tail[1] - ahead[1]) == 1))
            print('duplicate positions in body: {0}'.format(
                len(positions) - len(set(positions))))
            print('regions: {0}, head_regions {1}, contains {2}, adjacent(tail) {3}, '
                  'adjacent(ahead) {4}'.format(
                      len(regions), head_regions, contains, adjacent,
                      get_adjacent_groups(regions, cols, ahead)))
            print('region sizes: {0}'.format([r.bit_count() for r in regions]))
            for y in range(moved.shape[0]):
                row = ''
                for x in range(moved.shape[1]):
                    cell = (x - 1, y - 1)
                    value2 = int(moved[y][x])
                    char = {0: '.', 1: 'F', 2: 'H', 3: '#', 4: 'W'}[value2]
                    if cell == tuple(new_head):
                        char = 'N'
                    elif cell == tail:
                        char = 't'
                    elif cell == ahead:
                        char = 'a'
                    row += char
                print('   ' + row)
            if shown >= 3:
                break
        best = None
        for action in ACTIONS:
            nxt = sim_step(body, food, move_dir, action)
            if nxt is None:
                continue
            area, tail_ok = branch_facts(nxt[0], nxt[1])
            gap = 0 if food is None else (abs(nxt[0][0][0] - food[0])
                                         + abs(nxt[0][0][1] - food[1]))
            rank = (1 if tail_ok else 0, -gap, area)
            if best is None or rank > best[0]:
                best = (rank, action)
        if best is None:
            break
        game.step(best[1])
        if game.finished:
            break
        body = [tuple(p) for p in game.snake.get_positions()]
        food = None if game.current_food == 'no food' else tuple(game.current_food.position)
        move_dir = game.head.move_dir
