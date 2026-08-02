"""Same validation as simcheck.py, but driven by the tail-following heuristic.

Random play dies around step 11, so it never tests the crowded late-game states the failure
attribution actually runs in. The heuristic fills the board, so this compares simulator and
game at every length up to a perfect game.
"""
import os
import random
import sys

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Snake import Game
from snake_constants import ACTIONS
from diag import branch_facts, sim_step

random.seed(4242)
game = Game(display=False, policy_name='simcheck')
steps = mismatches = 0
outcomes = {'perfect': 0, 'dead': 0, 'starved': 0}
max_len = 0

for episode in range(12):
    game.reset()
    body = [tuple(p) for p in game.snake.get_positions()]
    food = tuple(game.current_food.position)
    move_dir = game.head.move_dir
    while True:
        best = None
        for action in ACTIONS:
            nxt = sim_step(body, food, move_dir, action)
            if nxt is None:
                continue
            area, tail_ok = branch_facts(nxt[0], nxt[1])
            # Food-seeking on purpose: ranking by area alone avoids food (growing shrinks the
            # free area), so the snake circled at length 5 and starved, never exercising the
            # rule that a step which eats does *not* vacate the tail.
            if food is None:
                gap = 0
            else:
                gap = abs(nxt[0][0][0] - food[0]) + abs(nxt[0][0][1] - food[1])
            rank = (1 if tail_ok else 0, -gap, area)
            if best is None or rank > best[0]:
                best = (rank, action, nxt)
        if best is None:
            # Simulator says every move is fatal; the game should die on whatever we pick.
            action = 'forward'
            finished, _ = game.step(action)
            steps += 1
            if not finished:
                print('MISMATCH: simulator saw no legal move, game survived', body)
                mismatches += 1
            outcomes['dead'] += 1
            break

        action = best[1]
        predicted_body, predicted_food, predicted_dir = best[2]
        finished, _ = game.step(action)
        steps += 1
        max_len = max(max_len, len(game.snake_group))
        if game.perfect_game:
            outcomes['perfect'] += 1
            break
        if game.starved:
            outcomes['starved'] += 1
            break
        if finished:
            print('MISMATCH: game died on a move the simulator called legal', body, action)
            mismatches += 1
            outcomes['dead'] += 1
            break
        real_body = [tuple(p) for p in game.snake.get_positions()]
        if predicted_body != real_body:
            print('MISMATCH body at step {0} len {1}:\n  sim  {2}\n  game {3}'.format(
                game.current_step, len(real_body), predicted_body, real_body))
            mismatches += 1
            break
        body, move_dir = predicted_body, predicted_dir
        food = None if game.current_food == 'no food' else tuple(game.current_food.position)

print('{0} steps, max length {1}, outcomes {2}, mismatches {3}'.format(
    steps, max_len, outcomes, mismatches))
