"""Validates diag.py's simulator against the real game, step by step.

The failure attribution rests entirely on the simulator agreeing with Snake.step() about what
kills the snake, so this replays random games in both and compares body, food and death.
"""
import os
import random
import sys

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Snake import Game
from snake_constants import ACTIONS
from diag import sim_step

random.seed(99)
game = Game(display=False, policy_name='simcheck')
mismatches = 0
steps = 0
deaths = 0
agreements = 0

for episode in range(60):
    game.reset()
    body = [tuple(p) for p in game.snake.get_positions()]
    food = tuple(game.current_food.position)
    move_dir = game.head.move_dir
    while True:
        action = random.choice(ACTIONS)
        predicted = sim_step(body, food, move_dir, action)
        finished, _ = game.step(action)
        steps += 1
        if finished and not game.perfect_game and not game.starved:
            deaths += 1
            if predicted is not None:
                print('MISMATCH: game died, simulator said the move was legal', body, action)
                mismatches += 1
            else:
                agreements += 1
            break
        if predicted is None:
            print('MISMATCH: simulator predicted death, game survived', body, action)
            mismatches += 1
            break
        if game.perfect_game or game.starved:
            break
        body, sim_food, move_dir = predicted
        real_body = [tuple(p) for p in game.snake.get_positions()]
        if body != real_body:
            print('MISMATCH body at step {0}:\n  sim  {1}\n  game {2}'.format(
                game.current_step, body, real_body))
            mismatches += 1
            break
        # The game places a new food at random when one is eaten; adopt it, since the
        # simulator has no way to predict where it lands.
        food = None if game.current_food == 'no food' else tuple(game.current_food.position)
        if sim_food is not None and food != sim_food:
            print('MISMATCH food: sim {0} game {1}'.format(sim_food, food))
            mismatches += 1
            break

print('{0} steps, {1} deaths, {2} death calls agreed, {3} mismatches'.format(
    steps, deaths, agreements, mismatches))
