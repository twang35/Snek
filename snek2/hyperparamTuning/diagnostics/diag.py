"""How does the champion actually fail, and what could its observation vector have seen?

    PYTHONPATH=. python -u diag.py <policy> <step> <episodes> <seed> <out.json>

Three measurements, all on the greedy policy in the current environment:

  1. Failure attribution. Every lost episode replays its last 40 decisions through an
     independent simulator. At each one, for each legal move, a simple tail-following
     heuristic tries to survive 60 more steps. The latest decision where the *chosen* move
     loses and some alternative survives is the pinpointed mistake — and the facts about the
     board at that moment say which observation would have flagged it.

  2. Observation aliasing. Steps are grouped by their exact observation vector. Within a
     group, the spread of snake length and of reachable free area is information the policy
     provably does not have, because two states it cannot distinguish differ in it.

  3. Reachability of the food, which the observation space reserves a slot for and disables.
"""
import collections
import json
import os
import random
import sys

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from snake_constants import (ACTIONS, CURRENT_DIRECTION_MAPS, MOVE_VECTORS, PERFECT_SCORE,
                             POLICY_DIR, SCREENTILES)
from snake_environment import SnakeEnvironment
from snek2 import build_q_net
from state_helpers import (count_groups, get_adjacent_groups, get_grid_value,
                           get_relative_pos, update_grid)

OPEN = (0, 1)
NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))
WINDOW = 40      # decisions replayed before each loss
HORIZON = 60     # steps the heuristic must survive to call a branch survivable


# ---------------- an independent simulator, so the replay does not depend on pygame ----------

def build_grid(body, food):
    """The same padded grid Snake._rebuild_grid() builds, from a body list and food cell."""
    grid = np.zeros((SCREENTILES[1] + 3, SCREENTILES[0] + 3))
    grid[[0, -1], :] = 4
    grid[:, [0, -1]] = 4
    if food is not None:
        grid[food[1] + 1, food[0] + 1] = 1
    for index, position in enumerate(body):
        grid[position[1] + 1, position[0] + 1] = 2 if index == 0 else 3
    return grid


def sim_step(body, food, move_dir, action):
    """One move. Returns (body, food, move_dir) or None if the move is fatal.

    Mirrors Snake.step(): the tail vacates as the head advances, except on a step that eats,
    where add_segment() refills the tile it came from.
    """
    new_dir = CURRENT_DIRECTION_MAPS[move_dir][action]
    vector = MOVE_VECTORS[new_dir]
    new_head = (body[0][0] + vector[0], body[0][1] + vector[1])
    if not (0 <= new_head[0] <= SCREENTILES[0] and 0 <= new_head[1] <= SCREENTILES[1]):
        return None
    eats = food is not None and new_head == food
    occupied = body if eats else body[:-1]
    if new_head in occupied:
        return None
    return ([new_head] + list(occupied if eats else body[:-1]),
            None if eats else food, new_dir)


def branch_facts(body, food):
    """(reachable free area, is the tail reachable) for a post-move body."""
    grid = build_grid(body, food)
    cols = grid.shape[1]
    regions, _ = count_groups(grid)
    head_regions = get_adjacent_groups(regions, cols, body[0])
    tail_regions = get_adjacent_groups(regions, cols, body[-1])
    area = sum(regions[index].bit_count() for index in head_regions)
    return area, bool(head_regions & tail_regions)


def heuristic_survives(body, food, move_dir, horizon=HORIZON):
    """Plays tail-following, area-maximising moves and reports whether it lasts `horizon`.

    Survival here is a lower bound on what was possible: a better player may survive where
    this one dies, so 'no branch survived' is weaker evidence than 'this branch survived'.
    """
    for _ in range(horizon):
        if len(body) >= PERFECT_SCORE:
            return True
        best = None
        for action in ACTIONS:
            nxt = sim_step(body, food, move_dir, action)
            if nxt is None:
                continue
            area, tail_ok = branch_facts(nxt[0], nxt[1])
            rank = (1 if tail_ok else 0, area)
            if best is None or rank > best[0]:
                best = (rank, nxt)
        if best is None:
            return False
        body, food, move_dir = best[1]
    return True


# ---------------- what the observation leaves out, per action, in the live game -------------

def action_facts(game):
    grid = game.grid
    cols = grid.shape[1]
    head = game.head.tile_pos
    tail = tuple(game.tail.tile_pos)
    move_dir = game.head.move_dir
    food = game.current_food
    food_bit = None
    if food != 'no food':
        food_bit = 1 << ((food.position[1] + 1) * cols + (food.position[0] + 1))

    facts = []
    for action in ACTIONS:
        new_head = get_relative_pos(action, head, move_dir)
        value = get_grid_value(new_head, grid)
        safe = value in OPEN or tuple(new_head) == tail
        row = {'safe': bool(safe), 'eats': bool(value == 1), 'area': 0, 'regions': 0,
               'tail_reachable': False, 'food_reachable': None, 'degree': 0}
        if safe:
            moved = update_grid(action, head, tail, grid.copy(), move_dir)
            regions, count = count_groups(moved)
            head_regions = get_adjacent_groups(regions, cols, new_head)
            tail_regions = get_adjacent_groups(regions, cols, tail)
            row['regions'] = count
            row['area'] = sum(regions[index].bit_count() for index in head_regions)
            row['tail_reachable'] = bool(head_regions & tail_regions
                                         or tuple(new_head) == tail)
            if food_bit is not None and not row['eats']:
                row['food_reachable'] = any(regions[index] & food_bit
                                            for index in head_regions)
            row['degree'] = sum(
                1 for offset in NEIGHBOURS
                if get_grid_value((new_head[0] + offset[0], new_head[1] + offset[1]),
                                  moved) in OPEN)
        facts.append(row)
    return facts


def attribute(buffer):
    """Finds the latest buffered decision where the chosen move loses and another survives."""
    for offset in range(len(buffer) - 1, -1, -1):
        state = buffer[offset]
        body, food, move_dir = state['body'], state['food'], state['dir']
        survives = []
        for index, action in enumerate(ACTIONS):
            nxt = sim_step(body, food, move_dir, action)
            survives.append(bool(nxt) and heuristic_survives(*nxt))
        chosen = state['chosen']
        if not survives[chosen] and any(survives):
            alternatives = [i for i, ok in enumerate(survives) if ok]
            chosen_facts = state['facts'][chosen]
            best = max(alternatives, key=lambda i: state['facts'][i]['area'])
            alt_facts = state['facts'][best]
            return {
                'lead': len(buffer) - 1 - offset,
                'len': state['len'],
                'n_safe': sum(1 for f in state['facts'] if f['safe']),
                'chosen_safe': chosen_facts['safe'],
                'chosen_area': chosen_facts['area'],
                'alt_area': alt_facts['area'],
                'chosen_tail': chosen_facts['tail_reachable'],
                'alt_tail': alt_facts['tail_reachable'],
                'chosen_degree': chosen_facts['degree'],
                'alt_degree': alt_facts['degree'],
                'chosen_regions': chosen_facts['regions'],
                'alt_regions': alt_facts['regions'],
            }
    return None


def main():
    policy_name, step, episodes, seed, out_path = (
        sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    random.seed(seed)
    np.random.seed(seed)

    env = SnakeEnvironment(discount=0.99, display=False, policy_name=policy_name)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    action_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_spec.maximum - action_spec.minimum + 1
    global_step = tf.compat.v1.train.get_or_create_global_step()
    agent = dqn_agent.DdqnAgent(
        tf_env.time_step_spec(), tf_env.action_spec(),
        q_network=build_q_net(num_actions), epsilon_greedy=0.0,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        td_errors_loss_fn=common.element_wise_huber_loss,
        target_update_period=8, train_step_counter=global_step)
    agent.initialize()
    checkpoint = tf.train.Checkpoint(agent=agent, policy=agent.policy, global_step=global_step)
    checkpoint.restore(os.path.join(POLICY_DIR, policy_name,
                                    'ckpt-{0}'.format(step))).expect_partial()
    policy_action = common.function(agent.policy.action)

    game = env._game
    alias = {}
    geom_alias = {}
    outcomes = []
    counters = collections.Counter()
    choice = collections.Counter()

    for episode in range(episodes):
        time_step = tf_env.reset()
        buffer = collections.deque(maxlen=WINDOW)
        while not time_step.is_last():
            snake_len = len(game.snake_group)
            facts = action_facts(game)
            obs = [round(float(x), 6) for x in game.get_observation()]
            safe = [i for i, f in enumerate(facts) if f['safe']]
            areas = [facts[i]['area'] for i in safe]
            best_area = max(areas) if areas else 0

            key = hash(tuple(obs))
            geom_key = hash(tuple(obs[:18] + obs[19:]))
            for table, table_key in ((alias, key), (geom_alias, geom_key)):
                row = table.get(table_key)
                if row is None:
                    table[table_key] = [1, snake_len, snake_len, best_area, best_area]
                else:
                    row[0] += 1
                    row[1] = min(row[1], snake_len)
                    row[2] = max(row[2], snake_len)
                    row[3] = min(row[3], best_area)
                    row[4] = max(row[4], best_area)

            late = snake_len >= 50
            counters['steps'] += 1
            counters['steps_late'] += late
            reachable = [facts[i]['food_reachable'] for i in safe]
            if reachable and all(r is False for r in reachable):
                counters['food_sealed'] += 1
                counters['food_sealed_late'] += late
            if len(safe) > 1:
                counters['choices'] += 1
                spread = max(areas) - min(areas)
                if spread >= 5:
                    counters['choices_area_spread5'] += 1
                if not all(facts[i]['tail_reachable'] == facts[safe[0]]['tail_reachable']
                           for i in safe):
                    counters['choices_tail_differs'] += 1

            chosen = int(policy_action(time_step).action.numpy()[0])
            if len(safe) > 1 and max(areas) - min(areas) >= 5:
                if chosen not in safe:
                    choice['fatal'] += 1
                elif facts[chosen]['area'] == max(areas):
                    choice['largest_area'] += 1
                elif facts[chosen]['area'] == min(areas):
                    choice['smallest_area'] += 1
                else:
                    choice['middle_area'] += 1

            buffer.append({'body': [tuple(p) for p in game.snake.get_positions()],
                           'food': (None if game.current_food == 'no food'
                                    else tuple(game.current_food.position)),
                           'dir': game.head.move_dir, 'facts': facts,
                           'chosen': chosen, 'len': snake_len})
            time_step = tf_env.step(tf.constant([chosen], dtype=tf.int32))

        outcome = ('perfect' if game.perfect_game
                   else 'starved' if game.starved else 'collision')
        record = {'outcome': outcome, 'score': game.current_score, 'steps': game.current_step}
        if outcome != 'perfect':
            record['final_safe_moves'] = sum(1 for f in buffer[-1]['facts'] if f['safe'])
            record['blame'] = attribute(list(buffer))
        outcomes.append(record)
        print('  episode {0}: {1} score {2}{3}'.format(
            episode + 1, outcome, game.current_score,
            '' if outcome == 'perfect' else ' blame=' + str(record['blame'] and
                                                            record['blame']['lead'])),
            flush=True)

    json.dump({'policy': policy_name, 'step': step, 'seed': seed, 'window': WINDOW,
               'horizon': HORIZON, 'outcomes': outcomes, 'counters': dict(counters),
               'choice': dict(choice), 'alias': summarize(alias),
               'geom_alias': summarize(geom_alias)}, open(out_path, 'w'))
    print('wrote {0}'.format(out_path))


def summarize(table):
    steps = classes = repeated = 0
    len_spreads = []
    area_spreads = []
    for count, min_len, max_len, min_area, max_area in table.values():
        steps += count
        classes += 1
        if count > 1:
            repeated += count
            len_spreads.append(max_len - min_len)
            area_spreads.append(max_area - min_area)
    len_spreads.sort()
    area_spreads.sort()

    def at(values, quantile):
        if not values:
            return 0
        return values[min(len(values) - 1, int(len(values) * quantile))]

    return {'steps': steps, 'classes': classes, 'repeated_steps': repeated,
            'len_spread_median': at(len_spreads, 0.5), 'len_spread_p90': at(len_spreads, 0.9),
            'len_spread_max': at(len_spreads, 1.0),
            'area_spread_median': at(area_spreads, 0.5),
            'area_spread_p90': at(area_spreads, 0.9)}


if __name__ == '__main__':
    main()
