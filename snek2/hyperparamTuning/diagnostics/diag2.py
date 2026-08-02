"""Scores candidate observations against the decisions that actually lost the game.

    PYTHONPATH=. python -u diag2.py <policy> <step> <episodes> <seed> <out.json>

diag.py established that 70 of 72 losses turn on one identifiable decision, and that neither
the free area nor the existing tail-reachability flag separates the surviving move from the
chosen one — the flag reads False for both branches in 72% of those decisions, because it is a
static snapshot of a board whose cells are about to empty.

This runs the same 360 games and, at each blamed decision, asks of every candidate feature:
would it have ranked the surviving move above the one that lost? The headline number per
candidate is `right` vs `wrong` vs `tie`.

Candidates, all cheap enough to compute three times per step:

  timed_area   cells reachable by a walker for whom a body cell opens up once that segment has
               passed. Segment i (0 = head) vacates in len - i steps, so a cell is passable to
               a walker arriving at time d when d >= len - i. This is the free area the snake
               will actually have, rather than the free area it has this instant.
  timed_tail   whether that walker can reach the tail cell at all.
  timed_depth  how far into the future the walker gets before running out of board.
  corridor     open cells in a straight line ahead of the landing cell.
  degree       open neighbours of the landing cell.
  static_area  the instantaneous free area, for comparison.
"""
import collections
import json
import os
import sys

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import random

import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from snake_constants import ACTIONS, MOVE_VECTORS, POLICY_DIR, SCREENTILES
from snake_environment import SnakeEnvironment
from snek2 import build_q_net

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag import (WINDOW, action_facts, branch_facts, heuristic_survives, sim_step)

NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLS = SCREENTILES[0] + 1
ROWS = SCREENTILES[1] + 1


def timed_walk(body, horizon=None):
    """Breadth-first walk from the head where body cells open on the schedule they vacate.

    Returns (cells reachable, tail cell reachable, deepest time reached). Ignores food, so a
    step that grows the snake is treated as one that does not — the schedule it computes is the
    optimistic one, which is the right bias for a feature meant to spot a closing trap.
    """
    length = len(body)
    if horizon is None:
        horizon = length
    frees = {}
    for index, position in enumerate(body):
        frees[position] = length - index
    tail = body[-1]
    start = body[0]
    seen = {start}
    frontier = [start]
    depth = 0
    reached_tail = False
    while frontier and depth < horizon:
        depth += 1
        nxt = []
        for cell in frontier:
            for offset in NEIGHBOURS:
                candidate = (cell[0] + offset[0], cell[1] + offset[1])
                if candidate in seen:
                    continue
                if not (0 <= candidate[0] < COLS and 0 <= candidate[1] < ROWS):
                    continue
                blocked_until = frees.get(candidate)
                if blocked_until is not None and depth < blocked_until:
                    continue        # that segment has not passed yet
                seen.add(candidate)
                nxt.append(candidate)
                if candidate == tail:
                    reached_tail = True
        frontier = nxt
    return len(seen) - 1, reached_tail, depth


def corridor_length(body, food, move_dir):
    """Open cells in a straight line ahead of the head."""
    occupied = set(body)
    vector = MOVE_VECTORS[move_dir]
    cell = body[0]
    count = 0
    while True:
        cell = (cell[0] + vector[0], cell[1] + vector[1])
        if not (0 <= cell[0] < COLS and 0 <= cell[1] < ROWS):
            return count
        if cell in occupied:
            return count
        count += 1


def candidates(body, food, move_dir):
    """Every candidate feature for a post-move state."""
    area, tail_ok = branch_facts(body, food)
    timed_area, timed_tail, timed_depth = timed_walk(body)
    occupied = set(body)
    degree = sum(1 for offset in NEIGHBOURS
                 if 0 <= body[0][0] + offset[0] < COLS
                 and 0 <= body[0][1] + offset[1] < ROWS
                 and (body[0][0] + offset[0], body[0][1] + offset[1]) not in occupied)
    return {'static_area': area, 'static_tail': tail_ok, 'timed_area': timed_area,
            'timed_tail': timed_tail, 'timed_depth': timed_depth, 'degree': degree,
            'corridor': corridor_length(body, food, move_dir), 'len': len(body)}


def blame_with_candidates(buffer):
    """The latest decision where the chosen move loses and another survives, with features."""
    for offset in range(len(buffer) - 1, -1, -1):
        state = buffer[offset]
        branches = []
        for action in ACTIONS:
            nxt = sim_step(state['body'], state['food'], state['dir'], action)
            branches.append((nxt, bool(nxt) and heuristic_survives(*nxt)))
        chosen = state['chosen']
        if branches[chosen][1] or not any(ok for _, ok in branches):
            continue
        survivors = [index for index, (_, ok) in enumerate(branches) if ok]
        record = {'lead': len(buffer) - 1 - offset, 'len': state['len'],
                  'n_safe': sum(1 for f in state['facts'] if f['safe']),
                  'chosen_legal': branches[chosen][0] is not None}
        if branches[chosen][0] is not None:
            record['chosen'] = candidates(*branches[chosen][0])
        record['survivors'] = [candidates(*branches[index][0]) for index in survivors]
        return record
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
    blames = []
    saturation = collections.Counter()
    outcomes = collections.Counter()

    for episode in range(episodes):
        time_step = tf_env.reset()
        buffer = collections.deque(maxlen=WINDOW)
        while not time_step.is_last():
            snake_len = len(game.snake_group)
            facts = action_facts(game)
            body = [tuple(p) for p in game.snake.get_positions()]
            food = (None if game.current_food == 'no food'
                    else tuple(game.current_food.position))
            move_dir = game.head.move_dir

            # Saturation: how often does each flag still say "yes" for some legal move? A flag
            # that is 0 everywhere late in the game cannot be steering anything.
            if snake_len >= 50:
                saturation['late_steps'] += 1
                static_any = timed_any = False
                for index, action in enumerate(ACTIONS):
                    nxt = sim_step(body, food, move_dir, action)
                    if nxt is None:
                        continue
                    if facts[index]['tail_reachable']:
                        static_any = True
                    if timed_walk(nxt[0])[1]:
                        timed_any = True
                saturation['late_static_tail_any'] += static_any
                saturation['late_timed_tail_any'] += timed_any

            chosen = int(policy_action(time_step).action.numpy()[0])
            buffer.append({'body': body, 'food': food, 'dir': move_dir, 'facts': facts,
                           'chosen': chosen, 'len': snake_len})
            time_step = tf_env.step(tf.constant([chosen], dtype=tf.int32))

        outcome = ('perfect' if game.perfect_game
                   else 'starved' if game.starved else 'collision')
        outcomes[outcome] += 1
        if outcome != 'perfect':
            blame = blame_with_candidates(list(buffer))
            if blame:
                blame['score'] = game.current_score
                blames.append(blame)
            else:
                outcomes['unattributed'] += 1
        print('  episode {0}: {1}'.format(episode + 1, outcome), flush=True)

    json.dump({'policy': policy_name, 'step': step, 'seed': seed,
               'outcomes': dict(outcomes), 'saturation': dict(saturation),
               'blames': blames}, open(out_path, 'w'))
    print('wrote {0}'.format(out_path))


if __name__ == '__main__':
    main()
