"""Compares three versions of the "can I still reach my tail" test at the losing decisions.

    PYTHONPATH=. python -u diag3.py <policy> <step> <episodes> <seed> <out.json>

  observed  exactly what group_obs() puts in the observation vector: the regions adjacent to
            the *vacated tail cell*, intersected with the regions adjacent to the new head.
  fixed     the regions *containing* the vacated tail cell instead of those adjacent to it.
            One line different. In a coiled endgame the vacated cell is often surrounded by
            body, so "adjacent to it" is the empty set and the observed flag reads 0 whatever
            the head does.
  timed     a walk from the new head in which each body cell opens once that segment passes.

Also asserts on every step that the `observed` recomputation matches the actual observation
vector, so the comparison is against what the policy really sees rather than a lookalike.
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

from snake_constants import ACTIONS, POLICY_DIR
from snake_environment import SnakeEnvironment
from snek2 import build_q_net
from state_helpers import (count_groups, get_adjacent_groups, get_grid_value,
                           get_relative_pos, update_grid)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag import WINDOW, heuristic_survives, sim_step
from diag2 import timed_walk

OPEN = (0, 1)


def tail_tests(game):
    """(observed, fixed, timed, legal) per action, plus the region count per action."""
    grid = game.grid
    cols = grid.shape[1]
    head = game.head.tile_pos
    tail = tuple(game.tail.tile_pos)
    move_dir = game.head.move_dir
    body = [tuple(p) for p in game.snake.get_positions()]
    food = None if game.current_food == 'no food' else tuple(game.current_food.position)

    rows = []
    for action in ACTIONS:
        new_head = get_relative_pos(action, head, move_dir)
        value = get_grid_value(new_head, grid)
        legal = value in OPEN or tuple(new_head) == tail
        row = {'legal': bool(legal), 'observed': False, 'fixed': False, 'timed': False,
               'regions': 0}
        if legal:
            moved = update_grid(action, head, tail, grid.copy(), move_dir)
            regions, count = count_groups(moved)
            head_regions = get_adjacent_groups(regions, cols, new_head)
            row['regions'] = count
            # what group_obs does
            adjacent = get_adjacent_groups(regions, cols, tail)
            row['observed'] = bool(adjacent & head_regions or tuple(new_head) == tail)
            # The change: also count the region *holding* the vacated cell. Those are not the
            # same set — when the vacated cell has no open neighbour it is a singleton region,
            # so "adjacent to it" is empty while "holding it" is that singleton. The union is
            # needed rather than the replacement, because a move that eats does not vacate the
            # tail at all, and then only the adjacency test says anything.
            tail_bit = 1 << ((tail[1] + 1) * cols + (tail[0] + 1))
            holding = {index for index, region in enumerate(regions) if region & tail_bit}
            row['fixed'] = bool(head_regions & (holding | adjacent)
                                or tuple(new_head) == tail)
            nxt = sim_step(body, food, move_dir, action)
            if nxt is not None:
                row['timed'] = timed_walk(nxt[0])[1]
        rows.append(row)
    return rows


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
    counters = collections.Counter()
    outcomes = collections.Counter()

    for episode in range(episodes):
        time_step = tf_env.reset()
        buffer = collections.deque(maxlen=WINDOW)
        while not time_step.is_last():
            snake_len = len(game.snake_group)
            tests = tail_tests(game)
            obs = [float(x) for x in game.get_observation()]
            # group_obs writes [head_with_tail, lg(regions)] per action at 9..14.
            # Only legal moves: group_obs computes the flag for fatal moves too, where the
            # prospective head sits inside a wall or body and the test is meaningless, and the
            # recomputation above skips those.
            for index in range(3):
                if not tests[index]['legal']:
                    continue
                if bool(obs[9 + 2 * index]) != tests[index]['observed']:
                    counters['observed_mismatch'] += 1
                if obs[10 + 2 * index] != np.log2(tests[index]['regions'] + 1):
                    counters['regions_mismatch'] += 1
            counters['steps'] += 1
            if snake_len >= 50:
                counters['late'] += 1
                for name in ('observed', 'fixed', 'timed'):
                    legal = [t for t in tests if t['legal']]
                    if legal and any(t[name] for t in legal):
                        counters['late_any_' + name] += 1
                    if legal and all(not t[name] for t in legal):
                        counters['late_none_' + name] += 1

            chosen = int(policy_action(time_step).action.numpy()[0])
            buffer.append({'body': [tuple(p) for p in game.snake.get_positions()],
                           'food': (None if game.current_food == 'no food'
                                    else tuple(game.current_food.position)),
                           'dir': game.head.move_dir, 'tests': tests, 'chosen': chosen,
                           'len': snake_len})
            time_step = tf_env.step(tf.constant([chosen], dtype=tf.int32))

        outcome = ('perfect' if game.perfect_game
                   else 'starved' if game.starved else 'collision')
        outcomes[outcome] += 1
        if outcome != 'perfect':
            blame = attribute(list(buffer))
            if blame:
                blames.append(blame)
            else:
                outcomes['unattributed'] += 1
        print('  episode {0}: {1}'.format(episode + 1, outcome), flush=True)

    json.dump({'policy': policy_name, 'step': step, 'seed': seed,
               'outcomes': dict(outcomes), 'counters': dict(counters), 'blames': blames},
              open(out_path, 'w'))
    print('wrote {0}'.format(out_path))


def attribute(buffer):
    for offset in range(len(buffer) - 1, -1, -1):
        state = buffer[offset]
        survives = []
        for action in ACTIONS:
            nxt = sim_step(state['body'], state['food'], state['dir'], action)
            survives.append(bool(nxt) and heuristic_survives(*nxt))
        chosen = state['chosen']
        if survives[chosen] or not any(survives):
            continue
        return {'lead': len(buffer) - 1 - offset, 'len': state['len'],
                'chosen': state['tests'][chosen],
                'survivors': [state['tests'][i] for i, ok in enumerate(survives) if ok]}
    return None


if __name__ == '__main__':
    main()
