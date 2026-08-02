"""Four versions of "can the head still reach the tail", scored on the losing decisions.

    PYTHONPATH=. python -u diag4.py <policy> <step> <episodes> <seed> <out.json>

  observed  today's group_obs: regions adjacent to the *old* tail cell.
  holding   observed, plus the region that *contains* the old tail cell. Local to group_obs,
            needs no extra state.
  newtail   regions adjacent to the *post-move* tail — which is the old tail cell when the move
            eats (add_segment refills it) and the segment ahead of it otherwise. This is the
            root-cause fix: group_obs is handed one tail position and uses it for both cases.
  timed     a walk from the new head where each body cell opens as that segment passes.

Also counts how often `holding` and `newtail` actually disagree, over every legal action of
every step, since the argument that they are the same fix needs checking rather than asserting.
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
VARIANTS = ('observed', 'holding', 'newtail', 'both', 'timed')


def tail_tests(game):
    grid = game.grid
    cols = grid.shape[1]
    head = game.head.tile_pos
    tail = tuple(game.tail.tile_pos)
    move_dir = game.head.move_dir
    body = [tuple(p) for p in game.snake.get_positions()]
    food = None if game.current_food == 'no food' else tuple(game.current_food.position)
    # The segment ahead of the tail becomes the tail on any step that does not eat.
    ahead_of_tail = body[-2] if len(body) > 1 else tail

    rows = []
    for action in ACTIONS:
        new_head = get_relative_pos(action, head, move_dir)
        value = get_grid_value(new_head, grid)
        legal = value in OPEN or tuple(new_head) == tail
        row = {'legal': bool(legal), 'eats': bool(value == 1)}
        for name in VARIANTS:
            row[name] = False
        if legal:
            eats = value == 1
            moved = update_grid(action, head, tail, grid.copy(), move_dir)
            regions, _ = count_groups(moved)
            head_regions = get_adjacent_groups(regions, cols, new_head)

            # Every variant keeps group_obs's existing special case: stepping onto the cell the
            # tail is vacating is always safe, and no region test sees it because the cell ends
            # up holding the head. Dropping it made the newtail variant look far worse than it
            # is — that one move is most of where the two fixes appeared to disagree.
            follows_tail = tuple(new_head) == tail

            adjacent = get_adjacent_groups(regions, cols, tail)
            row['observed'] = bool(adjacent & head_regions) or follows_tail

            tail_bit = 1 << ((tail[1] + 1) * cols + (tail[0] + 1))
            contains = {index for index, region in enumerate(regions) if region & tail_bit}
            row['holding'] = bool(head_regions & (contains | adjacent)) or follows_tail

            post_move_tail = tail if eats else ahead_of_tail
            row['newtail'] = bool(get_adjacent_groups(regions, cols, post_move_tail)
                                  & head_regions) or follows_tail

            row['both'] = row['holding'] or row['newtail']

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
            late = snake_len >= 50
            counters['steps'] += 1
            counters['late'] += late

            for index in range(3):
                if not tests[index]['legal']:
                    continue
                counters['legal_actions'] += 1
                counters['legal_actions_late'] += late
                if bool(obs[9 + 2 * index]) != tests[index]['observed']:
                    counters['observed_mismatch'] += 1
                row = tests[index]
                if row['holding'] != row['newtail']:
                    counters['holding_vs_newtail_differ'] += 1
                    counters['holding_vs_newtail_differ_late'] += late
                    counters['newtail_true_holding_false'] += row['newtail']
                    counters['holding_true_newtail_false'] += row['holding']
                    if row['eats']:
                        counters['differ_on_eating_step'] += 1
                for name in VARIANTS:
                    if row[name] != row['timed']:
                        counters[name + '_vs_timed_differ'] += 1
                    if row[name]:
                        counters[name + '_true'] += 1

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
