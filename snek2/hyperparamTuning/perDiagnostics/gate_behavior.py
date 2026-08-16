"""Board-state metametrics per snake length, for comparing what a policy *does* around its training
gate. Written to ask why chase-safe gate 75 (b29) produced a record region while gate 85 (b27) was
null, with seed-matched checkpoints (same seed, only the gate differs).

At eval time there is no shaping — the policy is fixed — so any behavioural difference around a
length threshold is what the arm *learned*. This plays greedy episodes and logs, for every step where
snake length >= MIN_LENGTH, board-state quantities computed from the environment's own primitives
(`chase_safe_state`, `count_groups`, `get_adjacent_groups`) rather than reimplemented:

  phi         head+food+tail in one open region NOW (the gate-independent potential, 0/1)
  regions     number of open (free) connected components on the board
  one_piece   largest free component / total free cells  (packing: 1.0 = free space is one blob)
  size1       number of size-1 free pockets (dead single cells)
  iso_regions free components the head cannot step into this move (walled off from the head)
  iso_cells   total free cells in those isolated components
  food_reach  1 if the head can reach the food's component this move, else 0 (-1 if no food)
  headroom    starve budget left, in steps

Per episode it stores the outcome (perfect/starved/collision) and the full per-step arrays, so the
analysis can slice before/after any gate and reconstruct board quality at the gate-crossing step.

    cd snek2
    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/gate_behavior.py <out.json> <ckpt-path-or-policy> \
        <episodes> <seed> <gate>

Same checkpoint-resolution and seed semantics as `behaviour_profile.py`: a seed reproduces the food
sequence, so seed-matched checkpoints face the same games. Read-only: restores a checkpoint, writes
one JSON, starts no eval.
"""
import json
import os
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common

from eval_agent import build_eval_agent
from snake_environment import SnakeEnvironment
from state_helpers import steps_until_starve, count_groups, get_adjacent_groups
from under_the_hood import seed_process

# Reuse the exact grid/predicate the Phase 0 probe uses, so the potential and the region math match.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chase_safe_potential import build_grid, chase_safe_state, resolve

MIN_LENGTH = 40   # skip the wide-open early game; both gates (75, 85) sit well above this


def board_metrics(body, food):
    """Free-space packing/reachability of the *current* board, from count_groups regions."""
    grid = build_grid(body, food)
    regions, count = count_groups(grid)
    cols = grid.shape[1]
    sizes = [bin(r).count('1') for r in regions]
    total = sum(sizes)
    largest = max(sizes) if sizes else 0
    reachable = get_adjacent_groups(regions, cols, body[0])   # regions the head can step into
    isolated = [i for i in range(len(regions)) if i not in reachable]
    iso_cells = sum(sizes[i] for i in isolated)
    size1 = sum(1 for s in sizes if s == 1)
    if food is not None:
        food_bit = 1 << ((food[1] + 1) * cols + (food[0] + 1))
        food_reach = 1 if any(regions[i] & food_bit for i in reachable) else 0
    else:
        food_reach = -1
    return count, (largest / total if total else 1.0), size1, len(isolated), iso_cells, food_reach


def main():
    out_path, target = sys.argv[1], sys.argv[2]
    episodes, seed = int(sys.argv[3]), int(sys.argv[4])
    gate = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    seed_process(seed, stream=0)

    path, directory = resolve(target)
    py_env = SnakeEnvironment(discount=0.9975, display=False, policy_name='smoke')
    py_env.reset()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    agent, checkpoint, global_step = build_eval_agent(tf_env, py_env, directory)
    checkpoint.restore(path).expect_partial()
    gstep = int(global_step.numpy())
    policy_action = common.function(agent.policy.action)
    print('%s -> %s (global_step %d), %d episodes, seed %d, gate %d'
          % (target, os.path.basename(path), gstep, episodes, seed, gate), flush=True)

    outcomes = {'perfect': 0, 'starved': 0, 'collision': 0}
    per_episode = []

    for episode in range(episodes):
        time_step = tf_env.reset()
        rec = {'L': [], 'phi': [], 'reg': [], 'op': [], 's1': [], 'isoR': [], 'isoC': [],
               'fr': [], 'hr': []}
        steps = 0
        while True:
            snap = py_env.snapshot()
            length = len(snap.body)
            if length >= MIN_LENGTH:
                count, one_piece, size1, iso_r, iso_c, food_reach = board_metrics(snap.body, snap.food)
                rec['L'].append(length)
                rec['phi'].append(chase_safe_state(snap.body, snap.food))
                rec['reg'].append(count)
                rec['op'].append(round(one_piece, 4))
                rec['s1'].append(size1)
                rec['isoR'].append(iso_r)
                rec['isoC'].append(iso_c)
                rec['fr'].append(food_reach)
                rec['hr'].append(steps_until_starve(snap.current_step, snap.last_food_step, length))

            action_step = policy_action(time_step)
            time_step = tf_env.step(action_step.action)
            steps += 1
            if bool(time_step.is_last().numpy()[0]):
                break

        final = py_env.snapshot()
        outcome = ('perfect' if final.perfect_game
                   else 'starved' if final.starved else 'collision')
        outcomes[outcome] += 1
        rec['outcome'] = outcome
        rec['final_length'] = len(final.body)
        rec['steps'] = steps
        per_episode.append(rec)
        if (episode + 1) % 25 == 0:
            print('  %d/%d  %s' % (episode + 1, episodes, dict(outcomes)), flush=True)

    payload = {'policy': target, 'checkpoint': os.path.basename(path), 'global_step': gstep,
               'episodes': episodes, 'seed': seed, 'gate': gate, 'min_length': MIN_LENGTH,
               'outcomes': outcomes, 'per_episode': per_episode}
    with open(out_path, 'w') as handle:
        json.dump(payload, handle)
    print('wrote %s  outcomes=%s' % (out_path, outcomes), flush=True)


if __name__ == '__main__':
    main()
