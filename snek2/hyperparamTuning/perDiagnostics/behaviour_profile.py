"""What does a checkpoint actually *do*? A behavioural fingerprint, comparable across checkpoints.

Written to ask what separates the record checkpoints from mediocre ones, after the counterfactual
Q-value approach turned out to measure training duration rather than skill (see the demotion note in
`../findings.md`). This measures play instead of preference: it runs greedy episodes and records, per
step, what the *chosen* action's observation values were, then aggregates by snake length.

Everything logged is already computed by the environment for the policy's own benefit, so the
instrumentation costs nothing but the episode itself. No search, no extra network passes.

Per length band it reports:

  steps_per_food   steps between meals -- the efficiency that decides whether the starve clock binds
  headroom         starve budget left at the moment of eating, in steps
  hug              share of chosen moves that end against a wall or the body (packing)
  tail_reach       share of chosen moves that keep the tail reachable (the classic safety invariant)
  regions          lg(open regions) of the post-move board, scaled -- 1 is one region
  chase_safe       share of chosen moves where head, food and tail land in one region
  win_avail        share of states where some action wins outright
  forward          share of chosen moves that go straight

and per episode the outcome, final length, and total steps.

    cd snek2
    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/behaviour_profile.py <out.json> <ckpt-path-or-policy> \
        <episodes> <seed>

`<ckpt-path-or-policy>` takes the same two forms as `point_of_no_return.py`: a `hallOfFame/<name>` or
`savedPolicies/<name>` directory uses its newest checkpoint, and a path ending `ckpt-<step>` pins one.
Shard by seed; a seed reproduces the same food sequence under a greedy policy, so **two checkpoints
run on the same seeds face the same games**, which is what makes the columns comparable.

Read-only: restores checkpoints, writes one JSON, starts no eval.
"""
import collections
import glob
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
from snake_constants import PERFECT_SCORE
from state_helpers import steps_until_starve
from under_the_hood import seed_process

# Observation layout, current era. The per-action blocks are indexed by the chosen action so every
# number below describes the move the policy actually made, not the best case over three.
SAFE, CHASE, WIN, HUG, NOTTAIL = 6, 15, 18, 23, 26
TAIL_REACH, REGIONS = 9, 10          # interleaved: tail at 9+2a, regions at 10+2a
FILL = 22
BANDS = (('10-49', 10, 49), ('50-84', 50, 84), ('85-94', 85, 94), ('95-99', 95, 99))
FORWARD_ACTION = 2                   # TF_ACTION_TO_ACTIONS: 0 left, 1 right, 2 forward


def resolve(target):
    """(checkpoint path, directory holding arch.json).

    `hallOfFame/` entries carry only the two `ckpt-<step>.*` files and `arch.json` — no `checkpoint`
    index file, because they were copied out of a rotation rather than written by a Checkpointer. So
    `tf.train.latest_checkpoint` returns None for them and the glob below is the path that works.
    """
    if os.path.basename(target).startswith('ckpt-'):
        return target, os.path.dirname(target)
    directory = target if os.path.isdir(target) else os.path.join('savedPolicies', target)
    path = tf.train.latest_checkpoint(directory)
    if path is None:
        indexes = glob.glob(os.path.join(directory, 'ckpt-*.index'))
        if not indexes:
            raise SystemExit('no checkpoint in ' + directory)
        newest = max(indexes, key=lambda p: int(p.split('ckpt-')[1].split('.index')[0]))
        path = newest[:-len('.index')]
    return path, directory


def per_action(obs, first, action, stride=1):
    return float(obs[first + stride * action])


def step_record(obs, action):
    """The per-step metrics describing the move actually chosen.

    Separate from the episode loop so the index arithmetic is testable without an environment: the
    stride-2 reads for the interleaved tail/regions block are easy to get wrong and produce
    plausible-looking numbers when they are.
    """
    return {
        'hug': per_action(obs, HUG, action),
        'tail_reach': per_action(obs, TAIL_REACH, action, 2),
        'regions': per_action(obs, REGIONS, action, 2),
        'chase_safe': per_action(obs, CHASE, action),
        'safe': per_action(obs, SAFE, action),
        'win_avail': float(max(obs[WIN], obs[WIN + 1], obs[WIN + 2]) > 0.5),
        'forward': float(action == FORWARD_ACTION),
        'nottail': per_action(obs, NOTTAIL, action),
    }


STEP_METRICS = ('hug', 'tail_reach', 'regions', 'chase_safe', 'safe', 'win_avail',
                'forward', 'nottail')


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    out_path, target = sys.argv[1], sys.argv[2]
    episodes, seed = int(sys.argv[3]), int(sys.argv[4])
    seed_process(seed, stream=0)

    path, directory = resolve(target)
    py_env = SnakeEnvironment(discount=0.9975, display=False, policy_name='smoke')
    py_env.reset()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    agent, checkpoint, global_step = build_eval_agent(tf_env, py_env, directory)
    checkpoint.restore(path).expect_partial()
    step = int(global_step.numpy())
    policy_action = common.function(agent.policy.action)
    print('%s -> %s (global_step %d), %d episodes, seed %d'
          % (target, os.path.basename(path), step, episodes, seed), flush=True)

    outcomes = collections.Counter()
    per_episode = []
    # Accumulators keyed (band, metric) -> list of per-step or per-meal values.
    bucket = collections.defaultdict(list)
    meals = collections.defaultdict(list)

    for episode in range(episodes):
        time_step = tf_env.reset()
        previous_length = None
        previous_meal_step = 0
        steps = 0
        while True:
            snap = py_env.snapshot()
            obs = time_step.observation.numpy()[0]
            length = len(snap.body)
            action_step = policy_action(time_step)
            action = int(action_step.action.numpy()[0])

            band = next((name for name, low, high in BANDS if low <= length <= high), None)
            if band is not None:
                for metric, value in step_record(obs, action).items():
                    bucket[(band, metric)].append(value)

            time_step = tf_env.step(action_step.action)
            steps += 1
            after = py_env.snapshot()
            new_length = len(after.body)
            # A meal is a length increase. Recording it here rather than from the reward keeps this
            # independent of whatever shaping the arm trained with.
            if previous_length is not None and new_length > previous_length:
                eaten_band = next((n for n, lo, hi in BANDS if lo <= new_length <= hi), None)
                if eaten_band is not None:
                    meals[(eaten_band, 'steps_per_food')].append(after.current_step
                                                                 - previous_meal_step)
                    meals[(eaten_band, 'headroom')].append(
                        steps_until_starve(after.current_step, previous_meal_step, previous_length))
                previous_meal_step = after.current_step
            previous_length = new_length

            if bool(time_step.is_last().numpy()[0]):
                break

        final = py_env.snapshot()
        outcome = ('perfect' if final.perfect_game
                   else 'starved' if final.starved else 'collision')
        outcomes[outcome] += 1
        per_episode.append({'outcome': outcome, 'final_length': len(final.body),
                            'steps': steps, 'score': final.current_score})

    summary = {'target': target, 'checkpoint': os.path.basename(path), 'step': step,
               'seed': seed, 'episodes': episodes, 'outcomes': dict(outcomes),
               'perfect_score': PERFECT_SCORE, 'per_episode': per_episode, 'bands': {}}
    for name, _, _ in BANDS:
        row = {}
        for metric in STEP_METRICS:
            values = bucket[(name, metric)]
            row[metric] = float(np.mean(values)) if values else None
            row['n_steps'] = len(bucket[(name, 'hug')])
        # Starvation is a tail event: a policy with a fine median can still lose an episode to one
        # 500-step wander, so the percentiles are the point and the median is context.
        spf = meals[(name, 'steps_per_food')]
        head = meals[(name, 'headroom')]
        row['steps_per_food'] = float(np.median(spf)) if spf else None
        row['steps_per_food_mean'] = float(np.mean(spf)) if spf else None
        row['steps_per_food_p90'] = float(np.percentile(spf, 90)) if spf else None
        row['steps_per_food_p99'] = float(np.percentile(spf, 99)) if spf else None
        row['steps_per_food_max'] = float(np.max(spf)) if spf else None
        row['long_hunt_share'] = float(np.mean(np.array(spf) > 200)) if spf else None
        row['headroom'] = float(np.median(head)) if head else None
        row['headroom_p10'] = float(np.percentile(head, 10)) if head else None
        row['headroom_min'] = float(np.min(head)) if head else None
        row['n_meals'] = len(spf)
        summary['bands'][name] = row

    with open(out_path, 'w') as handle:
        json.dump(summary, handle, indent=1)
    print('%s: %s' % (os.path.basename(path), dict(outcomes)), flush=True)
    print('wrote ' + out_path)


if __name__ == '__main__':
    main()
