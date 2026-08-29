"""What range do the discounted returns actually occupy? The Phase 0 measurement for C51 — see
[`../../plans/distributional-c51.md`](../../plans/distributional-c51.md).

A categorical agent predicts a distribution over a **fixed** grid `[v_min, v_max]` of `num_atoms`
points, and `project_distribution` clamps anything outside it. So the grid is a modelling choice that
has to be made before training, and this env makes it awkward: the reward quantum is
`FOOD_REWARD = 1.0`, a death pays `-5.0`, and a perfect game pays `PERFECT_GAME_REWARD = 100` on its
terminal step. Cover 100 and most atoms sit where nothing lands; cover only the bulk and the win value
is clipped in the endgame, which is the part that decides a perfect game.

This script measures the quantity the grid has to contain, rather than arguing about it.

**It is the agent's own bootstrap target, not an approximation of it.** Both `dqn_agent` and
`categorical_dqn_agent` compute their target as `reward + gamma * next_time_step.discount * V(next)`,
so the return this walks back is

    G_t = r_{t+1} + d_{t+1} * G_{t+1}

with `r` and `d` taken from the `TimeStep` that `env.step()` *returns*, and `d = 0` on a terminal step
(`snake_environment.to_tensor_time_step` sets that, and it is the only thing that stops the
bootstrap). Nothing here reimplements a discount rule.

Reported four ways, because each answers a different question about the grid:

  percentiles      where the bulk is, so `num_atoms` can be chosen for a useful spacing *there*
  tail shares      how much mass sits above 10/25/50/75, i.e. how much of a wide grid is wasted
  by outcome       returns split perfect / collision / starved. C51's case here rests on the claim
                   that the endgame return is **bimodal**; this is the panel that tests it
  by length band   where in the game the tail lives

**One episode set serves every gamma.** The discount reaches the return only through the recorded `d`,
and it changes neither the greedy action (an argmax over Q from the observation) nor any reward, so
re-running the recursion with `d' = 0 if d == 0 else g` gives the exact return under `g`. Both the
b24/b25 config's **0.9975** and the older **0.99** are reported from the same episodes.

**`SNEK_FOOD_DISTANCE_REWARD` defaults to `0` here, not to the repo default of 0.001**, because the
config this measures for (b24/b25 and everything after batch 17) sets it to 0. Left at 0.001 the
measured returns would carry a shaping term the arm will not have. Every reward knob in effect is
printed and recorded in the payload, so a run cannot be read without knowing what it included.

    cd snek2
    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/return_distribution.py <out.json> <ckpt-path-or-policy> \
        <episodes-per-seed> <seed[,seed...]>

`<ckpt-path-or-policy>` takes the same two forms as `behaviour_profile.py` and
`chase_safe_potential.py`: a `hallOfFame/<name>` or `savedPolicies/<name>` directory uses its newest
checkpoint, a path ending `ckpt-<step>` pins one. **Run it on more than one checkpoint** — the grid
has to hold the returns seen across the whole training arc, not just a champion's, so a strong
checkpoint and a mid-skill one from the same arm bound it from both sides.

Read-only: restores checkpoints, writes one JSON, starts no eval, touches nothing under
`savedPolicies/`, `runs/`, `evals/` or `hallOfFame/`.
"""
import collections
import glob
import json
import math
import os
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
# Must precede the snake_constants import: `from snake_constants import *` binds a copy, so setting
# it afterwards would never reach the game. See the module docstring for why the default differs
# from the repo's.
os.environ.setdefault('SNEK_FOOD_DISTANCE_REWARD', '0')

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common

import snake_constants
from eval_agent import build_eval_agent
from snake_environment import SnakeEnvironment
from under_the_hood import seed_process

# The discount the arm being planned for runs at (`SNEK_DISCOUNT=0.9975` in the b24/b25 config), plus
# the older 0.99 for reference. Both come out of one episode set — see the docstring.
GAMMAS = (0.9975, 0.99)
# Reported quantiles. Dense in both tails: the top end decides `v_max` and the bottom end is where a
# death lands, and the middle is what `num_atoms` has to resolve.
QUANTILES = (0.0, 0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9, 100.0)
# Thresholds for the tail shares. Chosen against the reward structure rather than the data: 1 is one
# meal, 10 is a handful, and 50/75 can only be reached near a win, since only PERFECT_GAME_REWARD is
# larger than the food a policy can collect inside a discount horizon.
TAIL_THRESHOLDS = (1.0, 10.0, 25.0, 50.0, 75.0)
# Same bands as behaviour_profile.py and chase_safe_potential.py, so the three are read side by side.
BANDS = (('10-49', 10, 49), ('50-84', 50, 84), ('85-89', 85, 89),
         ('90-94', 90, 94), ('95-97', 95, 97), ('98-99', 98, 99))
# Atom counts to report spacing for. 51 is C51's and Rainbow's published value; 111 gives spacing 1.0
# — one atom per FOOD_REWARD — over a grid that covers the win; 21 and 201 bracket both.
CANDIDATE_ATOMS = (21, 51, 111, 201)
# Atoms the p5-p95 bulk should get, for the inverse question: how many atoms does a wanted bulk
# resolution imply, given a grid wide enough to hold the tail.
BULK_ATOM_TARGETS = (10, 20, 40)


def band_of(length):
    return next((name for name, low, high in BANDS if low <= length <= high), None)


def resolve(target):
    """(checkpoint path, directory holding arch.json). Same two forms as `chase_safe_potential.py`.

    `hallOfFame/` entries carry only `ckpt-<step>.*` and `arch.json` — no `checkpoint` index file,
    because they were copied out of a rotation rather than written by a Checkpointer, so
    `tf.train.latest_checkpoint` returns None and the glob is the path that works.
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


def discounted_returns(rewards, discounts, gamma):
    """`G_t = r_{t+1} + d_{t+1} * G_{t+1}` over one episode, walked backwards.

    `rewards[i]` and `discounts[i]` are the reward and discount carried by the `TimeStep` returned by
    stepping out of state `i`, so `returns[i]` is the return **from state i**, which is the value the
    grid has to represent for that state.

    `gamma` substitutes for the environment's discount while preserving the terminal zero: a stored
    `d` of 0.0 is the episode boundary and must stay 0 whatever gamma is asked for, or the return
    would bootstrap straight through the end of the episode. That is the same trap
    `to_tensor_time_step` documents from the other side.
    """
    returns = [0.0] * len(rewards)
    carried = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        step_discount = 0.0 if discounts[index] == 0.0 else gamma
        carried = rewards[index] + step_discount * carried
        returns[index] = carried
    return returns


def describe(values):
    """Percentiles, mean, and the tail shares, for one collection of returns."""
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        'n': int(array.size),
        'mean': float(array.mean()),
        'std': float(array.std()),
        'quantiles': {str(q): float(np.percentile(array, q)) for q in QUANTILES},
        'share_above': {str(t): float((array > t).mean()) for t in TAIL_THRESHOLDS},
    }


def grid_advice(stats):
    """Candidate grids, from the measured spread.

    Two directions, because the choice has two constraints that pull against each other. `v_min`/
    `v_max` are set by the **extremes** — anything outside them is clamped, and the clamped value here
    would be the win. `num_atoms` is set by the **bulk**, because that is where the spacing has to be
    fine enough for the distribution to carry information.

    Rounded outward to whole units, since `FOOD_REWARD` is 1.0 and a grid boundary between meals would
    be an odd choice to defend.
    """
    low, high = stats['quantiles']['0.0'], stats['quantiles']['100.0']
    v_min = float(math.floor(low) - 1)
    v_max = float(math.ceil(high) + 1)
    span = v_max - v_min
    bulk = stats['quantiles']['95.0'] - stats['quantiles']['5.0']
    per_atom = []
    for atoms in CANDIDATE_ATOMS:
        spacing = span / (atoms - 1)
        per_atom.append({
            'num_atoms': atoms,
            'spacing': spacing,
            'atoms_across_bulk': bulk / spacing if spacing else None,
            'atoms_per_food_reward': snake_constants.FOOD_REWARD / spacing if spacing else None,
        })
    implied = []
    for wanted in BULK_ATOM_TARGETS:
        spacing = bulk / wanted if wanted else None
        implied.append({
            'atoms_across_bulk': wanted,
            'spacing': spacing,
            'num_atoms': int(math.ceil(span / spacing) + 1) if spacing else None,
        })
    return {'v_min': v_min, 'v_max': v_max, 'span': span,
            'bulk_p5_p95': bulk, 'candidates': per_atom, 'implied_by_bulk': implied}


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    out_path, target = sys.argv[1], sys.argv[2]
    episodes = int(sys.argv[3])
    seeds = [int(part) for part in sys.argv[4].split(',')]

    path, directory = resolve(target)
    # The env's own discount is irrelevant to the measurement (see the docstring) but it is what the
    # recorded `d` holds, so it is set to the first requested gamma and the payload records it.
    py_env = SnakeEnvironment(discount=GAMMAS[0], display=False, policy_name='smoke')
    py_env.reset()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    agent, checkpoint, global_step = build_eval_agent(tf_env, py_env, directory)
    checkpoint.restore(path).expect_partial()
    step = int(global_step.numpy())
    policy_action = common.function(agent.policy.action)
    rewards_in_effect = {
        'FOOD_REWARD': snake_constants.FOOD_REWARD,
        'DEATH_REWARD': snake_constants.DEATH_REWARD,
        'STARVE_REWARD': snake_constants.STARVE_REWARD,
        'PERFECT_GAME_REWARD': snake_constants.PERFECT_GAME_REWARD,
        'FOOD_DISTANCE_REWARD': snake_constants.FOOD_DISTANCE_REWARD,
        'CHASE_SAFE_SHAPING': snake_constants.CHASE_SAFE_SHAPING,
    }
    print('%s -> %s (global_step %d), %d episodes x seeds %s'
          % (target, os.path.basename(path), step, episodes, seeds), flush=True)
    print('rewards in effect: %s' % rewards_in_effect, flush=True)

    outcomes = collections.Counter()
    per_episode = []
    # gamma -> list of per-state returns, and the same split by outcome and by length band.
    pooled = {gamma: [] for gamma in GAMMAS}
    by_outcome = {gamma: collections.defaultdict(list) for gamma in GAMMAS}
    by_band = {gamma: collections.defaultdict(list) for gamma in GAMMAS}

    for seed in seeds:
        seed_process(seed, stream=0)
        for _ in range(episodes):
            time_step = tf_env.reset()
            step_rewards, step_discounts, lengths = [], [], []
            while True:
                lengths.append(len(py_env.snapshot().body))
                action_step = policy_action(time_step)
                time_step = tf_env.step(action_step.action)
                # Both are the values the agent's target uses for the transition just taken, so they
                # are read off the returned TimeStep rather than recomputed.
                step_rewards.append(float(time_step.reward.numpy()[0]))
                step_discounts.append(float(time_step.discount.numpy()[0]))
                if bool(time_step.is_last().numpy()[0]):
                    break

            final = py_env.snapshot()
            outcome = ('perfect' if final.perfect_game
                       else 'starved' if final.starved else 'collision')
            outcomes[outcome] += 1
            episode_row = {'seed': seed, 'outcome': outcome, 'steps': len(step_rewards),
                           'score': final.current_score, 'final_length': len(final.body),
                           'undiscounted_total': float(sum(step_rewards))}
            for gamma in GAMMAS:
                returns = discounted_returns(step_rewards, step_discounts, gamma)
                pooled[gamma].extend(returns)
                by_outcome[gamma][outcome].extend(returns)
                for length, value in zip(lengths, returns):
                    band = band_of(length)
                    if band is not None:
                        by_band[gamma][band].append(value)
                episode_row['g0_gamma_%s' % gamma] = returns[0] if returns else None
                episode_row['gmax_gamma_%s' % gamma] = max(returns) if returns else None
            per_episode.append(episode_row)

    by_gamma = {}
    for gamma in GAMMAS:
        stats = describe(pooled[gamma])
        by_gamma[str(gamma)] = {
            'pooled': stats,
            'grid': grid_advice(stats) if stats else None,
            'by_outcome': {name: describe(values)
                           for name, values in sorted(by_outcome[gamma].items())},
            'by_band': {name: describe(by_band[gamma].get(name, []))
                        for name, _, _ in BANDS},
        }

    summary = {'target': target, 'checkpoint': os.path.basename(path), 'step': step,
               'seeds': seeds, 'episodes_per_seed': episodes, 'outcomes': dict(outcomes),
               'gammas': list(GAMMAS), 'rewards': rewards_in_effect,
               'by_gamma': by_gamma, 'per_episode': per_episode}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as handle:
        json.dump(summary, handle, indent=2)

    for gamma in GAMMAS:
        entry = by_gamma[str(gamma)]
        stats = entry['pooled']
        print('\n=== gamma %s — %d states over %d episodes %s'
              % (gamma, stats['n'], sum(outcomes.values()), dict(outcomes)))
        print('%-12s %8s %8s %8s %8s %8s %8s %8s %8s'
              % ('set', 'n', 'min', 'p5', 'p50', 'p95', 'p99.9', 'max', 'mean'))

        def line(label, row):
            if row is None:
                return
            q = row['quantiles']
            print('%-12s %8d %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f'
                  % (label, row['n'], q['0.0'], q['5.0'], q['50.0'], q['95.0'], q['99.9'],
                     q['100.0'], row['mean']))

        line('pooled', stats)
        for name in sorted(entry['by_outcome']):
            line('  ' + name, entry['by_outcome'][name])
        for name, _, _ in BANDS:
            line('  len ' + name, entry['by_band'][name])
        print('share of states above: %s'
              % {k: round(v, 4) for k, v in stats['share_above'].items()})
        grid = entry['grid']
        print('grid from the extremes: [%.1f, %.1f], span %.1f, bulk p5-p95 %.2f'
              % (grid['v_min'], grid['v_max'], grid['span'], grid['bulk_p5_p95']))
        for row in grid['candidates']:
            print('  N=%-4d spacing %7.3f  atoms across bulk %8.1f  atoms per food %6.2f'
                  % (row['num_atoms'], row['spacing'], row['atoms_across_bulk'],
                     row['atoms_per_food_reward']))
        for row in grid['implied_by_bulk']:
            print('  %d atoms across the bulk -> spacing %.3f -> N = %d'
                  % (row['atoms_across_bulk'], row['spacing'], row['num_atoms']))
    print('\nwrote ' + out_path, flush=True)


if __name__ == '__main__':
    main()
