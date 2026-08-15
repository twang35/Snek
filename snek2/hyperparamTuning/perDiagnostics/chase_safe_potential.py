"""How often does "head, food and tail in one region" flip? The Phase 0 measurement for
`CHASE_SAFE_SHAPING` — see [`../../plans/chase-safe-reward-shaping.md`](../../plans/chase-safe-reward-shaping.md).

Potential-based shaping adds `F = c * (gamma * Phi(s') - Phi(s))`, and **`c` cannot be calibrated from
the return**: the discounted sum telescopes exactly to `-c * Phi(s0)`, a constant, which is the whole
invariance argument. What `c` scales is the per-transition reward, and since `FOOD_DISTANCE_REWARD=0`
leaves the real reward at exactly 0.0 on ~94% of steps, `+/-c` *is* the reward on a step where Phi
flips. So the budget is set by the **flip rate per meal interval** — a meal being how often a real
`FOOD_REWARD = 1.0` arrives — and a held Phi is nearly free at `c * (gamma - 1)` per step.

That makes the flip rate, not the base rate, the number this script exists to produce:

    c = 0.25 / flips per meal interval

It reports both candidate potentials in one pass, because they need different denominators:

  variant A (global)      Phi = chase-safe
  variant B (length-gated) Phi = chase-safe AND snake_len >= GATE, so only meals at or above the
                          gate are shaped and only those meals belong in the budget

Per length band, split finely at the top because that is the band the change is for:

  phi_mean         share of acted-in states where head, food and tail share one open region
  flips_per_100    Phi transitions per 100 steps -- flip *frequency*, independent of meal duration
  flips_per_meal   the calibration denominator. Must be read beside flips_per_100: a weak policy
                   taking 86-226 steps per meal at 95-99 inflates this through duration alone
  steps_per_meal   the duration that does the inflating
  share_steps      share of the episode's steps in this band -- what decides whether the gate is
                   worth having, since a global c doses every band at once

**Phi is computed from `Game.snapshot()`, not from the live grid**, so this probe stays independent of
the production function the plan proposes for `state_helpers`. The two must agree; `build_grid` here is
`point_of_no_return`'s reconstruction of exactly what `Snake._rebuild_grid()` writes.

**Structural zero at the top, and it bounds what any `c` can do.** With one free cell there is no region
holding head, food and tail, so Phi is 0 at length 99 for every policy and near-0 at 98 — the last two
or three meals cannot be shaped by this potential at all. Expect the 98-99 row to read ~0 and do not
read it as a measurement failure.

    cd snek2
    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/chase_safe_potential.py <out.json> <ckpt-path-or-policy> \
        <episodes-per-seed> <seed[,seed...]>

`<ckpt-path-or-policy>` takes the same two forms as `behaviour_profile.py`: a `hallOfFame/<name>` or
`savedPolicies/<name>` directory uses its newest checkpoint, a path ending `ckpt-<step>` pins one.
Several seeds run in one process so three checkpoints face the same food draws in three invocations.

Read-only: restores checkpoints, writes one JSON, starts no eval, touches nothing under
`savedPolicies/`, `runs/`, `evals/` or `hallOfFame/`.
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
from snake_constants import PERFECT_SCORE, SCREENTILES
from state_helpers import count_groups, get_adjacent_groups
from under_the_hood import seed_process

# Split at the top: the old 85-94 / 95-99 pair is too coarse to site a gate with, and 98-99 is
# where Phi goes structurally zero.
BANDS = (('10-49', 10, 49), ('50-84', 50, 84), ('85-89', 85, 89),
         ('90-94', 90, 94), ('95-97', 95, 97), ('98-99', 98, 99))
# Candidate gate for variant B. 85 rather than 95 because Phi is structurally 0 at 99 and nearly so
# at 98, and because it matches FORK_MIN_LENGTH, so the shaped transitions are the ones the forking
# collector already oversamples.
GATE = 85
# The safety budget from the plan: summed |shaping| between two meals, as a share of FOOD_REWARD.
BUDGET = 0.25


def band_of(length):
    return next((name for name, low, high in BANDS if low <= length <= high), None)


def build_grid(body, food):
    """The padded grid `Snake._rebuild_grid()` builds, from a body tuple and food cell.

    Same reconstruction as `point_of_no_return.build_grid`. Duplicated rather than imported because
    that module builds a search over hypothetical boards and importing it here would pull the whole
    thing in for six lines.
    """
    grid = np.zeros((SCREENTILES[1] + 3, SCREENTILES[0] + 3))
    grid[[0, -1], :] = 4
    grid[:, [0, -1]] = 4
    if food is not None:
        grid[food[1] + 1, food[0] + 1] = 1
    for index, cell in enumerate(body):
        grid[cell[1] + 1, cell[0] + 1] = 2 if index == 0 else 3
    return grid


def chase_safe_state(body, food):
    """1 when the head, the food and the tail all sit in one open region of the current board.

    The *state* form of the per-action flag at observation indices 15-17. `group_obs` asks "would this
    move leave them in one region"; a potential function needs "are they in one region now".

    The head and tail cells are occupied, so they get `get_adjacent_groups`; the food cell is open and
    belongs to a region of its own, so it gets containment. That mirrors `group_obs` exactly — and
    testing the food against the *intersection* is the point, because the head can neighbour two
    regions at once and reaching the food through one while the tail is only reachable through the
    other is the trap the flag names.
    """
    if food is None:
        return 0
    grid = build_grid(body, food)
    regions, _ = count_groups(grid)
    cols = grid.shape[1]
    escape = (get_adjacent_groups(regions, cols, body[0])
              & get_adjacent_groups(regions, cols, body[-1]))
    if not escape:
        return 0
    food_bit = 1 << ((food[1] + 1) * cols + (food[0] + 1))
    return 1 if any(regions[index] & food_bit for index in escape) else 0


def resolve(target):
    """(checkpoint path, directory holding arch.json). Same two forms as `behaviour_profile.py`.

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


def divide(numerator, denominator, scale=1.0):
    return float(numerator) * scale / denominator if denominator else None


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    out_path, target = sys.argv[1], sys.argv[2]
    episodes = int(sys.argv[3])
    seeds = [int(part) for part in sys.argv[4].split(',')]

    path, directory = resolve(target)
    py_env = SnakeEnvironment(discount=0.9975, display=False, policy_name='smoke')
    py_env.reset()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    agent, checkpoint, global_step = build_eval_agent(tf_env, py_env, directory)
    checkpoint.restore(path).expect_partial()
    step = int(global_step.numpy())
    policy_action = common.function(agent.policy.action)
    print('%s -> %s (global_step %d), %d episodes x seeds %s'
          % (target, os.path.basename(path), step, episodes, seeds), flush=True)

    outcomes = collections.Counter()
    per_episode = []
    # (band, key) -> count. `steps` counts transitions, `flips`/`flips_gated` count Phi changes
    # attributed to the band of the state the transition left, `meals` counts length increases
    # attributed to the band of the new length, matching behaviour_profile.py.
    tally = collections.Counter()

    for seed in seeds:
        seed_process(seed, stream=0)
        for episode in range(episodes):
            time_step = tf_env.reset()
            snap = py_env.snapshot()
            phi = chase_safe_state(snap.body, snap.food)
            steps = 0
            while True:
                length = len(snap.body)
                band = band_of(length)
                gated = phi if length >= GATE else 0

                action_step = policy_action(time_step)
                time_step = tf_env.step(action_step.action)
                steps += 1

                after = py_env.snapshot()
                done = bool(time_step.is_last().numpy()[0])
                new_length = len(after.body)
                # Phi(terminal) = 0 -- the condition the invariance requires, and the same branch
                # the production term will take, since on a death the head is off the board.
                phi_next = 0 if done else chase_safe_state(after.body, after.food)
                gated_next = 0 if (done or new_length < GATE) else phi_next

                if band is not None:
                    tally[(band, 'steps')] += 1
                    tally[(band, 'phi')] += phi
                    tally[(band, 'flips')] += int(phi_next != phi)
                    tally[(band, 'flips_gated')] += int(gated_next != gated)
                    # Two kinds of flip carry no information and have to be separable, or the top
                    # band reads busy when it is not. A *terminal* flip is the mandatory
                    # Phi(terminal) = 0 the invariance requires: exactly one per episode, and for a
                    # perfect game it lands in the 98-99 row, which is how a band with a constant
                    # Phi can still show 40 flips per 100 steps. A *gate-crossing* flip exists only
                    # because the gated potential is 0 below the gate, so it fires once per episode
                    # at length 84->85 and says nothing about the board.
                    if done and phi_next != phi:
                        tally[(band, 'flips_terminal')] += 1
                    if done and gated_next != gated:
                        tally[(band, 'flips_gated_terminal')] += 1
                    if (length >= GATE) != (new_length >= GATE) and gated_next != gated:
                        tally[(band, 'flips_gate_cross')] += 1
                    if new_length > length:
                        meal_band = band_of(new_length)
                        if meal_band is not None:
                            tally[(meal_band, 'meals')] += 1

                phi, snap = phi_next, after
                if done:
                    break

            final = py_env.snapshot()
            outcome = ('perfect' if final.perfect_game
                       else 'starved' if final.starved else 'collision')
            outcomes[outcome] += 1
            per_episode.append({'seed': seed, 'outcome': outcome,
                                'final_length': len(final.body), 'steps': steps,
                                'score': final.current_score})

    total_steps = sum(tally[(name, 'steps')] for name, _, _ in BANDS)
    bands = {}
    for name, _, _ in BANDS:
        steps_in = tally[(name, 'steps')]
        meals_in = tally[(name, 'meals')]
        genuine = tally[(name, 'flips')] - tally[(name, 'flips_terminal')]
        bands[name] = {
            'n_steps': steps_in,
            'share_steps': divide(steps_in, total_steps),
            'phi_mean': divide(tally[(name, 'phi')], steps_in),
            'flips': tally[(name, 'flips')],
            'flips_terminal': tally[(name, 'flips_terminal')],
            'flips_genuine': genuine,
            'flips_gate_cross': tally[(name, 'flips_gate_cross')],
            'flips_per_100': divide(tally[(name, 'flips')], steps_in, 100.0),
            'genuine_per_100': divide(genuine, steps_in, 100.0),
            'meals': meals_in,
            'flips_per_meal': divide(tally[(name, 'flips')], meals_in),
            'genuine_per_meal': divide(genuine, meals_in),
            'flips_gated_per_meal': divide(tally[(name, 'flips_gated')], meals_in),
            'steps_per_meal': divide(steps_in, meals_in),
        }

    # The two calibrations. Variant A is dosed on every meal; variant B only on meals at or above the
    # gate, because below it Phi is identically 0 and no shaping is paid at all.
    gated_bands = [name for name, low, _ in BANDS if low >= GATE]
    flips_a = sum(tally[(name, 'flips')] for name, _, _ in BANDS)
    meals_a = sum(tally[(name, 'meals')] for name, _, _ in BANDS)
    flips_b = sum(tally[(name, 'flips_gated')] for name, _, _ in BANDS)
    meals_b = sum(tally[(name, 'meals')] for name in gated_bands)
    # Genuine = the board actually changed its chase-safety. Excludes the one mandatory terminal
    # flip per episode, and for the gated form the one gate crossing as well, since neither is a
    # signal the policy can act on and both would otherwise inflate a thin rate.
    genuine_a = flips_a - sum(tally[(name, 'flips_terminal')] for name, _, _ in BANDS)
    genuine_b = (flips_b - sum(tally[(name, 'flips_gated_terminal')] for name, _, _ in BANDS)
                 - sum(tally[(name, 'flips_gate_cross')] for name, _, _ in BANDS))
    rate_a, rate_b = divide(genuine_a, meals_a), divide(genuine_b, meals_b)
    calibration = {
        'budget': BUDGET, 'gate': GATE,
        'variant_a': {'flips': flips_a, 'genuine_flips': genuine_a, 'meals': meals_a,
                      'flips_per_meal': rate_a, 'c': BUDGET / rate_a if rate_a else None},
        'variant_b': {'flips': flips_b, 'genuine_flips': genuine_b,
                      'meals_at_or_above_gate': meals_b,
                      'flips_per_meal': rate_b, 'c': BUDGET / rate_b if rate_b else None},
        'share_steps_at_or_above_gate': divide(
            sum(tally[(name, 'steps')] for name in gated_bands), total_steps),
        'share_flips_at_or_above_gate': divide(
            sum(tally[(name, 'flips')] for name in gated_bands), flips_a),
    }

    summary = {'target': target, 'checkpoint': os.path.basename(path), 'step': step,
               'seeds': seeds, 'episodes_per_seed': episodes, 'outcomes': dict(outcomes),
               'perfect_score': PERFECT_SCORE, 'total_steps': total_steps,
               'calibration': calibration, 'bands': bands, 'per_episode': per_episode}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as handle:
        json.dump(summary, handle, indent=2)

    print('\n%-8s %8s %7s %8s %8s %9s %7s %9s %9s'
          % ('band', 'steps', 'share', 'phi', 'gen/100', 'gen/meal', 'meals', 'step/meal', 'term'))
    for name, _, _ in BANDS:
        row = bands[name]
        fmt = lambda value, spec: ('%' + spec) % value if value is not None else '        -'
        print('%-8s %8d %7s %8s %8s %9s %7d %9s %9d'
              % (name, row['n_steps'], fmt(row['share_steps'], '7.3f'),
                 fmt(row['phi_mean'], '8.3f'), fmt(row['genuine_per_100'], '8.2f'),
                 fmt(row['genuine_per_meal'], '9.2f'), row['meals'],
                 fmt(row['steps_per_meal'], '9.1f'), row['flips_terminal']))
    print('\noutcomes: %s' % dict(outcomes))
    for variant in ('variant_a', 'variant_b'):
        entry = calibration[variant]
        print('%s: %d genuine flips of %d, %.3f genuine/meal -> c = %s'
              % (variant, entry['genuine_flips'], entry['flips'],
                 entry['flips_per_meal'] or 0.0,
                 ('%.3f' % entry['c']) if entry['c'] else 'n/a'))
    print('share of steps at length >= %d: %.3f; share of flips there: %.3f'
          % (GATE, calibration['share_steps_at_or_above_gate'] or 0.0,
             calibration['share_flips_at_or_above_gate'] or 0.0))
    print('wrote ' + out_path, flush=True)


if __name__ == '__main__':
    main()
