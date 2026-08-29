"""Where a C51 head's value starts, and how long it takes to come down.

Answers the question `plans/distributional-c51.md` pre-registered and deferred: **a categorical head's
initial expected Q is the grid midpoint** — `(v_min + v_max) / 2`, so **57.5** on the shipped
`[-5, 120]` grid — because the atom logits start near-uniform. A scalar `ddqn` head starts at Q ~ 0
instead (`RandomUniform(+/-0.03)` on its final layer). So every c51 arm in this project began believing
an arbitrary state was worth more than half a perfect game, and the plan said to measure the wash-out
in phase 2 and decide whether `SNEK_C51_ZERO_INIT=1` should become the default.

Three things are reported, and the third is the one that decides it.

1. **`init`** rows: a *fresh* net on the arm's own architecture, with the standard init and with the
   `zero_init` ramp, on the same state set as everything else. Confirms the midpoint claim on the real
   network rather than in the abstract, and confirms the ramp does what it says.
2. **The wash-out ladder**: mean greedy `V = max_a Q(s, a)` at log-spaced checkpoints on a **fixed**
   state set, with `excess` = that mean minus the arm's *final* mean. This is the curve the plan asked
   for. The ladder is log-spaced because the interesting part is the first 100k steps, where a linear
   `--stride` would spend every point in the flat tail.
3. **`score`**: the arm's own `avg_score` at the nearest eval to each rung. The harm hypothesis is not
   "the value starts too high" — it is "the value is still wrong when the arm is trying to learn".
   Reading the two columns together is what separates a transient from a handicap.

**The state set must be shared when arms are compared, for the reason `c51_stability.py` documents at
length**: a set drawn from each arm's own play differs in mean snake length, and `V` is strongly
length-dependent (see `value_by_length.py`), so a per-arm set would compare arms on different states.
Here it matters even more than it does for churn, because `V` *is* the measurement rather than a
tiebreak margin. `--states-from` is therefore the default mode of use, and the script prints a NOTE
when it is omitted.

**What this cannot do.** It measures the offset and its decay on arms that all ran with the standard
init. It cannot attribute an arm's *score* to the init, because no arm has been run with the ramp — that
needs `SNEK_C51_ZERO_INIT=1` against a matched control. What it can do is bound the cost: an offset gone
before the arm scores anything cannot have cost the arm much, and one still present at the arm's peak
is a live suspect.

Usage:

    cd snek2
    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python hyperparamTuning/perDiagnostics/init_optimism.py \
      --policy b36a-c51fc320seed1 --policy b36b-c51fc320seed2 \
      --states-from hallOfFame/b29b-chase10g75seed2-ckpt1447000
"""
import argparse
import json
import os
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import policy_arch
import under_the_hood
from eval_agent import build_eval_agent
from snake_environment import SnakeEnvironment

import c51_stability
from c51_stability import (POLICIES_DIR, checkpoint_steps, collect_shared_states, collect_states,
                          q_values)

RUNS_DIR = 'runs'


def ladder_steps(steps, count, ratio=None):
    """`count` checkpoints spaced geometrically from the earliest to the newest.

    Log spacing rather than `c51_stability.sample_steps`' fixed stride: the offset decays fastest at the
    start, so a linear ladder over a 2M-step arm puts one point inside the transient and nineteen in the
    tail. Each target is snapped to the nearest *available* checkpoint and duplicates are dropped, so an
    arm whose early checkpoints are sparse gets fewer rungs rather than the same one repeated.
    """
    if not steps:
        return []
    lo, hi = float(steps[0]), float(steps[-1])
    if count < 2 or hi <= lo:
        return [steps[-1]]
    if ratio is None:
        ratio = (hi / lo) ** (1.0 / (count - 1))
    wanted = [lo * ratio ** i for i in range(count)]
    array = np.asarray(steps, dtype=float)
    kept = []
    for target in wanted:
        nearest = steps[int(np.argmin(np.abs(array - target)))]
        if nearest not in kept:
            kept.append(nearest)
    if steps[-1] not in kept:
        kept.append(steps[-1])
    return kept


def score_curve(policy_name):
    """`(step, avg_score)` pairs from the arm's own eval file, or `[]` if it has none.

    Read from `runs/<policy>_evals.json` rather than recomputed, because the point of the column is to
    align the value offset against *the arm's own learning as it was recorded* — a fresh eval here would
    be a different measurement with different episode counts.
    """
    path = os.path.join(RUNS_DIR, policy_name + '_evals.json')
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        payload = json.load(handle)
    return [(int(row['step']), float(row.get('avg_score', 0.0)))
            for row in payload.get('evals', []) if 'step' in row]


def score_at(curve, step):
    """The recorded `avg_score` at the eval nearest `step`, or nan when the arm has no eval file."""
    if not curve:
        return float('nan')
    return min(curve, key=lambda pair: abs(pair[0] - step))[1]


def mean_v(net, states, support):
    """Mean greedy value over the state set: `mean_s max_a Q(s, a)`.

    The greedy max rather than the mean over actions, because that is the quantity the Bellman target
    bootstraps from and the one an offset propagates through.
    """
    q = q_values(net, states, support)
    return float(np.mean(q.max(axis=-1))), float(np.mean(np.sort(q, axis=-1)[:, -1]
                                                          - np.sort(q, axis=-1)[:, -2]))


def fresh_rows(observation_spec, action_spec, arch, states, support, mean_len):
    """The two `init` rows: a brand-new net with the standard init, and with the `zero_init` ramp.

    Built from the *recorded* arch so the fresh net matches the arm being measured rather than whatever
    `SNEK_FC_LAYERS` happens to say — the same reason `policy_arch` exists.
    """
    # Set here rather than relied on from the arm loop: `atom_stats` reads a module global in
    # `c51_stability`, and a scalar arm measured last leaves it None. Whoever calls `atom_stats` owns
    # setting it — the first version of this script crashed after a full six-arm ladder for exactly that.
    c51_stability.ATOM_SUPPORT = np.asarray(support, dtype=np.float32)
    rows = []
    for label, zero_init in (('init-standard', False), ('init-zeroed', True)):
        net = under_the_hood.build_categorical_q_net(
            observation_spec, action_spec, arch['fc_layer_params'], arch['num_atoms'],
            v_min=arch['v_min'], v_max=arch['v_max'], zero_init=zero_init)
        value, gap = mean_v(net, states, support)
        mass_low, mass_high, effective_atoms = c51_stability.atom_stats(net, states)
        rows.append({'policy': label, 'step': 0, 'v': value, 'gap': gap, 'score': float('nan'),
                     'mlo': mass_low, 'mhi': mass_high, 'aeff': effective_atoms, 'len': mean_len})
    return rows


def measure(policy_name, points, states_wanted, max_episodes, shared=None):
    """One arm's ladder, newest checkpoint last so `excess` can be taken against its final value."""
    ckpt_dir = os.path.join(POLICIES_DIR, policy_name)
    if not os.path.isdir(ckpt_dir):
        ckpt_dir = policy_name
    steps = checkpoint_steps(ckpt_dir)
    if not steps:
        print('{0}: no checkpoints'.format(policy_name))
        return [], None
    rungs = ladder_steps(steps, points)

    spec_env = SnakeEnvironment(discount=0.99, display=False, policy_name=policy_name)
    tf_env = tf_py_environment.TFPyEnvironment(spec_env)
    agent, checkpoint, global_step = build_eval_agent(tf_env, spec_env, ckpt_dir)
    arch = policy_arch.read_arch(ckpt_dir)
    support = policy_arch.support_from_arch(arch)
    # A `ddqn` arm is measured too, and it is the control that makes the c51 rows readable. Its scalar
    # head starts at Q ~ 0 and *rises*, so where it settles is an independent reading of the true value
    # scale — and how long it takes to get there separates "the categorical init is slow to unlearn"
    # from "gamma 0.9975 makes every value function slow", which look identical on a c51 arm alone.
    categorical = support is not None
    c51_stability.ATOM_SUPPORT = np.asarray(support, dtype=np.float32) if categorical else None
    net = agent._q_network

    if shared is not None:
        states, mean_len = shared
    else:
        checkpoint.restore(os.path.join(ckpt_dir, 'ckpt-{0}'.format(steps[-1]))).expect_partial()
        states, mean_len = collect_states(spec_env, net, support, states_wanted, max_episodes)

    curve = score_curve(policy_name)
    rows = []
    for step in rungs:
        checkpoint.restore(os.path.join(ckpt_dir, 'ckpt-{0}'.format(step))).expect_partial()
        value, gap = mean_v(net, states, support)
        row = {'policy': policy_name, 'step': step, 'v': value, 'gap': gap,
               'score': score_at(curve, step), 'len': mean_len}
        if categorical:
            row['mlo'], row['mhi'], row['aeff'] = c51_stability.atom_stats(net, states)
        rows.append(row)
    final = rows[-1]['v'] if rows else None
    for row in rows:
        row['excess'] = row['v'] - final
    context = None
    if categorical:
        context = {'spec': (tf_env.observation_spec(), tf_env.action_spec()), 'arch': arch,
                   'support': support, 'states': states, 'len': mean_len}
    return rows, context


def report(rows, midpoint=None):
    """The ladder as a table, then the wash-out summary that answers the plan's question."""
    head = '%-34s %9s %8s %8s %7s %7s %6s %6s %6s' % (
        'policy', 'step', 'V', 'excess', 'gap', 'score', 'mlo', 'mhi', 'aeff')
    print(head)
    print('-' * len(head))
    for row in rows:
        # A scalar arm has no atoms, so its last three columns are blank rather than 0 — a 0 there would
        # read as "all the mass left the boundary", which is a measurement and not an absence.
        print('%-34s %9d %8.2f %8s %7.2f %7s %6s %6s %6s' % (
            row['policy'], row['step'], row['v'],
            '%8.2f' % row['excess'] if 'excess' in row else '       -',
            row['gap'],
            '  -   ' if row['score'] != row['score'] else '%6.1f' % row['score'],
            '%6.3f' % row['mlo'] if 'mlo' in row else '   -  ',
            '%6.3f' % row['mhi'] if 'mhi' in row else '   -  ',
            '%6.2f' % row['aeff'] if 'aeff' in row else '   -  '))
    if midpoint is not None:
        print('\ngrid midpoint (the standard init\'s expected Q): %.2f' % midpoint)


def washout(rows, tolerance):
    """The first rung whose `excess` is within `tolerance`, per arm — the wash-out step.

    Reported against the arm's own final value rather than against zero, because the question is when
    the value function stopped carrying its initial offset, not when it reached some absolute level.
    """
    out = {}
    for row in rows:
        if row['policy'].startswith('init-') or 'excess' not in row:
            continue
        if abs(row['excess']) <= tolerance and row['policy'] not in out:
            out[row['policy']] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--policy', action='append', required=True,
                        help='policy name under savedPolicies/, or a path. Repeatable.')
    parser.add_argument('--states-from', default=None,
                        help='draw the fixed state set from this policy and use it for every --policy. '
                             'Strongly preferred: V is length-dependent, so a per-arm set compares '
                             'arms on different states.')
    parser.add_argument('--states', type=int, default=1500)
    parser.add_argument('--episodes', type=int, default=200,
                        help='cap on episodes played while collecting the state set')
    parser.add_argument('--points', type=int, default=14, help='rungs on the log-spaced ladder')
    parser.add_argument('--tolerance', type=float, default=2.0,
                        help='reward units within the final value that count as washed out')
    args = parser.parse_args()

    shared = None
    if args.states_from:
        shared = collect_shared_states(args.states_from, args.states, args.episodes)
        print('state set: {0} states from {1}, mean snake length {2:.1f}\n'.format(
            len(shared[0]), args.states_from, shared[1]))
    else:
        print('NOTE: per-arm state sets. V depends strongly on snake length, so these rows are not '
              'comparable across arms — check the `len` column, and prefer --states-from.\n')

    all_rows = []
    first_context = None
    for policy_name in args.policy:
        rows, context = measure(policy_name, args.points, args.states, args.episodes, shared)
        all_rows.extend(rows)
        if context is not None and first_context is None:
            first_context = context
        # Printed per arm as well as in the final table: a six-arm ladder is minutes of restores, and a
        # failure in the last arm should not discard the five that succeeded.
        if rows:
            print('  {0}: V {1:.2f} -> {2:.2f} over {3} rungs'.format(
                policy_name, rows[0]['v'], rows[-1]['v'], len(rows)), flush=True)

    midpoint = None
    if first_context is not None:
        arch = first_context['arch']
        midpoint = (float(arch['v_min']) + float(arch['v_max'])) / 2.0
        observation_spec, action_spec = first_context['spec']
        all_rows = fresh_rows(observation_spec, action_spec, arch, first_context['states'],
                              first_context['support'], first_context['len']) + all_rows

    report(all_rows, midpoint)

    washed = washout(all_rows, args.tolerance)
    if washed:
        print('\nwashed out (|excess| <= %.1f) at:' % args.tolerance)
        for policy_name, row in washed.items():
            print('  %-34s step %8d   score there %s' % (
                policy_name, row['step'],
                '  -  ' if row['score'] != row['score'] else '%.1f' % row['score']))


if __name__ == '__main__':
    main()
