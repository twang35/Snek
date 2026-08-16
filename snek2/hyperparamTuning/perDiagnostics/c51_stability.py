"""Why a C51 arm's score moves so much more than a scalar arm's, measured on the policy rather than
on the score.

The pilot's eval curves are visibly more chaotic than their seed-matched `b25` control — 2.5x the
step-to-step movement in trailing score and 2.2x the drawdown depth, on the same config, environment
and seeds. An eval curve cannot say *why*: a score that swings could be a policy that keeps changing
its mind, or a stable policy being measured through a noisy 10-episode sample. Those want opposite
fixes, so this separates them.

Everything here is read from checkpoints already on disk. Nothing trains, nothing is written into
`runs/`, `evals/` or `savedPolicies/`.

## What it measures

**Churn** is the headline. For one arm, a set of states is collected once by playing its newest
checkpoint greedily; then every sampled checkpoint scores that *same* fixed state set, and adjacent
checkpoints are compared by how often their greedy action differs. Fixing the states is the whole
point — it removes the episode sampling that the eval curve cannot separate out, so what is left is
the policy changing. `agent.policy` is unshielded (`shielded_policy` masks the exploration draw
only), so a raw `argmax_a Q(s, a)` is exactly the action an eval would take, and a flip is a real
behaviour change rather than one the shield would have hidden.

**The action gap** says whether churn is cheap or expensive to produce. `gap` is
`mean(Q_top - Q_second)`; `gapn` divides it by the across-state spread of `Q_top`, which makes it
comparable between a scalar head and a categorical one whose Q is a weighted sum over atoms. A
*small* normalised gap means the ranking of actions is decided by a margin thinner than the noise in
the value estimate, which is the mechanism by which a well-trained network can still act
erratically.

**Boundary mass and effective atom count** are C51-specific and test the support directly. `mlo` and
`mhi` are the probability sitting on the outermost atoms. Mass on the top atom is the diagnostic that
matters here: `v_max=120` is above the highest return this task can pay (~104), so a well-calibrated
policy should put ~0 there, and mass that piles up instead means the projection is being clipped and
the expected value is biased upward with nowhere to go. `aeff` is `exp(entropy)` in atoms — the width
of the predicted return distribution. A collapsed `aeff` (near 1) is a network that has stopped
representing uncertainty, which for C51 is the same failure as a scalar head; a very wide one at a
state whose outcome is nearly determined is the opposite.

## Reading it

A ddqn arm on the same architecture and observation era is the reference, and `b30e-h`
(`fc 200,100,100`, era `b09c616`) are the closest available locally — the `b25` control itself ran on
the desktop, so no checkpoints for it exist here. Compare churn between the two algorithms at a
similar score, not at a similar step: churn falls as a policy converges, so an arm still climbing is
expected to churn more than one that has plateaued.

Two confounds worth stating rather than correcting for. The state sets differ between arms, because
each arm collects its own; `len` reports the mean snake length in the set so a mismatch is visible.
And checkpoint *spacing in steps* is what churn is per-unit-of, so `--stride` is held in steps rather
than in checkpoint index — an arm whose checkpoints are 1,000 steps apart and one whose are 5,000
apart are not otherwise comparable.

    PYTHONPATH=. python hyperparamTuning/perDiagnostics/c51_stability.py \
        --policy c51pilot-lr5e5seed1 --policy b30e-chase10fc200x100x100seed1

    --states N     states in the fixed set (default 1500)
    --points K     checkpoints sampled per arm (default 10)
    --stride S     steps between sampled checkpoints (default 5000); the newest K*S steps are used
    --episodes E   cap on collection episodes (default 60)
"""
import argparse
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

POLICIES_DIR = 'savedPolicies'


def checkpoint_steps(ckpt_dir):
    """Every step with a restorable checkpoint in `ckpt_dir`, ascending."""
    return sorted(int(f[len('ckpt-'):].split('.')[0])
                  for f in os.listdir(ckpt_dir)
                  if f.startswith('ckpt-') and f.endswith('.index'))


def sample_steps(steps, points, stride, end=None):
    """`points` checkpoints spaced about `stride` steps apart, ending at `end`, ascending.

    `end` defaults to the newest checkpoint. Passing it is how churn is resolved by *phase*: churn
    falls as a policy converges, so a tail-only reading answers a different question from the one an
    eval curve raises when it swings in mid-training. Anchoring two arms at the same `end` compares
    them at the same point in their own training rather than at the same distance from their cap.

    Spacing is in *steps*, not in list index, because churn is a rate per step and arms differ in how
    often they wrote. Walks down from the newest and takes the first checkpoint at least `stride`
    below the last one kept, so a gap in the series costs a point rather than shifting every later
    one.
    """
    kept = []
    if end is not None:
        steps = [s for s in steps if s <= end]
        if not steps:
            return []
    for step in reversed(steps):
        if not kept or kept[-1] - step >= stride:
            kept.append(step)
        if len(kept) == points:
            break
    return list(reversed(kept))


def snake_length(env):
    """The snake's current length, for the `len` column that makes a state-set mismatch visible.

    Reached through the private `_game` because `SnakeEnvironment` exposes no accessor, and a missing
    attribute is reported as 0 rather than raised — the column is context, not a measurement, and a
    diagnostic must not fail on it.
    """
    game = getattr(env, '_game', None)
    if game is None or not hasattr(game, 'snapshot'):
        return 0
    return len(game.snapshot().body)


def collect_states(env, net, support, count, max_episodes):
    """`count` observations from greedy play, plus the mean snake length over them.

    Played one step at a time in the parent rather than through a worker pool: the set has to be
    *fixed* and identical for every checkpoint, so it is built once and held as an array.
    """
    states = []
    lengths = []
    episodes = 0
    while len(states) < count and episodes < max_episodes:
        time_step = env.reset()
        episodes += 1
        while not time_step.is_last() and len(states) < count:
            obs = np.asarray(time_step.observation, dtype=np.float32)
            states.append(obs)
            lengths.append(snake_length(env))
            q = under_the_hood.expected_q(net, tf.constant(obs[None, :]), support=support)
            time_step = env.step(np.int32(np.argmax(q.numpy()[0])))
    return np.stack(states), float(np.mean(lengths)) if lengths else 0.0


def q_values(net, states, support, batch=512):
    """`Q(s, .)` for the whole state set, batched so a large set does not build one giant graph."""
    out = []
    for i in range(0, len(states), batch):
        chunk = tf.constant(states[i:i + batch])
        out.append(np.asarray(under_the_hood.expected_q(net, chunk, support=support)))
    return np.concatenate(out)


def atom_stats(net, states, batch=512):
    """`(mass_low, mass_high, effective_atoms)` for a categorical net, over the greedy action only.

    Restricted to the greedy action because that is the distribution the policy acts on; averaging
    over all three would dilute the boundary mass with actions the policy has already rejected.
    """
    los, his, effs = [], [], []
    for i in range(0, len(states), batch):
        chunk = tf.constant(states[i:i + batch])
        step_type = tf.fill([tf.shape(chunk)[0]], 1)
        logits, _ = net(chunk, step_type=step_type)
        probs = tf.nn.softmax(logits, axis=-1).numpy()        # [batch, actions, atoms]
        q = (probs * ATOM_SUPPORT[None, None, :]).sum(-1)
        best = q.argmax(-1)
        chosen = probs[np.arange(len(probs)), best]           # [batch, atoms]
        los.append(chosen[:, 0])
        his.append(chosen[:, -1])
        entropy = -(chosen * np.log(np.clip(chosen, 1e-12, None))).sum(-1)
        effs.append(np.exp(entropy))
    return (float(np.mean(np.concatenate(los))), float(np.mean(np.concatenate(his))),
            float(np.mean(np.concatenate(effs))))


ATOM_SUPPORT = None    # set per arm, so atom_stats' Q matches the arm's own support


def measure(policy_name, states_wanted, points, stride, max_episodes, end=None):
    """One arm: restores each sampled checkpoint against a fixed state set and returns its rows."""
    global ATOM_SUPPORT
    ckpt_dir = os.path.join(POLICIES_DIR, policy_name)
    steps = checkpoint_steps(ckpt_dir)
    if not steps:
        print('{0}: no checkpoints'.format(policy_name))
        return []
    chosen = sample_steps(steps, points, stride, end)
    if not chosen:
        print('{0}: no checkpoint at or below step {1}'.format(policy_name, end))
        return []

    spec_env = SnakeEnvironment(discount=0.99, display=False, policy_name=policy_name)
    tf_env = tf_py_environment.TFPyEnvironment(spec_env)
    agent, checkpoint, global_step = build_eval_agent(tf_env, spec_env, ckpt_dir)
    arch = policy_arch.read_arch(ckpt_dir)
    support = policy_arch.support_from_arch(arch)
    ATOM_SUPPORT = np.asarray(support, dtype=np.float32) if support is not None else None
    net = agent._q_network

    # Newest first, so the state set comes from the arm's most developed policy.
    checkpoint.restore(os.path.join(ckpt_dir, 'ckpt-{0}'.format(chosen[-1]))).expect_partial()
    states, mean_len = collect_states(spec_env, net, support, states_wanted, max_episodes)

    rows = []
    previous_action = None
    previous_q = None
    for step in chosen:
        checkpoint.restore(os.path.join(ckpt_dir, 'ckpt-{0}'.format(step))).expect_partial()
        if int(global_step.numpy()) != step:
            print('  warning: global_step reads {0}, expected {1}'.format(
                int(global_step.numpy()), step))
        q = q_values(net, states, support)
        order = np.sort(q, axis=-1)
        gap = float(np.mean(order[:, -1] - order[:, -2]))
        top = order[:, -1]
        action = q.argmax(-1)
        # How far the value function itself moved over `stride` steps. Held against `gap` rather than
        # reported alone: a flip needs the movement to exceed the margin between the top two actions,
        # so `dqrel` above ~1 is a value function lurching far enough to reorder actions by itself,
        # while a small `dqrel` with high churn would instead mean actions were near-tied.
        dq = float(np.mean(np.abs(q - previous_q))) if previous_q is not None else float('nan')
        row = {
            'policy': policy_name, 'step': step, 'gap': gap,
            'gapn': gap / float(np.std(top)) if np.std(top) > 0 else float('nan'),
            'qmean': float(np.mean(top)), 'qsd': float(np.std(top)),
            'churn': float(np.mean(action != previous_action)) if previous_action is not None
                     else float('nan'),
            'dq': dq, 'dqrel': dq / gap if gap > 0 else float('nan'),
            'len': mean_len, 'states': len(states),
        }
        if support is not None:
            row['mlo'], row['mhi'], row['aeff'] = atom_stats(net, states)
        rows.append(row)
        previous_action = action
        previous_q = q
    return rows


def report(rows):
    """Per-checkpoint table then a per-arm mean, since the mean is what compares two arms."""
    categorical = any('mhi' in r for r in rows)
    head = '%-34s %9s %7s %8s %7s %7s %7s %7s' % (
        'policy', 'step', 'churn', 'gap', 'dq', 'dqrel', 'qmean', 'qsd')
    if categorical:
        head += ' %6s %6s %6s' % ('mlo', 'mhi', 'aeff')
    print(head)
    for r in rows:
        line = '%-34s %9d %7.3f %8.3f %7.3f %7.3f %7.2f %7.2f' % (
            r['policy'][:34], r['step'], r['churn'], r['gap'], r['dq'], r['dqrel'],
            r['qmean'], r['qsd'])
        if categorical:
            line += (' %6.3f %6.3f %6.1f' % (r['mlo'], r['mhi'], r['aeff'])
                     if 'mhi' in r else ' %6s %6s %6s' % ('-', '-', '-'))
        print(line)

    print()
    print('%-34s %7s %8s %7s %7s %6s' % ('arm mean', 'churn', 'gap', 'dq', 'dqrel', 'len'))
    for name in dict.fromkeys(r['policy'] for r in rows):
        mine = [r for r in rows if r['policy'] == name]
        churn = [r['churn'] for r in mine if r['churn'] == r['churn']]
        extra = ''
        if any('mhi' in r for r in mine):
            extra = '  mhi %.3f  mlo %.3f  aeff %.1f' % (
                np.mean([r['mhi'] for r in mine]), np.mean([r['mlo'] for r in mine]),
                np.mean([r['aeff'] for r in mine]))
        finite = [r for r in mine if r['dq'] == r['dq']]
        print('%-34s %7.3f %8.3f %7.3f %7.3f %6.1f%s' % (
            name[:34], np.mean(churn) if churn else float('nan'),
            np.mean([r['gap'] for r in mine]),
            np.mean([r['dq'] for r in finite]) if finite else float('nan'),
            np.mean([r['dqrel'] for r in finite]) if finite else float('nan'),
            mine[0]['len'], extra))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--policy', action='append', required=True,
                        help='policy directory under savedPolicies/; repeatable')
    parser.add_argument('--states', type=int, default=1500)
    parser.add_argument('--points', type=int, default=10)
    parser.add_argument('--stride', type=int, default=5000)
    parser.add_argument('--episodes', type=int, default=60)
    parser.add_argument('--end', type=int, default=None,
                        help='anchor the newest sampled checkpoint at or below this step, so churn '
                             'can be read at a chosen phase rather than only at the cap')
    args = parser.parse_args()

    rows = []
    for name in args.policy:
        print('measuring {0}'.format(name))
        rows.extend(measure(name, args.states, args.points, args.stride, args.episodes,
                            args.end))
        print()
    report(rows)


if __name__ == '__main__':
    main()
