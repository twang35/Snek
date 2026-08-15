"""How does a policy's reading of one observation input change over training?

Built to ask whether a drawdown is where the endgame value function reorganises. The `b18` vs `b20`
comparison found that the record arm read "is it safe to chase the food" (observation 15-17) and the
`b20` arms did not; this script turns that one-shot measurement into a time series, so a run's
before / during / after a collapse can be compared on the same boards.

Reports per checkpoint, over a **fixed** board set:

  d_chase   counterfactual dQ for index 15+a, "head, food and tail all land in one region"
  d_safe    the same for index 6+a, "is this move safe" -- the POSITIVE CONTROL
  d_win     the same for index 18+a, "does this move win the game"
  ratio     d_chase / d_safe
  agree     fraction of boards where the greedy action matches the previous checkpoint's, and
            where it matches the LAST checkpoint in the ladder
  grad      mean |d max_a Q / d obs_i| for all 30 inputs, which is the only measure here that
            covers the continuous inputs a 0 -> 1 flip cannot address

`agree_final` is the one to read for "when did this policy become the policy it ended up as".
`agree_prev` is the churn rate, and needs its step gap divided out before two ladders are compared.

The control is what makes this readable. Q magnitude moves by an order of magnitude over training
and especially through a collapse, so a rising `d_chase` on its own cannot tell "started reading the
flag" from "every Q got bigger". `d_safe` is an input the network must be sensitive to at any skill
level, so the ratio is scale-free in the way the raw delta is not.

Counterfactual, not correlational: index 15+a is flipped 0 -> 1 on a real board and that action's Q
re-read, holding all 29 other values at their measured setting. A correlational split (mean Q where
the flag fires vs where it does not) cannot separate the input from the boards it fires on.

**The board set is held fixed across every checkpoint**, and by default comes from an arm that has
finished, so it cannot shift underfoot. That is the whole reason the series is comparable: the
question is what changed in the network, so nothing else may change. It does mean the boards are
off-policy for most checkpoints -- realistic endgame positions, not the ones that checkpoint would
reach itself. For "does this network read this input" that is what is wanted.

    cd snek2
    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/input_sensitivity_over_time.py \
        <out.json> <policy> <steps> [boards-policy]

`steps` is a comma list whose items are either `240000` or `200000-260000:5000`. `boards-policy`
defaults to `b18b-tgt1000seed2`, whose buffer is frozen and holds the length 90-99 boards the
original finding used.

Read-only with respect to `savedPolicies/` and `runs/`, and it starts no eval, so it cannot displace
the charts in `evals/`.
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

import policy_arch
from eval_agent import build_eval_agent
from snake_environment import SnakeEnvironment
from under_the_hood import expected_q

POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'savedPolicies')
DEFAULT_BOARDS = 'b18b-tgt1000seed2'
LENGTHS = [10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 93, 95, 96, 97, 98, 99]
PER_LENGTH = 400          # boards sampled per length, capped by what the buffer holds
ENDGAME = (95, 99)        # the pooled band the headline numbers come from

# Agreement is reported per band as well as overall. A collapse can be confined to one phase of the
# game -- an endgame-only board set showed 99% action agreement straight through a collapse from
# score 94 to 4, which says where the collapse was not, and nothing about where it was.
BANDS = (('early', 10, 49), ('mid', 50, 84), ('endgame', 85, 99))

CHASE, SAFE, WIN = 15, 6, 18
FLAGS = (('chase', CHASE), ('safe', SAFE), ('win', WIN))
BATCH = 8192


def parse_steps(spec):
    """`'240000,300000-400000:50000'` -> a sorted list of ints."""
    out = []
    for piece in spec.split(','):
        piece = piece.strip()
        if not piece:
            continue
        if '-' in piece:
            span, _, stride = piece.partition(':')
            lo, _, hi = span.partition('-')
            step = int(stride) if stride else 1000
            out.extend(range(int(lo), int(hi) + 1, step))
        else:
            out.append(int(piece))
    return sorted(set(out))


def load_boards(boards_policy, seed=0):
    """A fixed, length-stratified sample of real boards from a saved replay buffer.

    Only frames the policy actually acted on are eligible: `step_type == 1` drops both the episode
    boundaries and the terminal frames, whose Q is unconstrained by any target and therefore
    meaningless to probe (the `game_over` trap in the same shape).
    """
    path = os.path.join(POLICY_DIR, boards_policy, 'replay_buffer', 'buffer.npz')
    data = np.load(path, allow_pickle=True)['data'].item()
    obs = data['field1'][:, 0].astype(np.float32)
    step_type = data['field0'][:, 0]
    length = np.round(obs[:, 22] * 100).astype(int)

    rng = np.random.default_rng(seed)
    picks, lengths = [], []
    for target in LENGTHS:
        eligible = np.where((length == target) & (step_type == 1))[0]
        if len(eligible) == 0:
            continue
        take = eligible if len(eligible) <= PER_LENGTH else rng.choice(
            eligible, PER_LENGTH, replace=False)
        picks.append(obs[take])
        lengths.append(np.full(len(take), target))
    return np.concatenate(picks), np.concatenate(lengths)


def q_values(agent, obs, support=None):
    """Greedy-time Q for a batch of observations, MID step type throughout.

    Through `under_the_hood.expected_q`, so a c51 checkpoint reports `sum_i z_i p_i(s, a)` — the same
    reduction its own policy takes its argmax over — rather than a raw logit that would look like a Q
    value and not be one. `support` comes from the checkpoint's `arch.json` and is required for a
    categorical net; `expected_q` raises if it is missing or if it is supplied for a scalar one.
    """
    out = []
    for start in range(0, len(obs), BATCH):
        chunk = obs[start:start + BATCH]
        q = expected_q(agent._q_network, tf.convert_to_tensor(chunk), support=support,
                       step_type=tf.fill([len(chunk)], 1))
        out.append(q.numpy())
    return np.concatenate(out)


def saliency(agent, obs, support=None):
    """Mean |d max_a Q / d obs_i| per input, averaged over boards.

    The counterfactual flip only works on the binary flags. Indices 0-5 (food direction and
    distance), 21 (starve budget) and the region counts are continuous, so a gradient is the only
    way to ask whether the network reads them at all. Magnitudes are not comparable *between*
    indices whose units differ -- read each index against its own history.
    """
    total = np.zeros(obs.shape[1])
    for start in range(0, len(obs), BATCH):
        chunk = tf.convert_to_tensor(obs[start:start + BATCH])
        with tf.GradientTape() as tape:
            tape.watch(chunk)
            q = expected_q(agent._q_network, chunk, support=support,
                           step_type=tf.fill([len(chunk)], 1))
            best = tf.reduce_max(q, axis=1)
        total += np.abs(tape.gradient(best, chunk).numpy()).sum(axis=0)
    return total / len(obs)


def sensitivities(agent, obs, support=None):
    """Per flag, the mean dQ of flipping that flag 0 -> 1, pooled over the three actions.

    Also returns the mean max-Q, which is the scale the deltas should be read against.
    """
    result = {}
    for name, first in FLAGS:
        deltas = []
        for action in range(3):
            lo = obs.copy()
            lo[:, first + action] = 0.0
            hi = obs.copy()
            hi[:, first + action] = 1.0
            deltas.append(q_values(agent, hi, support)[:, action]
                          - q_values(agent, lo, support)[:, action])
        result['d_' + name] = np.stack(deltas, axis=1).mean(axis=1)
    q = q_values(agent, obs, support)
    result['max_q'] = q.max(axis=1)
    result['argmax'] = q.argmax(axis=1)
    return result


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    out_path, policy, steps_spec = sys.argv[1], sys.argv[2], sys.argv[3]
    boards_policy = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_BOARDS
    steps = parse_steps(steps_spec)

    obs, lengths = load_boards(boards_policy)
    print('boards: %d from %s, lengths %s' % (
        len(obs), boards_policy, sorted(set(lengths.tolist()))))

    ckpt_dir = os.path.join(POLICY_DIR, policy)
    py_env = SnakeEnvironment(discount=0.9975, display=False, policy_name='smoke')
    py_env.reset()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    agent, checkpoint, global_step = build_eval_agent(tf_env, py_env, ckpt_dir)
    # None for a scalar arm, which is what expected_q wants there. Read once: the support is a
    # property of the policy directory, not of a checkpoint inside it.
    support = policy_arch.support_from_arch(policy_arch.require_arch(ckpt_dir))

    endgame = (lengths >= ENDGAME[0]) & (lengths <= ENDGAME[1])
    rows = []
    argmaxes = []
    print('%9s %8s %8s %8s %7s %8s %9s' % (
        'step', 'd_chase', 'd_safe', 'd_win', 'ratio', 'max_q', 'agr_prev'))
    for step in steps:
        prefix = os.path.join(ckpt_dir, 'ckpt-{0}'.format(step))
        if not os.path.exists(prefix + '.index'):
            print('%9d  (no checkpoint)' % step)
            continue
        checkpoint.restore(prefix).expect_partial()
        restored = int(global_step.numpy())
        if restored != step:
            # A silent restore failure would produce a whole series of plausible numbers off one
            # network, which is exactly the kind of result that survives review.
            raise SystemExit('ckpt-%d restored global_step %d' % (step, restored))

        s = sensitivities(agent, obs, support)
        row = {'step': step}
        for key in ('d_chase', 'd_safe', 'd_win', 'max_q'):
            row[key] = float(s[key][endgame].mean())
        row['ratio'] = row['d_chase'] / row['d_safe'] if row['d_safe'] else float('nan')
        row['by_length'] = {
            str(L): {key: float(s[key][lengths == L].mean())
                     for key in ('d_chase', 'd_safe', 'd_win', 'max_q')}
            for L in sorted(set(lengths.tolist()))}
        row['grad'] = [float(g) for g in saliency(agent, obs, support)]
        if argmaxes:
            same = s['argmax'] == argmaxes[-1]
            row['agree_prev'] = float(same.mean())
            row['agree_prev_bands'] = {
                name: float(same[(lengths >= lo) & (lengths <= hi)].mean())
                for name, lo, hi in BANDS}
        else:
            row['agree_prev'] = None
        argmaxes.append(s['argmax'])
        rows.append(row)
        print('%9d %8.3f %8.3f %8.3f %7.3f %8.2f %9s   %s' % (
            step, row['d_chase'], row['d_safe'], row['d_win'], row['ratio'], row['max_q'],
            '-' if row['agree_prev'] is None else '%.3f' % row['agree_prev'],
            '' if row['agree_prev'] is None else ' '.join(
                '%s %.3f' % (name, row['agree_prev_bands'][name]) for name, _, _ in BANDS)))

    # Agreement with where the run ended up, which is the measure of when the final policy
    # actually formed. Filled in afterwards because it needs the last checkpoint.
    for row, argmax in zip(rows, argmaxes):
        same = argmax == argmaxes[-1]
        row['agree_final'] = float(same.mean())
        row['agree_final_bands'] = {name: float(same[(lengths >= lo) & (lengths <= hi)].mean())
                                    for name, lo, hi in BANDS}

    with open(out_path, 'w') as handle:
        json.dump({'policy': policy, 'boards_policy': boards_policy,
                   'n_boards': int(len(obs)), 'per_length': PER_LENGTH,
                   'endgame_band': list(ENDGAME), 'rows': rows}, handle, indent=1)
    print('wrote %s (%d checkpoints)' % (out_path, len(rows)))


if __name__ == '__main__':
    main()
