"""Does the value function pull the agent *toward* finishing, and does it read the inputs that say so?

Written to answer one question about batch 33 that `value_by_length.py` could not: the win-10 arm's `V`
falls with snake length, but is that because it **ignores** the board-fill input, or because it reads it
correctly and the objective genuinely makes progress unattractive? The answer decides whether the fix is
an observation change or a reward change, and they are very different pieces of work.

    cd snek2
    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/endgame_gradient.py <ckpt-or-policy> <states> <max-episodes>

Three panels, all on real greedy-play states (no off-manifold sweeps — those were tried first and are
easy to over-read, since an endgame board with board-fill forced to 0.3 is a state that cannot exist):

  saliency    mean `|dV/dobs_i|` per index, ranked. **Answers "is this input ignored?"** directly, and
              on the states where it matters rather than pooled over the whole game.
  occupancy   the share of states where each of a chosen set of indices is nonzero. A near-constant
              input has untrained weights whatever its saliency says — the `game_over` trap in
              CLAUDE.md, and index 29 is documented at 99.95% constant.
  by band     `V`, the **action gap** (`V` − second-best `Q`), and `dV` per band. The gap says what the
              decision rests on; `dV` says which way the value function points as the snake grows.

**Read `dV` and the gap as a pair, because they answer different questions and one without the other
misleads.** Batch 33's endgame gap is *large* (median 21-24 against a `V` of 15-20) — it separates fatal
from non-fatal moves perfectly. What it cannot do is prefer the move that makes progress, because `dV`
per band is **negative** (−0.8 to −3.8): eating costs more value than the meal pays. The win-100 control
has the same large gap and `dV` of **+1.8 to +12.8**. So the failing arm is not confused and not blind —
it is correctly maximising an objective in which finishing is a loss.

**The design rule this produced.** With `V(m) ≈ (meals remaining) + W·γ^(steps to finish)`, one meal of
progress trades ~1 unit of remaining-food for the win term's discount growth, so progress is locally
attractive only when

    W > 1 / (1 - γ^k)        k = steps per meal

At γ=0.9975 that is **~100 at k=4 and ~57 at k=7**, so the shipped `PERFECT_GAME_REWARD=100` sits just
above the threshold and 10 is an order of magnitude below it. **Shrinking the win reward requires
shrinking γ to match.** Full account:
[`../findings.md`](../findings.md#-falsified-2026-08-16-shrinking-the-win-reward-100--10-does-not-buy-c51-stability--it-teaches-the-agent-that-winning-is-a-mistake).

**Set the environment the checkpoint was trained in** — `SNEK_PERFECT_GAME_REWARD` in particular, since a
win-10 arm carries `v_max=40` and `check_support` refuses to build it against the default win of 100.

Read-only: restores a checkpoint, prints, starts no eval. Add `<out.json>` as a 4th argument to save.
"""
import json
import os
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import policy_arch
import under_the_hood
from snake_environment import SnakeEnvironment
from eval_agent import build_eval_agent
from behaviour_profile import resolve
from c51_stability import snake_length

# Finer than the shared BANDS at the top end on purpose: the whole question is what happens over the
# last few meals, which `return_distribution.BANDS` puts in two buckets.
BANDS = ((20, 49), (50, 79), (80, 89), (90, 93), (94, 96), (97, 99))

# Indices whose occupancy is worth printing every run: board-fill (the progress signal), the starve
# budget beside it, the three "this move wins" flags, and food-space as the known near-constant.
WATCH = (18, 19, 20, 21, 22, 29)
LABELS = {18: 'win flag, left', 19: 'win flag, right', 20: 'win flag, forward',
          21: 'starve budget', 22: 'board-fill', 29: 'food space'}


def collect(env, net, support, want, max_episodes):
    """Greedy play. Returns arrays of observations, snake lengths, V and the action gap."""
    obs_rows, lengths, vs, gaps = [], [], [], []
    episodes = 0
    while len(obs_rows) < want and episodes < max_episodes:
        time_step = env.reset()
        episodes += 1
        while not time_step.is_last() and len(obs_rows) < want:
            obs = np.asarray(time_step.observation, dtype=np.float32)
            q = under_the_hood.expected_q(net, tf.constant(obs[None, :]), support=support).numpy()[0]
            ranked = np.sort(q)[::-1]
            obs_rows.append(obs)
            lengths.append(snake_length(env))
            vs.append(float(ranked[0]))
            gaps.append(float(ranked[0] - ranked[1]))
            time_step = env.step(np.int32(np.argmax(q)))
    return (np.stack(obs_rows), np.asarray(lengths), np.asarray(vs), np.asarray(gaps), episodes)


def saliency(net, support, batch):
    """Mean `|dV/dobs|` per index over `batch`, where `V = max_a Q`."""
    if len(batch) == 0:
        return None
    x = tf.constant(batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        v = tf.reduce_max(under_the_hood.expected_q(net, x, support=support), axis=-1)
    return np.abs(tape.gradient(v, x).numpy()).mean(axis=0)


def measure(target, want, max_episodes):
    path, directory = resolve(target)
    spec_env = SnakeEnvironment(discount=0.99, display=False, policy_name='endgame_gradient')
    tf_env = tf_py_environment.TFPyEnvironment(spec_env)
    agent, checkpoint, global_step = build_eval_agent(tf_env, spec_env, directory)
    arch = policy_arch.read_arch(directory)
    support = policy_arch.support_from_arch(arch)
    checkpoint.restore(path).expect_partial()
    net = agent._q_network

    obs_rows, lengths, vs, gaps, episodes = collect(spec_env, net, support, want, max_episodes)
    endgame = obs_rows[lengths >= 90]
    sal = saliency(net, support, endgame)

    payload = {
        'target': target, 'step': int(global_step.numpy()), 'algo': arch.get('algo'),
        'v_max': arch.get('v_max'), 'perfect_game_reward': arch.get('perfect_game_reward'),
        'states': len(obs_rows), 'episodes': episodes,
        'length_min': int(lengths.min()), 'length_max': int(lengths.max()),
        'endgame_states': int(len(endgame)),
        'saliency': None if sal is None else [float(x) for x in sal],
        'occupancy': {str(i): float((obs_rows[:, i] != 0).mean()) for i in WATCH},
        'mean_input': {str(i): float(obs_rows[:, i].mean()) for i in WATCH},
        'by_band': {},
    }
    for low, high in BANDS:
        mask = (lengths >= low) & (lengths <= high)
        name = '{0}-{1}'.format(low, high)
        if mask.sum() == 0:
            payload['by_band'][name] = {'n': 0}
            continue
        payload['by_band'][name] = {
            'n': int(mask.sum()), 'v': float(vs[mask].mean()),
            'gap_p50': float(np.median(gaps[mask])), 'gap_mean': float(gaps[mask].mean())}
    return payload


def report(payload):
    print('=== {0}  step {1}  v_max {2}  win {3}'.format(
        payload['target'], payload['step'], payload['v_max'],
        'not recorded' if payload['perfect_game_reward'] is None else payload['perfect_game_reward']))
    print('{0} states / {1} episodes, length {2}-{3}, {4} endgame (>=90)'.format(
        payload['states'], payload['episodes'], payload['length_min'], payload['length_max'],
        payload['endgame_states']))

    sal = payload['saliency']
    if sal:
        sal = np.asarray(sal)
        order = np.argsort(-sal)
        total = sal.sum()
        print('\nsaliency at length >=90 — top 6 of {0} by mean |dV/dobs|'.format(len(sal)))
        for i in order[:6]:
            print('   idx %2d  %9.4f  %5.1f%% of total   %s' % (
                i, sal[i], 100.0 * sal[i] / total, LABELS.get(int(i), '')))

    print('\noccupancy — a near-constant input has untrained weights whatever its saliency')
    for i in WATCH:
        print('   idx %2d  nonzero %7.3f%%  mean %.4f   %s' % (
            i, 100.0 * payload['occupancy'][str(i)], payload['mean_input'][str(i)], LABELS.get(i, '')))

    print('\n%-10s %7s %9s %9s %9s   %s' % ('length', 'n', 'V', 'gap p50', 'gap/V %', 'dV vs previous'))
    previous = None
    for low, high in BANDS:
        entry = payload['by_band']['{0}-{1}'.format(low, high)]
        name = '{0}-{1}'.format(low, high)
        if entry['n'] == 0:
            print('%-10s %7d' % (name, 0))
            continue
        delta = '' if previous is None else '%+9.2f' % (entry['v'] - previous)
        print('%-10s %7d %9.2f %9.3f %9.1f   %s' % (
            name, entry['n'], entry['v'], entry['gap_p50'],
            100.0 * entry['gap_mean'] / abs(entry['v']) if entry['v'] else 0.0, delta))
        previous = entry['v']
    print('\nA negative dV column means progress lowers value — the agent is right to avoid finishing.')


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    target, want, max_episodes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    payload = measure(target, want, max_episodes)
    report(payload)
    if len(sys.argv) > 4:
        with open(sys.argv[4], 'w') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print('wrote ' + sys.argv[4])


if __name__ == '__main__':
    main()
