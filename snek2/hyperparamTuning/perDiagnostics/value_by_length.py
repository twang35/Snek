"""What does the network *believe* a state is worth, by snake length — and is it right?

The companion to [`return_distribution.py`](return_distribution.py). That script measures the
**realised** discounted return `G_t` from greedy play; this one reads the network's own
`V(s) = max_a Q(s, a)` on the same kind of states, in **the same length bands**. Neither number means
much alone. Put side by side they give the calibration error `V − G` per band, which is the only way to
distinguish "the agent is playing badly" from "the agent has been told the wrong thing and is playing it
correctly".

    cd snek2
    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/value_by_length.py <out.json> <ckpt-path-or-policy> \
        <states> <max-episodes>

`<ckpt-path-or-policy>` takes the same two forms as the rest of this directory: a `hallOfFame/<name>` or
`savedPolicies/<name>` directory uses its newest checkpoint, a path ending `ckpt-<step>` pins one.

**Set the environment the checkpoint was trained in.** `SNEK_PERFECT_GAME_REWARD` in particular — a
win-10 arm carries `v_max=40`, and `categorical_agent.check_support` refuses to build it against the
default win of 100. That refusal is the guard working; the fix is to pass the arm's own reward, not to
weaken the check.

**Bands are `return_distribution.BANDS` verbatim, imported rather than copied**, because the whole point
is reading the two outputs against each other and a band edge that drifted by one would silently compare
different populations.

**Read `V − G`, and read its sign.** This script was written for batch 33, where the win reward was cut
from 100 to 10. It found the answer in one table: at length 95-97 the network believed **16.65** where
the realised return was **0.02**. The win-100 control read 56.18 against a realised 56.22 in the band
where both have plenty of states.

**The sharpest reading needs no realised return at all.** `Snake.py` *replaces* the food reward on a
winning step rather than adding to it, so taking the win pays exactly `PERFECT_GAME_REWARD` and
terminates. At win=10 that is **10.0** against a believed 16.65 for carrying on — the network's own
values rank "keep playing" **67% above** "win now", and a policy that acts on them stalls on a
nearly-full board and dies of geometry. That is what
[`behaviour_profile.py`](behaviour_profile.py) had already measured (median 32-42 steps per meal in the
last band against the control's 2, and 73-90% of losses by collision, not starvation) without being able
to say why. **So compare `V` against the win's own payoff before reaching for a return measurement.**

**V is on the greedy policy's own state distribution**, so it is comparable with
`return_distribution.py` (also greedy) and *not* with a training-time value that saw epsilon-greedy
states. Both walk the same env, so a band with few visits is thin in both — check `n` before reading a
band, especially 98-99.

Read-only: restores a checkpoint, writes one JSON, starts no eval.
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
from snake_environment import SnakeEnvironment, OBS_ERA
from eval_agent import build_eval_agent
from behaviour_profile import resolve
from c51_stability import snake_length
from return_distribution import BANDS, band_of


def describe(values):
    if not values:
        return {'n': 0}
    array = np.asarray(values, dtype=np.float64)
    return {'n': int(array.size), 'mean': float(array.mean()), 'sd': float(array.std()),
            'p5': float(np.percentile(array, 5)), 'p50': float(np.percentile(array, 50)),
            'p95': float(np.percentile(array, 95)), 'min': float(array.min()),
            'max': float(array.max())}


def measure(target, states_wanted, max_episodes):
    """Returns `(rows, step, arch)` where `rows` is a list of `(snake_length, V(s))`."""
    path, directory = resolve(target)

    spec_env = SnakeEnvironment(discount=0.99, display=False, policy_name='value_by_length')
    tf_env = tf_py_environment.TFPyEnvironment(spec_env)
    agent, checkpoint, global_step = build_eval_agent(tf_env, spec_env, directory)
    arch = policy_arch.read_arch(directory)
    support = policy_arch.support_from_arch(arch)
    checkpoint.restore(path).expect_partial()
    net = agent._q_network

    rows = []
    episodes = 0
    while len(rows) < states_wanted and episodes < max_episodes:
        time_step = spec_env.reset()
        episodes += 1
        while not time_step.is_last() and len(rows) < states_wanted:
            obs = np.asarray(time_step.observation, dtype=np.float32)
            q = under_the_hood.expected_q(net, tf.constant(obs[None, :]), support=support).numpy()[0]
            rows.append((snake_length(spec_env), float(np.max(q))))
            time_step = spec_env.step(np.int32(np.argmax(q)))
    return rows, int(global_step.numpy()), arch, episodes


def report(payload):
    # A dir written before `arch.json` recorded the reward scale has no field, which is not the same
    # thing as a win of 0 — say so rather than printing None.
    win = payload['perfect_game_reward']
    print('{0} step {1}  v_max {2}  win {3}'.format(
        payload['target'], payload['step'], payload['v_max'],
        'not recorded' if win is None else win))
    print('%-10s %7s %9s %9s %9s %9s' % ('length', 'n', 'V mean', 'V sd', 'V p50', 'V p95'))
    for name, _, _ in BANDS:
        entry = payload['by_band'][name]
        if entry['n'] == 0:
            print('%-10s %7d %9s' % (name, 0, '-'))
            continue
        print('%-10s %7d %9.2f %9.2f %9.2f %9.2f' % (
            name, entry['n'], entry['mean'], entry['sd'], entry['p50'], entry['p95']))
    print('{0} states over {1} episodes'.format(payload['states'], payload['episodes']))
    print('Pair with return_distribution.py at the same gamma to read V - G per band.')


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    out_path, target = sys.argv[1], sys.argv[2]
    states_wanted, max_episodes = int(sys.argv[3]), int(sys.argv[4])

    rows, step, arch, episodes = measure(target, states_wanted, max_episodes)
    by_band = {name: [] for name, _, _ in BANDS}
    for length, value in rows:
        name = band_of(length)
        if name is not None:
            by_band[name].append(value)

    payload = {
        'target': target, 'step': step, 'states': len(rows), 'episodes': episodes,
        'obs_era': OBS_ERA, 'algo': arch.get('algo'), 'v_max': arch.get('v_max'),
        'perfect_game_reward': arch.get('perfect_game_reward'),
        'by_band': {name: describe(by_band[name]) for name, _, _ in BANDS},
        'all': describe([value for _, value in rows]),
    }
    report(payload)
    with open(out_path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print('wrote ' + out_path)


if __name__ == '__main__':
    main()
