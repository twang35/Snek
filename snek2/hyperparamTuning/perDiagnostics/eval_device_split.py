"""Where a vec_eval step's time goes, and whether a GPU could take any of it.

**The question this settles.** A batched eval that spends its life calling a neural network looks like
an obvious GPU workload, and on this project it is not -- so the answer needs to be a measurement
somebody can re-take rather than a claim in a doc. Two numbers:

* **The split**, at a given `--width`: the numpy env step and observation build against the TensorFlow
  policy call. Only the policy can move to a GPU, so `1 / (1 - policy_share)` is the Amdahl ceiling on
  *any* accelerator, however fast.
* **The policy call per device**, across batch widths. Run in an env with `tensorflow-metal` installed
  and `/GPU:0` appears; run in the plain `snek` env and only the CPU row prints, which is itself the
  answer to "is MPS even available here".

**Measured on the laptop 2026-08-24 (M-series, 14 cores), width 1024**: the policy is **8.2%** of a
step (413 us of 5050 us) and the observation build alone is 4296 us -- so the ceiling is **1.09x**. The
policy call is **2.3x slower** on MPS at that width (933 us against 407 us) and only overtakes the CPU
above ~8000 rows, because MPS carries a fixed ~900 us a call: batches of 256 and 1024 cost nearly the
same, which is dispatch and transfer rather than arithmetic. The net is 11,650 MACs a row, so a
1024-row batch is ~24 MFLOP -- far too little to amortise a round trip.

**So MPS is a ~10% regression at the operating width, and the crossover does not rescue it**: the numpy
env cost scales with width, so at 16384 lanes the step is ~74 ms and the policy share falls to ~2%,
making the 722 us saved worth under 1%. A wave also runs `VEC_WAVE_PROCS` shards that would contend for
one GPU. **The bottleneck is not on the GPU's side of the fence** -- it is a bitboard flood fill in
numpy, which is not a tensor program.

Two details that make the numbers trustworthy, both easy to get wrong:

* **`.numpy()` is inside the timed region.** It is the synchronisation point; timing the `tf.function`
  alone on a GPU measures the dispatch and not the work, and would have reported MPS as far faster.
* **The network is built inside `tf.device(...)`** as well as called there, so its weights live on the
  device under test. Building on the CPU and calling on the GPU times a weight transfer every call.

**‡ `--verify <policy> <step>` is the more important half, and it is why MPS is disqualified rather
than merely unhelpful.** It asks whether the greedy policy on the visible device still agrees with
`argmax` over its own Q-values, and reports the Q-gap where it does not. On CPU the agreement is total.
**On MPS 23 of 64 states disagree, and the discarded action is worse by a median 0.64 in Q** -- so it is
not float tie-breaking, which this codebase does see (the Q-values themselves agree to 5.7e-06, and
bare `argmax`, graph-mode `argmax` and `tfp.Categorical.mode` are each individually correct on Metal).
End to end that turned four hall-of-fame champions measuring **97-98% perfect** into **0.0%**, with no
error and a *faster* wall clock. **Run this before trusting any new device, TF version or accelerator
build**, because the failure mode is a silent zero, which looks exactly like a bad arm.

Usage (from `snek2/`):

    PYTHONPATH=. python -u hyperparamTuning/perDiagnostics/eval_device_split.py            # both
    PYTHONPATH=. python -u hyperparamTuning/perDiagnostics/eval_device_split.py --width 2048
    PYTHONPATH=. python -u hyperparamTuning/perDiagnostics/eval_device_split.py --split-only
    PYTHONPATH=. python -u hyperparamTuning/perDiagnostics/eval_device_split.py \
        --verify savedPolicies/<policy> <step>
"""

import argparse
import os
import time

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import numpy as np
import tensorflow as tf

# The project's real network shape. Reading it from `snek2.tuned()` would drag in the whole trainer for
# four integers, and a restored checkpoint is not needed: the timing depends on the shape, not the
# weights.
LAYERS = (50, 100, 50)


def build_act(device, obs_len, num_actions=3):
    """A greedy policy of the project's shape, built *and* called on `device`."""
    with tf.device(device):
        net = tf.keras.Sequential(
            [tf.keras.layers.Dense(n, activation='relu') for n in LAYERS]
            + [tf.keras.layers.Dense(num_actions)])
        net.build((None, obs_len))

        @tf.function(input_signature=[tf.TensorSpec([None, obs_len], tf.float32)])
        def act(observation):
            return tf.argmax(net(observation), axis=1, output_type=tf.int32)
    return act


def bench(fn, repeats, warmup=5):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def timed_policy(act, obs, device, repeats):
    with tf.device(device):
        return bench(lambda: act(tf.convert_to_tensor(obs)).numpy(), repeats)


def devices():
    """`(label, spec)` for every device worth timing, CPU first.

    **`get_visible_devices`, not `list_physical_devices`.** The latter still reports a GPU that has been
    hidden with `set_visible_devices([], 'GPU')` -- which is how the CPU arm of an A/B is built -- so
    this header would claim a GPU in the run that was specifically constructed not to have one.
    """
    found = [('CPU', '/CPU:0')]
    if tf.config.get_visible_devices('GPU'):
        found.append(('GPU/MPS', '/GPU:0'))
    return found


def report_split(width, repeats):
    """The numpy/TensorFlow split at `width`, and the Amdahl ceiling it implies."""
    from vectorized import vec_env, config
    env = vec_env.VecSnake(width, seed=0)
    env.reset_all()
    obs = env.observe()
    actions = np.zeros(width, dtype=np.int64)

    act = build_act('/CPU:0', config.OBS_LEN)
    t_policy = timed_policy(act, obs, '/CPU:0', repeats)
    t_full = bench(lambda: env.step(actions, observe=True), max(10, repeats // 2))
    t_bare = bench(lambda: env.step(actions, observe=False), max(10, repeats // 2))
    total = t_full + t_policy

    print('\nsplit at width {0}'.format(width))
    print('  policy (TensorFlow, the only GPU-able part) : {0:9.1f} us  {1:5.1f}%'.format(
        t_policy * 1e6, 100 * t_policy / total))
    print('  env step + observation (numpy)              : {0:9.1f} us  {1:5.1f}%'.format(
        t_full * 1e6, 100 * t_full / total))
    print('    observation build alone                   : {0:9.1f} us'.format(
        (t_full - t_bare) * 1e6))
    print('  total per step                              : {0:9.1f} us'.format(total * 1e6))
    print('  ceiling if the policy became FREE           : {0:9.2f}x'.format(total / t_full))


def report_devices(widths, obs_len, repeats):
    print('\npolicy call per device')
    print('  {0:>8} {1:>10} {2:>11} {3:>13} {4:>10}'.format(
        'batch', 'device', 'us/call', 'rows/s', 'vs CPU'))
    for width in widths:
        obs = np.random.rand(width, obs_len).astype(np.float32)
        baseline = None
        for label, spec in devices():
            try:
                seconds = timed_policy(build_act(spec, obs_len), obs, spec, repeats)
            except Exception as error:            # an unsupported op falls back, or raises
                print('  {0:>8} {1:>10}   FAILED {2}'.format(width, label, type(error).__name__))
                continue
            ratio = '-'
            if baseline is None:
                baseline = seconds
            else:
                ratio = '{0:.2f}x'.format(baseline / seconds)
            print('  {0:>8} {1:>10} {2:>11.1f} {3:>13.0f} {4:>10}'.format(
                width, label, seconds * 1e6, width / seconds, ratio))
    if len(devices()) > 1:
        print('\n  Read "vs CPU" against the split above rather than on its own: a policy that is 2x '
              'slower\n  costs the whole eval only its own share of a step.')


def verify_policy(policy_dir, step, rows):
    """Does the greedy policy still agree with `argmax` over its own Q-values, on this device?

    The comparison has to be **policy against its own network**, not device against device: a
    device-to-device diff cannot separate "wrong" from "different", and the whole question is whether
    the composed `GreedyPolicy(QPolicy)` graph still means what it means on the CPU.

    The Q-gap at each disagreement is what makes the answer unambiguous. This codebase really does have
    near-ties whose argmax flips under float32 reassociation, so a bare disagreement count would be
    consistent with a harmless build; a gap of 0.64 in reward units is not.
    """
    import numpy as np
    from tf_agents.environments import tf_py_environment
    from tf_agents.trajectories import time_step as ts
    import eval_agent
    from snake_environment import SnakeEnvironment
    from vectorized import vec_env

    env = vec_env.VecSnake(rows, seed=7)
    env.reset_all()
    for _ in range(5):
        env.step(np.zeros(rows, dtype=np.int64))
    obs = env.observe().astype(np.float32)

    py_env = SnakeEnvironment()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    agent, checkpoint, _ = eval_agent.build_eval_agent(tf_env, py_env, policy_dir)
    checkpoint.restore(os.path.join(policy_dir, 'ckpt-{0}'.format(step))).expect_partial()
    policy, obs_len = agent.policy, obs.shape[1]

    @tf.function(input_signature=[tf.TensorSpec([None, obs_len], tf.float32)])
    def act(observation):
        count = tf.shape(observation)[0]
        step_type = tf.fill([count], tf.constant(ts.StepType.MID, dtype=tf.int32))
        return policy.action(ts.TimeStep(
            step_type=step_type, reward=tf.zeros([count], dtype=tf.float32),
            discount=tf.ones([count], dtype=tf.float32), observation=observation)).action

    tensor = tf.convert_to_tensor(obs)
    actions = act(tensor).numpy()
    # Repeat, because a stochastic policy would make the whole comparison meaningless and the eval
    # non-reproducible; every policy this project evaluates greedily must be deterministic here.
    if not (actions == act(tensor).numpy()).all():
        print('\n  ** the policy is NOT deterministic on this device - nothing below is meaningful **')
    q = np.asarray(agent._q_network(tensor)[0]).reshape(len(actions), -1)
    best = q.argmax(axis=1)
    disagree = np.flatnonzero(actions != best)

    print('\nverify {0} at step {1}, {2} states'.format(policy_dir, step, rows))
    print('  greedy action agrees with argmax(Q) : {0} of {1}'.format(
        len(actions) - len(disagree), len(actions)))
    if not len(disagree):
        print('  this device computes the policy correctly')
        return
    gaps = np.array([q[i, best[i]] - q[i, actions[i]] for i in disagree])
    print('  Q-gap discarded at those states     : min {0:.2e}  median {1:.4f}  max {2:.4f}'.format(
        gaps.min(), float(np.median(gaps)), gaps.max()))
    print('  gaps under 1e-4 (a true tie)        : {0} of {1}'.format(
        int((gaps < 1e-4).sum()), len(gaps)))
    print('  gaps over 0.1  (a real error)       : {0} of {1}'.format(
        int((gaps > 0.1).sum()), len(gaps)))
    print('  ** this device does not compute the policy correctly - do not measure on it **'
          if (gaps > 0.1).any() else '  disagreements are float-level ties, not errors')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--verify', nargs=2, metavar=('POLICY_DIR', 'STEP'),
                        help='check the greedy policy against argmax(Q) on the visible device')
    parser.add_argument('--verify-states', type=int, default=64)
    parser.add_argument('--width', type=int, default=1024, help='env lanes for the split (vec default)')
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--split-only', action='store_true')
    parser.add_argument('--batches', type=int, nargs='+', default=[256, 1024, 4096, 16384])
    args = parser.parse_args()

    print('TensorFlow {0}, devices: {1}'.format(
        tf.__version__, ', '.join(label for label, _ in devices())))
    if args.verify:
        verify_policy(args.verify[0], int(args.verify[1]), args.verify_states)
        return
    report_split(args.width, args.repeats)
    if not args.split_only:
        from vectorized import config
        report_devices(args.batches, config.OBS_LEN, args.repeats)


if __name__ == '__main__':
    main()
