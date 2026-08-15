"""Can this checkpoint still *fit a new target*? The direct plasticity test, not a proxy for it.

`plasticity.py` measures the three published signatures — dormant units, feature rank, weight norm.
Every one of them is a **correlate**: they are the things that are usually true of a network that has
lost plasticity. This asks the question itself, the way Lyle et al. and Kumar et al. define it — take
the network, give it a fresh target function, let it train, and see how much of that target it can
fit compared with a randomly initialised network of the same shape given the same budget.

    fit = 1 - final_mse / target_variance      (1.0 = fitted perfectly, 0.0 = learned nothing)
    relative = fit(checkpoint) / fit(fresh net)

**`relative` is the number that matters**, because the absolute `fit` depends on the budget: with
`STEPS` gradient steps a fresh net does not reach 1.0 either, and the question is not "is this network
perfect" but "has it lost ground against where it started". Below 1.0 means the trained network is
*worse* at learning something new than an untrained one — the definition of plasticity loss.

**The target is another network of the same shape**, and `REPEATS` of them, seeded and identical for
every checkpoint of every arm. Same shape keeps the target in a function class the architecture can
represent, so a low `fit` cannot be explained by the target being impossible; averaging over several
teachers keeps a difference between two checkpoints from being one target's luck. Random per-board
labels would be a memorisation test instead — a different question, and one the wide nets win by
construction.

**The probe trains a copy and never touches the checkpoint.** Weights are read out, a scratch network
is built, and the copy is what gets fitted — the same restore is then reused for the next probe.
Adam at the training learning rate, so "could training have moved it" is asked with training's own
step size rather than a large one that would hide a small-step problem.

Two things to know before reading the output:

- **A `fit` near 1.0 for everything means the budget is too generous**, and a `fit` near 0 for
  everything means it is too small; either way `relative` stops resolving anything. `STEPS` was chosen
  by checking a fresh net lands mid-range. It is reported in the payload, and comparing numbers across
  different budgets is meaningless.
- **The head is zeroed and the target standardised**, so every network starts at exactly
  `mse_start = 1.0`. A trained Q head is scaled to Q values 20-30x its own initialisation, and against
  a raw teacher target the trained nets read `fit = -283`: the probe was measuring the gap between two
  output scales. `copy_of` and `teacher_targets` carry the detail.
- **Two variants, and the pair is the point.** `fit` trains the whole network — that is plasticity, the
  ability to adapt. `fit_frozen` freezes the hidden stack and fits the head alone — that is whether the
  features it already has can support a new target unchanged, which is what feature rank is a proxy
  for. A network can be poor at the first and fine at the second, and only measuring both distinguishes
  "cannot move" from "has nothing useful to move to".

Usage, from `snek2/`:

    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/plasticity_probe.py <out.json> <policy> [stride] [extra]

Same `<policy>` rules as `plasticity.py`: a name under `savedPolicies/`, or any directory holding
`arch.json` and `ckpt-*`. Read-only with respect to `savedPolicies/` and `runs/`, and starts no eval.
"""
import json
import os
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment

import policy_arch
from eval_agent import build_eval_agent
from input_sensitivity_over_time import load_boards, parse_steps, DEFAULT_BOARDS, POLICY_DIR
from plasticity import build_ladder, layer_stack, seeded_reinit, trailing_curve
from snake_environment import SnakeEnvironment
from under_the_hood import build_q_net

# Gradient steps per probe. **Calibrated, not guessed**, on b18b at three checkpoints:
#
#   steps   fresh fit   trained fit   relative
#     400       0.038     0.31-0.35    8.1-9.1     fresh has barely started; measures fitting *speed*
#    2000       0.527     0.62-0.63    1.17-1.19   fresh mid-range; measures fitting *reach*
#
# 2000 is the default because a control at 0.53 is the informative middle — at 400 the ratio is
# dominated by how slowly a fresh net starts. Both budgets agree on the direction, which is the reason
# to trust either. `relative` is only comparable at equal budget, so the value is in the payload.
STEPS = int(os.environ.get('PROBE_STEPS', 2000))
MINIBATCH = 256
# Independent probes per checkpoint, each with its own teacher and minibatch order, averaged. One
# probe is deterministic but arbitrary: it answers "how well does this network fit *that* target",
# and a difference of a few points between checkpoints could be one target's luck. The sd across
# repeats is reported so a within-arm trend can be read against it.
REPEATS = int(os.environ.get('PROBE_REPEATS', 3))
# Training's own optimiser settings (snek2.py: Adam, SNEK_LEARNING_RATE default 1e-5). A probe at a
# larger step size would answer a question training never asks.
LEARNING_RATE = 1e-5
# Base seed for the teachers. Fixed so every checkpoint of every arm is fitted to the identical set
# of target functions — the comparison is between networks, not between targets.
TEACHER_SEED = 771
# Fresh-net controls averaged for the denominator of `relative`.
FRESH_DRAWS = 5
FRESH_SEED = 20260814


def teacher_targets(fc, obs, seed=TEACHER_SEED):
    """The target function: a seeded random network of the same shape, standardised per output.

    Same shape as the student on purpose — the target is then known to be representable, so a low fit
    is the student's inability rather than the target's impossibility.

    **Standardised to zero mean and unit variance, and this was forced by a wrong first answer.** The
    teacher's head is `RandomUniform(±0.03)`, so its raw outputs have variance 0.003 while a trained
    checkpoint emits Q values of order 1. Against that target the trained nets read `fit = -283`: the
    probe was measuring the distance between two output scales, which says nothing about plasticity.
    Standardising the target and zeroing the student's head (see `copy_of`) puts every network at
    exactly `mse_start = 1.0`, so what is left is how much of the target it can learn.
    """
    teacher = build_q_net(3, tuple(fc))
    teacher(tf.zeros([2, obs.shape[1]], tf.float32), step_type=tf.fill([2], 1))
    seeded_reinit(teacher, seed)
    hidden, head = layer_stack(teacher)
    activation = tf.convert_to_tensor(obs)
    for layer in hidden:
        activation = layer(activation)
    raw = head(activation).numpy()
    return (raw - raw.mean(axis=0)) / raw.std(axis=0)


def copy_of(net, fc, obs_len, weights=None):
    """A scratch network holding `net`'s weights with a **zeroed** head, never the restored one.

    Zeroing the head — rather than redrawing it — is what makes checkpoints comparable: every student
    then starts at output 0, so `mse_start` is exactly the target's variance for all of them and `fit`
    is not a mixture of "how much did it learn" and "where did it start". A trained head is scaled to
    Q values 20-30x its own initialisation, and leaving it in place made the probe read `fit = -283`.

    Gradients still flow. With the head at zero the hidden layers see no gradient on the very first
    step, because `dL/dh` runs through the head's kernel; the head itself moves on that step, and
    everything below moves from the second. That costs one step out of `STEPS` and costs it identically
    for every network being compared.
    """
    scratch = build_q_net(3, tuple(fc))
    scratch(tf.zeros([2, obs_len], tf.float32), step_type=tf.fill([2], 1))
    source = weights if weights is not None else [layer.get_weights() for layer in net.layers]
    for layer, values in zip(scratch.layers, source):
        layer.set_weights([np.array(v) for v in values])
    _, head = layer_stack(scratch)
    head.set_weights([np.zeros(head.kernel.shape, dtype=head.kernel.dtype.as_numpy_dtype),
                      np.zeros(head.bias.shape, dtype=head.bias.dtype.as_numpy_dtype)])
    return scratch


def fit_target(net, obs, targets, steps=STEPS, minibatch=MINIBATCH, seed=0, head_only=False):
    """Fraction of the target's variance this network removes in `steps` Adam steps.

    The minibatch order is seeded, so two probes of the same weights give the same answer and a
    difference between checkpoints is not sampling noise in the probe itself.

    `head_only` freezes the hidden stack and fits the head alone. That separates the two things a low
    full-network fit could mean: **`fit`** is whether the network can still *adapt* — plasticity in the
    sense the question is about — while **`fit_frozen`** is whether the features it already has are
    rich enough to support a new target without changing them, which is what feature rank is a proxy
    for. A network can be poor at the first and fine at the second.
    """
    # **`legacy.Adam` and `tf.function`, both for speed and neither changing the answer.** TF warns
    # that the v2.11+ Adam "runs slowly on M1/M2 Macs", and an eager loop pays Python overhead on every
    # one of `steps` iterations: the first version of this probe needed ~18 minutes per arm, which is
    # three hours across the arm set. Compiled and on the legacy optimiser it is ~10x faster, and
    # `tests/test_plasticity_probe.py` pins the numbers this produces.
    optimiser = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
    layers = list(net.layers)
    trainable = layers[-1:] if head_only else layers
    variables = [v for layer in trainable for v in layer.trainable_variables]
    obs_tensor = tf.convert_to_tensor(obs)
    target_tensor = tf.convert_to_tensor(targets)
    variance = float(np.var(targets))
    rng = np.random.default_rng(seed)

    def forward(batch):
        activation = batch
        for layer in layers:
            activation = layer(activation)
        return activation

    @tf.function
    def train_step(rows):
        batch = tf.gather(obs_tensor, rows)
        wanted = tf.gather(target_tensor, rows)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(forward(batch) - wanted))
        optimiser.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    start = float(np.mean((forward(obs_tensor).numpy() - targets) ** 2))
    # The row draws stay in numpy so the minibatch order depends only on `seed`, not on any TF state.
    for rows in rng.integers(0, len(obs), size=(steps, minibatch)):
        train_step(tf.convert_to_tensor(rows))
    end = float(np.mean((forward(obs_tensor).numpy() - targets) ** 2))
    return {'mse_start': start, 'mse_end': end, 'variance': variance,
            'fit': 1.0 - end / variance if variance else None,
            'reduction': (start - end) / start if start else None}


def fresh_net(fc, obs_len, seed):
    net = build_q_net(3, tuple(fc))
    net(tf.zeros([2, obs_len], tf.float32), step_type=tf.fill([2], 1))
    seeded_reinit(net, seed)
    return net


def fresh_control(fc, obs, target_sets):
    """The denominator of `relative`: what a randomly initialised net of this shape achieves.

    Averaged over both axes the checkpoints are averaged over — `FRESH_DRAWS` initialisations and every
    teacher in `target_sets` — so numerator and denominator have faced the same targets.
    """
    draws = []
    for draw in range(FRESH_DRAWS):
        seed = FRESH_SEED + 100000 * draw
        for repeat, targets in enumerate(target_sets):
            # Zeroed head here too, or the control would start from a different error than the
            # checkpoints it is the denominator for.
            net = copy_of(fresh_net(fc, obs.shape[1], seed), fc, obs.shape[1])
            entry = fit_target(net, obs, targets, seed=repeat)
            frozen = copy_of(fresh_net(fc, obs.shape[1], seed), fc, obs.shape[1])
            entry['fit_frozen'] = fit_target(frozen, obs, targets, seed=repeat,
                                             head_only=True)['fit']
            draws.append(entry)
    keys = ('fit', 'fit_frozen', 'mse_end', 'reduction')
    out = {key: float(np.mean([d[key] for d in draws])) for key in keys}
    out['spread'] = {key: float(np.std([d[key] for d in draws])) for key in keys}
    out['stderr'] = {key: out['spread'][key] / np.sqrt(len(draws)) for key in keys}
    out.update({'step': -1, 'fresh': True, 'draws': len(draws), 'trailing': None})
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out_path, policy = sys.argv[1], sys.argv[2]
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 250000
    extra = parse_steps(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else []
    boards_policy = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_BOARDS

    obs, _ = load_boards(boards_policy)
    ckpt_dir = policy if os.path.isdir(policy) else os.path.join(POLICY_DIR, policy)
    # The teacher and the scratch student are scalar nets of the arm's own shape, which a categorical
    # head is not. Refuse before the ladder is built.
    policy_arch.refuse_categorical(ckpt_dir, 'plasticity_probe.py')
    arm = os.path.basename(os.path.normpath(ckpt_dir))
    ladder, skipped = build_ladder(ckpt_dir, stride, extra)
    curve = trailing_curve(arm)
    print('%s: %d probe points, %d skipped, %d boards' % (arm, len(ladder), len(skipped), len(obs)),
          flush=True)

    py_env = SnakeEnvironment(discount=0.9975, display=False, policy_name='smoke')
    py_env.reset()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    agent, checkpoint, global_step = build_eval_agent(tf_env, py_env, ckpt_dir)
    fc = [layer.units for layer in layer_stack(agent._q_network)[0]]

    target_sets = [teacher_targets(fc, obs, TEACHER_SEED + 1000 * r) for r in range(REPEATS)]
    control = fresh_control(fc, obs, target_sets)
    rows = [control]
    header = '%9s %8s %8s %9s %9s %9s' % ('step', 'trailing', 'fit', 'relative', 'fit_froz',
                                          'rel_froz')
    print('  fc %s, %d teachers x %d fresh draws, %d Adam steps at lr %g'
          % (fc, REPEATS, FRESH_DRAWS, STEPS, LEARNING_RATE), flush=True)
    print(header, flush=True)
    print('%9s %8s %8.4f %9s %9.4f %9s   <- fresh net x%d (se %.4f / %.4f)'
          % ('fresh', '-', control['fit'], '1.000', control['fit_frozen'], '1.000', FRESH_DRAWS,
             control['stderr']['fit'], control['stderr']['fit_frozen']), flush=True)

    for step in ladder:
        checkpoint.restore(os.path.join(ckpt_dir, 'ckpt-{0}'.format(step))).expect_partial()
        restored = int(global_step.numpy())
        if restored != step:
            raise SystemExit('ckpt-%d restored global_step %d' % (step, restored))
        saved = [[np.array(v) for v in layer.get_weights()] for layer in agent._q_network.layers]
        fits, frozens = [], []
        for repeat, targets in enumerate(target_sets):
            fits.append(fit_target(copy_of(agent._q_network, fc, obs.shape[1], saved), obs, targets,
                                   seed=repeat)['fit'])
            frozens.append(fit_target(copy_of(agent._q_network, fc, obs.shape[1], saved), obs,
                                      targets, seed=repeat, head_only=True)['fit'])
        row = {'step': step, 'fresh': False, 'trailing': curve.get(step),
               'fit': float(np.mean(fits)), 'fit_frozen': float(np.mean(frozens)),
               'fit_sd': float(np.std(fits)), 'fit_frozen_sd': float(np.std(frozens)),
               'fits': [float(f) for f in fits], 'frozens': [float(f) for f in frozens]}
        row['relative'] = row['fit'] / control['fit'] if control['fit'] else None
        row['rel_frozen'] = (row['fit_frozen'] / control['fit_frozen']
                             if control.get('fit_frozen') else None)
        rows.append(row)
        print('%9d %8s %8.4f %9.3f %9.4f %9.3f   (sd %.4f)'
              % (step, '%.1f' % row['trailing'] if row['trailing'] is not None else '-',
                 row['fit'], row['relative'], row['fit_frozen'], row['rel_frozen'],
                 row['fit_sd']), flush=True)

    payload = {'policy': arm, 'ckpt_dir': ckpt_dir, 'fc_layer_params': fc,
               'boards_policy': boards_policy, 'boards': int(len(obs)), 'stride': stride,
               'probe_steps': STEPS, 'minibatch': MINIBATCH, 'learning_rate': LEARNING_RATE,
               'teacher_seed': TEACHER_SEED, 'repeats': REPEATS, 'fresh_draws': FRESH_DRAWS,
               'skipped': skipped, 'rows': rows}
    with open(out_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=float)
    print('wrote %s' % out_path, flush=True)


if __name__ == '__main__':
    main()
