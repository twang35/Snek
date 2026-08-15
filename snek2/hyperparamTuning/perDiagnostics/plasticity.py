"""Is the network losing plasticity as it trains, and does that explain the decline?

Every arm in this project has a lifetime — peak around 2.5-3M steps, dead by ~7M — and thirteen
batches of optimiser, PER and architecture knobs have not moved the ceiling once. The
loss-of-plasticity literature describes that shape: a network that has trained for a long time on a
non-stationary target loses the *ability* to fit a new one, and no hyperparameter recovers it. This
measures the three standard signatures against training step, over a **fixed** board set.

| metric | what it is | bad direction |
|---|---|---|
| `dormant` | ReDo dormant-unit fraction per hidden layer (Sokar et al. 2023): a unit is τ-dormant when its mean absolute activation, normalised by the layer mean, is ≤ τ. τ = 0.025 | **up** — dormant units carry no gradient, so capacity is being retired |
| `dead` | units whose post-ReLU activation is **exactly 0 on every board**. The τ = 0 limit, and unambiguous | **up** |
| `srank` | effective rank of the penultimate feature matrix (Kumar et al. 2021): fewest singular values covering 99% of the spectrum | **down** — the features are collapsing onto a subspace |
| `stable` | stable rank, `‖Φ‖_F² / σ₁²`, a smooth companion to `srank` with no threshold in it | **down** |
| `wnorm` | per-layer ‖W‖_F as a **growth factor over its own initialisation** | **up** — large weights saturate ReLU units and shrink the effective step size |

**The fresh-network row is what makes any of this readable**, so it is always measured first and
printed as `step -1`: an untrained net of the same shape on the same boards. Every number below is
read as a departure from it, which is the only way to say whether `srank 41` is high or low.

**That control is averaged over 15 *seeded* draws, and both parts were forced by measurement.** The
trained rows are byte-identical between two runs of the same arm, but an unseeded 5-draw control moved
its own mean by 0.174 -> 0.143 dormant and 40.6 -> 42.2 srank_c between those same two runs — a third
of the trained-vs-fresh difference it exists to measure. The printed second row is the **standard
error** of the control, which is the bar a departure has to clear.

**`wnorm` is normalised against the initialisation each layer actually got**, not a constant. Hidden
layers use `VarianceScaling(2.0, fan_in)`, whose expected ‖W‖_F is `sqrt(2 · units)`; the Q head uses
`RandomUniform(±0.03)`, whose expected ‖W‖_F is `0.03 · sqrt(fan_in · units / 3)`. Both read ~1.00 on
a fresh net, which `tests/test_plasticity.py` pins — a single constant would make the head look 10x
smaller than its initialisation forever.

**The ladder is uniform in step, on purpose.** The questions this was built for are "does plasticity
fall before the peak", "is a late drawdown worse than an early one" and "does it fall even where the
perfect rate is flat" — all of which are about *where* in training something happens, so choosing
where to look would answer them by construction. A uniform stride looks everywhere equally and lets
the analysis align each row against the arm's own trailing-30 curve, which is joined in as
`trailing` for exactly that purpose.

**The board set is fixed across every checkpoint and every arm**, and comes from a finished arm's
replay buffer via `input_sensitivity_over_time.load_boards`, so nothing moves underfoot: the question
is what changed *in the network*. That makes the boards off-policy for most checkpoints, which is
what is wanted — a policy is being asked how it represents a fixed, realistic set of positions, not
the ones it would choose. See that script's docstring for why the same choice is made there.

Usage, from `snek2/`:

    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/plasticity.py \
        <out.json> <policy> [stride] [extra-steps] [boards-policy]

`stride` defaults to 50000. `extra-steps` is a comma list merged into the ladder, for hall-of-fame
checkpoints. A ladder point with no checkpoint snaps to the nearest one within `SNAP` steps, and is
reported as skipped otherwise — `SNEK_MIN_CHECKPOINT_SCORE` means an arm writes nothing before it
first scores 40, so the early rise is simply not on disk.

`<policy>` is a name under `savedPolicies/`, or **any directory holding `arch.json` and some
`ckpt-*`** — a `hallOfFame/` entry, or a partial ladder staged outside the repo. That second form is
what the desktop arms need: batch 20's wide shapes and all of batch 24 trained on `the-claw-den`, so
only their eval curves came back over the git bus, and a 50k ladder is ~11 MB of the 527 MB directory.
Stage those **outside `savedPolicies/`** — a directory there with holes in it reads as a real arm to
every other tool in this folder. The arm's eval curve is still looked up by the directory's basename,
so keep the staged directory named after the arm.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment

import policy_arch
from eval_agent import build_eval_agent
from input_sensitivity_over_time import load_boards, parse_steps, DEFAULT_BOARDS, POLICY_DIR
from snake_environment import SnakeEnvironment
from under_the_hood import build_q_net

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'runs')
BATCH = 8192
# ReDo's threshold. 0.025 is the paper's default and the value its Atari results use.
TAU = 0.025
# Fraction of the spectrum `srank` must cover. 0.01 is Kumar et al.'s delta.
DELTA = 0.01
# Rows fed to the SVD. The feature matrix is at most 320 wide, so this is far more than enough to
# resolve its rank, and it keeps one SVD in the milliseconds.
SVD_ROWS = 6000
# How far a ladder point may snap to find a checkpoint. Checkpoints land every 1000 steps while an
# arm is scoring above SNEK_MIN_CHECKPOINT_SCORE, so 5000 closes small gaps without silently
# measuring somewhere else entirely.
SNAP = 5000
# Trailing window the project smooths perfect-rate with everywhere else.
TRAILING = 30
# Random initialisations averaged for the fresh-network control row.
FRESH_DRAWS = 15
# Seed for those draws. **Measured:** at 5 unseeded draws the control's own mean moved 0.174 -> 0.143
# dormant and 40.6 -> 42.2 srank_c between two runs of the same arm — a third of the trained-vs-fresh
# difference it is the reference for. Seeding makes the control identical for every arm of a given
# shape, so a cross-arm comparison is against one number rather than a fresh random variable.
FRESH_SEED = 20260814


def layer_stack(q_net):
    """(hidden Dense layers, the linear Q head).

    The head is separated because every metric here treats it differently: it has no ReLU so it has
    no dormant units, it is not the feature layer whose rank is measured, and its initialisation is
    `RandomUniform` rather than `VarianceScaling`.
    """
    layers = list(q_net.layers)
    return layers[:-1], layers[-1]


def activations(hidden, obs):
    """Post-ReLU activations per hidden layer, as a list of (boards x units) arrays."""
    out = [[] for _ in hidden]
    for start in range(0, len(obs), BATCH):
        h = tf.convert_to_tensor(obs[start:start + BATCH])
        for index, layer in enumerate(hidden):
            h = layer(h)
            out[index].append(h.numpy())
    return [np.concatenate(chunks) for chunks in out]


def dormant_fraction(acts, tau=TAU):
    """(τ-dormant share, exactly-dead share) for one layer's activations.

    ReDo's score is the unit's mean absolute activation divided by the layer's mean of that, so it is
    scale-free — which matters here, because Q magnitudes move by an order of magnitude over training
    and an unnormalised threshold would track that instead of dormancy.
    """
    score = np.abs(acts).mean(axis=0)
    mean = score.mean()
    if mean == 0:
        return 1.0, 1.0                       # the whole layer is silent
    dormant = float((score / mean <= tau).mean())
    dead = float((acts.max(axis=0) == 0).mean())
    return dormant, dead


def effective_rank(feats, delta=DELTA):
    """(srank, stable rank) of a feature matrix.

    `srank` is the fewest singular values covering `1 - delta` of the spectrum's mass; stable rank is
    `‖Φ‖_F² / σ₁²`. Both are reported because `srank` steps in integers and can sit still through a
    real collapse, while stable rank moves continuously and has no threshold to argue about.
    """
    if feats.shape[0] > SVD_ROWS:
        feats = feats[:SVD_ROWS]
    singular = np.linalg.svd(feats, compute_uv=False)
    total = singular.sum()
    if total == 0:
        return 0, 0.0
    covered = np.cumsum(singular) / total
    srank = int(np.searchsorted(covered, 1.0 - delta) + 1)
    stable = float((singular ** 2).sum() / (singular[0] ** 2)) if singular[0] > 0 else 0.0
    return srank, stable


def centred_rank(feats):
    """The same two ranks after subtracting each unit's mean — the ones that actually move.

    **Measured before trusting the raw version:** post-ReLU features carry a large common offset, so
    σ₁ is mostly the mean vector and stable rank pins near **1.1 for a fresh net and a trained one
    alike**. Centring asks how many directions the features *vary* in, which is what "representation
    collapse" means. Kumar et al. compute srank on Φ directly; that is kept above for comparability
    and this is the sensitive companion, not a replacement.
    """
    if feats.shape[0] > SVD_ROWS:
        feats = feats[:SVD_ROWS]
    return effective_rank(feats - feats.mean(axis=0, keepdims=True))


def constant_fraction(acts, tol=0.01):
    """Share of units whose activation barely varies across the boards.

    A unit stuck at a constant carries no information however large that constant is, so this catches
    what a magnitude-based dormancy score misses: `dormant` asks "is this unit quiet", this asks "is
    this unit *saying* anything". Scale-free — the per-unit std is compared against the layer's mean
    std, not against an absolute.
    """
    std = acts.std(axis=0)
    mean = std.mean()
    if mean == 0:
        return 1.0
    return float((std / mean <= tol).mean())


def expected_norm(kernel, is_head):
    """The Frobenius norm this layer's initialiser would have produced.

    Hidden layers: `VarianceScaling(scale=2, fan_in)`, so per-weight std is `sqrt(2 / fan_in)` and
    the expected norm is `sqrt(fan_in · units · 2 / fan_in) = sqrt(2 · units)`. Head:
    `RandomUniform(±0.03)`, variance `0.03² / 3`. Keras corrects the truncated normal's variance
    back to the nominal one, so no truncation factor belongs here — `tests/test_plasticity.py`
    checks both against a freshly built network rather than trusting this comment.
    """
    fan_in, units = kernel.shape
    if is_head:
        return 0.03 * np.sqrt(fan_in * units / 3.0)
    return np.sqrt(2.0 * units)


def weight_stats(hidden, head):
    """Per-layer ‖W‖_F, its growth factor over initialisation, and max |w|."""
    rows = []
    for index, layer in enumerate(list(hidden) + [head]):
        kernel = layer.get_weights()[0]
        norm = float(np.linalg.norm(kernel))
        rows.append({'layer': index, 'units': int(kernel.shape[1]),
                     'is_head': layer is head,
                     'wnorm': norm,
                     'growth': norm / float(expected_norm(kernel, layer is head)),
                     'max_abs': float(np.abs(kernel).max())})
    return rows


def hidden_kernels(hidden):
    """Every hidden kernel flattened into one vector, for measuring movement between checkpoints."""
    return np.concatenate([layer.get_weights()[0].ravel() for layer in hidden])


def measure(agent, obs):
    """Every metric for whatever is currently loaded into `agent`."""
    hidden, head = layer_stack(agent._q_network)
    acts = activations(hidden, obs)
    layers = []
    for index, layer_acts in enumerate(acts):
        dormant, dead = dormant_fraction(layer_acts)
        layers.append({'layer': index, 'units': int(layer_acts.shape[1]),
                       'dormant': dormant, 'dead': dead,
                       'constant': constant_fraction(layer_acts),
                       'zero_rate': float((layer_acts == 0).mean()),
                       'mean_act': float(layer_acts.mean())})
    srank, stable = effective_rank(acts[-1])
    srank_c, stable_c = centred_rank(acts[-1])
    weights = weight_stats(hidden, head)
    return {'layers': layers, 'weights': weights,
            'srank': srank, 'srank_frac': srank / float(acts[-1].shape[1]),
            'stable_rank': stable,
            'srank_c': srank_c, 'srank_c_frac': srank_c / float(acts[-1].shape[1]),
            'stable_rank_c': stable_c,
            'kernels': hidden_kernels(hidden),
            'constant_all': float(np.average([l['constant'] for l in layers],
                                             weights=[l['units'] for l in layers])),
            # Whole-network summaries, so a shape with three hidden layers and one with a single
            # layer can be put in the same table. Dormancy is unit-weighted rather than
            # layer-averaged: a 320-unit layer retiring 10% of itself is not the same event as a
            # 30-unit layer doing so.
            'dormant_all': float(np.average([l['dormant'] for l in layers],
                                            weights=[l['units'] for l in layers])),
            'dead_all': float(np.average([l['dead'] for l in layers],
                                         weights=[l['units'] for l in layers])),
            'growth_hidden': float(np.mean([w['growth'] for w in weights if not w['is_head']])),
            'growth_head': float([w['growth'] for w in weights if w['is_head']][0])}


def trailing_curve(policy):
    """{step: trailing-30 perfect rate} for an arm, from its eval log."""
    path = os.path.join(RUNS_DIR, '%s_evals.json' % policy)
    if not os.path.exists(path):
        return {}
    rows = json.load(open(path))['evals']
    steps = [r['step'] for r in rows]
    perfect = [float(r['perfect_percent']) for r in rows]
    out = {}
    for index, step in enumerate(steps):
        window = perfect[max(0, index - TRAILING + 1):index + 1]
        out[step] = sum(window) / len(window)
    return out


def available_steps(ckpt_dir):
    import re
    steps = []
    for name in os.listdir(ckpt_dir):
        found = re.match(r'ckpt-(\d+)\.index$', name)
        if found:
            steps.append(int(found.group(1)))
    return sorted(steps)


def build_ladder(ckpt_dir, stride, extra):
    """Uniform-in-step ladder over what is on disk, snapped to real checkpoints."""
    have = available_steps(ckpt_dir)
    if not have:
        raise SystemExit('no checkpoints in ' + ckpt_dir)
    wanted = list(range(0, have[-1] + stride, stride)) + list(extra)
    have_array = np.array(have)
    ladder, skipped = [], []
    for step in sorted(set(wanted)):
        nearest = int(have_array[np.argmin(np.abs(have_array - step))])
        if abs(nearest - step) <= SNAP:
            ladder.append(nearest)
        else:
            skipped.append(step)
    return sorted(set(ladder)), skipped


def seeded_reinit(net, seed):
    """Redraw every weight in `net` from its own initialiser, with an explicit seed.

    **`tf.random.set_seed` is not enough**, and this was measured rather than assumed: two calls with
    the same global seed produced dormant 0.111 and 0.139, because an op-level seed is derived from a
    per-process op counter, so the draw depends on how many ops have been created before it. Seeding
    each initialiser directly removes that dependence.

    The initialisers are **cloned from the layers' own configs** — `VarianceScaling(2.0, fan_in)` for
    the hidden layers and `RandomUniform(±0.03)` for the head — so this cannot drift from
    `under_the_hood.dense_layer`. Anything with no `seed` in its config (the `Zeros` biases) is left to
    produce its one deterministic answer.
    """
    for index, layer in enumerate(net.layers):
        fresh = []
        for offset, (initialiser, variable) in enumerate(
                ((layer.kernel_initializer, layer.kernel),
                 (layer.bias_initializer, layer.bias))):
            config = initialiser.get_config()
            if 'seed' in config:
                config['seed'] = seed + 1000 * index + offset
                initialiser = initialiser.__class__.from_config(config)
            fresh.append(initialiser(variable.shape, dtype=variable.dtype).numpy())
        layer.set_weights(fresh)


def fresh_baseline(agent, obs, fc, draws=FRESH_DRAWS, seed=FRESH_SEED):
    """The untrained control row: same shape, same boards, no training.

    **Averaged over `draws` initialisations and seeded**, because a single draw is itself a random
    variable — two runs of the same arm read dormant 0.230 and 0.170 on the same shape. Unseeded, the
    *mean* of 5 draws still moved 0.174 -> 0.143 between two runs, against a trained-vs-fresh
    difference of 0.08, so the reference was contributing a third of the effect it was meant to
    measure. Seeding the draws makes it one fixed number per shape; `spread` is kept so a reader can
    see how much of any departure is real.
    """
    fresh_rows = []
    saved = agent._q_network
    try:
        for draw in range(draws):
            fresh = build_q_net(3, tuple(fc))
            fresh(tf.zeros([2, obs.shape[1]], tf.float32), step_type=tf.fill([2], 1))
            seeded_reinit(fresh, seed + 100000 * draw)
            agent._q_network = fresh
            drawn = measure(agent, obs)
            del drawn['kernels']
            fresh_rows.append(drawn)
    finally:
        agent._q_network = saved
    scalars = ['dormant_all', 'dead_all', 'constant_all', 'srank', 'srank_c', 'stable_rank',
               'stable_rank_c', 'growth_hidden', 'growth_head', 'srank_frac', 'srank_c_frac']
    baseline = {key: float(np.mean([r[key] for r in fresh_rows])) for key in scalars}
    baseline['spread'] = {key: float(np.std([r[key] for r in fresh_rows])) for key in scalars}
    # The standard error of the mean is what a departure has to clear, and it is the number the
    # printed sd row would otherwise be mistaken for.
    baseline['stderr'] = {key: baseline['spread'][key] / np.sqrt(draws) for key in scalars}
    baseline.update({'step': -1, 'trailing': None, 'fresh': True, 'move': None,
                     'draws': draws, 'seed': seed, 'layers': fresh_rows[0]['layers'],
                     'weights': fresh_rows[0]['weights']})
    return baseline


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out_path, policy = sys.argv[1], sys.argv[2]
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
    extra = parse_steps(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else []
    boards_policy = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_BOARDS

    obs, lengths = load_boards(boards_policy)
    print('boards: %d from %s, lengths %s' % (len(obs), boards_policy,
                                              sorted(set(lengths.tolist()))), flush=True)

    # A name under savedPolicies/, or a directory anywhere. The eval curve is keyed on the arm name
    # either way, which is why a staged ladder keeps the arm's directory name.
    ckpt_dir = policy if os.path.isdir(policy) else os.path.join(POLICY_DIR, policy)
    # The fresh-network comparison builds a same-shaped *scalar* net, and the layer walk expects a
    # plain Sequential. Neither holds for a categorical head, so refuse before doing any work.
    policy_arch.refuse_categorical(ckpt_dir, 'plasticity.py')
    arm = os.path.basename(os.path.normpath(ckpt_dir))
    ladder, skipped = build_ladder(ckpt_dir, stride, extra)
    curve = trailing_curve(arm)
    if not curve:
        print('note: no runs/%s_evals.json, so `trailing` will be null' % arm, flush=True)
    print('%s: %d ladder points, %d skipped (no checkpoint within %d)'
          % (policy, len(ladder), len(skipped), SNAP), flush=True)

    py_env = SnakeEnvironment(discount=0.9975, display=False, policy_name='smoke')
    py_env.reset()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    agent, checkpoint, global_step = build_eval_agent(tf_env, py_env, ckpt_dir)
    fc = [int(l['units']) for l in
          [{'units': layer.units} for layer in layer_stack(agent._q_network)[0]]]

    rows = []
    header = '%9s %8s %8s %6s %6s %7s %8s %8s %8s %9s' % (
        'step', 'trailing', 'dormant', 'dead', 'const', 'srank_c', 'stable_c', 'growth', 'g_head',
        'move/100k')

    baseline = fresh_baseline(agent, obs, fc)
    rows.append(baseline)
    print(header, flush=True)
    print('%9s %8s %8.3f %6.3f %6.3f %7.1f %8.2f %8.3f %8.3f %9s   <- fresh net x%d, the control'
          % ('fresh', '-', baseline['dormant_all'], baseline['dead_all'],
             baseline['constant_all'], baseline['srank_c'], baseline['stable_rank_c'],
             baseline['growth_hidden'], baseline['growth_head'], '-', FRESH_DRAWS), flush=True)
    # The standard error of that mean, not the sd: this is the bar a departure from the control has to
    # clear, and it is ~4x tighter than the per-draw spread at 15 draws.
    print('%9s %8s %8.3f %6.3f %6.3f %7.1f %8.2f %8.3f %8.3f %9s   <- se of the control'
          % ('', '', baseline['stderr']['dormant_all'], baseline['stderr']['dead_all'],
             baseline['stderr']['constant_all'], baseline['stderr']['srank_c'],
             baseline['stderr']['stable_rank_c'], baseline['stderr']['growth_hidden'],
             baseline['stderr']['growth_head'], '-'), flush=True)

    previous = None
    for step in ladder:
        prefix = os.path.join(ckpt_dir, 'ckpt-{0}'.format(step))
        checkpoint.restore(prefix).expect_partial()
        restored = int(global_step.numpy())
        if restored != step:
            # A silent restore failure would produce a whole plausible series off one network, which
            # is exactly the kind of result that survives review. Same guard as
            # input_sensitivity_over_time.py.
            raise SystemExit('ckpt-%d restored global_step %d' % (step, restored))
        row = measure(agent, obs)
        # Relative weight movement since the previous ladder point, per 100k steps. The only measure
        # here that is *not* a correlate of plasticity — a network that has stopped moving has
        # stopped learning, whatever its dormancy reads. Normalised by the gap because the ladder
        # snaps and the gaps are not exactly equal.
        move = None
        if previous is not None:
            gap = step - previous[0]
            if gap > 0:
                move = float(np.linalg.norm(row['kernels'] - previous[1])
                             / np.linalg.norm(previous[1]) * 100000.0 / gap)
        previous = (step, row['kernels'])
        del row['kernels']
        row.update({'step': step, 'trailing': curve.get(step), 'fresh': False, 'move': move})
        rows.append(row)
        print('%9d %8s %8.3f %6.3f %6.3f %7d %8.2f %8.3f %8.3f %9s'
              % (step, '%.1f' % row['trailing'] if row['trailing'] is not None else '-',
                 row['dormant_all'], row['dead_all'], row['constant_all'], row['srank_c'],
                 row['stable_rank_c'], row['growth_hidden'], row['growth_head'],
                 '%.4f' % move if move is not None else '-'), flush=True)

    payload = {'policy': arm, 'ckpt_dir': ckpt_dir, 'fc_layer_params': fc,
               'boards_policy': boards_policy,
               'boards': int(len(obs)), 'stride': stride, 'tau': TAU, 'delta': DELTA,
               'skipped': skipped, 'rows': rows}
    with open(out_path, 'w') as handle:
        json.dump(payload, handle)
    print('wrote %s' % out_path)


if __name__ == '__main__':
    main()
