"""What the PER priority signal actually does, measured from saved buffers and checkpoints.

Answers three questions that the batch 18 vs 19/20 comparison raised but could not settle from
training curves alone:

1. **Do the two signals prioritize different transitions?** No, and provably not: Huber is
   strictly monotone in `|td_error|`, so both rank every transition identically. Measured as a
   top-k Jaccard, which comes out at exactly 1.0 -- if it ever does not, the mechanism story
   here is wrong and the loss function has changed.
2. **How much mass does the top of the buffer get, after importance sampling?** This is the real
   difference between the batches, and it is *not* the priority alone -- IS weights partly undo
   prioritization, completely so in the limit. Reported as the concentration of the expected
   update, which is the quantity the network actually sees.
3. **Where is the value function wrong?** Mean max-Q against snake length, which is what
   separated the batches most sharply.

Priorities are **not** recoverable from a saved buffer: `cpprb.save_transitions()` keeps the
transitions and resets priorities to the max. So this recomputes them fresh, by restoring the
arm's own final checkpoint and running its own buffer through the same `DdqnAgent` loss that
`training.py` calls -- `extra.td_error` and `extra.td_loss` are the exact two tensors
`PRIORITY_SIGNAL` chooses between. The real in-buffer priorities were **staler** than these,
because a transition's priority is only refreshed when it is sampled, so treat the numbers here
as the sharpest the config could be rather than exactly what it was.

Usage, from `snek2/`:

    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/per_priorities.py <out_dir> [policy ...]

With no policies it does batch 18 against batch 20 wave 1, which is the seed-matched pair the
finding rests on. Writes `<out_dir>/<policy>_per.npz` per arm plus a summary chart, and prints
the tables. **Read-only with respect to `savedPolicies/` and `runs/`** -- it restores
checkpoints and never writes one, so it is safe to run beside a live arm.
"""
import json
import os
import sys

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np

# Both batches ran (50, 100, 50). Set before importing anything that builds a network, because
# build_q_net reads SNEK_FC_LAYERS at call time and a mismatch restores silently under
# expect_partial() -- see eval_agent.py.
os.environ.setdefault('SNEK_FC_LAYERS', '50,100,50')

import tensorflow as tf
from tf_agents.environments import tf_py_environment

import policy_arch
from eval_agent import build_eval_agent
from snake_constants import PERFECT_SCORE
from snake_environment import SnakeEnvironment

POLICY_DIR = 'savedPolicies'
CHUNK = 4096

# cpprb adds this to the raw priority before raising it to alpha, so a zero TD error still gets
# sampled. Measured rather than read off the docs: two items at priorities 0 and 1 with alpha 1.0
# sample in a 1.0001e-4 ratio.
CPPRB_EPS = 1e-4

DEFAULT_ARMS = ['b18a-tgt1000seed1', 'b18b-tgt1000seed2', 'b18c-tgt1000seed3',
                'b18d-tgt1000seed4', 'b20a-fc50seed1', 'b20b-fc50seed2',
                'b20c-fc50seed3', 'b20d-fc50seed4']

CATS = ['won the game', 'died (wall/body)', 'starved', 'ate food',
        'ordinary, len >= 80', 'ordinary, len < 80']


def load_buffer(policy):
    """The six flat fields of a saved replay buffer, in `tf.nest.flatten(Trajectory)` order.

    That order is step_type, observation, action, next_step_type, reward, discount --
    `policy_info` is `()` here and flattens to nothing, which is why six fields cover a
    seven-field namedtuple.
    """
    path = os.path.join(POLICY_DIR, policy, 'replay_buffer', 'buffer.npz')
    data = np.load(path, allow_pickle=True)['data'].item()
    return [data['field{0}'.format(i)] for i in range(6)]


def categorise(reward, snake_len):
    """One mutually exclusive bucket per transition, from the reward it earned.

    Ordering matters: the win test runs last so a winning move is never counted as food, even
    though it also eats. `PERFECT_GAME_REWARD` is 100 and `FOOD_REWARD` is 1.0, so `> 50`
    separates them with room to spare.
    """
    cats = np.full(len(reward), 'ordinary', dtype=object)
    cats[np.isclose(reward, 1.0)] = 'ate food'
    cats[np.isclose(reward, -0.5)] = 'starved'
    cats[np.isclose(reward, -5.0)] = 'died (wall/body)'
    cats[reward > 50] = 'won the game'
    ordinary = cats == 'ordinary'
    cats[ordinary & (snake_len >= 80)] = 'ordinary, len >= 80'
    cats[ordinary & (snake_len < 80)] = 'ordinary, len < 80'
    return cats


def sampling_probs(raw, alpha):
    p = np.power(raw + CPPRB_EPS, alpha)
    return p / p.sum()


def exposures(abs_err, huber, alpha=0.6):
    """Expected share of the update each transition receives, per config.

    Sampling draws transition `i` with probability proportional to `raw_i ** alpha`. With
    importance sampling on, its gradient is then scaled by `w_i`, and cpprb's weights are
    `(N p_i) ** -beta` rescaled by `normalize_is_weights` to average 1.0 -- so `w_i` is
    proportional to `p_i ** -beta`. The product is what the network sees:

        exposure_i  ~  p_i * p_i ** -beta  =  raw_i ** (alpha * (1 - beta))

    **At beta = 1.0 that exponent is zero**, so the expected update is uniform and the priority
    signal cannot matter. This is not an approximation of PER being weak at high beta; it is
    prioritization cancelling exactly, in expectation. Verified against realised cpprb draws in
    `is_flattening_check()`, which finds a small residue the algebra misses.
    """
    n = len(abs_err)
    return {
        'b18: td_loss, no IS': sampling_probs(huber, alpha),
        'td_error, no IS (untested)': sampling_probs(abs_err, alpha),
        'b19/b20: td_error, IS beta=0.4': sampling_probs(abs_err, alpha * 0.6),
        'b19/b20: td_error, IS beta=1.0': np.full(n, 1.0 / n),
    }


def concentration(p):
    n = len(p)
    order = np.sort(p)[::-1]
    cum = np.cumsum(np.sort(p))
    return {
        'top0.1pct': float(order[:max(1, n // 1000)].sum()),
        'top1pct': float(order[:n // 100].sum()),
        'top10pct': float(order[:n // 10].sum()),
        # Effective sample size as a fraction of the buffer: 1.0 is uniform, and 0.25 means the
        # update behaves as though the buffer were a quarter of its size.
        'ess_frac': float(1.0 / (n * np.sum(p ** 2))),
        'gini': float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n),
    }


def measure(policy, agent, checkpoint, global_step):
    # Every number below is a scalar-head quantity — |TD error|, Huber td_loss, max Q. A c51 arm has
    # none of them, so it is refused here rather than measured wrongly.
    policy_arch.refuse_categorical(os.path.join(POLICY_DIR, policy), 'per_priorities.py')
    latest = tf.train.latest_checkpoint(os.path.join(POLICY_DIR, policy))
    if latest is None:
        raise SystemExit('no checkpoint under ' + os.path.join(POLICY_DIR, policy))
    checkpoint.restore(latest).expect_partial()
    fields = load_buffer(policy)
    rows = len(fields[0])

    abs_err = np.empty(rows)
    huber = np.empty(rows)
    qmax = np.empty(rows)
    for start in range(0, rows, CHUNK):
        stop = min(start + CHUNK, rows)
        flat = [tf.convert_to_tensor(f[start:stop]) for f in fields]
        experience = tf.nest.pack_sequence_as(agent.collect_data_spec, flat)
        # training.py's call exactly. gamma stays at the agent's 1.0 because the discount
        # travels in the trajectory; weights=None because the per-sample signals are wanted,
        # not a weighted mean.
        _, extra = agent._loss(experience,
                               td_errors_loss_fn=agent._td_errors_loss_fn,
                               gamma=agent._gamma,
                               reward_scale_factor=agent._reward_scale_factor,
                               weights=None, training=False)
        abs_err[start:stop] = np.abs(extra.td_error.numpy())
        huber[start:stop] = extra.td_loss.numpy()
        q, _ = agent._q_network(tf.convert_to_tensor(fields[1][start:stop, 0]),
                               step_type=tf.convert_to_tensor(fields[0][start:stop, 0]))
        qmax[start:stop] = np.max(q.numpy(), axis=1)

    return {
        'checkpoint_step': int(global_step.numpy()),
        'abs_td_error': abs_err,
        'td_loss': huber,
        'qmax': qmax,
        'reward': fields[4][:, 0].astype(np.float64),
        # observation index 22 is snake_len / PERFECT_SCORE -- see state_helpers.get_observations.
        'snake_len': np.round(fields[1][:, 0, 22] * PERFECT_SCORE).astype(int),
    }


def summarise(policy, m, alpha=0.6):
    abs_err, huber = m['abs_td_error'], m['td_loss']
    exp = exposures(abs_err, huber, alpha)
    cats = categorise(m['reward'], m['snake_len'])

    # Huber is monotone in |delta|, so both signals must rank identically. A Jaccard below 1.0
    # means the loss function is no longer element-wise Huber and this whole reading is stale.
    k = 1000
    top_err = set(np.argsort(-abs_err)[:k].tolist())
    top_loss = set(np.argsort(-huber)[:k].tolist())

    live = abs_err > 1e-6
    slope = float(np.polyfit(np.log(abs_err[live]), np.log(huber[live] + 1e-12), 1)[0])

    table = {}
    for cat in CATS:
        mask = cats == cat
        row = {'count': int(mask.sum()), 'buffer_share': float(mask.mean())}
        for name, p in exp.items():
            row[name] = float(p[mask].sum())
        row['mean_abs_td'] = float(abs_err[mask].mean()) if mask.any() else None
        table[cat] = row

    by_len = {}
    for length in range(0, PERFECT_SCORE + 1):
        mask = m['snake_len'] == length
        if mask.sum() >= 20:
            by_len[length] = {'n': int(mask.sum()),
                              'mean_qmax': float(m['qmax'][mask].mean()),
                              'mean_abs_td': float(abs_err[mask].mean())}

    top = np.argsort(-abs_err)[:100]
    return {
        'policy': policy,
        'checkpoint_step': m['checkpoint_step'],
        'abs_td_error': {'mean': float(abs_err.mean()), 'median': float(np.median(abs_err)),
                         'p99': float(np.percentile(abs_err, 99)), 'max': float(abs_err.max()),
                         'frac_below_1': float((abs_err < 1.0).mean())},
        'mean_qmax': float(m['qmax'].mean()),
        'top1000_jaccard': len(top_err & top_loss) / len(top_err | top_loss),
        'huber_loglog_slope': slope,
        'effective_alpha': alpha * slope,
        'concentration': {name: concentration(p) for name, p in exp.items()},
        'categories': table,
        'by_length': by_len,
        'top100_mean_len': float(m['snake_len'][top].mean()),
        'buffer_mean_len': float(m['snake_len'].mean()),
    }


def is_flattening_check(abs_err, huber, alpha=0.6, n=20000, batch=128, draws=6000):
    """Realised exposure from actual cpprb draws, against a same-effort uniform noise floor.

    `exposures()` derives the beta=1.0 case as exactly uniform. That derivation runs through
    cpprb's C++ weight computation and `normalize_is_weights`, neither of which is in the
    algebra, so it is checked rather than trusted. The floor matters because a finite number of
    draws never reads as perfectly uniform.

    The residue this finds is real and worth knowing: `normalize_is_weights` divides by the
    **batch** mean rather than a global constant, so the cancellation is per-batch and imperfect,
    and it leaves more concentration behind the sharper the priorities are.
    """
    from cpprb import PrioritizedReplayBuffer
    from prioritized_replay_buffer import normalize_is_weights

    abs_err, huber = abs_err[:n], huber[:n]

    def run(raw, beta, weight):
        rb = PrioritizedReplayBuffer(n, {'x': {'shape': (1,), 'dtype': np.float32}}, alpha=alpha)
        for i in range(n):
            rb.add(x=np.array([i], dtype=np.float32))
        rb.update_priorities(np.arange(n), raw)
        total = np.zeros(n)
        for _ in range(draws):
            got = rb.sample(batch, beta=beta)
            np.add.at(total, got['indexes'],
                      normalize_is_weights(got['weights']) if weight else 1.0)
        return concentration(total / total.sum())

    return {
        'noise floor (flat priorities)': run(np.ones(n), 0.0, False),
        'b18: td_loss, no IS': run(huber, 0.0, False),
        'td_error, no IS (untested)': run(abs_err, 0.0, False),
        'b19/b20: td_error, IS beta=0.4': run(abs_err, 0.4, True),
        'b19/b20: td_error, IS beta=1.0': run(abs_err, 1.0, True),
        'td_loss, IS beta=1.0 (never run)': run(huber, 1.0, True),
    }


def chart(summaries, path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(13, 5.2))
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2)

    for s in summaries:
        b18 = s['policy'].startswith('b18')
        lengths = sorted(s['by_length'])
        left.plot(lengths, [s['by_length'][x]['mean_qmax'] for x in lengths],
                  color='#1f77b4' if b18 else '#d62728',
                  linestyle='-' if b18 else '--', linewidth=1.4, alpha=0.85,
                  label=s['policy'] if b18 else None)
    left.set_xlabel('snake length (100 = perfect game)')
    left.set_ylabel('mean max Q over the buffer')
    left.set_title('Value against progress through the game\n'
                   'blue solid = batch 18 (td_loss, no IS), red dashed = batch 20 (td_error, IS)')
    left.grid(alpha=0.3)
    left.axhline(0, color='black', linewidth=0.8)
    # The last two lengths spike to 50-140 as the 100-point terminal reward comes into view, on as
    # few as 20 samples. Left off-scale rather than dropped: it compresses everything below 95 to a
    # flat band, which is the part of the curve the two batches disagree about.
    left.set_ylim(-4, 58)
    left.annotate('length 99-100 runs off-scale\n(terminal reward, n as low as 20)',
                  xy=(0.03, 0.05), xycoords='axes fraction', fontsize=8, color='#444444')

    schemes = ['b18: td_loss, no IS', 'td_error, no IS (untested)',
               'b19/b20: td_error, IS beta=0.4', 'b19/b20: td_error, IS beta=1.0']
    width = 0.2
    spots = np.arange(3)
    for i, scheme in enumerate(schemes):
        vals = [100 * np.mean([s['concentration'][scheme][m] for s in summaries])
                for m in ('top0.1pct', 'top1pct', 'top10pct')]
        right.bar(spots + i * width, vals, width, label=scheme)
    right.set_xticks(spots + 1.5 * width)
    right.set_xticklabels(['top 0.1%', 'top 1%', 'top 10%'])
    right.set_ylabel('share of the expected update (%)')
    right.set_title('Concentration of the expected update\n'
                    'sampling probability x importance-sampling weight')
    right.legend(fontsize=8)
    right.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print('wrote', path)


def main():
    out_dir = sys.argv[1]
    arms = sys.argv[2:] or DEFAULT_ARMS
    os.makedirs(out_dir, exist_ok=True)

    py_env = SnakeEnvironment(discount=0.9975, display=False, policy_name='smoke')
    py_env.reset()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    agent, checkpoint, global_step = build_eval_agent(tf_env, py_env)

    summaries = []
    for policy in arms:
        m = measure(policy, agent, checkpoint, global_step)
        np.savez_compressed(os.path.join(out_dir, policy + '_per.npz'),
                            checkpoint_step=m['checkpoint_step'],
                            abs_td_error=m['abs_td_error'], td_loss=m['td_loss'],
                            qmax=m['qmax'], reward=m['reward'], snake_len=m['snake_len'])
        summaries.append(summarise(policy, m))
        print('measured %-22s ckpt %-9d mean|td| %.4f  mean maxQ %.2f' % (
            policy, m['checkpoint_step'], m['abs_td_error'].mean(), m['qmax'].mean()), flush=True)

    first = np.load(os.path.join(out_dir, arms[0] + '_per.npz'))
    realised = is_flattening_check(np.abs(first['abs_td_error']), first['td_loss'])

    with open(os.path.join(out_dir, 'summary.json'), 'w') as handle:
        json.dump({'arms': summaries, 'realised_exposure': realised}, handle, indent=1)
    chart(summaries, os.path.join(out_dir, 'per-priorities.png'))
    report(summaries, realised)


def report(summaries, realised):
    print('\n=== ordering: both signals rank identically (Jaccard must be 1.0) ===')
    for s in summaries:
        print('%-22s top-1000 Jaccard %.4f   huber log-log slope %.3f -> effective alpha %.3f'
              % (s['policy'], s['top1000_jaccard'], s['huber_loglog_slope'],
                 s['effective_alpha']))

    print('\n=== concentration of the expected update, mean over the arms measured ===')
    print('%-34s %9s %9s %9s %8s %7s' % ('scheme', 'top0.1%', 'top1%', 'top10%', 'ESS/N', 'gini'))
    for scheme in summaries[0]['concentration']:
        c = {m: np.mean([s['concentration'][scheme][m] for s in summaries])
             for m in ('top0.1pct', 'top1pct', 'top10pct', 'ess_frac', 'gini')}
        print('%-34s %8.2f%% %8.2f%% %8.2f%% %8.3f %7.3f' % (
            scheme, 100 * c['top0.1pct'], 100 * c['top1pct'], 100 * c['top10pct'],
            c['ess_frac'], c['gini']))

    print('\n=== realised exposure from actual cpprb draws (one arm) ===')
    print('%-34s %9s %8s' % ('scheme', 'top1%', 'ESS/N'))
    for scheme, c in realised.items():
        print('%-34s %8.2f%% %8.3f' % (scheme, 100 * c['top1pct'], c['ess_frac']))

    print('\n=== where the mass goes, and where the buffer sits ===')
    for s in summaries:
        print('%-22s top-100 by priority: mean length %.1f (buffer mean %.1f)' % (
            s['policy'], s['top100_mean_len'], s['buffer_mean_len']))


if __name__ == '__main__':
    main()
