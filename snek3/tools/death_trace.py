"""Play a checkpoint greedily and keep every episode's action and food sequence, so failures replay.

Phase 1 of plans/obs-history.md. The measured policy is the argmax and the game is deterministic
given its food sequence, so `(initial food, actions, food placed after each eat)` rebuilds any episode
exactly — `replay()` below does it through `VecSnake.step(forced_food=...)`.

Episode accounting follows `vectorized/engine.py`: exactly `episodes` episodes are *started* and every
one runs to completion, so no episode is selected on its length. Lanes that have nothing left to start
sit finished and are masked out of the policy call.

    PYTHONPATH=. python -m tools.death_trace trace hallOfFame/<entry> --step N --episodes 30000 \
        --out logs/obs-history/<name>.npz

The `.npz` holds flat arrays plus per-episode offsets; `load_traces()` returns a list of dicts.
"""
import argparse
import os
import sys
import time

import numpy as np

from tools import sidecar_env  # noqa: E402  -- must precede anything that imports env.constants
sidecar_env.adopt_from_argv(sys.argv)  # a history checkpoint's depth, from its sidecar

from vectorized.vec_env import VecSnake


def trace(policy_fn, episodes, lanes=512, seed=11, log=sys.stderr):
    n = min(lanes, episodes)
    env = VecSnake(n, seed=seed)
    # Per-lane growing buffers. Python appends at 512 rows/step cost about a minute per 40M steps,
    # which is a small fraction of the observation cost, so nothing cleverer is warranted.
    acts = [[] for _ in range(n)]
    ates = [[] for _ in range(n)]
    foods = [[] for _ in range(n)]
    first_food = env.food.copy()
    started = n
    live = np.ones(n, dtype=bool)
    done_eps = []                     # (first_food, acts, ates, foods, outcome, score)
    obs = env.observe()
    t0 = time.time()
    steps = 0
    while live.any():
        rows = np.flatnonzero(live)
        actions = np.zeros(n, dtype=np.int64)
        actions[rows] = policy_fn(obs[rows])
        obs, _, finished, info = env.step(actions, autoreset=False)
        steps += 1
        a_l = actions.tolist(); e_l = info['ate'].tolist(); f_l = env.food.tolist()
        for i in rows.tolist():
            acts[i].append(a_l[i]); ates[i].append(e_l[i]); foods[i].append(f_l[i])
        fin = np.flatnonzero(finished & live)
        if fin.size:
            for i in fin.tolist():
                outcome = ('perfect' if info['perfect'][i] else
                           'starved' if info['starved'][i] else 'died')
                done_eps.append((int(first_food[i]), np.array(acts[i], np.uint8),
                                 np.array(ates[i], bool), np.array(foods[i], np.int16),
                                 outcome, int(env.score[i])))
                acts[i] = []; ates[i] = []; foods[i] = []
            # Restart only as many as remain to be started; the rest go idle.
            k = min(fin.size, episodes - started)
            if k > 0:
                env.reset_rows(fin[:k])
                first_food[fin[:k]] = env.food[fin[:k]]
                started += k
                obs = env.observe()
            live[fin[k:]] = False
        if steps % 500 == 0:
            n_done = len(done_eps)
            fails = sum(1 for e in done_eps if e[4] != 'perfect')
            print('{0:6d} steps  {1:6d}/{2} done  {3} failures  {4:.0f} s'.format(
                steps, n_done, episodes, fails, time.time() - t0), file=log, flush=True)
    return done_eps


def save_traces(path, eps):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    lengths = np.array([len(e[1]) for e in eps], np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    np.savez_compressed(
        path,
        first_food=np.array([e[0] for e in eps], np.int16),
        actions=np.concatenate([e[1] for e in eps]) if eps else np.zeros(0, np.uint8),
        ate=np.concatenate([e[2] for e in eps]) if eps else np.zeros(0, bool),
        food=np.concatenate([e[3] for e in eps]) if eps else np.zeros(0, np.int16),
        offsets=offsets,
        outcome=np.array([e[4] for e in eps]),
        score=np.array([e[5] for e in eps], np.int16))


def load_traces(path):
    # Read each member once: an `NpzFile` decompresses the whole member on every `z[name]`, so
    # indexing it inside the loop would decompress the 35M-element arrays 30,000 times.
    z = dict(np.load(path))
    out = []
    off = z['offsets']
    for k in range(len(z['outcome'])):
        s, e = int(off[k]), int(off[k + 1])
        out.append({'first_food': int(z['first_food'][k]), 'actions': z['actions'][s:e],
                    'ate': z['ate'][s:e], 'food': z['food'][s:e],
                    'outcome': str(z['outcome'][k]), 'score': int(z['score'][k])})
    return out


def replay(ep, on_step=None, upto=None):
    """Re-run one traced episode in a one-lane `VecSnake`, calling `on_step(t, env, obs, action)`
    *before* each action is applied. Returns the env in its final state."""
    env = VecSnake(1, seed=0)
    env.reset_all(forced_food=np.array([ep['first_food']], np.int64))
    obs = env.observe()
    T = len(ep['actions']) if upto is None else min(upto, len(ep['actions']))
    for t in range(T):
        a = np.array([int(ep['actions'][t])], np.int64)
        if on_step is not None:
            on_step(t, env, obs, int(a[0]))
        obs, _, finished, info = env.step(a, forced_food=np.array([int(ep['food'][t])], np.int64),
                                          autoreset=False)
        if bool(finished[0]) and t != T - 1:
            raise RuntimeError('replay finished early at step {0} of {1}'.format(t, T))
    return env, obs


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    tr = sub.add_parser('trace')
    tr.add_argument('policy')
    tr.add_argument('--step', type=int, default=None)
    tr.add_argument('--episodes', type=int, default=30000)
    tr.add_argument('--lanes', type=int, default=512)
    tr.add_argument('--seed', type=int, default=11)
    tr.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.cmd == 'trace':
        from tools import restore
        policy_fn, arch, step = restore.restore(restore.policy_dir(args.policy), step=args.step)
        print('restored {0} @ {1}'.format(args.policy, step), file=sys.stderr, flush=True)
        eps = trace(policy_fn, args.episodes, lanes=args.lanes, seed=args.seed)
        save_traces(args.out, eps)
        outs = [e[4] for e in eps]
        print('saved {0}: {1} episodes, perfect {2}, died {3}, starved {4}'.format(
            args.out, len(eps), outs.count('perfect'), outs.count('died'), outs.count('starved')),
            file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
