"""Phase 1 of plans/archive/obs-history.md: how do the best checkpoints die, and was a zigzag involved?

Reads a `tools.death_trace` file. Perfect games give the matched-board-fill control; each failure is
replayed exactly and classified:

    starved               500 steps without eating; reports the cycle length if the head is looping
    blunder               stepped into body/wall while a legal move existed at that very step
    sealed-food collision the trap the snake entered held the food (or it ate inside the trap)
    forced collision      it walled itself in with no food involved

The point of no return (PONR) is the last step at which some action still left the head able to reach
its tail (the `can reach tail` flags of the connectivity block); the action taken there did not. The fatal wall is the body
cells bounding the region the head then entered; segment k behind the head was laid by the action
k+1 steps earlier, so the wall indexes straight into the action sequence.

    PYTHONPATH=. python -m tools.death_analyze logs/obs-history/<name>.npz [--json out.json]
"""
import argparse
import collections
import json
import sys

import numpy as np

from env.constants import block_ranges
from tools.death_trace import load_traces, replay
from vectorized.vec_env import CAP, GRID, NB, PAD

# Observation blocks by name, so this survives a layout change (2026-09-07 dropped two blocks and
# moved everything after index 17 down three). The traces themselves are layout-independent: they
# hold actions and food, and the observation is rebuilt by replay under the current layout.
_R = block_ranges()
_LEGAL = slice(*_R['body_and_wall'])
_GROUPS = slice(*_R['head_with_tail_groups'])
_CHASE = slice(*_R['safe_to_chase_food'])
_NOT_TAIL = slice(*_R['not_following_tail'])

LEFT, RIGHT, FWD = 0, 1, 2
FILL_BUCKETS = 10          # board fill in tenths: length / 100


# ----------------------------------------------------------------- path measures

def reversal_flags(actions, k=2):
    """1 where a turn is the opposite of a turn within the previous k steps."""
    a = np.asarray(actions, np.int64)
    n = len(a)
    out = np.zeros(n, bool)
    for lag in range(1, k + 1):
        prev = np.full(n, -1, np.int64)
        prev[lag:] = a[:-lag]
        out |= ((a == LEFT) & (prev == RIGHT)) | ((a == RIGHT) & (prev == LEFT))
    return out


def lengths_along(ep):
    """Snake length before each action."""
    return 5 + np.concatenate([[0], np.cumsum(ep['ate'][:-1])]) if len(ep['ate']) else np.zeros(0)


def path_measures(actions):
    a = np.asarray(actions, np.int64)
    n = len(a)
    if n == 0:
        return {'steps': 0, 'turn_density': np.nan, 'reversal_rate': np.nan, 'mean_run': np.nan}
    turns = a != FWD
    rev = reversal_flags(a)
    # straight runs: steps between turns
    idx = np.flatnonzero(turns)
    runs = np.diff(np.concatenate([[-1], idx, [n]])) - 1
    return {'steps': int(n), 'turn_density': float(turns.mean()),
            'reversal_rate': float(rev.mean()), 'mean_run': float(runs.mean())}


def control_by_fill(eps):
    """Perfect games' path measures pooled per fill bucket: {bucket: {turn_density, reversal_rate}}."""
    turns = np.zeros(FILL_BUCKETS); revs = np.zeros(FILL_BUCKETS); steps = np.zeros(FILL_BUCKETS)
    for ep in eps:
        if ep['outcome'] != 'perfect':
            continue
        a = ep['actions'].astype(np.int64)
        b = np.minimum(lengths_along(ep) * FILL_BUCKETS // 100, FILL_BUCKETS - 1).astype(int)
        r = reversal_flags(a)
        np.add.at(steps, b, 1); np.add.at(turns, b, a != FWD); np.add.at(revs, b, r)
    with np.errstate(invalid='ignore', divide='ignore'):
        return {'steps': steps, 'turn_density': turns / steps, 'reversal_rate': revs / steps}


# ----------------------------------------------------------------- replay tools

def flood(open_row, start):
    """Cells reachable from `start` over open cells (start included)."""
    seen = np.zeros(PAD, bool)
    stack = [int(start)]
    seen[start] = True
    while stack:
        c = stack.pop()
        for d in NB:
            nb = c + int(d)
            if 0 <= nb < PAD and not seen[nb] and open_row[nb]:
                seen[nb] = True
                stack.append(nb)
    return seen


def body_cells(env):
    """Body cells head-first: index k is k segments behind the head."""
    L = int(env.length[0]); hp = int(env.hp[0])
    return env.body[0, (hp - np.arange(L)) % CAP]


def cycle_length(heads_dirs):
    """Shortest repeat period of (head, dir) over the tail of the sequence, or 0."""
    seq = list(heads_dirs)
    n = len(seq)
    for p in range(1, min(200, n // 2) + 1):
        if seq[-p:] == seq[-2 * p:-p]:
            return p
    return 0


def analyse_failure(ep, control, window=200):
    T = len(ep['actions'])
    rec = {'outcome': ep['outcome'], 'score': ep['score'], 'steps': T}
    per_step = []          # (legal[3], reach[3], chase[3], food_space, head, dir, length)

    def on_step(t, env, obs, a):
        o = obs[0]
        # food_space (old index 29) was removed on 2026-09-07; its slot here reads nan.
        per_step.append((o[_LEGAL].copy(), o[_GROUPS][0::2].copy(), o[_CHASE].copy(), float('nan'),
                         int(env.body[0, env.hp[0]]), int(env.head_dir[0]), int(env.length[0])))

    replay(ep, on_step=on_step)
    legal = np.array([p[0] for p in per_step]); reach = np.array([p[1] for p in per_step])
    chase = np.array([p[2] for p in per_step]); fspace = np.array([p[3] for p in per_step])
    heads = [p[4] for p in per_step]; dirs = [p[5] for p in per_step]
    lengths = np.array([p[6] for p in per_step])
    a = ep['actions'].astype(np.int64)
    rev = reversal_flags(a)
    rec['final_fill'] = float(lengths[-1] / 100.0)

    if ep['outcome'] == 'starved':
        rec['category'] = 'starved'
        eats = np.flatnonzero(ep['ate'])
        last_food = int(eats[-1]) + 1 if eats.size else 0     # first step of the starve window
        rec['cycle_len'] = cycle_length(list(zip(heads[last_food:], dirs[last_food:])))
        rec['food_space_at_end'] = float(fspace[-1])
        rec['chase_safe_any_in_window'] = float(np.any(chase[last_food:] > 0.5))
        w = a[max(0, last_food - window):last_food]
        rec['pre_window'] = path_measures(w)
        rec['fill_bucket'] = int(min(lengths[last_food] * FILL_BUCKETS // 100, FILL_BUCKETS - 1))
        return rec

    # --- a collision. Was a legal move available at the fatal step?
    fatal_legal_alt = bool(legal[-1].sum() > 0)
    rec['fatal_step_had_legal_alt'] = fatal_legal_alt
    # PONR: last step where any action kept the tail reachable.
    any_reach = np.flatnonzero(reach.max(axis=1) > 0.5)
    if any_reach.size == 0:
        rec['category'] = 'never-reachable'
        return rec
    ponr = int(any_reach[-1])
    rec['ponr'] = ponr
    rec['trap_steps'] = T - 1 - ponr
    rec['ponr_chose_reach'] = bool(reach[ponr, a[ponr]] > 0.5)
    ate_in_trap = bool(ep['ate'][ponr:].any())
    rec['ate_in_trap'] = ate_in_trap
    rec['chase_safe_alt_at_ponr'] = bool(np.any((chase[ponr] > 0.5) & (reach[ponr] > 0.5)))
    rec['food_space_at_ponr'] = float(fspace[ponr])

    # Region entered after the PONR move, and the body wall bounding it.
    env, _ = replay(ep, upto=ponr + 1)
    head = int(env.body[0, env.hp[0]])
    open_row = env.open_[0].copy(); open_row[head] = True
    region = flood(open_row, head)
    food = int(env.food[0])
    rec['food_in_trap'] = bool(food >= 0 and region[food])
    rec['trap_size'] = int(region.sum() - 1)
    body = body_cells(env)                       # k segments behind head at state ponr+1
    wall_k = [k for k in range(1, len(body))
              if any(0 <= body[k] + int(d) < PAD and region[body[k] + int(d)] for d in NB)]
    # segment k at state s=ponr+1 was laid by action index s-1-k
    s = ponr + 1
    wall_idx = [s - 1 - k for k in wall_k if s - 1 - k >= 0]
    rest_idx = [s - 1 - k for k in range(1, len(body)) if k not in wall_k and s - 1 - k >= 0]
    rec['wall_segments'] = len(wall_k)
    rec['wall_reversal_rate'] = float(rev[wall_idx].mean()) if wall_idx else np.nan
    rec['wall_turn_density'] = float((a[wall_idx] != FWD).mean()) if wall_idx else np.nan
    rec['body_rest_reversal_rate'] = float(rev[rest_idx].mean()) if rest_idx else np.nan
    rec['pre_window'] = path_measures(a[max(0, ponr - window):ponr])
    rec['fill_bucket'] = int(min(lengths[ponr] * FILL_BUCKETS // 100, FILL_BUCKETS - 1))
    rec['control_reversal_rate'] = float(control['reversal_rate'][rec['fill_bucket']])
    rec['control_turn_density'] = float(control['turn_density'][rec['fill_bucket']])

    if fatal_legal_alt and rec['trap_steps'] == 0:
        rec['category'] = 'blunder'
    elif fatal_legal_alt and not rec['ponr_chose_reach'] and (ate_in_trap or rec['food_in_trap']):
        rec['category'] = 'sealed-food collision'
    elif ate_in_trap or rec['food_in_trap']:
        rec['category'] = 'sealed-food collision'
    else:
        rec['category'] = 'forced collision'
    return rec


def _fmt(x):
    return 'nan' if x is None or (isinstance(x, float) and np.isnan(x)) else (
        '{0:.3f}'.format(x) if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--json', default=None)
    ap.add_argument('--window', type=int, default=200)
    args = ap.parse_args()
    eps = load_traces(args.trace)
    outcomes = collections.Counter(e['outcome'] for e in eps)
    print('{0}: {1} episodes  {2}'.format(args.trace, len(eps), dict(outcomes)))
    control = control_by_fill(eps)
    print('\nperfect-game control by board fill (tenths):')
    print('  fill   steps   turn_density  reversal_rate')
    for b in range(FILL_BUCKETS):
        print('  {0:.1f}  {1:8.0f}   {2:.3f}         {3:.3f}'.format(
            b / FILL_BUCKETS, control['steps'][b], control['turn_density'][b],
            control['reversal_rate'][b]))
    fails = [e for e in eps if e['outcome'] != 'perfect']
    recs = []
    for i, ep in enumerate(fails):
        recs.append(analyse_failure(ep, control, window=args.window))
        print('.', end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)
    cats = collections.Counter(r['category'] for r in recs)
    print('\nfailures by category: {0}'.format(dict(cats)))
    print('\nper failure:')
    keys = ['category', 'score', 'final_fill', 'ponr', 'trap_steps', 'trap_size', 'food_in_trap',
            'ate_in_trap', 'fatal_step_had_legal_alt', 'chase_safe_alt_at_ponr', 'food_space_at_ponr',
            'wall_segments', 'wall_reversal_rate', 'body_rest_reversal_rate', 'control_reversal_rate',
            'cycle_len']
    print('  ' + '  '.join(keys))
    for r in recs:
        print('  ' + '  '.join(_fmt(r.get(k)) for k in keys))
        pw = r.get('pre_window')
        if pw:
            print('      pre-window: turn_density {0:.3f} reversal_rate {1:.3f} mean_run {2:.2f}'
                  ' (control at fill {3}: rev {4:.3f})'.format(
                      pw['turn_density'], pw['reversal_rate'], pw['mean_run'], r['fill_bucket'],
                      control['reversal_rate'][r['fill_bucket']]))
    # aggregates
    print('\naggregates:')
    for cat in cats:
        rs = [r for r in recs if r['category'] == cat]
        pre = [r['pre_window']['reversal_rate'] for r in rs if r.get('pre_window')]
        ctl = [control['reversal_rate'][r['fill_bucket']] for r in rs if 'fill_bucket' in r]
        wall = [r['wall_reversal_rate'] for r in rs if not np.isnan(r.get('wall_reversal_rate', np.nan))]
        rest = [r['body_rest_reversal_rate'] for r in rs
                if not np.isnan(r.get('body_rest_reversal_rate', np.nan))]
        print('  {0:24s} n={1:3d}  pre-window rev {2}  matched control rev {3}  '
              'wall rev {4}  rest-of-body rev {5}'.format(
                  cat, len(rs), _fmt(float(np.mean(pre))) if pre else 'nan',
                  _fmt(float(np.mean(ctl))) if ctl else 'nan',
                  _fmt(float(np.mean(wall))) if wall else 'nan',
                  _fmt(float(np.mean(rest))) if rest else 'nan'))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'outcomes': dict(outcomes), 'categories': dict(cats),
                       'control': {k: (v.tolist()) for k, v in control.items()},
                       'failures': [{k: (None if isinstance(v, float) and np.isnan(v) else v)
                                     for k, v in r.items()} for r in recs]}, f, indent=1)


if __name__ == '__main__':
    main()
