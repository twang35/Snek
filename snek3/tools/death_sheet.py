"""A contact sheet of a trace's failures: the final board of each, the head's path over the last
`window` steps coloured by time, the fatal wall outlined, the point of no return marked.

    PYTHONPATH=. python -m tools.death_sheet logs/obs-history/<name>.npz --out charts/…png
"""
import argparse
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, patches
import numpy as np

from tools.death_analyze import analyse_failure, body_cells, control_by_fill, flood
from tools.death_trace import load_traces, replay
from vectorized.vec_env import GRID, NB, PAD


def xy(f):
    return int(f) % GRID, int(f) // GRID


def draw(ax, ep, rec, window=200):
    T = len(ep['actions'])
    heads = []

    def on_step(t, env, obs, a):
        heads.append(int(env.body[0, env.hp[0]]))

    env, _ = replay(ep, on_step=on_step)
    heads.append(int(env.body[0, env.hp[0]]))       # the fatal / final head
    body = body_cells(env)
    for c in body[1:]:
        x, y = xy(c)
        ax.add_patch(patches.Rectangle((x - .5, y - .5), 1, 1, color='#bbbbbb'))
    if env.food[0] >= 0:
        x, y = xy(env.food[0]); ax.add_patch(patches.Circle((x, y), .3, color='red'))
    # fatal wall, from the PONR state
    ponr = rec.get('ponr')
    if ponr is not None:
        env2, _ = replay(ep, upto=ponr + 1)
        head2 = int(env2.body[0, env2.hp[0]])
        op = env2.open_[0].copy(); op[head2] = True
        region = flood(op, head2)
        for c in body_cells(env2)[1:]:
            if any(0 <= c + int(d) < PAD and region[c + int(d)] for d in NB):
                x, y = xy(c)
                ax.add_patch(patches.Rectangle((x - .5, y - .5), 1, 1, fill=False,
                                               edgecolor='orange', lw=1.5))
        for c in np.flatnonzero(region):
            x, y = xy(c)
            ax.add_patch(patches.Rectangle((x - .5, y - .5), 1, 1, color='#ffe9b3', zorder=0))
        px, py = xy(heads[ponr]); ax.plot(px, py, 'x', color='blue', ms=8, mew=2, zorder=5)
    path = heads[max(0, len(heads) - window - 1):]
    cols = cm.viridis(np.linspace(0, 1, len(path)))
    for i in range(len(path) - 1):
        (x0, y0), (x1, y1) = xy(path[i]), xy(path[i + 1])
        ax.plot([x0, x1], [y0, y1], color=cols[i], lw=1.2, zorder=3)
    hx, hy = xy(heads[-1]); ax.plot(hx, hy, 's', color='black', ms=6, zorder=6)
    ax.set_xlim(.5, GRID - 1.5); ax.set_ylim(GRID - 1.5, .5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    title = '{0}  s{1}'.format(rec['category'], rec['score'])
    if ponr is not None:
        title += '  trap {0}'.format(rec['trap_steps'])
        if not np.isnan(rec.get('wall_reversal_rate', np.nan)):
            title += '  wall rev {0:.2f}'.format(rec['wall_reversal_rate'])
    if rec.get('cycle_len'):
        title += '  cycle {0}'.format(rec['cycle_len'])
    ax.set_title(title, fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace'); ap.add_argument('--out', required=True)
    ap.add_argument('--window', type=int, default=200)
    ap.add_argument('--max', type=int, default=120)
    args = ap.parse_args()
    eps = load_traces(args.trace)
    control = control_by_fill(eps)
    fails = [e for e in eps if e['outcome'] != 'perfect'][:args.max]
    cols = 10; rows = max(1, math.ceil(len(fails) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.1))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(fails):]:
        ax.axis('off')
    for ax, ep in zip(axes, fails):
        rec = analyse_failure(ep, control, window=args.window)
        draw(ax, ep, rec, window=args.window)
    fig.suptitle('{0}: {1} failures of {2}. grey body, red food, black head; path of last {3} '
                 'steps dark->light; blue x = point of no return; orange = fatal wall, '
                 'yellow = trap region'.format(args.trace, len(fails), len(eps), args.window),
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
