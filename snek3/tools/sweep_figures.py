"""The sweep's figures, from `viewer/sweep.json` (tools/sweep_analysis.py) into `charts/sweep/`.

    PYTHONPATH=. python -m tools.sweep_analysis figures                 # every batch: curve + traces, then levers, peaks, scatters
    PYTHONPATH=. python -m tools.sweep_analysis figures b17 b19         # those batches only
    PYTHONPATH=. python -m tools.sweep_analysis figures --cell b17 clipannealhold80   # one cell: 4 seeds across, metrics down

| figure | file | what |
|---|---|---|
| curve | `<batch>-curve.png` | x = the knob's cells in value order; one panel per metric; four seed dots, the median line, the reference at its own x with a band across the panel at its seed range |
| traces | `<batch>-traces.png` | x = transitions; one row per stage-A metric; every cell overlaid as its seed-median per bin, coloured light→dark by axis position; the reference thick grey |
| cell | `<batch>-<slug>-cell.png` | four seeds across, metrics down, the reference's seeds grey behind |
| levers | `levers.png` | one row per knob: its best cell's delta from the reference on density, drawdown and hof5000 best, as the four seeds' range |
| peaks | `peaks.png` | knobs × metrics: the winning cell, shaded by how far outside the reference's seed range it sits |
| scatters | `scatter-ev-density.png`, `scatter-kl-drawdown.png` | every arm of the sweep: does the critic's fit predict density, does the KL tail predict collapses |

Drawn with matplotlib's object API, never `pyplot`, for the reason `progress_chart.py` gives. Colours
follow the `dataviz` skill's reference palette: one blue ramp for the ordered cells (the knob's
order is the magnitude), slot-1 blue for a single series, slot-2 orange for the reference where it
must stand apart, text in ink tokens, hairline grid.
"""

import argparse
import math
import os
import statistics
import sys

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from tools import sweep_analysis as sa

# --- palette (dataviz reference instance, light surface)
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
BLUE = '#2a78d6'          # slot 1
ORANGE = '#eb6834'        # slot 2: the reference
REF_GREY = '#898781'
# The sequential blue ramp, steps 250..700 (an ordinal ramp on the light surface starts no lighter than 250).
RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']

DPI = 150
FONT = 7
RC = {'font.size': FONT, 'font.family': 'sans-serif', 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
      'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK, 'text.color': INK,
      'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
      'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
      'lines.antialiased': True, 'text.antialiased': True}

CURVE_PANELS = ['density98', 'hof_best', 'hof30k_best', 'best30', 'stage_a_98', 'drawdown80', 'drawdown50',
                'onset90_m', 'ev_late', 'kl_p99', 'entropy_end']
TRACE_ROWS = [('perfect', 'perfect rate /100, bin mean', None), ('perfect_min', 'worst eval in bin', None),
              ('score', 'average score', None), ('stage_b', 'stage-B rows ≥98 /500, share per 2.5M', None),
              ('value_loss', 'value loss', 'log'), ('ev', 'explained variance', None),
              ('kl', 'approx KL', 'log'), ('clipfrac', 'clip fraction', None), ('entropy', 'entropy', None)]


def _hex_to_rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


def ramp_colours(n):
    """`n` colours light→dark along the blue ramp, interpolated between its steps."""
    if n <= 1:
        return [RAMP[-1]]
    pts = [_hex_to_rgb(h) for h in RAMP]
    out = []
    for i in range(n):
        t = i / (n - 1) * (len(pts) - 1)
        lo, hi = int(math.floor(t)), min(len(pts) - 1, int(math.ceil(t)))
        f = t - lo
        rgb = tuple(pts[lo][k] * (1 - f) + pts[hi][k] * f for k in range(3))
        out.append('#%02x%02x%02x' % tuple(int(round(c * 255)) for c in rgb))
    return out


def cell_colours(batch):
    """A colour per non-reference cell: the blue ramp by axis position when the cells are ordered (any
    numeric cell), the categorical palette when they are switches (b19)."""
    cells = [c for c in batch['cells'] if not c['reference']]
    ordered = any(c['value'] is not None for c in cells)
    if ordered or len(cells) > len(CATEGORICAL):
        colours = ramp_colours(len(cells))
    else:
        colours = CATEGORICAL[:len(cells)]
    return {c['slug']: colours[i] for i, c in enumerate(cells)}


def _canvas(fig):
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(SURFACE)


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    return path


def _metric(sweep, key):
    return next(m for m in sweep['metrics'] if m['key'] == key)


def _steps(arms, key, sweep):
    """The x grid (M transitions) of a series: its first arm's bins -- every arm of a batch shares the batch's
    grid, and a reference from another batch keeps its own, drawn clipped to this batch's horizon."""
    block = 'stage_b' if key == 'stage_b' else 'trace'
    return np.array(sweep['arms'][arms[0]][block]['steps']) / 1e6


def _m(transitions):
    """`250k`, `2.5M`, `1M`: a bin size as the axis prints it."""
    if transitions >= 1_000_000:
        v = transitions / 1e6
        return ('%g' % v) + 'M'
    return ('%g' % (transitions / 1e3)) + 'k'


def _median_over(arms, key, sweep, block='trace'):
    """Per bin, the median over the cell's arms of `trace[key]` (None-aware). For `stage_b`, the share
    of rows ≥98 pooled over the arms per bin (None where no rows)."""
    if not arms:
        return None
    if key == 'stage_b':
        rows = np.array([sweep['arms'][p]['stage_b']['rows'] for p in arms], dtype=float).sum(axis=0)
        hi = np.array([sweep['arms'][p]['stage_b']['rows98'] for p in arms], dtype=float).sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            share = np.where(rows > 0, 100.0 * hi / rows, np.nan)
        return share
    stack = np.array([[np.nan if v is None else v for v in sweep['arms'][p]['trace'][key]] for p in arms], dtype=float)
    with np.errstate(all='ignore'):
        return np.nanmedian(stack, axis=0) if len(stack) else None


# ------------------------------------------------------------------------------------ the curve
def curve_figure(sweep, batch, panels=CURVE_PANELS):
    cells = batch['cells']
    ref = next((c for c in cells if c['reference']), None)
    n = len(panels)
    with _rc():
        fig = Figure(figsize=(7.2, 1.15 * n + 0.9))
        _canvas(fig)
        axes = fig.subplots(n, 1, sharex=True)
        fig.subplots_adjust(left=0.24, right=0.98, top=1 - 0.55 / (1.15 * n + 0.9), bottom=0.65 / (1.15 * n + 0.9), hspace=0.28)
        xs = [c['x'] for c in cells]
        for ax, key in zip(axes, panels):
            m = _metric(sweep, key)
            ax.set_facecolor(SURFACE)
            ax.grid(True, axis='y')
            ax.grid(False, axis='x')
            if ref is not None:
                rs = ref['stats'][key]
                if rs['min'] is not None:
                    ax.axhspan(rs['min'], rs['max'], color=ORANGE, alpha=0.10, lw=0, zorder=0)
                    ax.axhline(rs['median'], color=ORANGE, lw=0.8, alpha=0.7, zorder=1)
            med_x, med_y = [], []
            for c in cells:
                st = c['stats'][key]
                vals = [v for v in st['values'] if v is not None]
                colour = ORANGE if c['reference'] else BLUE
                if vals:
                    ax.scatter([c['x']] * len(vals), vals, s=14, facecolors=SURFACE if c['reference'] else colour,
                               edgecolors=colour, linewidths=1.0, zorder=3)
                    if not c['reference'] and key != 'hof30k_best':      # hof30k: a line through the few cells with rows would read as a trend
                        med_x.append(c['x']); med_y.append(st['median'])
                elif key in ('hof30k_best', 'hof_best', 'onset90_m', 'step98_m'):
                    # nothing measured: an open tick at the bottom says "no rows" / "not reached"
                    ax.plot([c['x']], [0], marker='|', color=MUTED, ms=6, mew=0.8, zorder=2, clip_on=False,
                            transform=ax.get_xaxis_transform())
            if med_x:
                ax.plot(med_x, med_y, color=BLUE, lw=1.4, zorder=2)
            label = m['label'] + (' (' + m['unit'] + ')' if m['unit'] and m['unit'] != 'M' else '')
            if key == 'value_loss_late' and not batch['value_loss_comparable']:
                label += ' — not comparable across this batch'
            ax.set_ylabel(label, fontsize=FONT - 0.5, rotation=0, ha='right', va='center', labelpad=6)
            ax.tick_params(length=2, pad=2, labelsize=FONT - 1)
            if key in ('kl_p99',):
                ax.set_yscale('log')
        axes[-1].set_xticks(xs)
        axes[-1].set_xticklabels([c['label'] for c in cells], rotation=35 if len(cells) > 9 or any(len(c['label']) > 6 for c in cells) else 0,
                                 ha='right' if len(cells) > 9 else 'center', fontsize=FONT - 1)
        axes[-1].set_xlim(-0.6, len(cells) - 0.4)
        axes[-1].set_xlabel(batch['knob'] + ' — cells in value order, evenly spaced', color=INK2)
        title = '{0} — {1}\nfour seeds (dots) and their median (line); the reference {2} hollow orange, its seed range the band'.format(
            batch['batch'], batch['knob'], ref['label'] if ref else '')
        fig.suptitle(title, fontsize=FONT + 0.5, x=0.24, ha='left', color=INK)
    return fig


# ----------------------------------------------------------------------------------- the traces
def traces_figure(sweep, batch, rows=TRACE_ROWS):
    cells = [c for c in batch['cells'] if not c['reference']]
    ref = next((c for c in batch['cells'] if c['reference']), None)
    colours = cell_colours(batch)
    n = len(rows)
    with _rc():
        fig = Figure(figsize=(7.2, 1.25 * n + 0.8))
        _canvas(fig)
        axes = fig.subplots(n, 1, sharex=True)
        fig.subplots_adjust(left=0.20, right=0.70, top=1 - 0.6 / (1.25 * n + 0.8), bottom=0.5 / (1.25 * n + 0.8), hspace=0.25)
        for ax, (key, label, scale) in zip(axes, rows):
            ax.set_facecolor(SURFACE)
            ax.grid(True, axis='y'); ax.grid(False, axis='x')
            label = label.replace('2.5M', _m(batch['stage_b_bin']))
            if ref is not None and ref['arms']:
                y = _median_over(ref['arms'], key, sweep)
                if y is not None:
                    ax.plot(_steps(ref['arms'], key, sweep), y, color=REF_GREY, lw=2.2, alpha=0.8, zorder=1, label='reference ' + ref['label'])
            for c in cells:
                y = _median_over(c['arms'], key, sweep)
                if y is None:
                    continue
                ax.plot(_steps(c['arms'], key, sweep), y, color=colours[c['slug']], lw=1.0, zorder=2, label=c['label'])
            if scale:
                ax.set_yscale(scale)
            ax.set_ylabel(label, fontsize=FONT - 0.5, rotation=0, ha='right', va='center', labelpad=6)
            ax.tick_params(length=2, pad=2, labelsize=FONT - 1)
        axes[-1].set_xlabel('transitions (M)', color=INK2)
        axes[-1].set_xlim(0, batch['horizon'] / 1e6)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.705, 0.5), fontsize=FONT - 1,
                   title=batch['knob'][:26], title_fontsize=FONT - 0.5, handlelength=1.6, labelspacing=0.35)
        ordered = any(c['value'] is not None for c in cells) or len(cells) > len(CATEGORICAL)
        fig.suptitle('{0} — {1}\neach cell as its seed-median per {3} transitions, {2}; the reference grey'.format(
            batch['batch'], batch['knob'], 'light → dark in value order' if ordered else 'one colour per switch', _m(batch['bin'])),
            fontsize=FONT + 0.5, x=0.20, ha='left', color=INK)
    return fig


# ------------------------------------------------------------------------------------- the cell
CELL_ROWS = [('perfect', 'perfect rate /100', None), ('perfect_min', 'worst eval in bin', None), ('score', 'average score', None),
             ('stage_b', 'stage-B ≥98 share /2.5M', None), ('value_loss', 'value loss', 'log'), ('ev', 'explained variance', None),
             ('kl', 'approx KL', 'log'), ('clipfrac', 'clip fraction', None), ('entropy', 'entropy', None)]


def cell_figure(sweep, batch, slug, rows=CELL_ROWS):
    cell = next(c for c in batch['cells'] if c['slug'] == slug)
    ref = next((c for c in batch['cells'] if c['reference']), None)
    arms = cell['arms']
    ncol = max(1, len(arms))
    with _rc():
        fig = Figure(figsize=(2.1 * ncol + 1.5, 1.1 * len(rows) + 0.8))
        _canvas(fig)
        axes = fig.subplots(len(rows), ncol, sharex=True, sharey='row', squeeze=False)
        fig.subplots_adjust(left=1.35 / (2.1 * ncol + 1.5), right=0.99, top=1 - 0.5 / (1.1 * len(rows) + 0.8),
                            bottom=0.5 / (1.1 * len(rows) + 0.8), hspace=0.25, wspace=0.08)
        for r, (key, label, scale) in enumerate(rows):
            label = label.replace('2.5M', _m(batch['stage_b_bin']))
            for c, policy in enumerate(arms):
                ax = axes[r][c]
                ax.set_facecolor(SURFACE); ax.grid(True, axis='y'); ax.grid(False, axis='x')
                if ref is not None:
                    for rp in ref['arms']:
                        y = _median_over([rp], key, sweep)
                        if y is not None:
                            ax.plot(_steps([rp], key, sweep), y, color=REF_GREY, lw=0.7, alpha=0.5, zorder=1)
                y = _median_over([policy], key, sweep)
                if y is not None:
                    ax.plot(_steps([policy], key, sweep), y, color=BLUE, lw=1.0, zorder=2)
                if scale:
                    ax.set_yscale(scale)
                ax.tick_params(length=2, pad=2, labelsize=FONT - 1.5)
                if r == 0:
                    ax.set_title(policy, fontsize=FONT, color=INK)
                if c == 0:
                    ax.set_ylabel(label, fontsize=FONT - 0.5, rotation=0, ha='right', va='center', labelpad=6)
                if r == len(rows) - 1:
                    ax.set_xlabel('transitions (M)', fontsize=FONT - 1, color=INK2)
                    ax.set_xlim(0, batch['horizon'] / 1e6)
        fig.suptitle('{0} {1} = {2}: each seed (blue) over the reference\'s seeds (grey)'.format(
            batch['batch'], batch['knob'], cell['label']), fontsize=FONT + 1, x=0.02, ha='left', color=INK)
    return fig


# ----------------------------------------------------------------------------------- the levers
LEVER_METRICS = ['density98', 'drawdown80', 'hof_best']


def levers_figure(sweep, metrics=LEVER_METRICS):
    rows = sa.peaks(sweep)
    # sort knobs by their density delta, largest first
    def delta(row, key):
        e = row['metrics'].get(key) or {}
        return e.get('delta')
    rows.sort(key=lambda r: -(delta(r, 'density98') or -1e9))
    with _rc():
        fig = Figure(figsize=(7.2, 0.34 * len(rows) + 1.3))
        _canvas(fig)
        axes = fig.subplots(1, len(metrics), sharey=True)
        fig.subplots_adjust(left=0.30, right=0.98, top=1 - 0.75 / (0.34 * len(rows) + 1.3), bottom=0.7 / (0.34 * len(rows) + 1.3), wspace=0.18)
        y = np.arange(len(rows))[::-1]
        for ax, key in zip(axes, metrics):
            m = _metric(sweep, key)
            ax.set_facecolor(SURFACE); ax.grid(True, axis='x'); ax.grid(False, axis='y')
            ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
            for yi, row in zip(y, rows):
                e = row['metrics'].get(key)
                if not e or e.get('ref_median') is None:
                    continue
                batch = next(b for b in sweep['batches'] if b['batch'] == row['batch'])
                cell = next(c for c in batch['cells'] if c['slug'] == e['slug'])
                ref = next(c for c in batch['cells'] if c['reference'])
                rs, cs = ref['stats'][key], cell['stats'][key]
                # the reference's seed range, as deltas from its own median: the noise the win must clear
                ax.plot([rs['min'] - rs['median'], rs['max'] - rs['median']], [yi, yi], color=ORANGE, lw=5, alpha=0.18, solid_capstyle='butt', zorder=1)
                vals = [v - rs['median'] for v in cs['values'] if v is not None]
                ax.plot([min(vals), max(vals)], [yi, yi], color=BLUE, lw=1.4, zorder=2)
                ax.scatter(vals, [yi] * len(vals), s=9, color=BLUE, zorder=3)
                ax.scatter([e['delta']], [yi], s=26, facecolors=SURFACE, edgecolors=BLUE, linewidths=1.2, zorder=4)
                ax.annotate(e['cell'], (max(vals), yi), xytext=(4, 0), textcoords='offset points', va='center', fontsize=FONT - 1.5, color=INK2)
            ax.set_title(m['label'] + (' (' + m['unit'] + ')' if m['unit'] else ''), fontsize=FONT, color=INK, loc='left')
            ax.tick_params(length=2, pad=2, labelsize=FONT - 1)
            lo, hi = ax.get_xlim(); ax.set_xlim(lo, hi + (hi - lo) * 0.25)   # room for the cell label
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(['{0} {1}'.format(r['batch'], r['knob'][:34]) for r in rows], fontsize=FONT - 0.5)
        fig.suptitle("Each knob's best cell, as its delta from the reference's median: four seeds (dots), their median (hollow),\n"
                     "the reference's own seed range (orange band) as the noise a win must clear. Sorted by the density delta.",
                     fontsize=FONT + 0.5, x=0.02, ha='left', color=INK)
    return fig


# ------------------------------------------------------------------------------------ the peaks
PEAK_METRICS = ['density98', 'hof_best', 'hof30k_best', 'best30', 'stage_a_98', 'drawdown80', 'drawdown50', 'onset90_m',
                'ev_late', 'kl_p99']


def peaks_figure(sweep, metrics=PEAK_METRICS):
    rows = sa.peaks(sweep)
    labels = {m['key']: m['label'] for m in sweep['metrics']}
    with _rc():
        fig = Figure(figsize=(7.6, 0.36 * len(rows) + 1.6))
        _canvas(fig)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.22, right=0.99, top=1 - 1.05 / (0.36 * len(rows) + 1.6), bottom=0.05)
        ax.set_facecolor(SURFACE); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ramp = ramp_colours(6)
        for r, row in enumerate(rows):
            for c, key in enumerate(metrics):
                e = row['metrics'].get(key)
                shade = SURFACE
                text = '–'
                ink = INK
                if e:
                    outside = e.get('outside')
                    level = 0 if outside is None or outside <= 0 else min(5, 1 + int(outside * 2))   # 0..0.5 -> 1, .. >=2 -> 5
                    shade = GRID if level == 0 else ramp[level]
                    ink = INK if level <= 2 else '#ffffff'
                    text = e['cell']
                    if e['p'] is not None and e['p'] <= 0.03:
                        text += ' *'
                ax.add_patch(Rectangle(
                    (c + 0.03, len(rows) - 1 - r + 0.05), 0.94, 0.9, facecolor=shade, edgecolor='none'))
                size = FONT - 1.5 if len(text) <= 8 else FONT - 3 if len(text) <= 14 else FONT - 3.5
                if len(text) > 19:
                    text = text[:18] + '…'
                ax.text(c + 0.5, len(rows) - 1 - r + 0.5, text, ha='center', va='center', fontsize=size, color=ink)
        ax.set_xlim(0, len(metrics)); ax.set_ylim(0, len(rows))
        ax.set_xticks([c + 0.5 for c in range(len(metrics))])
        ax.set_xticklabels([labels[k] for k in metrics], rotation=30, ha='left', fontsize=FONT - 1)
        ax.xaxis.tick_top()
        ax.set_yticks([len(rows) - 1 - r + 0.5 for r in range(len(rows))])
        ax.set_yticklabels(['{0} {1}'.format(r['batch'], r['knob'][:30]) for r in rows], fontsize=FONT - 0.5)
        ax.tick_params(length=0)
        fig.suptitle("Where each metric peaked, per knob. Grey: the winner's median is inside the reference's seed range; darker blue: further outside it, in units of that range. * = every seed past every reference seed (p = 0.029).",
                     fontsize=FONT, x=0.02, ha='left', color=INK, wrap=True)
    return fig


# ---------------------------------------------------------------------------------- the scatters
def _spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float('nan')


def scatter_figure(sweep, xkey, ykey, batches=None):
    xs, ys, refs = [], [], []
    ref_arms = {p for b in sweep['batches'] for p in b['reference_arms']}
    for policy, arm in sweep['arms'].items():
        if batches and arm['batch'] not in batches:
            continue
        x, y = arm['scalars'].get(xkey), arm['scalars'].get(ykey)
        if x is None or y is None:
            continue
        (refs if policy in ref_arms else xs).append((x, y))
    mx, my = _metric(sweep, xkey), _metric(sweep, ykey)
    allx = [p[0] for p in xs] + [p[0] for p in refs]
    ally = [p[1] for p in xs] + [p[1] for p in refs]
    rho = _spearman(allx, ally) if len(allx) > 2 else float('nan')
    with _rc():
        fig = Figure(figsize=(4.6, 3.4))
        _canvas(fig)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.14, right=0.97, top=0.86, bottom=0.14)
        ax.set_facecolor(SURFACE)
        ax.scatter([p[0] for p in xs], [p[1] for p in xs], s=9, color=BLUE, alpha=0.55, linewidths=0, zorder=2, label='sweep arm')
        if refs:
            ax.scatter([p[0] for p in refs], [p[1] for p in refs], s=22, facecolors=SURFACE, edgecolors=ORANGE, linewidths=1.2, zorder=3, label='reference arm')
        if xkey.startswith('kl'):
            ax.set_xscale('log')
        ax.set_xlabel(mx['label'], color=INK2); ax.set_ylabel(my['label'] + (' (%)' if my['unit'] == '%' else ''), color=INK2)
        ax.tick_params(length=2, pad=2, labelsize=FONT - 1)
        ax.legend(loc='best', fontsize=FONT - 1)
        fig.suptitle('{0} against {1} — {2} arms, Spearman ρ = {3:.2f}'.format(my['label'], mx['label'], len(allx), rho),
                     fontsize=FONT + 0.5, x=0.02, ha='left', color=INK)
    return fig, rho


# --------------------------------------------------------------------------------------- driver
class _rc:
    def __enter__(self):
        import matplotlib
        self._ctx = matplotlib.rc_context(RC)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def main(argv=None):
    parser = argparse.ArgumentParser(description='the sweep figures')
    parser.add_argument('batches', nargs='*', help='batches to draw (default all)')
    parser.add_argument('--cell', nargs=2, metavar=('BATCH', 'SLUG'), help='draw one cell figure and stop')
    parser.add_argument('--out', default=sa.FIGURES_DIR)
    parser.add_argument('--json', default=None)
    args = parser.parse_args(argv)
    sweep = sa.load(args.json)
    written = []
    if args.cell:
        batch = next(b for b in sweep['batches'] if b['batch'] == args.cell[0])
        written.append(_save(cell_figure(sweep, batch, args.cell[1]), os.path.join(args.out, '{0}-{1}-cell.png'.format(*args.cell))))
    else:
        wanted = set(args.batches) if args.batches else None
        for batch in sweep['batches']:
            if wanted and batch['batch'] not in wanted:
                continue
            written.append(_save(curve_figure(sweep, batch), os.path.join(args.out, batch['batch'] + '-curve.png')))
            written.append(_save(traces_figure(sweep, batch), os.path.join(args.out, batch['batch'] + '-traces.png')))
        if not wanted:
            written.append(_save(levers_figure(sweep), os.path.join(args.out, 'levers.png')))
            written.append(_save(peaks_figure(sweep), os.path.join(args.out, 'peaks.png')))
            for xkey, ykey, name in (('ev_late', 'density98', 'scatter-ev-density'), ('kl_p99', 'drawdown80', 'scatter-kl-drawdown')):
                fig, rho = scatter_figure(sweep, xkey, ykey)
                written.append(_save(fig, os.path.join(args.out, name + '.png')))
                print('{0}: Spearman rho = {1:.3f}'.format(name, rho))
    for path in written:
        print(os.path.relpath(path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
