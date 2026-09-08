"""The hyperparameter sweep (b9-b21) reduced to one file, and the figures drawn from it.

    PYTHONPATH=. python -m tools.sweep_analysis reduce             # runs/ -> viewer/sweep.json + viewer/sweep.js
    PYTHONPATH=. python -m tools.sweep_analysis figures [b9 ...]   # sweep.json -> charts/sweep/*.png (tools/sweep_figures.py)
    PYTHONPATH=. python -m tools.sweep_analysis peaks              # the peaks table as Markdown, on stdout

One reducer, two consumers: `docs/sweep.md`'s figures and the local page `viewer/sweep.html` both read
`sweep.json` / `sweep.js`, so the page and the report cannot disagree. The design is
`plans/sweep-analysis.md`.

What one arm reduces to:

| block | contents |
|---|---|
| `scalars` | the docs tables' numbers (`viewer_manifest`'s definitions, reused, never redefined) plus the ones only this file computes: onset by trailing-30 >= 90, crossings, late window diagnostics from the `ppo` block, worst post-onset eval, end drop, late sd |
| `trace` | every stage-A metric binned to `BIN` transitions -- the mean per bin, and for the perfect rate **the min too**: a collapse is one eval at 0% among fourteen at 95%, and a bin mean alone hides exactly what the drawdown rows exist to show |
| `stage_b` | rows >= 98 and rows in all, per `STAGE_B_BIN` transitions -- density over time |
| `hof5000`, `hof30k` | `[step, perfect_percent]` per row |

The knob value on the x axis comes from `plans/hyperparam-sweep.json`, not from the arm's name: a
cell's `env` against the batch's `control_value`. Cells are **evenly spaced in value order** (numeric
cells sorted, then the categorical ones in manifest order): the grids are dense where they matter
and a linear axis squashes lambda's 0.9-1.0 plateau into a tenth of the width. A batch's reference
cell (`viewer/references.json`) is slotted in at its own value.

**Late** means the last `LATE_FRACTION` of an arm's evals (~10M of 50M), so the diagnostics describe
the endgame the record region lives in rather than the climb.
"""

import datetime
import itertools
import json
import math
import os
import re
import statistics
import sys

from env import constants
from tools import viewer_manifest as vm

MANIFEST = os.path.join(constants.ROOT, 'plans', 'hyperparam-sweep.json')
# Batches added to the sweep view after the plan closed, in the manifest's `batches` shape (`batch`, `knob`,
# `control_value`, `cells` of `slug`/`env`/`prediction`), one entry per batch. b26 was the first (user,
# 2026-09-07: "I know it wasn't part of the sweep, but I'd like to see the metrics"). Kept apart from the
# plan so `plans/hyperparam-sweep.json` stays what it is -- the sweep as it was designed.
EXTRA_MANIFEST = os.path.join(constants.ROOT, 'plans', 'sweep-extra.json')
OUT_JSON = os.path.join(constants.ROOT, 'viewer', 'sweep.json')
OUT_JS = os.path.join(constants.ROOT, 'viewer', 'sweep.js')
FIGURES_DIR = os.path.join(constants.ROOT, 'charts', 'sweep')

BIN = 250_000                 # transitions per trace bin: 200 bins over the sweep's 50M
STAGE_B_BIN = 2_500_000       # transitions per stage-B density bin: 20 over 50M
HORIZON = 50_003_968
LATE_FRACTION = 0.2
TRAILING = 30
ONSET_TRAILING = 90.0         # the plan's onset: first step with trailing-30 >= 90
ONSET_EVAL = 80.0             # the manifest's onset: first eval >= 80 (what results.md's drawdowns use)

# stage-A metric -> where it sits in an eval row. The perfect rate also gets a per-bin min.
TRACE_METRICS = {
    'perfect': ('perfect_percent',),
    'score': ('avg_score',),
    'value_loss': ('ppo', 'value_loss'),
    'ev': ('ppo', 'explained_variance'),
    'kl': ('ppo', 'approx_kl'),
    'clipfrac': ('ppo', 'clip_fraction'),
    'entropy': ('ppo', 'entropy'),
}

# The scalar metrics a cell is compared to its reference on, with the direction that is "better".
# Everything in the peaks/levers tables is one of these, in this order.
METRICS = [
    # key, label, higher_is_better, unit
    ('density98', '≥98%/500 density', True, '%'),
    ('density99', '≥99%/500 density', True, '%'),
    ('hof_best', 'hof5000 best', True, ''),
    ('hof_mean', 'hof5000 mean', True, ''),
    ('hof30k_best', 'hof30k best', True, ''),
    ('best30', 'best30', True, ''),
    ('stage_a_98', 'stage-A ≥98, post-onset', True, '%'),
    ('drawdown80', 'evals < 80, post-onset', False, '%'),
    ('drawdown50', 'evals < 50, post-onset', False, '%'),
    ('worst_post', 'worst post-onset eval', True, ''),
    ('end_drop', 'best30 − trailing at cap', False, ''),
    ('late_sd', 'sd of last 300 evals', False, ''),
    ('onset90_m', 'onset (trailing-30 ≥ 90), M', False, 'M'),
    ('step98_m', 'first trailing-30 ≥ 98, M', False, 'M'),
    ('ev_late', 'explained variance, late', True, ''),
    ('value_loss_late', 'value loss, late', False, ''),
    ('kl_late', 'approx KL, late', False, ''),
    ('kl_p99', 'approx KL p99, post-onset', False, ''),
    ('clipfrac_late', 'clip fraction, late', None, ''),
    ('entropy_end', 'entropy at cap', None, ''),
]
METRIC_KEYS = [m[0] for m in METRICS]

# Batches whose value loss is not comparable across cells (a different function, or a different
# value scale), so `value_loss_late` is drawn but never ranked there. `ev_late` is the scale-free version.
VALUE_LOSS_INCOMPARABLE = {'b10', 'b19'}


# --------------------------------------------------------------------------------------- helpers
def _get(row, path):
    for key in path:
        row = row.get(key) if isinstance(row, dict) else None
        if row is None:
            return None
    return row


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _r(v, digits=3):
    return None if v is None else round(v, digits)


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _percentile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def trailing_mean(values, window=TRAILING):
    """Element i is the mean of the last `window` values up to and including i."""
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def first_crossing(steps, trailing, threshold):
    """The step of the first trailing value at or above `threshold`, or None."""
    for step, value in zip(steps, trailing):
        if value >= threshold:
            return step
    return None


def mann_whitney_exact(a, b):
    """Two-sided exact Mann-Whitney p for two small samples (n=4 vs 4 here: 70 arrangements, floor
    0.029). Ties count half. None when either side is empty."""
    a = [v for v in a if v is not None]
    b = [v for v in b if v is not None]
    if not a or not b:
        return None
    pooled = a + b
    n = len(a)

    def u_of(chosen):
        u = 0.0
        rest = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        for x in (pooled[i] for i in chosen):
            for y in rest:
                u += 1.0 if x > y else 0.5 if x == y else 0.0
        return u

    observed = u_of(set(range(n)))
    centre = n * len(b) / 2.0
    dev = abs(observed - centre)
    count = total = 0
    for chosen in itertools.combinations(range(len(pooled)), n):
        total += 1
        if abs(u_of(set(chosen)) - centre) >= dev - 1e-9:
            count += 1
    return round(count / total, 4)


# ------------------------------------------------------------------------------------ one arm
def bin_trace(evals, bin_size=BIN, horizon=HORIZON):
    """Each stage-A metric averaged per `bin_size` transitions, plus `perfect_min`; `steps` is the bin's
    right edge in transitions. Empty bins are None (a rollout-1024 arm has 382 evals for 200 bins)."""
    n_bins = max(1, math.ceil(horizon / bin_size))
    sums = {k: [0.0] * n_bins for k in TRACE_METRICS}
    counts = {k: [0] * n_bins for k in TRACE_METRICS}
    mins = [None] * n_bins
    for e in evals:
        t = e.get('transitions', e.get('step'))
        if t is None:
            continue
        b = min(n_bins - 1, max(0, (t - 1) // bin_size))
        for k, path in TRACE_METRICS.items():
            v = _get(e, path)
            if v is None:
                continue
            sums[k][b] += v
            counts[k][b] += 1
        p = e.get('perfect_percent')
        if p is not None:
            mins[b] = p if mins[b] is None else min(mins[b], p)
    out = {'steps': [(i + 1) * bin_size for i in range(n_bins)], 'perfect_min': mins}
    for k in TRACE_METRICS:
        digits = 4 if k in ('kl', 'clipfrac', 'ev', 'entropy') else 2
        out[k] = [round(sums[k][i] / counts[k][i], digits) if counts[k][i] else None for i in range(n_bins)]
    return out


def stage_b_density(rows, bin_size=STAGE_B_BIN, horizon=HORIZON):
    """`{'steps', 'rows', 'rows98'}` per `bin_size` transitions of stage-B rows -- the record region over time."""
    n_bins = max(1, math.ceil(horizon / bin_size))
    total, hi = [0] * n_bins, [0] * n_bins
    for r in rows:
        step = r.get('step')
        if step is None:
            continue
        b = min(n_bins - 1, max(0, (step - 1) // bin_size))
        total[b] += 1
        if r.get('perfect_percent', 0) >= 98:
            hi[b] += 1
    return {'steps': [(i + 1) * bin_size for i in range(n_bins)], 'rows': total, 'rows98': hi}


def scalars(stage_a, stage_b, hof, h30):
    """One arm's numbers. The docs-table ones come from `viewer_manifest`'s functions."""
    evals = (stage_a or {}).get('evals') or []
    summary = (stage_a or {}).get('summary') or {}
    best30 = summary.get('best_perfect30') or {}
    perfect = [e.get('perfect_percent', 0) for e in evals]
    steps = [e.get('step') for e in evals]
    trailing = trailing_mean(perfect) if perfect else []
    onset_i = next((i for i, p in enumerate(perfect) if p >= ONSET_EVAL), None)
    post = evals[onset_i:] if onset_i is not None else []
    post_perfect = perfect[onset_i:] if onset_i is not None else []
    late_n = max(1, int(round(len(evals) * LATE_FRACTION)))
    late = evals[-late_n:] if evals else []

    def late_mean(path, digits=4):
        vals = [_get(e, path) for e in late]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), digits) if vals else None

    onset90 = first_crossing(steps, trailing, ONSET_TRAILING)
    step95 = first_crossing(steps, trailing, 95.0)
    step98 = first_crossing(steps, trailing, 98.0)
    trailing_now = summary.get('trailing_now')
    out = {
        'step': summary.get('step'), 'evals': len(evals),
        'best30': best30.get('value'), 'best30_step': best30.get('step'),
        'trailing_now': trailing_now, 'sef': summary.get('strong_eval_fraction'),
        'end_drop': _r(best30.get('value') - trailing_now, 2) if best30.get('value') is not None and trailing_now is not None else None,
        'late_sd': _r(statistics.pstdev(perfect[-300:]), 2) if len(perfect) >= 2 else None,
        'drawdown50': vm.drawdown(evals, 50), 'drawdown80': vm.drawdown(evals, 80),
        'worst_post': min(post_perfect) if post_perfect else None,
        'stage_a_98': round(100.0 * sum(p >= 98 for p in post_perfect) / len(post_perfect), 1) if post_perfect else None,
        'onset80': vm.onset(evals), 'onset90': onset90, 'step95': step95, 'step98': step98,
        'onset90_m': _r(onset90 / 1e6, 2) if onset90 is not None else None,
        'step98_m': _r(step98 / 1e6, 2) if step98 is not None else None,
        'value_loss_late': late_mean(('ppo', 'value_loss')),
        'ev_late': late_mean(('ppo', 'explained_variance')),
        'kl_late': late_mean(('ppo', 'approx_kl'), 5),
        'kl_p99': _r(_percentile([_get(e, ('ppo', 'approx_kl')) for e in post], 0.99), 5),
        'clipfrac_late': late_mean(('ppo', 'clip_fraction')),
        'entropy_end': _get(evals[-1], ('ppo', 'entropy')) if evals else None,
        'epochs_run_mean': late_mean(('ppo', 'epochs_run'), 2),
    }
    rows = (stage_b or {}).get('rows') or []
    scores = [r.get('perfect_percent', 0) for r in rows]
    out.update({
        'rows': len(rows) if stage_b is not None else None,
        'density98': round(100.0 * sum(s >= 98 for s in scores) / len(scores), 1) if scores else None,
        'density99': round(100.0 * sum(s >= 99 for s in scores) / len(scores), 2) if scores else None,
        'cands99': sum(s >= 99 for s in scores) if scores else None,
        'best_row': max(scores) if scores else None,
    })
    hof_scores = [r.get('perfect_percent', 0) for r in ((hof or {}).get('rows') or [])]
    out.update({
        'hof_rows': len(hof_scores) if hof is not None else None,
        'hof_mean': round(sum(hof_scores) / len(hof_scores), 2) if hof_scores else None,
        'hof_best': max(hof_scores) if hof_scores else None,
        'hof_9873': sum(s >= 98.73 for s in hof_scores) if hof_scores else None,
    })
    h30_rows = (h30 or {}).get('rows') or []
    best = max(h30_rows, key=lambda r: r.get('perfect_percent', 0)) if h30_rows else None
    out.update({
        'hof30k_rows': len(h30_rows) if h30 is not None else None,
        'hof30k_mean': round(sum(r.get('perfect_percent', 0) for r in h30_rows) / len(h30_rows), 2) if h30_rows else None,
        'hof30k_best': best.get('perfect_percent') if best else None,
        'hof30k_best_step': best.get('step') if best else None,
    })
    return out


def arm_record(policy, runs_dir):
    """Everything the page and the figures need about one arm, or None without a stage-A file."""
    stage_a = vm._read(os.path.join(runs_dir, policy + '_evals.json'))
    if not stage_a or 'summary' not in stage_a:
        return None
    stage_b = vm._read(os.path.join(runs_dir, policy + '_checkpoint_evals.json'))
    hof = vm._read(os.path.join(runs_dir, policy + '_checkpoint_evals_hof5000.json'))
    h30 = vm._read(os.path.join(runs_dir, policy + '_checkpoint_evals_hof30k.json'))
    evals = stage_a.get('evals') or []
    return {
        'policy': policy, 'batch': vm.batch_of(policy), 'cell': vm.knob_of(policy), 'seed': vm.seed_of(policy),
        'scalars': scalars(stage_a, stage_b, hof, h30),
        'trace': bin_trace(evals),
        'stage_b': stage_b_density((stage_b or {}).get('rows') or []),
        'hof5000': [[r.get('step'), r.get('perfect_percent')] for r in ((hof or {}).get('rows') or [])],
        'hof30k': [[r.get('step'), r.get('perfect_percent')] for r in ((h30 or {}).get('rows') or [])],
    }


# --------------------------------------------------------------------------------- the batches
def cell_value(cell_env, control_value):
    """`(numeric, label)` for a cell. Numeric only when the batch has one control key and the cell
    sets exactly that key to a number -- b21's gates and doses share a batch and must not sort together,
    and an anneal cell has two keys. The label is what the axis prints."""
    if len(control_value) == 1 and len(cell_env) == 1:
        (key, raw), = cell_env.items()
        if key in control_value:
            value = _num(raw)
            if value is not None:
                return value, raw
    return None, None


def label_of(slug, env, control_value):
    """A short axis label: the numeric value for a numeric cell, else the slug."""
    value, raw = cell_value(env, control_value)
    return raw if raw is not None else slug


def layout_cells(batch, reference):
    """The batch's cells in axis order with the reference slotted in, as a list of
    `{slug, label, x, env, prediction, reference}`. Numeric cells sorted by value, then the categorical
    ones in manifest order; the reference goes at its value among the numeric cells, else after
    `references.json`'s `after`, else first."""
    control = batch.get('control_value') or {}
    numeric, categorical = [], []
    for c in batch['cells']:
        value, raw = cell_value(c.get('env') or {}, control)
        entry = {'slug': c['slug'], 'label': raw if raw is not None else c['slug'], 'value': value,
                 'env': c.get('env') or {}, 'prediction': c.get('prediction', ''), 'reference': False}
        (numeric if value is not None else categorical).append(entry)
    numeric.sort(key=lambda e: e['value'])
    ref = None
    if reference and reference.get('arms'):
        ref_value = _num(reference.get('value')) if len(control) == 1 else None
        ref = {'slug': 'reference', 'label': str(reference.get('value', 'ref')), 'value': ref_value,
               'env': dict(control), 'prediction': reference.get('label', ''), 'reference': True,
               'arms': list(reference['arms'])}
    cells = numeric + categorical
    if ref is not None:
        if ref['value'] is not None and numeric:
            at = next((i for i, e in enumerate(numeric) if e['value'] > ref['value']), len(numeric))
        elif reference.get('after') and any(e['slug'] == reference['after'] for e in cells):
            at = next(i for i, e in enumerate(cells) if e['slug'] == reference['after']) + 1
        else:
            at = 0
        cells.insert(at, ref)
    for i, e in enumerate(cells):
        e['x'] = i
    return cells


def cell_stats(arm_scalars, ref_scalars):
    """Per metric: the cell's four values, their median, and the exact Mann-Whitney p against the
    reference's four. `arm_scalars`/`ref_scalars` are lists of scalar dicts."""
    out = {}
    for key in METRIC_KEYS:
        vals = [s.get(key) for s in arm_scalars]
        refs = [s.get(key) for s in ref_scalars]
        present = [v for v in vals if v is not None]
        out[key] = {
            'values': vals, 'median': _r(_median(present), 3),
            'min': min(present) if present else None, 'max': max(present) if present else None,
            'n': len(present),
            'p': mann_whitney_exact(present, [v for v in refs if v is not None]) if refs and present else None,
        }
    return out


def manifest_batches(manifest_path=None, extra_path=None):
    """The plan's batches followed by the extra manifest's, in file order; the extra file is optional."""
    with open(manifest_path or MANIFEST) as handle:
        batches = list(json.load(handle)['batches'])
    extra = EXTRA_MANIFEST if extra_path is None else extra_path
    if extra and os.path.exists(extra):
        with open(extra) as handle:
            batches += list(json.load(handle)['batches'])
    return batches


def build(runs_dir=None, manifest_path=None, references_path=None, extra_path=None):
    runs_dir = runs_dir or constants.RUNS_DIR
    refs = vm.references(references_path)
    arms = {}
    batches = []
    for batch in manifest_batches(manifest_path, extra_path):
        name = batch['batch']
        reference = refs.get(name) or {}
        cells = layout_cells(batch, reference)
        policies = sorted(os.path.basename(p)[:-len('_evals.json')]
                          for p in _glob(runs_dir, name + '*_evals.json') if '_checkpoint' not in p)
        by_cell = {}
        for policy in policies:
            if vm.batch_of(policy) != name:
                continue
            rec = arm_record(policy, runs_dir)
            if rec is None:
                continue
            arms[policy] = rec
            by_cell.setdefault(rec['cell'], []).append(policy)
        ref_arms = []
        for policy in reference.get('arms', []):
            rec = arms.get(policy) or arm_record(policy, runs_dir)
            if rec is not None:
                arms[policy] = rec
                ref_arms.append(policy)
        ref_scalars = [arms[p]['scalars'] for p in ref_arms]
        for cell in cells:
            cell['arms'] = ref_arms if cell['reference'] else sorted(by_cell.get(cell['slug'], []),
                                                                       key=lambda p: (arms[p]['seed'] or 0, p))
            cell['stats'] = cell_stats([arms[p]['scalars'] for p in cell['arms']], ref_scalars)
        batches.append({
            'batch': name, 'knob': batch.get('knob', ''), 'control_value': batch.get('control_value') or {},
            'expected_optimum': batch.get('expected_optimum', ''),
            'reference_label': reference.get('label', ''), 'reference_arms': ref_arms,
            'value_loss_comparable': name not in VALUE_LOSS_INCOMPARABLE,
            'cells': cells,
        })
    return {
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'bin': BIN, 'stage_b_bin': STAGE_B_BIN, 'horizon': HORIZON, 'late_fraction': LATE_FRACTION,
        'metrics': [{'key': k, 'label': l, 'higher': h, 'unit': u} for k, l, h, u in METRICS],
        'batches': batches, 'arms': arms,
    }


def _glob(runs_dir, pattern):
    import glob
    return glob.glob(os.path.join(runs_dir, pattern))


# ------------------------------------------------------------------------------------ the peaks
def peaks(sweep):
    """Per batch, per metric: the winning cell (best median in the metric's direction), its median, the
    reference's, whether the winner's four seeds all sit past the reference's four (p = 0.029), and how
    far the winner's median sits outside the reference's seed range, in units of that range."""
    out = []
    for batch in sweep['batches']:
        ref = next((c for c in batch['cells'] if c['reference']), None)
        row = {'batch': batch['batch'], 'knob': batch['knob'], 'metrics': {}}
        for m in sweep['metrics']:
            key, higher = m['key'], m['higher']
            if higher is None or (key == 'value_loss_late' and not batch['value_loss_comparable']):
                continue
            cands = [c for c in batch['cells'] if not c['reference'] and c['stats'][key]['median'] is not None]
            if not cands:
                continue
            win = (max if higher else min)(cands, key=lambda c: c['stats'][key]['median'])
            st = win['stats'][key]
            entry = {'cell': win['label'], 'slug': win['slug'], 'median': st['median'], 'p': st['p'],
                     'ref_median': None, 'outside': None}
            if ref is not None and ref['stats'][key]['median'] is not None:
                rs = ref['stats'][key]
                entry['ref_median'] = rs['median']
                span = (rs['max'] - rs['min']) if rs['max'] is not None and rs['min'] is not None else None
                if span is not None:
                    d = (st['median'] - rs['max']) if higher else (rs['min'] - st['median'])
                    entry['outside'] = round(d / span, 2) if span > 0 else (1.0 if d > 0 else 0.0)
                    entry['delta'] = round(st['median'] - rs['median'], 3)
            row['metrics'][key] = entry
        out.append(row)
    return out


def _sig(v):
    """A number at the precision its size deserves: 30.2, 98.45, 0.0035."""
    if v is None:
        return '–'
    a = abs(v)
    return '{0:.1f}'.format(v) if a >= 10 else '{0:.2f}'.format(v) if a >= 1 else '{0:.3g}'.format(v)


def peaks_markdown(sweep, keys=('density98', 'hof_best', 'hof30k_best', 'best30', 'stage_a_98', 'drawdown80',
                                'drawdown50', 'onset90_m', 'ev_late', 'kl_p99')):
    labels = {m['key']: m['label'] for m in sweep['metrics']}
    lines = ['| knob | ' + ' | '.join(labels[k] for k in keys) + ' |',
             '|---|' + '---|' * len(keys)]
    for row in peaks(sweep):
        cells = []
        for k in keys:
            e = row['metrics'].get(k)
            if not e:
                cells.append('–')
                continue
            text = '`{0}` {1}'.format(e['cell'], _sig(e['median']))
            if e.get('delta') is not None:
                text += ' ({0}{1})'.format('+' if e['delta'] >= 0 else '−', _sig(abs(e['delta'])))
            if e['p'] is not None and e['p'] <= 0.03:
                text = '**' + text + '** p=' + str(e['p'])
            cells.append(text)
        lines.append('| {0} {1} | '.format(row['batch'], row['knob']) + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------- main
def write(sweep, json_path=None, js_path=None):
    json_path = json_path or OUT_JSON
    js_path = js_path or OUT_JS
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as handle:
        json.dump(sweep, handle, separators=(',', ':'))
    with open(js_path, 'w') as handle:
        handle.write('window.SNEK_SWEEP = ')
        json.dump(sweep, handle, separators=(',', ':'))
        handle.write(';\n')
    return json_path, js_path


def load(json_path=None):
    with open(json_path or OUT_JSON) as handle:
        return json.load(handle)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv else 'reduce'
    if command == 'reduce':
        sweep = build()
        json_path, js_path = write(sweep)
        n_cells = sum(len(b['cells']) for b in sweep['batches'])
        print('{0}: {1} arms, {2} batches, {3} cells -> {4} ({5:.1f} MB), {6}'.format(
            sweep['generated'], len(sweep['arms']), len(sweep['batches']), n_cells,
            os.path.relpath(json_path), os.path.getsize(json_path) / 1e6, os.path.relpath(js_path)))
        return 0
    if command == 'peaks':
        print(peaks_markdown(load()))
        return 0
    if command == 'figures':
        from tools import sweep_figures
        return sweep_figures.main(argv[1:])
    print('usage: sweep_analysis reduce | figures [batch ...] | peaks', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
