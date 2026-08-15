"""Reads `plasticity.py`'s payloads and answers the six questions they were measured for.

    1. does plasticity loss come *before* an arm's peak, or after it
    2. is it worse for the wide configs, and did they lose plasticity before reaching endgame
    3. does it happen around 1M steps, or are the nets still plastic at 3M
    4. does it coincide with a drawdown the arm never recovers from
    5. do *later* drawdowns cost more plasticity than earlier ones of similar depth
    6. does plasticity fall with step even across stretches where the perfect rate is flat

Questions 5 and 6 are the ones that need care, because step position and drawdown are confounded:
every late drawdown is also a drawdown at a higher weight norm, and an arm that is declining is also
an arm that is old. So the two are separated explicitly — drawdown events are matched on **depth**
before early and late are compared, and the flat-stretch table asks what happens to the metrics
where the perfect rate is *not* moving at all.

**A drawdown event runs peak -> trough -> recovery**, with events separated by new all-time highs in
the arm's own trailing-30 curve, so they never overlap and each one has a single depth. `recovered` is
whether the curve came back to the old peak before the arm stopped — question 4's whole subject, and
the reason the events are cut this way rather than by fixed windows.

The curve comes from `runs/<arm>_evals.json` at full eval resolution, not from the ladder's sparse
`trailing` column: a 10k-step ladder would place a trough up to 10k steps off and can miss a short one
entirely. Metrics are then read at the ladder row nearest each event step, and an event whose peak or
trough has no row within `NEAR` is reported as unmatched rather than silently attached to a row
somewhere else.

Usage, from `snek2/`:

    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/plasticity_analysis.py <payload_dir> [out.png]

Reads every `*.json` in `<payload_dir>` that looks like a plasticity payload. Writes a summary next to
it as `plasticity_summary.json`, and the four-panel figure if a path is given.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plasticity import trailing_curve

# A fall of this many points below a running peak opens a drawdown event. 15 pp is well outside
# eval-to-eval noise on a trailing-30 curve, which moves a few points between evals.
MIN_DEPTH = 15.0
# How far a ladder row may be from an event's step to be read as that step's metrics. Adaptive,
# because the ladders are not all the same density: the desktop-trained arms are staged at 50k, so a
# fixed 15k dropped every one of their events as unmatched — the metrics were there, the tolerance was
# not. Half a stride plus the snap window is the widest gap a rung can legitimately sit at.
NEAR = 15000
# A stretch is "flat" when its trailing-30 range stays inside this many points ...
FLAT_TOL = float(os.environ.get('FLAT_TOL', 6.0))
# ... over at least this many steps. Long enough that a metric trend is not one eval's wobble.
FLAT_SPAN = int(os.environ.get('FLAT_SPAN', 400000))
# **Both are worth varying, and the defaults are strict.** At 6 pp / 400k only one arm has any flat
# stretch at all, because a trailing-30 curve wobbles several points between evals even when the
# policy is not going anywhere. Question 6's answer should be read across settings — 10 pp / 300k
# gives many more stretches and says the same thing — which is why these are env-overridable rather
# than constants to be edited.
# Metrics reported everywhere, and the direction that means *worse* plasticity.
METRICS = [('dormant_all', 'dormant', +1), ('dead_all', 'dead', +1),
           ('constant_all', 'const', +1), ('srank_c', 'srank_c', -1),
           ('stable_rank_c', 'stable_c', -1), ('growth_hidden', 'growth', +1),
           ('growth_head', 'g_head', +1), ('move', 'move/100k', -1)]


def load_payloads(directory):
    """Every plasticity payload in `directory`, keyed by arm, newest wins on a repeat."""
    out = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith('.json') or name.startswith('plasticity_summary'):
            continue
        try:
            with open(os.path.join(directory, name)) as handle:
                payload = json.load(handle)
        except ValueError:
            continue
        if isinstance(payload, dict) and 'rows' in payload and 'fc_layer_params' in payload:
            # A denser stride is a strict improvement on the same arm, so prefer it.
            previous = out.get(payload['policy'])
            if previous is None or len(payload['rows']) >= len(previous['rows']):
                out[payload['policy']] = payload
    return out


def split_rows(payload):
    """(the fresh control row, the trained rows sorted by step)."""
    fresh = [r for r in payload['rows'] if r.get('fresh')]
    trained = sorted((r for r in payload['rows'] if not r.get('fresh')), key=lambda r: r['step'])
    return (fresh[0] if fresh else None), trained


def drawdown_events(curve, min_depth=MIN_DEPTH):
    """Peak -> trough -> recovery events in a trailing curve, separated by new all-time highs.

    `curve` is [(step, trailing)] in step order. An event opens when the curve falls `min_depth`
    below its running peak and closes when it makes a new high; the last one stays open, which is
    what an arm that never recovered looks like. Events cannot overlap, so each has one depth and one
    step position — the two things questions 5 and 6 compare.
    """
    events = []
    if not curve:
        return events
    peak_step, peak_value = curve[0]
    trough_step, trough_value = curve[0]
    opened = False
    for step, value in curve:
        if value >= peak_value:
            if opened and peak_value - trough_value >= min_depth:
                events.append({'peak_step': peak_step, 'peak': peak_value,
                               'trough_step': trough_step, 'trough': trough_value,
                               'depth': peak_value - trough_value,
                               'recovered': True, 'recovery_step': step})
            opened = False
            peak_step, peak_value = step, value
            trough_step, trough_value = step, value
            continue
        if value < trough_value:
            trough_step, trough_value = step, value
        if peak_value - value >= min_depth:
            opened = True
    if opened and peak_value - trough_value >= min_depth:
        events.append({'peak_step': peak_step, 'peak': peak_value,
                       'trough_step': trough_step, 'trough': trough_value,
                       'depth': peak_value - trough_value,
                       'recovered': False, 'recovery_step': None})
    return events


def near_for(payload):
    """This payload's match tolerance — half its stride, so a sparse ladder is not silently dropped."""
    return max(NEAR, payload.get('stride', 0) // 2 + 5000)


def row_at(trained, step, near=NEAR):
    """The ladder row nearest `step`, or None if the nearest is further than `near`."""
    if not trained:
        return None
    best = min(trained, key=lambda r: abs(r['step'] - step))
    return best if abs(best['step'] - step) <= near else None


def rows_between(trained, start, end):
    return [r for r in trained if start <= r['step'] <= end]


def flat_stretches(trained, curve, tol=FLAT_TOL, span=FLAT_SPAN):
    """Maximal stretches where the perfect rate is not moving, with each metric's drift across them.

    This is question 6 with the drawdown removed: if plasticity falls here too, step count is doing
    the damage on its own rather than the collapse being what damages it. Greedy and maximal from
    each start, then overlapping stretches are dropped so a long flat region counts once.
    """
    lookup = dict(curve)
    steps = [r['step'] for r in trained if lookup.get(r['step']) is not None]
    out, taken_to = [], -1
    for index, start in enumerate(steps):
        if start <= taken_to:
            continue
        end = start
        values = [lookup[start]]
        for later in steps[index + 1:]:
            candidate = values + [lookup[later]]
            if max(candidate) - min(candidate) > tol:
                break
            values, end = candidate, later
        if end - start < span:
            continue
        inside = [r for r in trained if start <= r['step'] <= end]
        stretch = {'start': start, 'end': end, 'span': end - start,
                   'mean_trailing': float(np.mean(values)),
                   'range': float(max(values) - min(values)), 'rows': len(inside)}
        for key, label, _ in METRICS:
            series = [(r['step'], r[key]) for r in inside if r.get(key) is not None]
            stretch[label] = per_million(series)
        out.append(stretch)
        taken_to = end
    return out


def per_million(series):
    """Least-squares slope of a (step, value) series, per 1M steps. None if too short."""
    if len(series) < 3:
        return None
    steps = np.array([s for s, _ in series], dtype=float)
    values = np.array([v for _, v in series], dtype=float)
    if steps.max() == steps.min():
        return None
    return float(np.polyfit(steps, values, 1)[0] * 1e6)


def arm_table(payload, curve):
    """One arm's control, peak and end readings, plus where its peak sits."""
    fresh, trained = split_rows(payload)
    if not trained:
        return None
    peak_step, peak_value = max(curve, key=lambda pair: pair[1]) if curve else (None, None)
    at_peak = row_at(trained, peak_step, near_for(payload)) if peak_step is not None else None
    before = [r for r in trained if peak_step is not None and r['step'] < peak_step]
    rise = before[len(before) // 2] if before else None
    return {'policy': payload['policy'], 'fc': payload['fc_layer_params'],
            'params': hidden_params(payload['fc_layer_params'], 30),
            'steps': trained[-1]['step'], 'rows': len(trained),
            'peak_step': peak_step, 'peak': peak_value,
            'end_trailing': curve[-1][1] if curve else None,
            'fresh': fresh, 'rise': rise, 'at_peak': at_peak, 'end': trained[-1],
            'reached_endgame': bool(peak_value is not None and peak_value >= 50.0)}


def hidden_params(fc, obs_len):
    """Weights in the hidden stack — the size measure the batch-20 sweep varied."""
    widths = [obs_len] + list(fc)
    return sum(widths[i] * widths[i + 1] for i in range(len(fc)))


def half_slopes(trained, curve):
    """Each metric's drift per 1M steps over the first and second half of the arm's step range."""
    if len(trained) < 6:
        return None, None
    middle = (trained[0]['step'] + trained[-1]['step']) / 2.0
    halves = []
    for rows in ([r for r in trained if r['step'] <= middle],
                 [r for r in trained if r['step'] > middle]):
        entry = {'from': rows[0]['step'], 'to': rows[-1]['step']} if rows else {}
        for key, label, _ in METRICS:
            entry[label] = per_million([(r['step'], r[key]) for r in rows
                                        if r.get(key) is not None])
        halves.append(entry)
    return halves[0], halves[1]


def describe_events(payload, curve):
    """Every drawdown event with the metric change across it, and its `move` while falling."""
    _, trained = split_rows(payload)
    near = near_for(payload)
    out = []
    for event in drawdown_events(curve):
        peak_row = row_at(trained, event['peak_step'], near)
        trough_row = row_at(trained, event['trough_step'], near)
        entry = dict(event)
        entry['policy'] = payload['policy']
        entry['matched'] = bool(peak_row and trough_row)
        entry['span'] = event['trough_step'] - event['peak_step']
        if peak_row and trough_row:
            for key, label, _ in METRICS:
                if key == 'move':
                    continue
                if peak_row.get(key) is None or trough_row.get(key) is None:
                    entry['d_' + label] = None
                else:
                    entry['d_' + label] = trough_row[key] - peak_row[key]
            inside = rows_between(trained, event['peak_step'], event['trough_step'])
            moves = [r['move'] for r in inside if r.get('move') is not None]
            entry['mean_move'] = float(np.mean(moves)) if moves else None
            entry['dormant_at_peak'] = peak_row['dormant_all']
            entry['srank_c_at_peak'] = peak_row['srank_c']
            entry['growth_at_peak'] = peak_row['growth_hidden']
        out.append(entry)
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    directory = sys.argv[1]
    chart_path = sys.argv[2] if len(sys.argv) > 2 else None

    payloads = load_payloads(directory)
    if not payloads:
        sys.exit('no plasticity payloads in ' + directory)
    curves = {arm: sorted(trailing_curve(arm).items()) for arm in payloads}

    arms, events, flats, halves = [], [], {}, {}
    for arm, payload in payloads.items():
        curve = curves[arm]
        summary = arm_table(payload, curve)
        if summary is None:
            continue
        arms.append(summary)
        events.extend(describe_events(payload, curve))
        _, trained = split_rows(payload)
        flats[arm] = flat_stretches(trained, curve)
        first, second = half_slopes(trained, curve)
        halves[arm] = {'first': first, 'second': second}
    arms.sort(key=lambda a: a['params'])

    print('\n=== the arms, by hidden-layer parameter count')
    print('%-30s %-18s %8s %8s %9s %8s %7s' % ('arm', 'fc', 'params', 'steps', 'peak', 'at step',
                                               'ends'))
    for arm in arms:
        print('%-30s %-18s %8d %8d %9s %8s %7s' % (
            arm['policy'], ','.join(str(x) for x in arm['fc']), arm['params'], arm['steps'],
            '%.1f' % arm['peak'] if arm['peak'] is not None else '-',
            arm['peak_step'] if arm['peak_step'] is not None else '-',
            '%.1f' % arm['end_trailing'] if arm['end_trailing'] is not None else '-'))

    print('\n=== control -> peak -> end, per arm  (control is a fresh net of the same shape)')
    head = '%-30s %-7s %8s %6s %6s %8s %8s %8s %9s'
    print(head % ('arm', 'when', 'dormant', 'dead', 'const', 'srank_c', 'stable_c', 'growth',
                  'move/100k'))
    for arm in arms:
        for label, row in (('fresh', arm['fresh']), ('rise', arm['rise']),
                           ('peak', arm['at_peak']), ('end', arm['end'])):
            if row is None:
                continue
            print(head % (arm['policy'] if label == 'fresh' else '', label,
                          '%.3f' % row['dormant_all'], '%.3f' % row['dead_all'],
                          '%.3f' % row['constant_all'], '%.1f' % row['srank_c'],
                          '%.2f' % row['stable_rank_c'], '%.3f' % row['growth_hidden'],
                          '%.4f' % row['move'] if row.get('move') is not None else '-'))

    print('\n=== question 2: the metrics at the point each arm first reached a 50%% perfect rate')
    print('    (an arm that never got there has no endgame row — that is the comparison)')
    print('%-30s %8s %9s %8s %6s %8s %8s %9s' % ('arm', 'params', 'crossed@', 'dormant', 'dead',
                                                 'srank_c', 'growth', 'move/100k'))
    for arm in arms:
        payload = payloads[arm['policy']]
        _, trained = split_rows(payload)
        crossing = next((step for step, value in curves[arm['policy']] if value >= 50.0), None)
        if crossing is None:
            print('%-30s %8d %9s %8s %6s %8s %8s %9s   <- never reached 50%%, peak %.1f' % (
                arm['policy'], arm['params'], 'never', '-', '-', '-', '-', '-', arm['peak']))
            continue
        row = row_at(trained, crossing, near_for(payload))
        if row is None:
            print('%-30s %8d %9d   <- no ladder row within %dk' % (
                arm['policy'], arm['params'], crossing, near_for(payload) // 1000))
            continue
        print('%-30s %8d %9d %8.3f %6.3f %8.1f %8.3f %9s' % (
            arm['policy'], arm['params'], crossing, row['dormant_all'], row['dead_all'],
            row['srank_c'], row['growth_hidden'],
            fmt(row.get('move'), '%.4f')))

    print('\n=== drawdown events (>= %.0f pp below a running peak)' % MIN_DEPTH)
    print('%-30s %9s %9s %6s %6s %4s %8s %8s %8s %8s' % (
        'arm', 'peak@', 'trough@', 'depth', 'span/M', 'rec', 'd_dorm', 'd_srank', 'd_growth',
        'move'))
    for event in sorted(events, key=lambda e: (e['policy'], e['peak_step'])):
        print('%-30s %9d %9d %6.1f %6.2f %4s %8s %8s %8s %8s' % (
            event['policy'], event['peak_step'], event['trough_step'], event['depth'],
            event['span'] / 1e6, 'y' if event['recovered'] else 'NO',
            fmt(event.get('d_dormant'), '%+.3f'), fmt(event.get('d_srank_c'), '%+.1f'),
            fmt(event.get('d_growth'), '%+.3f'), fmt(event.get('mean_move'), '%.4f')))

    print('\n=== question 5: later drawdowns vs earlier ones, matched on depth')
    matched = [e for e in events if e.get('matched')]
    report_early_late(matched)

    print('\n=== question 6: stretches where the perfect rate is flat (range <= %.0f pp over >= %dk)'
          % (FLAT_TOL, FLAT_SPAN // 1000))
    print('%-30s %9s %9s %7s %7s %9s %9s %9s %9s' % (
        'arm', 'from', 'to', 'mean', 'range', 'dorm/M', 'srank/M', 'growth/M', 'move/M'))
    for arm in arms:
        for stretch in flats.get(arm['policy'], []):
            print('%-30s %9d %9d %7.1f %7.1f %9s %9s %9s %9s' % (
                arm['policy'], stretch['start'], stretch['end'], stretch['mean_trailing'],
                stretch['range'], fmt(stretch.get('dormant'), '%+.4f'),
                fmt(stretch.get('srank_c'), '%+.2f'), fmt(stretch.get('growth'), '%+.3f'),
                fmt(stretch.get('move/100k'), '%+.4f')))

    print('\n=== drift per 1M steps, first half vs second half of each arm')
    print('%-30s %-7s %9s %9s %9s %9s %9s' % ('arm', 'half', 'dorm/M', 'srank/M', 'stable/M',
                                              'growth/M', 'move/M'))
    for arm in arms:
        entry = halves.get(arm['policy'], {})
        for label in ('first', 'second'):
            part = entry.get(label)
            if not part:
                continue
            print('%-30s %-7s %9s %9s %9s %9s %9s' % (
                arm['policy'] if label == 'first' else '', label,
                fmt(part.get('dormant'), '%+.4f'), fmt(part.get('srank_c'), '%+.2f'),
                fmt(part.get('stable_c'), '%+.3f'), fmt(part.get('growth'), '%+.3f'),
                fmt(part.get('move/100k'), '%+.4f')))

    out_path = os.path.join(directory, 'plasticity_summary.json')
    with open(out_path, 'w') as handle:
        json.dump({'arms': arms, 'events': events, 'flat': flats, 'halves': halves,
                   'min_depth': MIN_DEPTH, 'flat_tol': FLAT_TOL, 'flat_span': FLAT_SPAN},
                  handle, indent=1, default=float)
    print('\nwrote %s' % out_path)

    if chart_path:
        draw(payloads, curves, arms, chart_path)
        print('wrote %s' % chart_path)


def fmt(value, spec):
    return '-' if value is None else spec % value


def report_early_late(matched):
    """Split the events at the median peak step and compare their metric costs at matched depth."""
    if len(matched) < 4:
        print('  only %d matched events — not enough to split' % len(matched))
        return
    median_step = float(np.median([e['peak_step'] for e in matched]))
    early = [e for e in matched if e['peak_step'] <= median_step]
    late = [e for e in matched if e['peak_step'] > median_step]
    print('  split at peak step %.0f: %d early, %d late' % (median_step, len(early), len(late)))
    print('  %-10s %6s %7s %9s %9s %9s %9s' % ('group', 'n', 'depth', 'd_dorm', 'd_srank',
                                               'd_growth', 'move'))
    for label, group in (('early', early), ('late', late)):
        print('  %-10s %6d %7.1f %9s %9s %9s %9s' % (
            label, len(group), float(np.mean([e['depth'] for e in group])),
            fmt(mean_of(group, 'd_dormant'), '%+.3f'), fmt(mean_of(group, 'd_srank_c'), '%+.1f'),
            fmt(mean_of(group, 'd_growth'), '%+.3f'), fmt(mean_of(group, 'mean_move'), '%.4f')))
    # Depth is the confound: a deeper drawdown moves more of everything. So report the per-point
    # cost as well, and say plainly when the depths are not comparable.
    depths = [float(np.mean([e['depth'] for e in group])) for group in (early, late)]
    print('  mean depth %.1f early vs %.1f late — %s' % (
        depths[0], depths[1],
        'comparable' if abs(depths[0] - depths[1]) < 5 else 'NOT comparable, read per-point instead'))
    print('  %-10s %6s %7s %9s %9s %9s' % ('group', 'n', 'depth', 'dorm/pp', 'srank/pp',
                                           'growth/pp'))
    for label, group in (('early', early), ('late', late)):
        print('  %-10s %6d %7.1f %9s %9s %9s' % (
            label, len(group), float(np.mean([e['depth'] for e in group])),
            fmt(per_point(group, 'd_dormant'), '%+.4f'),
            fmt(per_point(group, 'd_srank_c'), '%+.3f'),
            fmt(per_point(group, 'd_growth'), '%+.4f')))


def mean_of(group, key):
    values = [e[key] for e in group if e.get(key) is not None]
    return float(np.mean(values)) if values else None


def per_point(group, key):
    values = [e[key] / e['depth'] for e in group if e.get(key) is not None and e['depth']]
    return float(np.mean(values)) if values else None


def draw(payloads, curves, arms, path):
    """Trailing rate over four plasticity metrics, one column per arm group, shared step axis."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    panels = [('trailing', 'trailing-30 perfect %', None),
              ('dormant_all', 'dormant fraction', 'dormant'),
              ('srank_c', 'centred srank', 'srank_c'),
              ('growth_hidden', 'hidden weight growth', 'growth'),
              ('move', 'kernel movement /100k', None)]
    figure = Figure(figsize=(13, 3.0 * len(panels)), dpi=110)
    colours = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#17becf',
               '#e377c2', '#7f7f7f']
    for index, (key, title, _) in enumerate(panels):
        axis = figure.add_subplot(len(panels), 1, index + 1)
        for order, arm in enumerate(arms):
            payload = payloads[arm['policy']]
            fresh, trained = split_rows(payload)
            colour = colours[order % len(colours)]
            label = '%s (%s)' % (arm['policy'], ','.join(str(x) for x in arm['fc']))
            if key == 'trailing':
                curve = curves[arm['policy']]
                axis.plot([s / 1e6 for s, _ in curve], [v for _, v in curve], color=colour,
                          linewidth=0.9, label=label)
                continue
            series = [(r['step'], r[key]) for r in trained if r.get(key) is not None]
            axis.plot([s / 1e6 for s, _ in series], [v for _, v in series], color=colour,
                      linewidth=1.1, label=label)
            if fresh and fresh.get(key) is not None:
                axis.axhline(fresh[key], color=colour, linestyle=':', linewidth=0.7, alpha=0.5)
            if arm['peak_step'] is not None:
                axis.axvline(arm['peak_step'] / 1e6, color=colour, linestyle='--', linewidth=0.6,
                             alpha=0.35)
        axis.set_ylabel(title, fontsize=9)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=8)
        if index == 0:
            axis.legend(fontsize=7, ncol=3, loc='upper right')
            axis.set_title('Plasticity metrics against step. Dotted = that arm\'s fresh-net control; '
                           'dashed vertical = its peak.', fontsize=10)
        if index == len(panels) - 1:
            axis.set_xlabel('training step (millions)', fontsize=9)
    figure.tight_layout()
    FigureCanvasAgg(figure).print_png(path)


if __name__ == '__main__':
    main()
