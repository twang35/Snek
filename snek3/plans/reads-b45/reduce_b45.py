"""One-off: reduce b45's pass files (the reads) into the self-contained page `viewer/pages/reads-b45.html`.

    python reduce_b45.py <dir with *_checkpoint_evals_reads-*.json> <template.html> <out.html>

Not a project tool on purpose (`plans/quantile-reads.md`: a one-off page). The numbers are the same
`tools/read_compare.py` prints: paired Δ per checkpoint against `reads-meanfixed`, a bootstrap
interval over checkpoints, and the death / starve split off the rows' own counts.
"""
import datetime
import glob
import json
import os
import random
import re
import statistics
import sys

CELLS = [
    ('c51local', 'C51 (51 atoms)', 'b37', ['b37a-c51local-seed1', 'b37b-c51local-seed2', 'b37c-c51local-seed3', 'b37d-c51local-seed4']),
    ('qrdqnlocal', 'QR-DQN (N 32)', 'b37', ['b37e-qrdqnlocal-seed5', 'b37f-qrdqnlocal-seed6', 'b37g-qrdqnlocal-seed7', 'b37h-qrdqnlocal-seed8']),
    ('mqrdqnlocal', 'M-QR-DQN (N 32)', 'b40', ['b40e-mqrdqnlocal-seed5', 'b40f-mqrdqnlocal-seed6', 'b40g-mqrdqnlocal-seed7', 'b40h-mqrdqnlocal-seed8']),
    ('iqnlocal', 'IQN (N 8)', 'b38', ['b38a-iqnlocal-seed1', 'b38b-iqnlocal-seed2', 'b38c-iqnlocal-seed3', 'b38d-iqnlocal-seed4']),
    ('fqflocal', 'FQF (N 8)', 'b39', ['b39a-fqflocal-seed1', 'b39b-fqflocal-seed2', 'b39c-fqflocal-seed3', 'b39d-fqflocal-seed4']),
]
# (slug, variant, group, one line). Order is the page's order.
READS = [
    ('meanfixed', 'mean:fixed', 'control', 'the control: argmax of the mean, IQN at fixed fractions'),
    ('cvar05', 'cvar:0.5', 'down', 'mean of the lowest half of the mass'),
    ('cvar025', 'cvar:0.25', 'down', 'mean of the lowest quarter of the mass'),
    ('mixmass07', 'mixmass:0.7', 'down', '0.7 x losses\' share of the mean + 0.3 x gains\' share (mass kept)'),
    ('mix07', 'mix:0.7', 'down', '0.7 x mean of the losses + 0.3 x mean of the gains (mass discarded)'),
    ('leastneg', 'leastneg', 'down', 'losses\' share of the mean closest to 0 wins; ties by the gains\' share'),
    ('leastneg-2', 'leastneg:-2', 'down', 'as leastneg, counting only losses below -2 (death-sized)'),
    ('leastnegmean', 'leastnegmean', 'down', 'mean of the losses closest to 0 wins (mass discarded); ties by the mean of the gains'),
    ('leastnegmean-2', 'leastnegmean:-2', 'down', 'as leastnegmean, counting only losses below -2'),
    ('mix05', 'mix:0.5', 'mixed', '0.5 / 0.5 mix of the two conditional means'),
    ('mix03', 'mix:0.3', 'up', '0.3 x mean of the losses + 0.7 x mean of the gains'),
    ('above10', 'above:10', 'up', 'share of the mean above 10; the mean where no action has any'),
    ('above30', 'above:30', 'up', 'share of the mean above 30'),
    ('above60', 'above:60', 'up', 'share of the mean above 60'),
    ('abovemean10', 'abovemean:10', 'up', 'mean of the quantiles above 10 (mass discarded)'),
    ('abovemean30', 'abovemean:30', 'up', 'mean of the quantiles above 30 (mass discarded)'),
]
CONTROL = 'meanfixed'


def read_rows(path):
    with open(path) as handle:
        payload = json.load(handle)
    rows = payload.get('rows', [])
    if isinstance(rows, dict):
        rows = list(rows.values())
    return rows


def compact(row):
    ci = row.get('perfect_ci95') or [None, None]
    return {'p': row['perfect_percent'], 'lo': ci[0], 'hi': ci[1], 'n': row['episodes'],
            'd': row.get('deaths'), 's': row.get('starves'), 'score': row.get('avg_score'),
            'sec': row.get('seconds'), 'hist': row.get('score_counts') or {}}


def bootstrap(differences, draws=2000, seed=0):
    if len(differences) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(differences) for _ in differences) for _ in range(draws))
    return [round(means[int(0.025 * draws)], 2), round(means[int(0.975 * draws) - 1], 2)]


def summarise(control, rows):
    keys = sorted(set(control) & set(rows))
    diffs = [rows[k]['p'] - control[k]['p'] for k in keys]
    have_counts = [k for k in rows if rows[k]['d'] is not None]
    eps = sum(rows[k]['n'] for k in have_counts)
    ceps = sum(control[k]['n'] for k in keys if control[k]['d'] is not None)
    death = 100.0 * sum(rows[k]['d'] for k in have_counts) / eps if eps else None
    starve = 100.0 * sum(rows[k]['s'] for k in have_counts) / eps if eps else None
    cdeath = 100.0 * sum(control[k]['d'] for k in keys if control[k]['d'] is not None) / ceps if ceps else None
    cstarve = 100.0 * sum(control[k]['s'] for k in keys if control[k]['d'] is not None) / ceps if ceps else None
    best_key = max(rows, key=lambda k: (rows[k]['p'], k)) if rows else None
    cbest = max((control[k]['p'] for k in control), default=None)
    return {
        'best': rows[best_key]['p'] if best_key else None, 'best_ckpt': best_key,
        'dbest': None if best_key is None or cbest is None else round(rows[best_key]['p'] - cbest, 2),
        'ckpts': len(rows), 'paired': len(keys),
        'perfect': round(statistics.fmean(r['p'] for r in rows.values()), 2) if rows else None,
        'delta': round(statistics.fmean(diffs), 2) if diffs else None,
        'ci': bootstrap(diffs),
        'wins': sum(1 for d in diffs if d > 0), 'losses': sum(1 for d in diffs if d < 0), 'level': sum(1 for d in diffs if d == 0),
        'death': None if death is None else round(death, 2), 'starve': None if starve is None else round(starve, 2),
        'ddeath': None if death is None or cdeath is None else round(death - cdeath, 2),
        'dstarve': None if starve is None or cstarve is None else round(starve - cstarve, 2),
        'score': round(statistics.fmean(r['score'] for r in rows.values() if r['score'] is not None), 2) if rows else None,
    }


def main(src, template, out):
    data = {'batch': 'b45', 'built': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'cells': [{'key': k, 'name': n, 'batch': b, 'arms': arms} for k, n, b, arms in CELLS],
            'reads': [{'key': s, 'variant': v, 'group': g, 'desc': d} for s, v, g, d in READS],
            'rows': {}, 'summary': {}}
    for key, _, _, arms in CELLS:
        data['rows'][key] = {}
        data['summary'][key] = {}
        for slug, _, _, _ in READS:
            rows = {}
            for arm in arms:
                path = os.path.join(src, '{0}_checkpoint_evals_reads-{1}.json'.format(arm, slug))
                if os.path.exists(path):
                    for row in read_rows(path):
                        rows['{0}@{1}'.format(arm, row['step'])] = compact(row)
            if rows:
                data['rows'][key][slug] = rows
        control = data['rows'][key].get(CONTROL, {})
        for slug in data['rows'][key]:
            data['summary'][key][slug] = summarise(control, data['rows'][key][slug])
    with open(template) as handle:
        page = handle.read()
    blob = json.dumps(data, separators=(',', ':'))
    assert page.count('/*DATA*/') == 1
    page = page.replace('/*DATA*/', blob)
    with open(out, 'w') as handle:
        handle.write(page)
    passes = sum(len(v) for v in data['rows'].values())
    print('wrote {0}: {1} cell-reads, {2:.1f} MB'.format(out, passes, len(page) / 1e6))
    for cell in data['summary']:
        for slug, s in data['summary'][cell].items():
            print('  {0:12s} {1:16s} ckpts {2:3d} perfect {3} delta {4} ci {5} death {6} starve {7}'.format(
                cell, slug, s['ckpts'], s['perfect'], s['delta'], s['ci'], s['death'], s['starve']))


if __name__ == '__main__':
    main(*sys.argv[1:4])
