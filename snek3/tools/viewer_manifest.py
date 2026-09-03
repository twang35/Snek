"""Writes `viewer/manifest.js`: one compact record per arm, for the static chart viewer.

    PYTHONPATH=. python -m tools.viewer_manifest            # runs/ -> viewer/manifest.js

The viewer (`viewer/index.html`) is a page GitHub Pages serves straight out of the repo, so it can
load nothing that is not committed and nothing large: an arm's `_evals.json` is ~1.6 MB and its
stage-B file ~2.8 MB, which over 64 arms is 280 MB per page load. This walks `runs/` once and reduces
each arm to the dozen numbers the docs tables quote — the same definitions as `docs/results.md`, so
the page and the tables cannot disagree:

| field | definition |
|---|---|
| `best30` | `summary.best_perfect30`, the peak 30-eval trailing perfect rate |
| `sef` | `summary.strong_eval_fraction`, share of stage-A evals at >=80% perfect |
| `rows`, `density98`, `cands985`, `best_row` | stage-B row count, share at >=98/500, count at >=98.5, max |
| `drawdown50`, `drawdown80` | share of post-competence stage-A evals (onset = first >=80%) below 50 / 80 |

The output is JavaScript rather than JSON — `window.SNEK_MANIFEST = {...}` — because a `<script src>`
loads from `file://` and `fetch()` does not, and the page has to work opened from disk as well as from
Pages. Re-run after every progress update; the manifest is committed beside the page.
"""

import datetime
import glob
import json
import os
import re

from env import constants

MANIFEST_PATH = os.path.join(os.path.dirname(constants.RUNS_DIR), 'viewer', 'manifest.js')
_BATCH_RE = re.compile(r'^([a-z]+\d+)')
_SEED_RE = re.compile(r'-seed(\d+)$')


def batch_of(policy):
    """`b9ce-lam999-seed1` -> `b9`; anything without a letters-digits prefix keys on its first token."""
    match = _BATCH_RE.match(policy)
    return match.group(1) if match else policy.split('-')[0]


def knob_of(policy):
    """The middle of the name — `b9ce-lam999-seed1` -> `lam999`, `b3k-fc200x100` -> `fc200x100`."""
    parts = _SEED_RE.sub('', policy).split('-')
    return '-'.join(parts[1:]) if len(parts) > 1 else ''


def seed_of(policy):
    match = _SEED_RE.search(policy)
    return int(match.group(1)) if match else None


def drawdown(evals, below):
    """Share (%) of stage-A evals after the first >=80% one that fall below `below`; None before onset."""
    onset = next((i for i, e in enumerate(evals) if e.get('perfect_percent', 0) >= 80), None)
    if onset is None:
        return None
    post = evals[onset:]
    return round(100.0 * sum(e.get('perfect_percent', 0) < below for e in post) / len(post), 2)


def _read(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def arm_record(policy, runs_dir):
    """The manifest row for one arm, or None if it has no chart to show."""
    png = os.path.join(runs_dir, policy + '.png')
    if not os.path.exists(png):
        return None
    record = {'policy': policy, 'batch': batch_of(policy), 'knob': knob_of(policy),
              'seed': seed_of(policy),
              'stage_b_png': os.path.exists(os.path.join(runs_dir, policy + '_checkpoint_evals.png'))}
    stage_a = _read(os.path.join(runs_dir, policy + '_evals.json')) or {}
    summary = stage_a.get('summary') or {}
    best30 = summary.get('best_perfect30') or {}
    record.update({
        'step': summary.get('step'),
        'best30': best30.get('value'), 'best30_step': best30.get('step'),
        'sef': summary.get('strong_eval_fraction'), 'trailing': summary.get('trailing_now'),
        'evals': summary.get('evals'),
        'drawdown50': drawdown(stage_a.get('evals') or [], 50),
        'drawdown80': drawdown(stage_a.get('evals') or [], 80),
    })
    stage_b = _read(os.path.join(runs_dir, policy + '_checkpoint_evals.json'))
    rows = (stage_b or {}).get('rows') or []
    scores = [r.get('perfect_percent', 0) for r in rows]
    record.update({
        'rows': len(rows) if stage_b is not None else None,
        'density98': round(100.0 * sum(s >= 98 for s in scores) / len(scores), 1) if scores else None,
        'cands985': sum(s >= 98.5 for s in scores) if scores else None,
        'best_row': max(scores) if scores else None,
    })
    return record


def build(runs_dir=None):
    runs_dir = runs_dir or constants.RUNS_DIR
    # A policy name has hyphens and never an underscore; every derived chart (`_checkpoint_evals`,
    # `_checkpoint_evals_hof5000`, `_eval_progress`) has one. So the stem decides.
    policies = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(runs_dir, '*.png'))
                      if '_' not in os.path.basename(p))
    arms = [rec for rec in (arm_record(p, runs_dir) for p in policies) if rec]
    return {'generated': datetime.datetime.now().isoformat(timespec='seconds'), 'arms': arms}


def render(manifest):
    return 'window.SNEK_MANIFEST = ' + json.dumps(manifest, separators=(',', ':')) + ';\n'


def main(argv=None):
    manifest = build()
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, 'w') as handle:
        handle.write(render(manifest))
    batches = {}
    for arm in manifest['arms']:
        batches[arm['batch']] = batches.get(arm['batch'], 0) + 1
    print('{0}: {1} arms in {2} batches -> {3}'.format(
        manifest['generated'], len(manifest['arms']), len(batches), os.path.relpath(MANIFEST_PATH)))
    print(' '.join('{0}:{1}'.format(k, v) for k, v in sorted(batches.items())))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
