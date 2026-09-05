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
| `rows`, `density98`, `cands99`, `best_row` | stage-B row count, share at >=98/500, `hof5000` candidates at >=99, max |
| `drawdown50`, `drawdown80` | share of post-competence stage-A evals (onset = first >=80%) below 50 / 80 |
| `hof_rows`, `hof_mean`, `hof_best`, `hof_9873` | the `hof5000` pass: rows, mean, max, count at >=98.73 (the snek2 champion) |
| `hof30k_rows`, `hof30k_mean`, `hof30k_best`, `hof30k_best_step` | the `hof30k` pass (30,000 episodes, seed 7): rows, mean, max and where it is |
| `hof_99` | `hof5000` rows at >=99 /5,000 — the `hof30k` candidate cut |
| `status` | `{a, b, h, k}`: one word per view, see `pass_state` — so the page can say whether a missing panel is a pass still to come or one that found nothing |

The status of a pass is read off the files, plus two liveness sources: the laptop's own `.live/` pid
registry (`tools/live_runs.py`) and a snapshot of the desktop's `status.json` that `tools/progress_update.py`
saves at `runs/.live/desktop/status.json` on every sync. Both are optional — with neither, every pass is
`done`, `pending`, `none` or `upstream`, which is still right about what the files say.

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
from tools import live_runs

MANIFEST_PATH = os.path.join(constants.ROOT, 'viewer', 'manifest.js')
# Which earlier arms are a batch's control cell, so the page can show them beside the batch's own.
REFERENCES_PATH = os.path.join(constants.ROOT, 'viewer', 'references.json')
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


def onset(evals):
    """The step of the first stage-A eval at >=80% perfect, or None."""
    for e in evals:
        if e.get('perfect_percent', 0) >= 80:
            return e.get('step')
    return None


def stage_a_share(evals, at_least):
    """Share (%) of all stage-A evals at or above `at_least` percent perfect; None with no evals."""
    if not evals:
        return None
    return round(100.0 * sum(e.get('perfect_percent', 0) >= at_least for e in evals) / len(evals), 1)


def drawdown(evals, below):
    """Share (%) of stage-A evals after the first >=80% one that fall below `below`; None before onset."""
    onset = next((i for i, e in enumerate(evals) if e.get('perfect_percent', 0) >= 80), None)
    if onset is None:
        return None
    post = evals[onset:]
    return round(100.0 * sum(e.get('perfect_percent', 0) < below for e in post) / len(post), 2)


LIVE_SUBDIR = os.path.join('.live', 'desktop')
DESKTOP_STATUS = os.path.join(LIVE_SUBDIR, 'status.json')
# The pass each view shows, its file label, and the desktop job id suffix the daemon gives that pass.
PASSES = {'b': (None, '-stageb'), 'h': ('hof5000', '-hof5000'), 'k': ('hof30k', '-hof30k')}
STATES = ('done', 'running', 'queued', 'pending', 'none', 'upstream')


def shard_files(runs_dir, policy, label=None):
    """The shard files of a pass in flight — `<policy>_checkpoint_evals[_<label>]-s<i>of<n>.json`.
    Same rule as `tools.results.shard_paths`, which is pinned to `constants.RUNS_DIR` and so cannot
    serve a manifest built from another directory. The regex keeps `_hof5000-s1of8` out of the
    unlabelled pass's list."""
    stem = policy + '_checkpoint_evals' + ('_' + label if label else '')
    exact = re.compile(re.escape(stem) + r'-s(\d+)of(\d+)\.json$')
    return sorted(p for p in glob.glob(os.path.join(runs_dir, stem + '-s*of*.json'))
                  if exact.search(os.path.basename(p)))


def desktop_ledger(runs_dir):
    """What the desktop was doing when the progress update last looked: `{'iso', 'jobs', 'running'}`
    — `jobs` is the ledger (`{job id: 'queued'|'running'|'done'|'failed'}`), `running` maps each view
    to the policies a running job of that kind covers. Empty when no snapshot has been saved."""
    status = _read(os.path.join(runs_dir, DESKTOP_STATUS)) or {}
    running = {'a': set(), 'b': set(), 'h': set(), 'k': set()}
    for job in status.get('running') or []:
        policies = set(job.get('policies') or ([job['policy']] if job.get('policy') else []))
        job_id = job.get('id') or ''
        kind = 'a' if job.get('type') == 'train' else next(
            (k for k, (_label, suffix) in PASSES.items() if job_id.endswith(suffix)), None)
        if kind:
            running[kind] |= policies
    return {'iso': status.get('iso'), 'jobs': status.get('ledger') or {}, 'running': running}


def pass_state(have_file, have_shards, candidates, in_running_job, ledger_state):
    """One word for where a pass stands, for one arm:

    | state | meaning |
    |---|---|
    | `done` | the pass's file exists |
    | `running` | shard files exist, or a running desktop job names the arm |
    | `queued` | the desktop ledger has the batch's pass queued (or running, but not yet on this arm) |
    | `pending` | nothing has run and the arm has candidates for it |
    | `none` | nothing has run and the arm has no candidates — a panel will never appear |
    | `upstream` | the pass it selects from has not happened yet |

    `candidates` is None when the upstream pass is missing, else the count that clears its cut."""
    if have_file:
        return 'done'
    if have_shards or in_running_job:
        return 'running'
    if ledger_state in ('queued', 'running'):
        return 'queued'
    if candidates is None:
        return 'upstream'
    return 'pending' if candidates else 'none'


def _measurement(runs_dir, name):
    """`runs/<name>` if it exists, else the live desktop snapshot `runs/.live/desktop/<name>` that
    `tools/progress_update.py` pulls for an arm still training on the box. `runs/` itself never holds a
    live desktop arm's JSON (it would be committed by accident and block the box's deploy), and the
    close-out's file, once imported, takes precedence over any snapshot."""
    path = os.path.join(runs_dir, name)
    if os.path.exists(path):
        return path
    return os.path.join(runs_dir, LIVE_SUBDIR, name)


def _read(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def arm_record(policy, runs_dir, desktop=None, laptop_live=frozenset()):
    """The manifest row for one arm, or None if it has no chart to show. `desktop` is
    `desktop_ledger(runs_dir)` and `laptop_live` the policies training on this box; `build` passes
    both so they are read once per manifest rather than once per arm."""
    png = os.path.join(runs_dir, policy + '.png')
    if not os.path.exists(png):
        return None
    desktop = desktop if desktop is not None else desktop_ledger(runs_dir)
    batch = batch_of(policy)
    record = {'policy': policy, 'batch': batch, 'knob': knob_of(policy),
              'seed': seed_of(policy),
              'stage_b_png': os.path.exists(os.path.join(runs_dir, policy + '_checkpoint_evals.png')),
              'hof_png': os.path.exists(os.path.join(runs_dir, policy + '_checkpoint_evals_hof5000.png')),
              'hof30k_png': os.path.exists(os.path.join(runs_dir, policy + '_checkpoint_evals_hof30k.png'))}
    stage_a = _read(_measurement(runs_dir, policy + '_evals.json')) or {}
    summary = stage_a.get('summary') or {}
    best30 = summary.get('best_perfect30') or {}
    record.update({
        'step': summary.get('step'),
        'best30': best30.get('value'), 'best30_step': best30.get('step'),
        'sef': summary.get('strong_eval_fraction'), 'trailing': summary.get('trailing_now'),
        'evals': summary.get('evals'),
        'drawdown50': drawdown(stage_a.get('evals') or [], 50),
        'drawdown80': drawdown(stage_a.get('evals') or [], 80),
        'stage_a_98': stage_a_share(stage_a.get('evals') or [], 98),
        'onset_step': onset(stage_a.get('evals') or []),
    })
    stage_b = _read(_measurement(runs_dir, policy + '_checkpoint_evals.json'))
    rows = (stage_b or {}).get('rows') or []
    scores = [r.get('perfect_percent', 0) for r in rows]
    record.update({
        'rows': len(rows) if stage_b is not None else None,
        'density98': round(100.0 * sum(s >= 98 for s in scores) / len(scores), 1) if scores else None,
        'cands99': sum(s >= 99 for s in scores) if scores else None,
        'best_row': max(scores) if scores else None,
    })
    hof = _read(os.path.join(runs_dir, policy + '_checkpoint_evals_hof5000.json'))
    hof_scores = [r.get('perfect_percent', 0) for r in ((hof or {}).get('rows') or [])]
    record.update({
        'hof_rows': len(hof_scores) if hof is not None else None,
        'hof_mean': round(sum(hof_scores) / len(hof_scores), 2) if hof_scores else None,
        'hof_best': max(hof_scores) if hof_scores else None,
        'hof_9873': sum(s >= 98.73 for s in hof_scores) if hof_scores else None,
    })
    h30 = _read(os.path.join(runs_dir, policy + '_checkpoint_evals_hof30k.json'))
    h30_rows = (h30 or {}).get('rows') or []
    best = max(h30_rows, key=lambda r: r.get('perfect_percent', 0)) if h30_rows else None
    record.update({
        'hof30k_rows': len(h30_rows) if h30 is not None else None,
        'hof30k_mean': round(sum(r.get('perfect_percent', 0) for r in h30_rows) / len(h30_rows), 2) if h30_rows else None,
        'hof30k_best': best.get('perfect_percent') if best else None,
        'hof30k_best_step': best.get('step') if best else None,
        'hof_99': sum(s >= 99 for s in hof_scores) if hof is not None else None,
    })
    # Where each view stands. Stage A is live if this box or the desktop is training it, or if its
    # measurements are still the desktop snapshot rather than the close-out's file.
    jobs, running = desktop['jobs'], desktop['running']
    job = jobs.get(policy)                      # the desktop ledger's entry for the training itself
    snapshot_only = bool(stage_a) and not os.path.exists(os.path.join(runs_dir, policy + '_evals.json'))
    if policy in laptop_live or policy in running['a'] or job == 'running' or (job is None and snapshot_only):
        stage_a_state = 'running'
    else:
        stage_a_state = 'queued' if job == 'queued' else 'done'
    # Each pass selects from the one before it: stage B needs the training finished, hof5000 needs
    # stage-B rows at >=99 /500, hof30k needs hof5000 rows at >=99 /5,000.
    candidates = {'b': None if stage_a_state != 'done' else 1,
                  'h': sum(s >= 99 for s in scores) if stage_b is not None else None,
                  'k': record['hof_99']}
    record['status'] = {'a': stage_a_state}
    for kind, (label, suffix) in PASSES.items():
        have = {'b': stage_b, 'h': hof, 'k': h30}[kind] is not None
        record['status'][kind] = pass_state(have, bool(shard_files(runs_dir, policy, label)), candidates[kind],
                                            policy in running[kind], jobs.get(batch + suffix))
    # A pass whose upstream found nothing will never run either: say `none`, not `upstream`.
    for kind, before in (('h', 'b'), ('k', 'h')):
        if record['status'][kind] == 'upstream' and record['status'][before] == 'none':
            record['status'][kind] = 'none'
    return record


def references(path=None):
    """`{batch: {'arms': [...], 'label': str}}` from `viewer/references.json`, or {} if absent."""
    path = path or REFERENCES_PATH
    data = _read(path) or {}
    return {k: v for k, v in data.items() if not k.startswith('_')}


def build(runs_dir=None, references_path=None):
    runs_dir = runs_dir or constants.RUNS_DIR
    # A policy name has hyphens and never an underscore; every derived chart (`_checkpoint_evals`,
    # `_checkpoint_evals_hof5000`, `_eval_progress`) has one. So the stem decides.
    policies = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(runs_dir, '*.png'))
                      if '_' not in os.path.basename(p))
    desktop = desktop_ledger(runs_dir)
    laptop_live = frozenset(policy for policy, _pid in live_runs.live(runs_dir, prune=False))
    arms = [rec for rec in (arm_record(p, runs_dir, desktop, laptop_live) for p in policies) if rec]
    known = {a['policy'] for a in arms}
    refs = {batch: {'arms': [a for a in ref.get('arms', []) if a in known], 'label': ref.get('label', ''),
                    'after': ref.get('after')}
            for batch, ref in references(references_path).items()}
    return {'generated': datetime.datetime.now().isoformat(timespec='seconds'), 'arms': arms,
            'references': refs, 'desktop_iso': desktop['iso']}


def render(manifest, charts_dir='../runs/'):
    """The manifest as a script. `charts_dir` is where the page finds the PNGs relative to itself:
    `../runs/` beside the source viewer, `charts/` on the published site (`tools/publish_pages.py`)."""
    payload = dict(manifest, charts_dir=charts_dir)
    return 'window.SNEK_MANIFEST = ' + json.dumps(payload, separators=(',', ':')) + ';\n'


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
