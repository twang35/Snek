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
| `hof_rows`, `hof_stopped`, `hof_mean`, `hof_best`, `hof_9873`, `hof_996` | the `hof5000` pass: rows, rows stopped early, mean over the full rows, max, count at >=98.73 (the snek2 champion), count at >=99.6 (the `hof30k` cut) |
| `hof30k_rows`, `hof30k_stopped`, `hof30k_mean`, `hof30k_best`, `hof30k_best_step`, `hof30k_998` | the `hof30k` pass (30,000 episodes, seed 7): rows, rows stopped early, mean over the full rows, max and where it is, count at >=99.8 |
| `hof_99` | `hof5000` rows at >=99 /5,000 — the `hof30k` candidate cut until 2026-09-08 |

A row stopped early (`abandoned`, `plans/archive/early-stop.md`) is a short sample that is only ever below its
pass's target: it counts as a row and never in a mean, and it sits below every `>=` count by arithmetic.
| `status` | `{a, b, h, k}`: one word per view, see `pass_state` — so the page can say whether a missing panel is a pass still to come or one that found nothing |
| `status_box` | `{a, b, h, k}`: the box a running or queued view is on (`desktop`, `laptop`, or None when unknown or unclaimed), so the caption names the right box |

The status of a pass is read off the files, plus three liveness sources: the laptop's own `.live/` pid
registry (`tools/live_runs.py`), this box's scheduler status (`runs/.live/.status.json`), and a snapshot
of the desktop's published `status.json` — which carries both boxes' running jobs, each tagged `box` — that
`tools/progress_update.py` saves at `runs/.live/desktop/status.json` on every sync. All are optional —
with none, every pass is `done`, `pending`, `none` or `upstream`, which is still right about what the
files say.

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
# Which box trained each arm (`{policy: 'desktop' | 'laptop'}`), written beside the status by whoever
# builds the manifest's runs directory -- `tools/site_build.py` from which feed carried the arm,
# `tools/progress_update.py` the same -- since a batch can span both boxes wave by wave
# (`tools/claims.py`) and the two boxes' steps/s differ enough to read as an effect.
BOXES_PATH = os.path.join('.live', 'boxes.json')
# The pass each view shows, its file label, and the desktop job id suffix the daemon gives that pass.
PASSES = {'b': (None, '-stageb'), 'h': ('hof5000', '-hof5000'), 'k': ('hof30k', '-hof30k')}
STATES = ('done', 'running', 'queued', 'pending', 'none', 'upstream')


def shard_files(runs_dir, policy, label=None, names=None):
    """The shard files of a pass in flight — `<policy>_checkpoint_evals[_<label>]-s<i>of<n>.json`.
    Same rule as `tools.results.shard_paths`, which is pinned to `constants.RUNS_DIR` and so cannot
    serve a manifest built from another directory. The regex keeps `_hof5000-s1of8` out of the
    unlabelled pass's list. `names` is the directory's listing when the caller already has one: `build`
    lists `runs/` once and passes it, because a glob per arm per pass listed the 4,400-file directory
    1,800 times a build (14 s of 39, measured 2026-09-06)."""
    stem = policy + '_checkpoint_evals' + ('_' + label if label else '')
    exact = re.compile(re.escape(stem) + r'-s(\d+)of(\d+)\.json$')
    if names is None:
        names = os.listdir(runs_dir) if os.path.isdir(runs_dir) else []
    return sorted(os.path.join(runs_dir, n) for n in names if exact.search(n))


def boxes(runs_dir):
    """`{policy: box}` from `runs/.live/boxes.json`, or {} when nobody has written one."""
    data = _read(os.path.join(runs_dir, BOXES_PATH)) or {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def write_boxes(runs_dir, mapping):
    """Writes `{policy: box}` where `boxes` reads it. Atomic, so a manifest build mid-write reads whole."""
    path = os.path.join(runs_dir, BOXES_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.partial', 'w') as handle:
        json.dump(dict(sorted(mapping.items())), handle)
    os.replace(path + '.partial', path)
    return path


def ledger_snapshot(runs_dir):
    """What both boxes were doing when last looked: `{'iso', 'jobs', 'running', 'job_boxes'}`.

    Two sources, merged: the desktop's published `status.json` snapshot (`runs/.live/desktop/`), whose
    `running` list carries both boxes' jobs since 2026-09-07, each tagged `box`; and this box's own
    scheduler status (`runs/.live/.status.json`), so the local viewer is right without a snapshot. `jobs`
    is the ledger (`{job id: 'queued'|'running'|'done'|'failed'}`), `running` maps each view to
    `{policy: box}` for the policies a running job of that kind covers, and `job_boxes` maps each
    running job id to its box. Empty when neither file exists."""
    desktop = _read(os.path.join(runs_dir, DESKTOP_STATUS)) or {}
    local = _read(live_runs.status_path(runs_dir)) or {}
    running = {'a': {}, 'b': {}, 'h': {}, 'k': {}}
    job_boxes = {}
    for source, default_box in ((desktop, 'desktop'), (local, local.get('box'))):
        for job in source.get('running') or []:
            policies = set(job.get('policies') or ([job['policy']] if job.get('policy') else []))
            job_id = job.get('id') or ''
            box = job.get('box') or default_box
            # A pass job is `<batch><suffix>` for wave 1 and `<batch><suffix>-wN` after, so the suffix
            # is matched with the wave allowed for: `endswith` saw only wave 1, and every later wave's
            # arms fell through to the cross-wave guess below (b27 on 2026-09-08: wave 2's hof5000 ran
            # on the laptop and the page said "queued on the desktop", wave 3's box).
            kind = 'a' if job.get('type') == 'train' else next(
                (k for k, (_label, suffix) in PASSES.items() if _PASS_JOB(suffix).match(job_id)), None)
            if kind:
                running[kind].update({p: box for p in policies})
            if job_id:
                job_boxes[job_id] = box
    return {'iso': desktop.get('iso'), 'jobs': desktop.get('ledger') or {}, 'running': running,
            'job_boxes': job_boxes}


def _PASS_JOB(suffix, batch=''):
    """The job ids of a pass: `<batch><suffix>` for wave 1, `<batch><suffix>-wN` after. With no batch,
    any batch's."""
    head = re.escape(batch + suffix) if batch else r'.*' + re.escape(suffix)
    return re.compile(head + r'(-w\d+)?$')


def ledger_pass_state(jobs, batch, suffix):
    """The desktop ledger's state for a batch's pass, across every wave of it.

    The daemon names a batch's second wave `<batch>-stageb-w2` (and `-hof5000-w2`, ...), so a bare
    `jobs.get(batch + suffix)` saw only the first wave and called an arm of wave 3 done as soon as
    wave 1 was. With several waves the most informative state wins: a running one says the pass is
    under way on the box, a queued one says this arm's turn is still to come, and only when every
    wave is finished does a missing file mean the arm is owed its pass.
    """
    pattern = re.compile(re.escape(batch + suffix) + r'(-w\d+)?$')
    states = [state for job_id, state in jobs.items() if pattern.match(job_id)]
    for wanted in ('running', 'queued'):
        if wanted in states:
            return wanted
    return states[0] if states else None


def pass_state(have_file, have_shards, candidates, in_running_job, ledger_state):
    """One word for where a pass stands, for one arm:

    | state | meaning |
    |---|---|
    | `done` | the pass's file exists |
    | `running` | shard files exist, or a running job on either box names the arm |
    | `queued` | the ledger has the batch's pass queued (or running, but not yet on this arm) |
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


def arm_record(policy, runs_dir, desktop=None, laptop_live=frozenset(), arm_boxes=None, names=None):
    """The manifest row for one arm, or None if it has no chart to show. `desktop` is
    `ledger_snapshot(runs_dir)`, `laptop_live` the policies training on this box and `arm_boxes` the
    `boxes(runs_dir)` mapping, `names` the runs directory's listing; `build` passes all four so they are
    read once per manifest rather than once per arm."""
    png = os.path.join(runs_dir, policy + '.png')
    if not os.path.exists(png):
        return None
    desktop = desktop if desktop is not None else ledger_snapshot(runs_dir)
    arm_boxes = arm_boxes if arm_boxes is not None else boxes(runs_dir)
    batch = batch_of(policy)
    record = {'policy': policy, 'batch': batch, 'knob': knob_of(policy),
              'seed': seed_of(policy), 'box': arm_boxes.get(policy),
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
    hof_rows = (hof or {}).get('rows') or []
    hof_scores = [r.get('perfect_percent', 0) for r in hof_rows]
    hof_full = [r.get('perfect_percent', 0) for r in hof_rows if not r.get('abandoned')]
    record.update({
        'hof_rows': len(hof_scores) if hof is not None else None,
        'hof_stopped': len(hof_scores) - len(hof_full) if hof is not None else None,
        'hof_mean': round(sum(hof_full) / len(hof_full), 2) if hof_full else None,
        'hof_best': max(hof_scores) if hof_scores else None,
        'hof_9873': sum(s >= 98.73 for s in hof_scores) if hof_scores else None,
        'hof_996': sum(s >= 99.6 for s in hof_scores) if hof is not None else None,
    })
    h30 = _read(os.path.join(runs_dir, policy + '_checkpoint_evals_hof30k.json'))
    h30_rows = (h30 or {}).get('rows') or []
    h30_full = [r.get('perfect_percent', 0) for r in h30_rows if not r.get('abandoned')]
    best = max(h30_rows, key=lambda r: r.get('perfect_percent', 0)) if h30_rows else None
    record.update({
        'hof30k_rows': len(h30_rows) if h30 is not None else None,
        'hof30k_stopped': len(h30_rows) - len(h30_full) if h30 is not None else None,
        'hof30k_mean': round(sum(h30_full) / len(h30_full), 2) if h30_full else None,
        'hof30k_best': best.get('perfect_percent') if best else None,
        'hof30k_best_step': best.get('step') if best else None,
        'hof30k_998': sum(r.get('perfect_percent', 0) >= 99.8 for r in h30_rows) if h30 is not None else None,
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
    # Which box a running or queued view is on, per view (None when unknown or unclaimed), so the caption
    # can say "queued on the laptop" rather than assuming the desktop.
    record['status_box'] = {'a': running['a'].get(policy) or record['box']}
    for kind, (label, suffix) in PASSES.items():
        have = {'b': stage_b, 'h': hof, 'k': h30}[kind] is not None
        record['status'][kind] = pass_state(have, bool(shard_files(runs_dir, policy, label, names)), candidates[kind],
                                            policy in running[kind],
                                            ledger_pass_state(jobs, batch, suffix))
        # A running job names its arms' box. Otherwise a queued pass runs where the arm's wave was
        # claimed, which is the box the arm trained on -- never another wave's running job, whose box
        # can differ (b27's waves 2 and 3 ran their hof5000 on the laptop and the desktop at once).
        state = record['status'][kind]
        record['status_box'][kind] = running[kind].get(policy) or (record['box'] if state in ('queued', 'pending') else None)
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
    desktop = ledger_snapshot(runs_dir)
    laptop_live = frozenset(policy for policy, _pid in live_runs.live(runs_dir, prune=False))
    arm_boxes = boxes(runs_dir)
    names = os.listdir(runs_dir) if os.path.isdir(runs_dir) else []
    arms = [rec for rec in (arm_record(p, runs_dir, desktop, laptop_live, arm_boxes, names) for p in policies) if rec]
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
