"""The mechanical half of a progress update, in one run.

    PYTHONPATH=. python -m tools.progress_update                 # sync, tables, docs skeletons, digest
    PYTHONPATH=. python -m tools.progress_update --no-sync       # offline: tables and docs from what runs/ holds
    PYTHONPATH=. python -m tools.progress_update --adopt b10     # replace a hand-written charts.md section

What it does, in order — every step something an update used to type out afresh, and got slightly
different each time:

| step | what |
|---|---|
| sync | `git fetch` both results feeds, both statuses, `ops` and `claims`; save the desktop's status with the shared queue's derived ledger to `runs/.live/desktop/status.json` for the viewer's pass states, and which box trained each arm to `runs/.live/boxes.json`; import every finished job's files that `runs/` lacks from either feed; `rsync` the live desktop batches' charts into `runs/` and their live JSON into the gitignored `runs/.live/desktop/` |
| the site | not here: the desktop daemon builds the `site` branch from both boxes' results feeds on every network cycle (`tools/site_build.py`); `ssh the-claw-den 'Snek/snek3/desktop/trigger'` builds it now |
| tables | one canonical per-batch table: knob value **read from the spec's env on `ops`**, rows, density, per-seed shares, `hof5000` candidates, best row, best30, sef, drawdown, stage-A ≥98 share, onset, plus the reference cell's row |
| charts.md | regenerates the sections it owns (marked) — table, reading slot, every panel including the reference group — and can adopt a hand-written one |
| results.md | inserts a skeleton for a batch that just closed and has no section: tables and a reading slot |
| digest | prints both boxes, what closed, what is live, each table, the top rows, the spec's predictions for the cells that just closed, and an ETA from the wave cadence |

**It writes numbers and structure, never readings.** The reading slots (`<!-- reading -->` … `<!-- /reading -->`)
are preserved across regenerations verbatim; that is where the person or model doing the update writes.
Definitions are `tools/viewer_manifest.py`'s, so the page and the docs cannot disagree.
"""

import argparse
import collections
import datetime
import glob
import json
import os
import re
import statistics
import subprocess
import sys

from env import constants
from tools import batch_state
from tools import claims
from tools import publish_pages
from tools import viewer_manifest

SNEK3 = constants.ROOT
REPO = os.path.dirname(SNEK3)
DOCS = os.path.join(SNEK3, 'docs')
BOX = 'the-claw-den'
# The box's arms write under desktop/runs/ (SNEK_RUNS_DIR, 2026-09-03), not runs/, which on the box is
# the static archive master tracks. Pulling from runs/ there would fetch nothing newer than the last commit.
BOX_RUNS = 'Snek/snek3/desktop/runs/'
PENDING = 'snek3/desktop/queue/pending/'
WAVE_ARMS = 8
DEFAULT_WAVE_SECONDS = 2.5 * 3600      # b10's measured cadence: training ~1 h 50 m + stage B ~40 m
MARK = '<!-- progress_update: batch {0} -->'
END_MARK = '<!-- /progress_update: batch {0} -->'
SHARD = re.compile(r'-s\d+of\d+\.json$|\.partial$')
READING = '<!-- reading -->'
END_READING = '<!-- /reading -->'


# ------------------------------------------------------------------------------------------ git/ssh
def git(*args, check=True):
    result = subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError('git {0}: {1}'.format(' '.join(args), result.stderr.strip()))
    return result.stdout


FEEDS = dict(claims.FEEDS)          # box -> results branch


def read_bus():
    """One fetch of everything the update reads, then `{'status', 'view'}`: the desktop's `status.json`
    off `ops-status` (both boxes' lines, since the daemon folds the laptop's in) and the shared queue's
    view (`tools.claims.gather`: specs, claims, both feeds' published ids, both statuses' running ids)."""
    git('fetch', '-q', 'origin', 'ops-status', 'ops')
    for branch in ('claims', 'laptop-status') + tuple(FEEDS.values()):
        git('fetch', '-q', 'origin', branch, check=False)
    status = json.loads(git('show', 'origin/ops-status:status.json'))
    view = claims.gather(repo=REPO, fetch=False)
    return {'status': status, 'view': view}


def feed_trees():
    """`[(feed, path)]` for every file on either results feed."""
    out = []
    for feed in FEEDS.values():
        for path in git('ls-tree', '-r', '--name-only', 'origin/' + feed, check=False).splitlines():
            out.append((feed, path))
    return out


def import_closed_waves(trees, runs_dir=None):
    """Copies every published job's files that `runs/` does not have yet -- an arm's at its cap, a pass's
    merged files, an eval spec's -- from either feed. Returns the count. A job on a feed is finished by
    construction: the scheduler publishes an arm at its cap and a pass with its merged files, nothing
    earlier. Pre-rename jobs (`p2-hof5000` is b5's) stay on `results` by design -- results.md, "Where the
    raw rows are"."""
    runs_dir = runs_dir or constants.RUNS_DIR
    copied = 0
    for feed, path in trees:
        parts = path.split('/')
        if len(parts) != 3 or parts[0] != 'results' or re.match(r'p\d+-', parts[1]) or not re.match(r'^b\d+', parts[1]):
            continue
        if SHARD.search(parts[2]):          # a shard's partial rows; the merged file is what runs/ keeps
            continue
        target = os.path.join(runs_dir, parts[2])
        if os.path.exists(target):
            continue
        with open(target, 'wb') as handle:
            handle.write(subprocess.run(['git', 'show', 'origin/{0}:{1}'.format(feed, path)], cwd=REPO,
                                        capture_output=True, check=True).stdout)
        copied += 1
    return copied


def live_batches(view):
    """Batches with anything still to do on either box -- the ones whose newest charts only a box has."""
    return batch_state.live_batches(view)


def desktop_live_batches(view):
    """Of those, the ones with a wave held by the desktop that is not closed: what `pull_live_charts` rsyncs."""
    out = []
    for name in batch_state.live_batches(view):
        if any(w['box'] == 'desktop' and w['state'] != 'closed' for w in batch_state.batch(view, name)['waves']):
            out.append(name)
    return out


def pull_live_charts(batches, runs_dir=None):
    """rsync the live batches' pictures into `runs/` and their measurements into `runs/.live/desktop/`.
    Returns (ok, message); off-LAN is not an error.

    The pictures land in `runs/` because they are committed on every update. The JSON is kept out of
    `runs/`: a live arm's `_evals.json` and `_checkpoint_evals.json` must never be committed (the box's
    deploy refuses to merge over a differing JSON), and an untracked copy in `runs/` is one `git add`
    away from that — so it goes to the gitignored side directory, where `viewer_manifest` reads it only
    when `runs/` has no close-out file for the arm."""
    if not batches:
        return True, 'no live batch'
    runs_dir = runs_dir or constants.RUNS_DIR
    live_dir = os.path.join(runs_dir, viewer_manifest.LIVE_SUBDIR)
    os.makedirs(live_dir, exist_ok=True)
    jobs = [(['.png', '.md'], runs_dir), (['_evals.json', '_checkpoint_evals.json'], live_dir)]
    pulled = 0
    for suffixes, target in jobs:
        argv = ['rsync', '-a']
        for batch in batches:
            argv += ['--include={0}*{1}'.format(batch, suffix) for suffix in suffixes]
        argv += ['--exclude=*', '-e', 'ssh -o ConnectTimeout=15 -o BatchMode=yes',
                 '{0}:{1}'.format(BOX, BOX_RUNS), target + '/']
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return False, 'rsync timed out after 120 s: live charts and measurements NOT refreshed — off-LAN?'
        if result.returncode != 0:
            return False, 'rsync failed ({0}): live charts and measurements NOT refreshed — off-LAN?'.format(
                result.returncode)
        pulled += 1
    return True, 'live charts and measurements pulled for ' + ', '.join(batches)


def save_desktop_status(status, runs_dir=None, view=None):
    """Keeps the status just fetched at `runs/.live/desktop/status.json`, where `tools.viewer_manifest`
    reads it to mark each arm's passes running, queued or still to come; its `ledger` is the shared
    queue's derived one when a `view` is given (done from the feeds, running from both statuses). Also
    writes which box trained each arm (`runs/.live/boxes.json`). Gitignored with the rest of `.live/`,
    and never in `runs/` itself, so `drop_superseded_snapshots` leaves it alone."""
    runs_dir = runs_dir or constants.RUNS_DIR
    path = os.path.join(runs_dir, viewer_manifest.DESKTOP_STATUS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(status)
    if view is not None:
        payload['ledger'] = dict(status.get('ledger') or {}, **(view.get('ledger') or {}))
        # the laptop's running jobs too, so a laptop arm's stage A reads `running` on the page
        payload['running'] = list(status.get('running') or []) + [
            {'id': job_id, 'type': 'train' if job_id in (view.get('specs') or {}) else 'eval', 'policy': job_id,
             'policies': [job_id]} for job_id, box in (view.get('running') or {}).items()
            if box == 'laptop' and job_id not in {j.get('id') for j in status.get('running') or []}]
        viewer_manifest.write_boxes(runs_dir, {job_id: box for job_id, box in (view.get('published') or {}).items()
                                               if re.match(r'^b\d+[a-z]+-', job_id) and '-stageb' not in job_id
                                               and '-hof' not in job_id})
    with open(path, 'w') as handle:
        json.dump(payload, handle)
    return path


def drop_superseded_snapshots(runs_dir=None):
    """Removes every live snapshot whose close-out file has since been imported into `runs/`."""
    runs_dir = runs_dir or constants.RUNS_DIR
    live_dir = os.path.join(runs_dir, viewer_manifest.LIVE_SUBDIR)
    dropped = 0
    for name in os.listdir(live_dir) if os.path.isdir(live_dir) else []:
        if os.path.exists(os.path.join(runs_dir, name)):
            os.remove(os.path.join(live_dir, name))
            dropped += 1
    return dropped


LOCAL_SPECS = os.path.join(SNEK3, 'logs')     # `logs/<batch>specs/*.json`, or `logs/laptop-queue/<batch>/*.json`


def read_spec(policy):
    """The arm's spec: from `ops` pending, else from a local `logs/*/<policy>.json` (a batch `scheduler`
    ran after it was dequeued from the desktop — b13, 2026-09-03). None when neither has it."""
    result = subprocess.run(['git', 'show', 'origin/ops:{0}{1}.json'.format(PENDING, policy)],
                            cwd=REPO, capture_output=True, text=True)
    text = result.stdout if result.returncode == 0 else None
    if text is None:
        for path in sorted(glob.glob(os.path.join(LOCAL_SPECS, '*', policy + '.json'))
                           + glob.glob(os.path.join(LOCAL_SPECS, '*', '*', policy + '.json'))):
            with open(path) as handle:
                text = handle.read()
            break
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def spec_envs(policies):
    """`{policy: env}` from each arm's spec (`ops`, else local); arms without a spec are absent."""
    out = {}
    for policy in policies:
        spec = read_spec(policy)
        if spec is not None:
            out[policy] = dict(spec.get('env', {}), _max_steps=spec.get('max_steps'))
    return out


def spec_notes(policy):
    spec = read_spec(policy)
    return spec.get('notes', '') if spec else ''


# ------------------------------------------------------------------------------------------ tables
def knob_key(envs):
    """The one `SNEK_*` key that varies across a batch's specs, seed aside. None if nothing does."""
    varying = collections.defaultdict(set)
    for env in envs.values():
        for key, value in env.items():
            varying[key].add(value)
    keys = [k for k, v in varying.items() if len(v) > 1 and k != 'SNEK_SEED' and not k.startswith('_')]
    if not keys:
        return None
    if len(keys) > 1 and all(len(varying[k]) <= 2 for k in keys):
        return None        # a switches batch (b19: adv-norm, huber/mse, eps, vf): the name token is the cell
    return sorted(keys, key=lambda k: -len(varying[k]))[0]


def knob_value(env, key, policy):
    """The knob's value for an arm: the spec's, or the name's middle token when there is no spec."""
    if env is not None and key and key in env:
        return env[key]
    return viewer_manifest.knob_of(policy)


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _off_knob(envs, key):
    """`{key: mode}` for every env key that varies across the batch besides the knob and the seed — an
    anneal's `_FINAL`, say. An arm off the mode on one of these is its own cell, named by its name token,
    rather than folded into the knob value it starts from (b15's `entanneal01` is not the fixed 0.01)."""
    keys = {k for env in envs.values() for k in env}
    seen = {k: [env.get(k) for env in envs.values()] for k in keys}        # absent counts as a value
    return {k: collections.Counter(v).most_common(1)[0][0] for k, v in seen.items()
            if k not in (key, 'SNEK_SEED') and not k.startswith('_') and len(set(v)) > 1}


def group_arms(arms, envs, key):
    """`[(value, [arm records])]` in knob order — numeric values first, in order, then the named cells."""
    off = _off_knob(envs, key) if key else {}
    groups = collections.OrderedDict()
    for arm in arms:
        env = envs.get(arm['policy'])
        value = knob_value(env, key, arm['policy'])
        if env is not None and any(env.get(k) != mode for k, mode in off.items()):
            value = viewer_manifest.knob_of(arm['policy'])
        groups.setdefault(value, []).append(arm)
    values = list(groups)
    numeric = sorted([v for v in values if _num(v) is not None], key=_num)
    return [(v, groups[v]) for v in numeric + [v for v in values if _num(v) is None]]


def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 2) if values else None


def group_row(value, arms, reference=False):
    rows = sum(a['rows'] or 0 for a in arms)
    with_rows = [a for a in arms if a['rows']]
    dens = [a['density98'] for a in with_rows]
    per_seed = ' '.join('{0:.1f}'.format(a['density98']) if a['rows'] else '–' for a in arms)
    pooled = (sum(a['density98'] * a['rows'] for a in with_rows) / rows) if rows else None
    best30 = [a['best30'] for a in arms if a['best30'] is not None]
    sef = [a['sef'] for a in arms if a['sef'] is not None]
    return {
        'value': value, 'reference': reference, 'n': len(arms), 'rows': rows,
        'density98': round(pooled, 1) if pooled is not None else None, 'per_seed': per_seed,
        'cands99': sum(a['cands99'] or 0 for a in arms),
        'best_row': max([a['best_row'] for a in with_rows if a['best_row'] is not None], default=None),
        'best30_mean': round(statistics.mean(best30), 2) if best30 else None,
        'best30_min': min(best30) if best30 else None, 'best30_max': max(best30) if best30 else None,
        'sef': round(statistics.mean(sef), 1) if sef else None,
        'drawdown50': _median([a['drawdown50'] for a in arms]),
        'drawdown80': _median([a['drawdown80'] for a in arms]),
        'stage_a_98': round(statistics.mean([a['stage_a_98'] for a in arms if a['stage_a_98'] is not None]), 1)
                      if any(a['stage_a_98'] is not None for a in arms) else None,
        'onset': [None if a['onset_step'] is None else round(a['onset_step'] / 1e6, 1) for a in arms],
        'arms': arms,
    }


def batch_table(batch, manifest, envs, reference=None, ref_envs=None):
    """The canonical table for one batch: a group row per knob value, the reference cell's row slotted
    in at its value, and the flat arm list."""
    arms = [a for a in manifest['arms'] if a['batch'] == batch]
    if (reference or {}).get('knob') == 'name':        # a switches batch (b19): the name token is the cell
        key = None
    else:
        key = knob_key({p: e for p, e in envs.items() if p in {a['policy'] for a in arms}})
    groups = [group_row(v, g) for v, g in group_arms(arms, envs, key)]
    if reference and reference.get('arms'):
        by = {a['policy']: a for a in manifest['arms']}
        ref_arms = [by[p] for p in reference['arms'] if p in by]
        if ref_arms:
            value = reference.get('value')
            if value is None:
                ref_env = (ref_envs or {}).get(ref_arms[0]['policy'])
                value = knob_value(ref_env, key, ref_arms[0]['policy']) if ref_env else 'reference'
            row = group_row(value, ref_arms, reference=True)
            row['label'] = reference.get('label', '')
            values = [g['value'] for g in groups]
            if _num(value) is not None:        # among the numeric cells; the named cells (anneals) follow them
                at = sum(1 for v in values if _num(v) is not None and _num(v) < _num(value))
            else:
                at = len(groups)
            groups.insert(at, row)
    return {'batch': batch, 'key': key, 'groups': groups, 'arms': arms}


def _fmt(v, suffix='', digits=None):
    if v is None:
        return '–'
    if digits is not None:
        return '{0:.{1}f}{2}'.format(v, digits, suffix)
    return '{0}{1}'.format(v, suffix)


def knob_label(key):
    return (key or 'knob').replace('SNEK_', '').replace('PPO_', '').lower()


def group_table_md(table):
    head = ('| {0} | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (mean, range) '
            '| sef | drawdown < 50% | < 80% | stage-A ≥98% |').format(knob_label(table['key']))
    lines = [head, '|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|']
    for g in table['groups']:
        name = '**{0}** (reference)'.format(g['value']) if g['reference'] else str(g['value'])
        rng = '' if g['best30_min'] is None else ' ({0}-{1})'.format(g['best30_min'], g['best30_max'])
        lines.append('| {0} | {1:,} | {2} | {3} | {4} | {5} | {6}{7} | {8} | {9} | {10} | {11} |'.format(
            name, g['rows'], _fmt(g['density98'], '%'), g['per_seed'], g['cands99'], _fmt(g['best_row']),
            _fmt(g['best30_mean'], digits=2), rng, _fmt(g['sef']), _fmt(g['drawdown50'], '%'),
            _fmt(g['drawdown80'], '%'), _fmt(g['stage_a_98'], '%')))
    return '\n'.join(lines) + '\n'


def arm_table_md(table):
    lines = ['| arm | {0} | rows | ≥98%/500 | ≥99 | best row | best30 @step | sef | drawdown < 50% |'.format(
        knob_label(table['key'])), '|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for g in table['groups']:
        if g['reference']:
            continue
        for a in g['arms']:
            step = '' if a['best30_step'] is None else ' @{0:.1f}M'.format(a['best30_step'] / 1e6)
            lines.append('| `{0}` | {1} | {2} | {3} | {4} | {5} | {6}{7} | {8} | {9} |'.format(
                a['policy'], g['value'], _fmt(a['rows']), _fmt(a['density98'], '%'), _fmt(a['cands99']),
                _fmt(a['best_row']), _fmt(a['best30']), step, _fmt(a['sef']), _fmt(a['drawdown50'], '%')))
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------------------------------ docs
def _reading(text):
    """The preserved reading block of a generated section, or an empty slot."""
    if READING in text and END_READING in text:
        return text[text.index(READING):text.index(END_READING) + len(END_READING)]
    return READING + '\n\n' + END_READING


def _adopted_reading(section):
    """A hand-written section's prose, wrapped as the reading slot. Its header, its tables, its panels
    and its group headings are dropped — the generator now writes those — and every other paragraph is
    kept verbatim, so nothing a person wrote is lost in the adoption."""
    kept = []
    for para in re.split(r'\n\s*\n', section.strip()):
        lines = [l for l in para.splitlines() if l.strip()]
        if not lines or lines[0].startswith('## ') or lines[0].startswith('<!-- '):
            continue
        if all(l.startswith('|') or l.startswith('![') for l in lines):
            continue
        if len(lines) == 1 and re.match(r'\*\*.*\*\* — .*:$', lines[0]):
            continue
        kept.append(para.strip())
    return READING + '\n' + ('\n\n'.join(kept) + '\n' if kept else '') + END_READING


def charts_section(table, title, status_line, reading, runs_dir=None):
    runs_dir = runs_dir or constants.RUNS_DIR
    out = [MARK.format(table['batch']), '## {0}'.format(title), '', status_line, '', group_table_md(table),
           reading]
    for g in table['groups']:
        head = ('**{0} {1}** — {2}:'.format(knob_label(table['key']), g['value'], g.get('label') or 'reference')
                if g['reference'] else
                '**{0} {1}** — `{2}`-`{3}`:'.format(knob_label(table['key']), g['value'],
                                                   g['arms'][0]['policy'].split('-')[0],
                                                   g['arms'][-1]['policy'].split('-')[0]))
        out += ['', head, '']
        for a in g['arms']:
            out.append('![{0}]({1}{0}.png)'.format(a['policy'], publish_pages.SITE_CHARTS_URL))
            if a['stage_b_png']:
                out.append('![{0} stage B]({1}{0}_checkpoint_evals.png)'.format(a['policy'], publish_pages.SITE_CHARTS_URL))
    out += ['', END_MARK.format(table['batch']), '']
    return '\n'.join(out)


def update_charts_md(text, table, title, status_line, adopt=False, runs_dir=None):
    """Regenerates the batch's owned section in charts.md, adopting a hand-written one on request, or
    inserting a new one above the first batch section. Returns the new text."""
    batch = table['batch']
    mark, end = MARK.format(batch), END_MARK.format(batch)
    if mark in text and end in text:
        i, j = text.index(mark), text.index(end) + len(end)
        section = charts_section(table, title, status_line, _reading(text[i:j]), runs_dir)
        return text[:i] + section.rstrip('\n') + text[j:]
    header = re.search(r'^## Batch {0}\b.*$'.format(re.escape(batch)), text, re.M)
    if header:
        if not adopt:
            return text
        i = header.start()
        after = re.search(r'^(## |<!-- progress_update: batch )', text[header.end():], re.M)
        j = header.end() + after.start() if after else len(text)
        section = charts_section(table, title, status_line, _adopted_reading(text[i:j]), runs_dir)
        return text[:i] + section + '\n' + text[j:]
    # Above the first batch section — which begins at its own start marker when it is a generated one, and
    # at its heading when hand-written. Inserting at the heading alone landed b13 *inside* b11's marker
    # pair on 2026-09-04, stacking three start markers at the top and displacing two sections' readings.
    first = re.search(r'^(## Batch |<!-- progress_update: batch )', text, re.M)
    i = first.start() if first else len(text)
    section = charts_section(table, title, status_line, _reading(''), runs_dir)
    return text[:i] + section + '\n' + text[i:]


PASS_SUFFIXES = ('_checkpoint_evals.json', '_checkpoint_evals_hof5000.json', '_checkpoint_evals_hof30k.json')


def files_closed(arms, runs_dir=None):
    """A batch with no claims on the bus -- one that closed before the shared queue, or a laptop-only batch --
    is closed when every arm has its stage-B, hof5000 and hof30k files in `runs/`. b19 closed 2026-09-05 under
    the old laptop queue and got no `results.md` skeleton because the bus knew nothing of it."""
    runs_dir = runs_dir or constants.RUNS_DIR
    return bool(arms) and all(os.path.exists(os.path.join(runs_dir, arm + suffix))
                              for arm in arms for suffix in PASS_SUFFIXES)


def results_skeleton(table, title, facts):
    return '\n'.join([MARK.format(table['batch']), '## {0}'.format(title), '', facts, '',
                      group_table_md(table), READING, '', '_Reading to be written: what the table says, '
                      'what it does not settle, what is next._', '', END_READING, '', '### Every arm', '',
                      arm_table_md(table), END_MARK.format(table['batch']), ''])


def insert_results_skeleton(text, table, title, facts):
    """Adds the skeleton above the first batch section if the batch has none. Returns (text, added)."""
    if re.search(r'^## Batch {0}\b'.format(re.escape(table['batch'])), text, re.M):
        return text, False
    first = re.search(r'^(## Batch |<!-- progress_update: batch )', text, re.M)   # above the first section's own marker, not between it and its heading (2026-09-04)
    i = first.start() if first else len(text)
    return text[:i] + results_skeleton(table, title, facts) + '\n' + text[i:], True


# ------------------------------------------------------------------------------------------ status
def batch_counts(view, batch):
    """running / queued / done counts for a batch's training arms and its stage-B waves, across both boxes."""
    return batch_state.counts(view, batch)


def wave_close_times(batch):
    """When each of the batch's stage-B waves landed on either feed, ascending."""
    times = {}
    for feed in FEEDS.values():
        out = git('log', '--format=%ct %s', 'origin/' + feed, check=False)
        for line in out.splitlines():
            ts, _, subject = line.partition(' ')
            match = re.match(r'results for ({0}-stageb\S*)'.format(re.escape(batch)), subject)
            if match:
                times[match.group(1)] = min(int(ts), times.get(match.group(1), 1 << 62))
    return sorted(times.values())


def eta(view, batch, now=None, fallback_seconds=DEFAULT_WAVE_SECONDS):
    """(remaining waves, seconds per wave, finish datetime) for a batch still training. With two boxes
    pulling, the remaining waves are what is left to claim plus what is claimed and untrained; the
    cadence is the batch's own wave-close spacing on either feed."""
    arms, _ = batch_counts(view, batch)
    remaining = -(-(arms['queued'] + arms['running']) // WAVE_ARMS)
    closes = wave_close_times(batch)
    gaps = [b - a for a, b in zip(closes, closes[1:])]
    per_wave = statistics.median(gaps) if gaps else fallback_seconds
    now = now or datetime.datetime.now()
    return remaining, per_wave, now + datetime.timedelta(seconds=remaining * per_wave)


def state_line(view, batch, status=None):
    """One line on where a batch stands across both boxes, wave by wave, with an ETA while it trains."""
    percent = {}
    for job in (status or {}).get('running') or []:
        if job.get('step') and job.get('max_steps'):
            percent[job['id']] = 100.0 * job['step'] / job['max_steps']
    text = batch_state.line(view, batch, percent=percent)
    if text.startswith('In flight'):
        arms, _ = batch_counts(view, batch)
        if arms['queued'] or arms['running']:
            remaining, per_wave, finish = eta(view, batch)
            text += '; {0} training wave(s) left at ~{1:.1f} h each -> ~{2:%Y-%m-%d %H:%M}'.format(
                remaining, per_wave / 3600, finish)
    return text


def running_arms(status):
    return [(r['id'], r.get('step') or 0, r.get('max_steps') or 0, r.get('elapsed_s') or 0)
            for r in status.get('running', [])]


# ------------------------------------------------------------------------------------------ main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--no-sync', action='store_true', help='no git fetch, import, rsync')
    parser.add_argument('--batches', default='', help='comma-separated batches to table (default: from references.json plus live)')
    parser.add_argument('--adopt', default='', help='comma-separated batches whose hand-written charts.md section to replace')
    parser.add_argument('--no-docs', action='store_true', help='print tables, touch no doc')
    args = parser.parse_args(argv)
    lines = []
    say = lines.append

    status, view = None, None
    if not args.no_sync:
        bus = read_bus()
        status, view = bus['status'], bus['view']
        save_desktop_status(status, view=view)
        copied = import_closed_waves(feed_trees())
        ok, msg = pull_live_charts(desktop_live_batches(view))
        dropped = drop_superseded_snapshots()
        say('sync: {0} finished-job files imported, {1} live snapshots superseded; {2}'.format(copied, dropped, msg))
    # The site is not built here: the desktop daemon builds the `site` branch from both boxes' results
    # feeds on every network cycle (`tools/site_build.py`); `desktop/trigger` builds it now.
    manifest = viewer_manifest.build()

    refs = viewer_manifest.references()
    charts_path, results_path = os.path.join(DOCS, 'charts.md'), os.path.join(DOCS, 'results.md')
    charts = open(charts_path).read()
    results = open(results_path).read()
    owned = set(re.findall(r'<!-- progress_update: batch (\S+) -->', charts))
    adopt = set(a for a in args.adopt.split(',') if a)
    hand_written = set(re.findall(r'^## Batch (\S+) ', charts, re.M)) - owned      # b8: prose, not a table
    batches = [b for b in args.batches.split(',') if b] or sorted(
        owned | adopt | (set(refs) - hand_written) | (set(live_batches(view)) if view else set()),
        key=lambda b: int(re.sub(r'\D', '', b) or 0))
    batches = [b for b in batches if any(a['batch'] == b for a in manifest['arms'])]

    if status:
        say('')
        say('desktop {0}: {1}'.format(status['iso'], '; '.join(status['at_a_glance']['running']) or 'idle'))
        for job, step, cap, elapsed in running_arms(status):
            if cap:
                say('  {0} {1:.1f}M/{2:.1f}M {3:.0f}% {4:.0f} min'.format(job, step / 1e6, cap / 1e6,
                                                                      100.0 * step / cap, elapsed / 60))
            else:
                say('  {0} {1:.0f} min'.format(job, elapsed / 60))
        glance = status['at_a_glance']
        say('laptop {0}: {1}'.format(glance.get('laptop_iso'), '; '.join(glance.get('laptop_running') or []) or 'idle'))
        for line in glance.get('pool') or []:
            say('  pool: ' + line)
        if glance.get('attention'):
            say('  ATTENTION: {0}'.format(glance['attention']))

    for batch in batches:
        arms = [a['policy'] for a in manifest['arms'] if a['batch'] == batch]
        ref = refs.get(batch, {})
        envs = spec_envs(arms + list(ref.get('arms', [])))
        table = batch_table(batch, manifest, envs, ref, envs)
        say('')
        say('=== {0}: {1} arms, knob {2}'.format(batch, len(arms), table['key']))
        where = state_line(view, batch, status) if view else 'Bus not read (--no-sync)'
        say(where)
        say(group_table_md(table))
        top = sorted(((r['perfect_percent'], a['policy'], r['step']) for a in table['arms']
                      for r in _stage_b_rows(a['policy'])), reverse=True)[:5]
        if top:
            say('top rows: ' + ', '.join('{0} {1} @{2}'.format(*t) for t in top))
        if not args.no_docs:
            closed = where.startswith('Closed') or (not where.startswith('In flight') and files_closed(arms))
            caps = {e.get('_max_steps') for p, e in envs.items() if p in arms} - {None}
            cap = '{0:.0f}M'.format(max(caps) / 1e6) if caps else '50M'
            title = '{0} — the `{1}` sweep, {2} values x {3} seeds, {4}'.format(
                batch, knob_label(table['key']), sum(1 for g in table['groups'] if not g['reference']),
                max((g['n'] for g in table['groups'] if not g['reference']), default=0), cap)
            if view and not closed:
                arm_states = batch_counts(view, batch)[0]
                if sum(arm_states.values()):
                    title += ' ({0} of {1} arms in)'.format(arm_states['done'], sum(arm_states.values()))
            status_line = ('**{0}** as of {1}. Reference group marked. Regenerated by `tools/progress_update.py`; '
                           'only the reading block is hand-written.').format(
                               where, datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
            charts = update_charts_md(charts, table, 'Batch ' + title, status_line, adopt=batch in adopt)
            if closed:
                facts = ('Closed on both boxes\' feeds; every arm has its stage-B measurement. One knob off the reference cell '
                         '(`{0}`, marked in the table). Numbers by `tools/progress_update.py`.').format(
                             ', '.join(ref.get('arms', [])) or 'none named')
                results, added = insert_results_skeleton(results, table, 'Batch ' + title + ', closed {0}'.format(
                    datetime.date.today().isoformat()), facts)
                if added:
                    say('results.md: skeleton inserted for {0} — write its reading'.format(batch))
            for g in table['groups']:
                if not g['reference'] and g['rows']:
                    note = spec_notes(g['arms'][0]['policy'])
                    i = note.find('Prediction:')
                    if i >= 0:
                        say('  prediction for {0}: {1}'.format(g['value'], note[i + 11:].split(' Batch note')[0].strip()[:160]))
    if not args.no_docs:
        open(charts_path, 'w').write(charts)
        open(results_path, 'w').write(results)
        say('')
        say('docs: charts.md sections regenerated for {0}'.format(', '.join(batches)))
    print('\n'.join(lines))
    return 0


def _stage_b_rows(policy):
    path = os.path.join(constants.RUNS_DIR, policy + '_checkpoint_evals.json')
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return json.load(handle).get('rows') or []


if __name__ == '__main__':
    raise SystemExit(main())
