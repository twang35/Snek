"""The shared queue: both boxes pull waves from `ops`, and a wave is claimed by pushing a small file to
the `claims` branch. Git's fast-forward check is the lock.

    PYTHONPATH=. python -m tools.claims show [--json]           # the pool: unclaimed work, each box's holdings
    PYTHONPATH=. python -m tools.claims release b21-w3          # a stranded wave back to the pool (a human decision)
    PYTHONPATH=. python -m tools.claims seed b17 desktop 6      # migration: waves 1-6 of b17 as the desktop's
    PYTHONPATH=. python -m tools.claims claim --box laptop --batch b21 --wave 4 --arms b21x-... b21y-...

**The mechanism.** A `git push` of a fast-forward commit is an atomic compare-and-swap on a ref: the
server takes it only if the branch is still where the pusher last saw it. So a box that wants work
fetches `ops` (the specs) and `claims` (who holds what), computes the next free wave, commits
`claims/<batch>/w<N>.json` naming itself and the wave's arms, and pushes. Two boxes racing from the
same parent commit both push; the server accepts one and rejects the other as non-fast-forward; the
loser resets to the new head, re-reads, and finds the wave taken -- so it claims the next one. No
coordinator, no lease, no tie-break, and it works from anywhere the git remote does. The design and the
alternatives it beat are `plans/archive/shared-queue.md`.

**What is claimed is a wave and its chain**: the arms of a wave and their stage-B, hof5000 and hof30k
passes are one unit, because the passes read the arms' checkpoints from the box's own `savedPolicies/`.
A batch may therefore span both boxes, wave by wave, and **wave numbers are global per batch**: N is
one more than the batch's highest existing claim, whichever box made it, so `b21-stageb-w2` means one
wave on both feeds and on the site. A released number is never reused. An **eval spec** (a hand pass) is
claimed whole, and only by a box that holds every one of its policies' checkpoints and none of whose
arms is still an unclaimed training spec.

**The `box` field on a spec pins it**: `"box": "desktop"` or `"laptop"`; absent means either. The box's
own name is `SNEK_BOX` (the daemon sets `desktop` from `host.env`; the laptop's default is `laptop`).

**The order is the scheduler's, over one queue**: batches by the lowest `priority` any of their specs
carries, then by name; within a batch by `priority`, then id. A wave is the next `wave_size` unclaimed,
eligible training specs of the first batch that has any.

**Claims are made just before launch, never ahead**, so the pool in the status is what is really free
and a claim held by a box that then goes quiet (a laptop lid) is at most the wave it was about to run.
There is **no expiry**: an expiry that fires while a box is merely slow is two boxes training the same
arms. A stranded claim is a line under `attention` and a human's `release`, which rewrites a wave's file
as a `released` tombstone: its arms go back to the pool, its number stays taken.

**Reading is from the refs, writing through a worktree** outside the checkout (`~/.snek3-laptop/claims`
on the laptop, beside `snek-bus/status` on the desktop; `SNEK_CLAIMS_WORKTREE`), made by
`gitbus.ensure_worktree` on first use like the status and results ones. Nothing here imports the
scheduler; the scheduler imports this.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

from desktop.daemon import gitbus
from desktop.daemon import launch
from desktop.daemon.daemon import batch_of, pass_of
from desktop.daemon.job import parse_job, JobError, BOXES
from env import constants

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
REMOTE = 'origin'
OPS_BRANCH = os.environ.get('SNEK_OPS_BRANCH', 'ops')
QUEUE_DIR = 'snek3/desktop/queue/pending'
BRANCH = os.environ.get('SNEK_CLAIMS_BRANCH', 'claims')
WORKTREE = os.environ.get('SNEK_CLAIMS_WORKTREE', os.path.expanduser('~/.snek3-laptop/claims'))
STATUS_BRANCHES = {'desktop': 'ops-status', 'laptop': 'laptop-status'}
FEEDS = {'desktop': 'results', 'laptop': 'laptop-results'}
DIR = 'claims'
# The record kinds that carry a wave number: a live claim, and the tombstone `release` leaves in its
# place so the number is never handed out again (decision 6 of the plan: `b21-stageb-w2` on a feed must
# mean one wave, and a wave retrained elsewhere after a release is `w4`, never a second `w2`).
WAVE_KINDS = ('wave', 'released')
# A spec with no `priority`, as `desktop/daemon/job.py` defaults it: lower runs first.
DEFAULT_PRIORITY = 100
# How many times a claim is recomputed after losing the race before this round gives up. Each loss
# means the other box took something, so a handful covers any realistic burst.
CLAIM_ATTEMPTS = 6
# A box whose status is older than this while it holds unfinished work is worth a line under attention.
STRANDED_AFTER_SECONDS = 2 * 3600


def _log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def box_name():
    """This box's name on the shared queue: `SNEK_BOX`, `laptop` when unset."""
    return os.environ.get('SNEK_BOX') or 'laptop'


# ---------------------------------------------------------------- the specs' order

def spec_priority(spec):
    """The spec's `priority`, lower first; `DEFAULT_PRIORITY` for a spec that names none or a bad one."""
    try:
        return int(spec.get('priority'))
    except (TypeError, ValueError):
        return DEFAULT_PRIORITY


def spec_order(spec):
    """Sort key: priority, then id. What orders the specs of a batch and the queued lines of the status."""
    return (spec_priority(spec), spec['id'])


def batch_order(specs):
    """The batches of `specs` (a dict id -> spec) in run order: lowest priority of any spec, then name."""
    batches = {}
    for spec in specs.values():
        batches.setdefault(batch_of(spec['id']), []).append(spec)
    return sorted(batches, key=lambda name: (min(spec_priority(spec) for spec in batches[name]), name))


# ---------------------------------------------------------------- reading ops

def read_specs(repo=REPO, remote=REMOTE, branch=OPS_BRANCH, queue_dir=QUEUE_DIR):
    """`(specs, malformed)`: every work spec on the fetched `ops` ref as the scheduler's dict (`id ->
    spec`, via `parse_job` and `launch.materialise`, so smoke and benchmark are already `train`), and
    `[(filename, error)]` for what did not parse. Actions (deploy, restart) are the daemon's and are
    skipped here."""
    host = {'GIT_REMOTE': remote, 'OPS_BRANCH': branch, 'QUEUE_DIR': queue_dir, 'REPO_PATH': repo}
    specs, malformed = {}, []
    for name, text in gitbus.read_pending_jobs(host):
        try:
            job = parse_job(text, name)
            spec = launch.materialise(job)
        except (JobError, ValueError) as error:
            malformed.append((name, str(error)))
            continue
        if spec is not None:
            specs[job.id] = spec
    return specs, malformed


# ---------------------------------------------------------------- the records

def wave_id(batch, number):
    return '{0}-w{1}'.format(batch, number)


def record_path(record):
    """Where a record lives on the branch: `claims/<batch>/w<N>.json`, or `claims/<batch>/eval-<id>.json`."""
    if record.get('kind', 'wave') in WAVE_KINDS:
        return '{0}/{1}/w{2}.json'.format(DIR, record['batch'], int(record['wave']))
    return '{0}/{1}/eval-{2}.json'.format(DIR, record['batch'], record['id'])


def record_id(record):
    """`b21-w3` for a wave, the spec's id for an eval: what `release` takes and the status names."""
    if record.get('kind', 'wave') in WAVE_KINDS:
        return wave_id(record['batch'], int(record['wave']))
    return record['id']


def wave_record(batch, number, box, arms, now=None):
    now = time.time() if now is None else now
    return {'kind': 'wave', 'batch': batch, 'wave': int(number), 'box': box, 'arms': list(arms),
            'claimed_iso': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(now)), 'claimed_ts': now}


def eval_record(batch, eval_id, box, policies, now=None):
    now = time.time() if now is None else now
    return {'kind': 'eval', 'batch': batch, 'id': eval_id, 'box': box, 'arms': list(policies),
            'claimed_iso': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(now)), 'claimed_ts': now}


def describe(record):
    if record.get('kind', 'wave') == 'released':
        return '{0} released (was {1}\'s)'.format(record_id(record), record['box'])
    if record.get('kind', 'wave') == 'wave':
        return '{0} ({1} arms: {2}) for {3}'.format(record_id(record), len(record['arms']),
                                                    ', '.join(record['arms']), record['box'])
    return 'eval {0} ({1} arms) for {2}'.format(record['id'], len(record['arms']), record['box'])


def read_records(worktree):
    """Every claim checked out in `worktree`, oldest wave first within a batch. A file that is not a
    claim is skipped with a line on stderr rather than raised: one bad commit must not stop both boxes."""
    root = os.path.join(worktree, DIR)
    records = []
    if not os.path.isdir(root):
        return records
    for batch in sorted(os.listdir(root)):
        folder = os.path.join(root, batch)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith('.json'):
                continue
            path = os.path.join(folder, name)
            try:
                with open(path) as handle:
                    record = json.load(handle)
                if not isinstance(record, dict) or record.get('box') not in BOXES or not record.get('batch'):
                    raise ValueError('not a claim record')
                record.setdefault('kind', 'wave')
                if record['kind'] in WAVE_KINDS:
                    record['wave'] = int(record['wave'])
                record['arms'] = [str(arm) for arm in record.get('arms') or []]
            except (OSError, ValueError, KeyError, TypeError) as error:
                sys.stderr.write('claims: skipping {0}: {1}\n'.format(path, error))
                continue
            record['_path'] = os.path.relpath(path, worktree)
            records.append(record)
    records.sort(key=lambda r: (r['batch'], r['kind'] != 'wave', r.get('wave', 0), r.get('id', '')))
    return records


# ---------------------------------------------------------------- the pure part

def claimed_ids(records):
    """Every spec id some claim covers: a wave's arms, an eval claim's spec id. A released wave's
    tombstone covers nothing -- its arms are back in the pool."""
    taken = set()
    for record in records:
        kind = record.get('kind', 'wave')
        if kind == 'wave':
            taken.update(record['arms'])
        elif kind == 'eval':
            taken.add(record['id'])
    return taken


def eligible(spec, box):
    """Whether `box` may claim `spec`: unpinned, or pinned to this box."""
    pin = spec.get('box')
    return not pin or pin == box


def mine(records, box):
    return [record for record in records if record.get('box') == box]


def default_has_checkpoints(policies, policy_dir=None):
    """Whether every policy has a checkpoint directory on this box -- what an eval spec needs here."""
    root = policy_dir or constants.POLICY_DIR
    return all(os.path.isdir(os.path.join(root, policy)) for policy in policies)


def next_claim(specs, records, box, wave_size, has_checkpoints=default_has_checkpoints, now=None):
    """The record `box` should claim next, or None when nothing in the pool is its to take.

    `specs` is `id -> spec` off `ops`; `records` every existing claim. Batches in run order; within the
    first batch that has anything eligible and unclaimed, its specs in `spec_order`: a training spec
    starts a wave of the batch's next `wave_size` unclaimed eligible training specs, numbered one past
    the batch's highest claim; an eval spec is taken whole if none of its policies is still an unclaimed
    training spec and this box has all their checkpoints, else passed over for the next candidate.
    Pure: no git, no filesystem but the `has_checkpoints` probe.
    """
    taken = claimed_ids(records)
    pool = [spec for spec in specs.values() if spec['id'] not in taken]
    if not pool:
        return None
    unclaimed_train = {spec['id'] for spec in pool if spec.get('type', 'train') == 'train'}
    for batch in batch_order(specs):
        candidates = sorted((spec for spec in pool if batch_of(spec['id']) == batch and eligible(spec, box)),
                            key=spec_order)
        for spec in candidates:
            if spec.get('type', 'train') == 'train':
                arms = [s['id'] for s in candidates if s.get('type', 'train') == 'train'][:max(1, int(wave_size))]
                number = 1 + max([r['wave'] for r in records
                                  if r.get('kind', 'wave') in WAVE_KINDS and r['batch'] == batch] or [0])
                return wave_record(batch, number, box, arms, now=now)
            policies = list(spec.get('policies') or [spec.get('policy')])
            if any(policy in unclaimed_train for policy in policies):
                continue                    # its arms are still to be trained; the wave comes first
            if not has_checkpoints(policies):
                continue                    # on the other box, or pruned: someone rsyncs, or the other box takes it
            return eval_record(batch, spec['id'], box, policies, now=now)
    return None


def ledger(specs, records, published, running):
    """`id -> state` for the tools that read one (`tools/viewer_manifest.py`): `done` when the id is on a
    feed, `running` when a box's status lists it, `queued` for anything else on `ops` or claimed. Pass
    ids (`b21-stageb-w2`) come from the feeds and the statuses, since no spec names them."""
    view = {}
    for spec_id in specs:
        view[spec_id] = 'queued'
    for record in records:
        for arm in record['arms']:
            view.setdefault(arm, 'queued')
        if record.get('kind', 'wave') == 'eval':
            view.setdefault(record['id'], 'queued')
    for job_id in published:
        view[job_id] = 'done'
    for job_id in running:
        view[job_id] = 'running'
    return view


POOL_LINE_MAX = 150      # a pool line with its description; the user reads status.json in a terminal


def batch_description(batch_specs):
    """What a batch's training specs say they are, from their `label`s: the text every label shares,
    without the `b30: ` prefix and cut back to the last comma or ` -- ` so it never ends mid-phrase.
    `sweep_specs` writes `<batch>: <what varies>, seed N of M -- wave ...`, so the shared part is the
    what-varies clause. '' when no spec carries a label."""
    labels = [str(spec.get('label') or '').strip() for spec in batch_specs]
    labels = [label for label in labels if label]
    if not labels:
        return ''
    shared = os.path.commonprefix(labels)
    if len(labels) > 1 or shared != labels[0]:
        cut = max(shared.rfind(', '), shared.rfind(' -- '))
        if cut <= 0:
            cut = shared.rfind(' ')     # labels that part mid-word (`hist4` / `hist8`): no fragment
        if cut > 0:
            shared = shared[:cut]
    shared = re.sub(r'^b\d+[a-z]*\s*[:-]\s*', '', shared).strip(' ,-')
    return shared


def with_description(line, description, limit=POOL_LINE_MAX):
    """`line | description`, the description shortened with `...` so the whole stays under `limit`."""
    if not description:
        return line
    room = limit - len(line) - len(' | ')
    if room < 8:
        return line
    if len(description) > room:
        description = description[:room - 3].rstrip(' ,-') + '...'
    return line + ' | ' + description


def pool_view(specs, records, published=frozenset(), running=None, status_ages=None, malformed=(),
              stranded_after=STRANDED_AFTER_SECONDS):
    """The pool as the status shows it.

    `published` is every job id on either feed (an arm at its cap, a pass with its merged files);
    `running` maps a job id to the box whose status lists it; `status_ages` maps a box to the age in
    seconds of its last status, or None when it has never published. Returns `{'unclaimed': [...],
    'held': {box: [...]}, 'lines': [...], 'attention': [...], 'ledger': {...}}`.
    """
    running = running or {}
    taken = claimed_ids(records)
    unclaimed = []
    for batch in batch_order(specs):
        group = [spec for spec in specs.values() if batch_of(spec['id']) == batch and spec['id'] not in taken]
        if not group:
            continue
        for phase, kind in (('training', 'train'), ('eval', 'eval')):
            these = sorted((spec for spec in group if spec.get('type', 'train') == kind), key=spec_order)
            if not these:
                continue
            pins = {}
            for spec in these:
                if spec.get('box'):
                    pins[spec['box']] = pins.get(spec['box'], 0) + 1
            unclaimed.append({'batch': batch, 'phase': phase, 'ids': [spec['id'] for spec in these],
                             'pins': pins,
                             'priority': min(spec_priority(spec) for spec in these)})
    held = {}
    for record in records:
        if record.get('kind', 'wave') == 'released':
            continue                    # a tombstone holds nothing; it only keeps its number taken
        arms = record['arms']
        if record.get('kind', 'wave') == 'wave':
            done = bool(arms) and all(arm in published for arm in arms) and all(
                '{0}-{1}{2}'.format(record['batch'], pass_name, '' if record['wave'] == 1 else '-w{0}'.format(record['wave']))
                in published for pass_name in ('stageb', 'hof5000', 'hof30k'))
        else:
            done = record['id'] in published
        held.setdefault(record['box'], []).append({
            'id': record_id(record), 'batch': record['batch'], 'kind': record.get('kind', 'wave'),
            'wave': record.get('wave'), 'arms': arms, 'done': done, 'claimed_iso': record.get('claimed_iso'),
            'running': any(arm in running for arm in arms) or record_id(record) in running})
    lines = []
    for entry in unclaimed:
        count = len(entry['ids'])
        pin = ''
        if entry['pins']:
            pin = ' | pinned ' + ', '.join('{0} {1}'.format(box, n) if n != count else box
                                            for box, n in sorted(entry['pins'].items()))
        # A line that names a batch and no box is unclaimed work by construction, so the word is dropped;
        # the count is out of the batch's training specs on ops, claimed or not (`16/24 arms`).
        if entry['phase'] == 'training':
            batch_specs = [spec for spec in specs.values()
                           if batch_of(spec['id']) == entry['batch'] and spec.get('type', 'train') == 'train']
            line = '{0} training | {1}/{2} arms{3}'.format(entry['batch'], count, len(batch_specs), pin)
            lines.append(with_description(line, batch_description(batch_specs)))
        else:
            lines.append('{0} eval | {1}{2}'.format(entry['batch'], ', '.join(entry['ids']), pin))
    for box in sorted(held):
        open_ones = [h for h in held[box] if not h['done']]
        if open_ones:
            # No running tag: whether a holding's wave or pass is live is the box's own running line,
            # a row up in the same block (user, 2026-09-06). `held[...]['running']` stays for the tools.
            lines.append('{0} holds {1}'.format(box, ', '.join(
                '{0} ({1} arm{2})'.format(h['id'], len(h['arms']), '' if len(h['arms']) == 1 else 's')
                for h in open_ones)))
    attention = []
    ages = status_ages or {}
    for box in sorted(held):
        open_ones = [h for h in held[box] if not h['done']]
        age = ages.get(box)
        if open_ones and age is not None and age >= stranded_after:
            attention.append('** {0} holds {1} but its status is {2:.1f}h old; if it is not coming back, '
                             '`python -m tools.claims release <id>` returns the work to the pool'.format(
                                 box, ', '.join(h['id'] for h in open_ones), age / 3600.0))
    for name, error in malformed:
        attention.append('** spec {0} on ops is malformed and is not run: {1}'.format(name, error))
    return {'unclaimed': unclaimed, 'held': held, 'lines': lines, 'attention': attention,
            'ledger': ledger(specs, records, published, running)}


# ---------------------------------------------------------------- the mirror

def mirror(queue_dir, specs, records, log=_log):
    """Writes this box's claims into the scheduler's queue directory -- `<queue>/<batch>/<id>.json` for
    every spec of every wave and eval it holds, each spec carrying `_wave` from its claim -- and removes
    any spec file there that no claim of ours covers. Markers (`.done-`, `.failed-`) are left alone.
    Returns the set of ids mirrored. `records` are already this box's (`mine`)."""
    wanted = {}
    for record in records:
        if record.get('kind', 'wave') == 'released':
            continue
        for arm in record['arms'] if record.get('kind', 'wave') == 'wave' else [record['id']]:
            spec = specs.get(arm)
            if spec is None:
                log('claims: {0} is held ({1}) but no longer on ops; not mirrored'.format(arm, record_id(record)))
                continue
            spec = dict(spec)
            if record.get('kind', 'wave') == 'wave':
                spec['_wave'] = record['wave']
            wanted[arm] = (record['batch'], spec)
    os.makedirs(queue_dir, exist_ok=True)
    present = {}
    for batch in sorted(os.listdir(queue_dir)):
        folder = os.path.join(queue_dir, batch)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.endswith('.json') and not name.startswith('.'):
                present[name[:-5]] = os.path.join(folder, name)
    for spec_id, (batch, spec) in wanted.items():
        folder = os.path.join(queue_dir, batch)
        path = os.path.join(folder, spec_id + '.json')
        text = json.dumps(spec, indent=1, sort_keys=True)
        os.makedirs(folder, exist_ok=True)
        current = present.pop(spec_id, None)
        if current and current != path:
            os.remove(current)
        try:
            with open(path) as handle:
                unchanged = handle.read() == text
        except OSError:
            unchanged = False
        if not unchanged:
            with open(path + '.partial', 'w') as handle:
                handle.write(text)
            os.replace(path + '.partial', path)
    for spec_id, path in present.items():
        try:
            os.remove(path)
            log('claims: {0} is not held here any more; removed from the queue'.format(spec_id))
        except OSError:
            pass
    return set(wanted)


# ---------------------------------------------------------------- the store

class Store(object):
    """The `claims` branch as this box writes it: a worktree outside the checkout, reset to the remote
    before every read, and a fast-forward push for every write. `try_claim` and `try_release` return
    `'won'`, `'lost'` (someone pushed first: reset and recompute) or `'failed'` (offline, auth: the
    local commit is dropped too, so the worktree never drifts from the remote). Every git call goes
    through `git`, injectable for tests."""

    def __init__(self, repo=REPO, remote=REMOTE, branch=BRANCH, worktree=WORKTREE, log=_log):
        self.repo, self.remote, self.branch, self.worktree, self.log = repo, remote, branch, worktree, log
        self._ready = False
        self.records = []           # as of the last `sync`
        self.head = None

    def _git(self, args, cwd=None, check=False):
        return gitbus._git(args, cwd=cwd or self.worktree, check=check)

    def ensure(self):
        if not self._ready:
            gitbus.ensure_worktree(self.repo, self.worktree, self.branch, self.remote)
            self._ready = True
        return self.worktree

    def remote_ref(self):
        return '{0}/{1}'.format(self.remote, self.branch)

    def sync(self):
        """Fetches the branch and resets the worktree to it. Returns the remote head, or None when the
        branch has never been pushed (the worktree then holds its empty root)."""
        self.ensure()
        gitbus.fetch_branch(self.repo, self.remote, self.branch)
        self.head = gitbus.ref_head(self.repo, self.remote_ref())
        if self.head:
            self._git(['reset', '-q', '--hard', self.remote_ref()], check=True)
        self.records = read_records(self.worktree)
        return self.head

    def _commit_and_push(self, message):
        self._git(['add', '-A', DIR])
        if not self._git(['status', '--porcelain']).strip():
            return 'won'                # nothing to write: the branch already says so
        self._git(['commit', '-q', '-m', message], check=True)
        outcome = gitbus.push_fast_forward(self.worktree, self.branch, self.remote)
        if outcome == 'pushed':
            return 'won'
        # Lost, or could not push: drop the local commit so the worktree tracks the remote again.
        gitbus.fetch_branch(self.repo, self.remote, self.branch)
        head = gitbus.ref_head(self.repo, self.remote_ref())
        if head:
            self._git(['reset', '-q', '--hard', self.remote_ref()])
        else:
            self._git(['reset', '-q', '--hard', 'HEAD~1'])
        return 'lost' if outcome == 'rejected' else 'failed'

    def try_claim(self, record):
        """Writes `record` and pushes. The caller has just `sync`ed and computed the record from what it read."""
        self.ensure()
        path = os.path.join(self.worktree, record_path(record))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as handle:
            json.dump({key: value for key, value in record.items() if not key.startswith('_')},
                      handle, indent=1, sort_keys=True)
            handle.write('\n')
        return self._commit_and_push('claim {0}'.format(record_id(record)))

    def try_release(self, claim_id):
        """Returns the claim named `claim_id` (`b21-w3`, or an eval spec's id) to the pool and pushes.

        A wave's file is rewritten as a `released` tombstone -- no arms, so `claimed_ids` frees them,
        but the number stays in the batch's count so the next claim is `w<N+1>`, never a second `w<N>`
        (the pass ids on the feeds would otherwise collide). An eval claim is simply deleted."""
        self.ensure()
        match = [record for record in self.records if record_id(record) == claim_id
                 and record.get('kind', 'wave') != 'released']
        if not match:
            return 'missing'
        record = match[0]
        path = os.path.join(self.worktree, record['_path'])
        try:
            if record.get('kind', 'wave') == 'wave':
                now = time.time()
                tombstone = {'kind': 'released', 'batch': record['batch'], 'wave': int(record['wave']),
                             'box': record['box'], 'arms': [], 'claimed_iso': record.get('claimed_iso'),
                             'released_iso': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(now)),
                             'released_ts': now}
                with open(path, 'w') as handle:
                    json.dump(tombstone, handle, indent=1, sort_keys=True)
                    handle.write('\n')
            else:
                os.remove(path)
        except OSError:
            return 'missing'
        outcome = self._commit_and_push('release {0}'.format(claim_id))
        self.records = read_records(self.worktree)     # won: the tombstone; lost or failed: the remote's
        return outcome


def claim_next(store, box, wave_size, has_checkpoints=default_has_checkpoints, read_specs=read_specs,
               attempts=CLAIM_ATTEMPTS, log=_log):
    """Claims the next wave or eval spec for `box`, racing the other box through the branch. Returns the
    record won, or None when the pool has nothing for this box (or the remote cannot be reached)."""
    for attempt in range(attempts):
        store.sync()
        specs, malformed = read_specs()
        for name, error in malformed:
            log('claims: spec {0} on ops is malformed and is skipped: {1}'.format(name, error))
        record = next_claim(specs, store.records, box, wave_size, has_checkpoints)
        if record is None:
            return None
        outcome = store.try_claim(record)
        if outcome == 'won':
            store.records = read_records(store.worktree)
            return record
        if outcome == 'failed':
            log('claims: could not push a claim for {0}; nothing claimed this round'.format(describe(record)))
            return None
        log('claims: lost the race for {0}; recomputing (attempt {1} of {2})'.format(
            record_id(record), attempt + 1, attempts))
    log('claims: lost {0} races in a row; nothing claimed this round'.format(attempts))
    return None


# ---------------------------------------------------------------- the feeds and statuses

def published_ids(repo=REPO, remote=REMOTE, feeds=FEEDS, fetch=True):
    """Every job id on either results feed: `results/<job-id>/` directories. `fetch` pulls the feeds first."""
    ids = {}
    for box, feed in feeds.items():
        if fetch:
            gitbus.fetch_branch(repo, remote, feed)
        ref = '{0}/{1}'.format(remote, feed)
        if not gitbus.ref_head(repo, ref):
            continue
        listing = gitbus._git(['ls-tree', '--name-only', ref, 'results/'], cwd=repo)
        for line in listing.splitlines():
            name = line.strip().split('/')[-1]
            if name:
                ids[name] = box
    return ids


def running_ids(repo=REPO, remote=REMOTE, branches=STATUS_BRANCHES, fetch=True, now=None):
    """`(running, ages)`: every job id a box's published status lists as running, mapped to the box, and
    each box's status age in seconds (None when it has never published). The desktop's status is the
    daemon's (`ops-status`), whose own running list is the scheduler's; the laptop's is `laptop-status`."""
    now = time.time() if now is None else now
    running, ages = {}, {}
    for box, branch in branches.items():
        if fetch:
            gitbus.fetch_branch(repo, remote, branch)
        text = gitbus._git(['show', '{0}/{1}:status.json'.format(remote, branch)], cwd=repo)
        try:
            status = json.loads(text or '')
        except ValueError:
            status = {}
        if not isinstance(status, dict):
            status = {}
        ts = status.get('ts')
        ages[box] = (now - float(ts)) if ts else None
        for job in status.get('running') or []:
            if job.get('id'):
                running[job['id']] = box
    return running, ages


def gather(repo=REPO, remote=REMOTE, store=None, fetch=True, now=None):
    """Everything the pool view needs, read once: specs and malformed off `ops`, the claims, the feeds'
    published ids, both statuses' running ids and ages. Returns the `pool_view` dict plus `specs`,
    `records`, `published` and `running` for callers that want the raw parts."""
    store = store or Store(repo=repo, remote=remote)
    if fetch:
        gitbus.fetch_branch(repo, remote, OPS_BRANCH)
    store.sync() if fetch else (store.ensure(), setattr(store, 'records', read_records(store.worktree)))
    specs, malformed = read_specs(repo=repo, remote=remote)
    published = published_ids(repo=repo, remote=remote, fetch=fetch)
    running, ages = running_ids(repo=repo, remote=remote, fetch=fetch, now=now)
    view = pool_view(specs, store.records, published=set(published), running=running, status_ages=ages,
                     malformed=malformed)
    view.update({'specs': specs, 'records': store.records, 'published': published, 'running': running,
                 'status_ages': ages, 'heads': {'ops': gitbus.ref_head(repo, '{0}/{1}'.format(remote, OPS_BRANCH)),
                                                'claims': store.head}})
    return view


# ---------------------------------------------------------------- the command line

def _seed(store, batch, box, waves, wave_size):
    """Migration: claims the first `waves` waves of `batch` for `box`, cut as the scheduler cut them
    (`spec_order`, `wave_size` at a time), so pass ids already on the feeds keep meaning the same wave."""
    store.sync()
    specs, _ = read_specs(repo=store.repo, remote=store.remote)
    arms = sorted((spec for spec in specs.values() if batch_of(spec['id']) == batch
                   and spec.get('type', 'train') == 'train'), key=spec_order)
    if not arms:
        print('no training specs for {0} on ops'.format(batch))
        return 1
    existing = {r['wave'] for r in store.records if r.get('kind', 'wave') in WAVE_KINDS and r['batch'] == batch}
    for number in range(1, int(waves) + 1):
        chunk = arms[(number - 1) * wave_size: number * wave_size]
        if not chunk:
            break
        if number in existing:
            print('{0} already claimed; skipped'.format(wave_id(batch, number)))
            continue
        record = wave_record(batch, number, box, [spec['id'] for spec in chunk])
        outcome = store.try_claim(record)
        print('{0}: {1}'.format(describe(record), outcome))
        if outcome != 'won':
            return 1
        store.records = read_records(store.worktree)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command')
    show = sub.add_parser('show', help='the pool: unclaimed work and each box\'s holdings')
    show.add_argument('--json', action='store_true', help='the whole view as JSON (what the daemon reads)')
    show.add_argument('--no-fetch', action='store_true', help='read the refs as they are')
    release = sub.add_parser('release', help='return a claim to the pool: b21-w3, or an eval spec\'s id')
    release.add_argument('claim_id')
    seed = sub.add_parser('seed', help='migration: claim the first N waves of a batch for a box, as they ran')
    seed.add_argument('batch')
    seed.add_argument('box', choices=BOXES)
    seed.add_argument('waves', type=int)
    seed.add_argument('--wave-size', type=int, default=8)
    claim = sub.add_parser('claim', help='write one explicit wave claim')
    claim.add_argument('--box', required=True, choices=BOXES)
    claim.add_argument('--batch', required=True)
    claim.add_argument('--wave', required=True, type=int)
    claim.add_argument('--arms', required=True, nargs='+')
    args = parser.parse_args(argv)
    store = Store()
    if args.command == 'show':
        view = gather(store=store, fetch=not args.no_fetch)
        if args.json:
            print(json.dumps({key: view[key] for key in ('unclaimed', 'held', 'lines', 'attention', 'ledger',
                                                          'heads', 'status_ages')}, indent=1, sort_keys=True))
            return 0
        for line in view['lines'] or ['(the pool is empty and nothing is held)']:
            print(line)
        for line in view['attention']:
            print(line)
        return 0
    if args.command == 'release':
        store.sync()
        for attempt in range(CLAIM_ATTEMPTS):
            outcome = store.try_release(args.claim_id)
            print('release {0}: {1}'.format(args.claim_id, outcome))
            if outcome != 'lost':
                return 0 if outcome == 'won' else 1
            store.sync()
        return 1
    if args.command == 'seed':
        return _seed(store, args.batch, args.box, args.waves, args.wave_size)
    if args.command == 'claim':
        store.sync()
        record = wave_record(args.batch, args.wave, args.box, args.arms)
        outcome = store.try_claim(record)
        print('{0}: {1}'.format(describe(record), outcome))
        return 0 if outcome == 'won' else 1
    parser.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
