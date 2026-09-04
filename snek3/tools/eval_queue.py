"""The stage-A work queue: a directory of checkpoints waiting to be measured.

**Why this exists.** Stage A is 66% of a training arm's wall clock and the cause is lane drain — one
checkpoint's 100 episodes start together and the batch empties toward width 1, so the last few
episodes carry the whole per-step numpy cost alone. The same episodes cost 3.3x less inside a
sustained `engine.measure_stream`, which refills a finished lane from the *next* checkpoint. Cutting
the episode count does not help: 4x fewer buys 1.6x, because the tail is set by episode length rather
than by count. See [`../docs/findings.md`](../docs/findings.md).

So the measurement moves out of the training loop and into shared worker processes that always have
several checkpoints in flight. Nothing else about stage A changes: same 100 episodes, same policy,
same row.

## One writer per file, which is the rule this repo already paid to learn

[`results.py`](results.py) records the incident: snek2's single controller banked every lane's
episodes and re-serialised the whole result file 125 times per measurement — 58 s of bookkeeping
against the 46 s four lanes needed to produce one — so it overtook its own workers and folded a
90-minute backlog with 16 of them idle. The rule that came out of it is that a file has exactly one
writer.

That rule is kept here, and it is what decides the whole layout:

| file | written by | holds |
|---|---|---|
| `runs/<policy>_evals.json`, `.md`, `.png` | **the trainer, still solely** | the arm's history |
| `runs/.evalq/<policy>/<step>.req` | that trainer | the trainer-side half of the row |
| `runs/.evalq/<policy>/<step>.claim.<pid>` | the worker that won the claim | nothing; it *is* the claim |
| `runs/.evalq/<policy>/<step>.done` | that same worker | the measured sample |

**The row is assembled from two halves and they are written by different processes.** The trainer's
half — `epsilon`, `guided_fraction`, `steps_per_second`, the fork counters — describes what the arm
was doing *while it collected that step*, and only the trainer knows it. The worker's half is the
measurement. They meet in the trainer, which merges and remains the only thing that writes a
`runs/` file anyone reads.

## Claiming is a rename, and there is no lock

`os.rename` is atomic on POSIX, so a worker moving `<step>.req` to `<step>.claim.<pid>` either wins or
gets `FileNotFoundError`. Two workers cannot measure the same checkpoint, and there is no lock file to
leave behind, no protocol to get wrong, and nothing to time out.

## Nothing here can deadlock a training, and that is by construction

**A trainer blocked on a full queue measures the oldest checkpoint itself, in-process.** In the steady
state it does not ask whether a worker is alive, or how many there are, or whether they are keeping up
— reaching the bound is itself the signal, so it simply takes the work back. Three things follow, and
the third is the point:

- with workers keeping up, the depth never reaches the bound and the trainer never steals;
- with no workers at all, the trainer steals every time and the arm behaves **exactly as it does
  today**, at today's speed;
- there is no state in which the arm waits for something that may never come.

So an eval worker is as disposable as the chart window, for the same reason and without the chart
window's caveat that a training genuinely does not depend on it.

**The tail is the one exception, and it inverts.** When an arm reaches its cap it holds the deepest
queue of the whole run, which is precisely when a worker is at its most efficient — a backlog of 16
costs ~1.1 s a checkpoint against ~4.2 s drained. So `Trainer._drain_patiently` *waits* there instead
of seizing, and reclaims only after 20 s of silence or when `live_workers()` is empty. The impatient
first version measured as a 1.13x end to end, because the trainer and the worker raced to redo the same
backlog serially.

## What this changes about the training, which is two things and not one

Both are pre-registered in [`../docs/runs.md`](../docs/runs.md) rather than slipped in as a speed-up,
because `perfect_percent` steers exploration ([`../docs/invariants.md`](../docs/invariants.md)
invariant 2) and is therefore a feedback loop rather than a readout.

1. **The schedule's feedback lags by up to `depth` evals.** Bounded, so the worst case is today's
   behaviour and never an unbounded drift.
2. **Stage-A rows stop being bit-reproducible from the arm's seed.** A queued measurement shares one
   wide `VecSnake` with the other checkpoints in its round, and lanes migrate between checkpoints —
   that migration is the entire reason streaming is faster — so a job cannot own its lanes or its
   RNG, and it does not see the boards `eval_seed(seed, step)` would have drawn. The two instruments
   agree to 0.09 standard errors over 3,222 rows (phase 2), so this costs nothing statistically, but
   **an arm run with the queue cannot be diffed byte-for-byte against one run without it.**

**`SNEK_EVAL_QUEUE` is on by default as of 2026-08-29**, because consequence 2 costs nothing
statistically and the queue is worth 3.83x on the metric that gates every experiment this project runs
— 2.38 h to 3M against 9.12 h, at 8 arms rather than 4. `SNEK_EVAL_QUEUE=0` restores the unqueued path
exactly and is what a byte-for-byte diff against b1 or b2 has to use; those two arms were measured
unqueued and stay the reproducible baseline. `SNEK_EVAL_QUEUE_DEPTH=0` is the middle option: queued
code path, reclaimed in the same drain, bit-identical rows (see `DEFAULT_DEPTH` below).
"""

import json
import os
import subprocess
import sys
import time

from env import constants
from tools import live_runs

DIR_NAME = '.evalq'
WORKERS_DIR = 'workers'
SLOT_CLAIM_GRACE_SECONDS = 30      # an empty slot younger than this is a claim mid-write, not a stale one
WORKER_MODULE = 'tools.eval_worker'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# How many checkpoints an arm may have unmeasured before it takes the work back.
#
# **16, because it is where an arm stops being eval-bound — a phase change rather than a trend.**
# Swept over 23 configurations on 2026-08-29 (2-12 arms, 2-10 workers, depth 8-24; see
# `docs/findings.md`). Every one of the 19 configurations at depth 8 sat pinned against its cap with
# the trainer waiting, at every arm and worker count tried. At 16 the queue drains to 11-13 and a
# four-arm arm runs at 94% of its unblocked rate: the workers are finally ahead of the trainers. 24
# regresses (2.44 h against 2.38 h) because there is nothing left to win, and 12 captures only 63% of
# the gain while still being eval-bound.
#
# The eval side is bounded by *resident* checkpoints — `arms x depth` — rather than by worker count,
# because a deeper `measure_stream` round packs its 1024 lanes better: the same four workers deliver
# 0.513 evals/s at two arms and 1.071 at eight. Depth is the only way to raise that population without
# adding arms, and unlike workers it makes the existing work *cheaper*: four arms and four workers go
# from 3.94 h to 2.59 h per arm **and** from 47% to 34.8% idle.
#
# **The cost is entirely feedback lag, and it is the reason this is not higher.** Depth is how many
# eval intervals behind the epsilon and guided schedules read; 16 is 16,000 counted steps, 0.5% of a
# 3M arm but proportionally much more over the early steps where those schedules move fastest. 8 was
# the original choice for exactly that reason. Nothing has measured whether the doubled lag harms
# learning — only that it is worth 0.83 h per arm.
# **0 is meaningful and is the verification mode.** At 0 the trainer reclaims each checkpoint in the
# same drain that offered it, so the row is measured from the just-written `ckpt-<step>.pt` before
# training resumes — the same weights, the same seed, the schedule applied at the same point. That
# makes a depth-0 queued arm bit-identical to an unqueued one on every measured field, which is what
# `tests/test_train.py` uses to pin the whole assembly path. It recovers none of the 5.3 h, and that is
# fine: it is a fixture, not a setting for an arm.
DEFAULT_DEPTH = 16

# Workers per box — a box-wide target, not per arm: every trainer asks for this many and they share
# whoever is already running.
#
# **6, and unlike depth this one turns over.** At four arms the sweep measured 2w 4.84 h, 4w 3.94 h,
# 6w 3.21 h, 8w 3.27 h, 10w 3.55 h: past six, extra workers starve the trainers, whose own unblocked
# rate falls 380 -> 306 st/s, and the box goes to 4% idle to run *slower*. Eight arms behave the same
# way (6w 3.67 h, 8w 3.64 h at 0.2% idle, with the trainers losing 342 -> 292 st/s).
#
# So **idle CPU is not the objective here, and targeting it selects badly**: the fastest configuration
# leaves ~20% of the box idle. Spare cores exist because there are not enough resident checkpoints to
# batch, and filling them with workers only makes each round shallower.
#
# 6 is measured on the laptop (14 cores: 10P + 4E). The desktop is 8 physical cores / 16 SMT threads
# and has not been swept; its own vec-eval optimum is 16 single-threaded shards, so 6 workers beside
# 8 trainers is in line with what that box already runs, but the number is inherited rather than
# measured there. Over-provisioning wastes a core and under-provisioning is absorbed by the trainer
# taking work back, so neither is a failure.
DEFAULT_WORKERS = 6

# A worker exits after this long with nothing to measure. Long enough to ride out the gap between one
# arm's evals, short enough that a finished batch does not leave processes on the box.
IDLE_EXIT_SECONDS = 300


# ---------------------------------------------------------------- where things live

def directory(runs_dir=None):
    """`runs/.evalq`. Beside `runs/.live` for the same reason that one is there.

    Machine-local and regenerable: the queue is in-flight work, not a result, and a test that
    redirects `RUNS_DIR` moves it along with the charts. It is gitignored, which also keeps it out of
    the way of the desktop's `git merge --ff-only` — an untracked file under a live arm's `runs/`
    path is exactly what aborts a deploy.
    """
    return os.path.join(runs_dir or constants.RUNS_DIR, DIR_NAME)


def policy_directory(policy, runs_dir=None):
    return os.path.join(directory(runs_dir), str(policy))


def _path(policy, step, suffix, runs_dir=None):
    return os.path.join(policy_directory(policy, runs_dir), '{0}.{1}'.format(int(step), suffix))


def request_path(policy, step, runs_dir=None):
    return _path(policy, step, 'req', runs_dir)


def claim_path(policy, step, runs_dir=None, pid=None):
    return _path(policy, step, 'claim.{0}'.format(os.getpid() if pid is None else pid), runs_dir)


def done_path(policy, step, runs_dir=None):
    return _path(policy, step, 'done', runs_dir)


def _write_json(path, payload):
    """Atomically, via a per-process staging name.

    The staging name carries the pid because two processes writing the same *final* path is a case
    this queue does allow — a worker's `.done` and a trainer's own `.done` for a checkpoint it took
    back — and a shared `.partial` would let one truncate the other's file mid-write.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    staging = '{0}.{1}.partial'.format(path, os.getpid())
    with open(staging, 'w') as handle:
        json.dump(payload, handle)
    os.replace(staging, path)


def _read_json(path):
    """The payload, or None if it is absent or half-written.

    None rather than raising on a bad parse: every caller's correct response is "not ready yet", and
    a queue file is scratch — the checkpoint it describes is still on disk and can be measured again.
    """
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- the trainer's side

def enqueue(policy, step, fields, episodes, runs_dir=None):
    """Offers `step` for measurement, carrying the trainer's half of the row and its episode count.

    Returns the request path. The checkpoint must already be on disk — a worker restores it by step —
    which is why `train.py` saves unconditionally and prunes afterwards rather than gating the save on
    an eval it no longer has.

    **`episodes` travels with the request and is not a worker setting.** It is the row's denominator,
    and a worker that measured 100 where the arm asked for 10 would put rows of two different depths
    in one history — which `docs/invariants.md` invariant 8 says are not comparable, and which the
    protocol exists to prevent. Found by a smoke test whose `SNEK_GRAPH_EVAL_EPISODES=10` arm silently
    received 100-episode rows.
    """
    path = request_path(policy, step, runs_dir)
    _write_json(path, {'policy': str(policy), 'step': int(step), 'fields': dict(fields),
                       'episodes': int(episodes)})
    return path


def landed(policy, step, runs_dir=None):
    """`{'held': sample, 'fields': trainer_half}` for `step`, or None if it is not back yet.

    **Both halves, and the `.done` carries the trainer's half too.** The alternative — the trainer
    keeping the fields in memory from when it offered the step — loses them on a resume, and the
    fields are exactly what cannot be reconstructed: `steps_per_second` and the fork counters describe
    a window of training that is over. A self-contained result file also means the queue survives a
    trainer restart with no in-memory state to rebuild.
    """
    return _read_json(done_path(policy, step, runs_dir))


def retire(policy, step, runs_dir=None):
    """Removes every trace of `step` once the trainer has merged it.

    Every suffix, not only the one expected: a checkpoint the trainer took back leaves a `.claim` from
    the worker that had it, and a worker that finished after the steal leaves a second `.done`.
    Neither is an error and both would otherwise accumulate for the length of the arm.
    """
    folder = policy_directory(policy, runs_dir)
    prefix = '{0}.'.format(int(step))
    try:
        names = os.listdir(folder)
    except OSError:
        return
    for name in names:
        if name.startswith(prefix):
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass


def outstanding(policy, runs_dir=None):
    """Every step this arm offered and has not retired, oldest first. For a resume.

    A killed trainer leaves its requests on disk with nobody assembling them. Re-adopting those steps
    keeps the arm's history from having a gap at exactly the resume boundary, which is where a gap is
    least welcome: the resume step is re-measured and would sit beside missing neighbours.

    **Nothing is reclaimed here, deliberately.** A `.claim` left by a live worker produces its `.done`
    normally; one left by a worker that is gone is resolved by the trainer's own forward-progress
    rule, which measures the oldest outstanding step itself once the queue reaches its bound. Renaming
    claims back to `.req` at startup would be one more moving part for a case that rule already
    covers.
    """
    try:
        names = os.listdir(policy_directory(policy, runs_dir))
    except OSError:
        return []
    steps = set()
    for name in names:
        stem, _, suffix = name.partition('.')
        if stem.isdigit() and not name.endswith('partial'):
            steps.add(int(stem))
    return sorted(steps)


def sweep(policy, keep=(), runs_dir=None):
    """Removes every queue file for `policy` except the steps in `keep`. Returns how many steps went.

    Called when an arm finishes. The `complete` guard makes a leftover file rare rather than
    impossible — a worker can be between its `os.remove` and its `_write_json` when the trainer
    reclaims — and this is the cheap sweep that makes "rare" into "gone" without a lock.
    """
    keep = {int(step) for step in keep}
    folder = policy_directory(policy, runs_dir)
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    removed = set()
    for name in names:
        stem, _, _rest = name.partition('.')
        if not stem.isdigit() or int(stem) in keep:
            continue
        try:
            os.remove(os.path.join(folder, name))
            removed.add(int(stem))
        except OSError:
            pass
    try:
        os.rmdir(folder)
    except OSError:
        pass                                # not empty, or already gone; neither matters
    return len(removed)


def take_back(policy, step, runs_dir=None):
    """Reclaims `step` for the caller to measure itself. Returns its fields, or None.

    None means a worker already holds the request. The caller may measure it anyway, and `train.py`
    does — a duplicated measurement costs one checkpoint's episodes and a stalled arm costs hours —
    which is why `fields_of` exists to read the payload without the rename. The duplicate is harmless:
    whichever sample the trainer merges first wins and `retire` removes the other.
    """
    target = claim_path(policy, step, runs_dir)
    try:
        os.rename(request_path(policy, step, runs_dir), target)
    except OSError:
        return None
    return (_read_json(target) or {}).get('fields')


def fields_of(policy, step, runs_dir=None):
    """The trainer's half for `step`, from whichever file holds it. None if nothing does.

    Checked in the order a step moves through: a `.done` is authoritative because it carries both
    halves, then a claim held by a worker, then an unclaimed request. The claim suffix has a pid in it
    and the pid is not ours, so the directory is scanned rather than a path built.
    """
    payload = _read_json(done_path(policy, step, runs_dir))
    if payload and payload.get('fields'):
        return payload['fields']
    folder = policy_directory(policy, runs_dir)
    prefix = '{0}.'.format(int(step))
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return None
    for name in names:
        if not name.startswith(prefix) or name.endswith('partial'):
            continue
        suffix = name[len(prefix):]
        if suffix == 'req' or suffix.startswith('claim'):
            payload = _read_json(os.path.join(folder, name))
            if payload and payload.get('fields'):
                return payload['fields']
    return None


# ---------------------------------------------------------------- the worker's side

def pending(runs_dir=None):
    """Every `(policy, step)` offered and not yet claimed, oldest step first within each policy.

    Grouped by policy and interleaved by the caller rather than here: a worker should not drain one
    arm's whole backlog before touching another's, because the arms' schedules all lag by their own
    queue depth and the point of the bound is that no arm's lag is much worse than another's.
    """
    root = directory(runs_dir)
    found = {}
    try:
        policies = sorted(os.listdir(root))
    except OSError:
        return found
    for policy in policies:
        folder = os.path.join(root, policy)
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        steps = []
        for name in names:
            stem, _, suffix = name.partition('.')
            if suffix == 'req' and stem.isdigit():
                steps.append(int(stem))
        if steps:
            found[policy] = sorted(steps)
    return found


def episodes_of(policy, step, runs_dir=None):
    """The episode count `step` was offered at, or None if the request has gone.

    Read before the claim, so a round can be grouped by depth — `measure_stream` takes one episode
    count for a whole call, so two arms at different depths cannot share a round and must not be
    silently forced into one.
    """
    payload = _read_json(request_path(policy, step, runs_dir))
    return int(payload['episodes']) if payload and 'episodes' in payload else None


def claim(policy, step, runs_dir=None):
    """Takes `step` for this process. Returns the request payload, or None if someone else won.

    The rename is the claim: it is atomic on POSIX, so the loser gets `FileNotFoundError` and moves
    on. Read *after* the rename — reading first and renaming second would let both workers read a
    payload and only then discover the race, which is the same amount of work and one more state.
    """
    source = request_path(policy, step, runs_dir)
    target = claim_path(policy, step, runs_dir)
    try:
        os.rename(source, target)
    except OSError:
        return None
    return _read_json(target)


def held_by_live_worker(policy, step, runs_dir=None):
    """Whether a process that still exists holds a claim on `step`.

    **This is what stops the trainer duplicating work a worker is already doing.** A round completes
    its checkpoints in whatever order lanes free up, not in step order, so the *front* of a trainer's
    queue is often among the last of a round to land while later steps pile up as `.done`. Without
    this the trainer hit its bound, waited out the grace period and re-measured the front itself —
    100 wasted episodes per eval, and the worker's own result for it then discarded. Measured as the
    difference between no speed-up and the real one.

    The pid is read out of the claim's filename, so a dead worker's claim reads as unheld and the
    trainer reclaims immediately — which is the forward-progress path.
    """
    folder = policy_directory(policy, runs_dir)
    prefix = '{0}.claim.'.format(int(step))
    try:
        names = os.listdir(folder)
    except OSError:
        return False
    for name in names:
        if not name.startswith(prefix) or name.endswith('partial'):
            continue
        tail = name[len(prefix):]
        if tail.isdigit() and live_runs.alive(int(tail)):
            return True
    return False


def complete(policy, step, held, fields, runs_dir=None):
    """Publishes the measurement together with the trainer's half, then drops this claim.

    `fields` is passed straight back from the request rather than interpreted: a worker has no
    business knowing what an epsilon is, and copying the half it was handed is what makes the `.done`
    a complete row's worth of input. See `landed`.
    """
    # **Nothing is published for a step this process no longer holds.** A trainer that reached its
    # depth bound reclaims the oldest checkpoint, measures it and `retire`s it — so a worker finishing
    # afterwards would drop a `.done` into a directory nobody is watching, and it would sit there for
    # the length of the batch. Checking our own claim first closes that: the claim is this process's
    # and the trainer's reclaim renames it away.
    try:
        os.remove(claim_path(policy, step, runs_dir))
    except OSError:
        return False
    _write_json(done_path(policy, step, runs_dir),
                {'step': int(step), 'held': held, 'fields': dict(fields or {})})
    return True


# ---------------------------------------------------------------- the worker slots

def workers_directory(runs_dir=None):
    return os.path.join(directory(runs_dir), WORKERS_DIR)


def worker_slot(index, runs_dir=None):
    return os.path.join(workers_directory(runs_dir), str(int(index)))


def live_workers(runs_dir=None):
    """The indexes of slots held by a process that still exists.

    Pid-based, exactly as `live_runs` is and for the reason given there: a pid we were handed cannot
    match the wrong thing the way a `pgrep` pattern can, and a dead slot is self-evidently dead
    rather than merely old.
    """
    held = []
    try:
        names = os.listdir(workers_directory(runs_dir))
    except OSError:
        return held
    for name in names:
        if not name.isdigit():
            continue
        pid = live_runs.read(worker_slot(name, runs_dir))
        if pid is not None and live_runs.alive(pid) and not live_runs.zombie(pid):
            held.append(int(name))
    return sorted(held)


def take_slot(index, runs_dir=None):
    """Claims worker slot `index` for this process, or False if a live worker already holds it.

    Creating the file *is* the claim, and the file is born holding the claimer's pid: the pid is
    written to a private temp file and `os.link`ed to the slot path, which fails with
    `FileExistsError` if the slot exists at all. Before 2026-09-03 the file was created with `O_EXCL`
    and the pid written a moment later, and in that moment the slot was *empty*: a rival arm reading
    it saw no pid, took the slot for stale, and claimed it too. Eight arms launched in the same
    instant on the laptop produced **seven slot-0 workers**, fourteen workers for eight slots, all at
    70% CPU. The `FileExistsError` branch handles a worker that exited and left its pid behind, which
    has to be re-claimable or a box that ran one batch would never start a worker again -- and an
    empty slot younger than `SLOT_CLAIM_GRACE_SECONDS` is a claim in progress by an older process,
    not a stale one.
    """
    path = worker_slot(index, runs_dir)
    temp = '{0}.claim.{1}'.format(path, os.getpid())
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(temp, 'w') as handle:
            handle.write('{0}\n'.format(os.getpid()))
        try:
            os.link(temp, path)
            return True
        except FileExistsError:
            pass
        finally:
            try:
                os.remove(temp)
            except OSError:
                pass
    except OSError:
        return False
    pid = live_runs.read(path)
    if pid is not None and live_runs.alive(pid) and not live_runs.zombie(pid):
        return False
    if pid is None and _slot_age(path) < SLOT_CLAIM_GRACE_SECONDS:
        return False
    return _write_slot(path, os.getpid())


def _slot_age(path):
    try:
        return time.time() - os.stat(path).st_mtime
    except OSError:
        return float('inf')


def _write_slot(path, pid):
    """Puts `pid` in a slot file atomically. Returns whether the record took.

    The pid written is **the worker's**, never the trainer's, and that is the whole reason this takes
    an argument. `take_slot` writes the claimer's pid so the slot is never briefly unowned, and
    `ensure_workers` overwrites it with the child's the moment there is a child — so `live_workers`
    tracks the process that is actually measuring. A slot left holding a trainer's pid would read as
    a live worker for the length of the arm while nothing drained the queue.
    """
    temp = '{0}.{1}'.format(path, os.getpid())
    try:
        with open(temp, 'w') as handle:
            handle.write('{0}\n'.format(int(pid)))
        os.replace(temp, path)
    except OSError:
        return False
    return live_runs.read(path) == int(pid)


def release_slot(index, runs_dir=None):
    try:
        os.remove(worker_slot(index, runs_dir))
    except OSError:
        pass


def spawn_worker(index, runs_dir=None, env=None):
    """Starts one worker and returns its Popen, or None if it would not start.

    `start_new_session` is the load-bearing argument, as it is for the chart window: it puts the
    worker in its own session and process group, so a Ctrl-C or a `kill` to a trainer's group leaves
    it alone, and killing a worker cannot signal a trainer. Failing to start one is not an error —
    the trainer measures its own checkpoints and says so.
    """
    argv = [sys.executable, '-u', '-m', WORKER_MODULE, '--slot', str(int(index))]
    if runs_dir:
        argv += ['--runs-dir', runs_dir]
    try:
        return subprocess.Popen(argv, cwd=ROOT, start_new_session=True,
                                stdin=subprocess.DEVNULL,
                                stdout=sys.stderr, stderr=subprocess.STDOUT,
                                env=dict(os.environ if env is None else env))
    except OSError as error:
        sys.stderr.write('eval worker did not start ({0}); the trainer will measure its own '
                         'checkpoints\n'.format(error))
        return None


def ensure_workers(target, runs_dir=None, env=None):
    """Brings the box up to `target` live workers. Returns the Popens this call started.

    `target` is required and has no environment fallback: `train.py` is the one place that reads
    `SNEK_EVAL_WORKERS`, through `tuned()`, so a caller that set it in a config could not be quietly
    overridden by the environment. It was, once — a fixture asking for zero workers got two, and they
    measured its checkpoints at their own seed.

    Called by every arm at startup, and idempotent because of the slot claim: four arms racing
    produce exactly `target` workers rather than four times as many. Returning only what *this*
    process started is what lets a trainer reap its own children without waiting on anyone else's.
    """
    target = int(target)
    started = []
    for index in range(target):
        if index in live_workers(runs_dir):
            continue
        if not take_slot(index, runs_dir):
            continue
        process = spawn_worker(index, runs_dir, env)
        if process is None:
            release_slot(index, runs_dir)
            continue
        # The slot now names the worker rather than this trainer. Until this line it named the
        # trainer, deliberately: a slot that is momentarily unowned would let a second arm claim it
        # and start a duplicate worker.
        _write_slot(worker_slot(index, runs_dir), process.pid)
        started.append(process)
    return started


def reap(processes):
    """Polls this arm's own workers so an exited one does not stay a zombie for the rest of the run.

    Never a `wait()`: a trainer is a long-lived parent and blocking it on a worker is the one thing
    this module exists to prevent.
    """
    return [process for process in processes or () if process.poll() is None]
