"""The unattended dispatcher. Runs under systemd (Restart=always) and drives the
whole box from git:

  1. fetch origin; re-read runtime.json (clamped; bad config -> keep last good)
  2. reap finished jobs; publish their artifacts to `results`
  3. reconcile: if the box is idle, open one wave -- same-type jobs (train OR eval,
     never both) up to that type's limit; while any job runs, launch nothing (no
     backfill), so waves have clean boundaries. Skipped if paused/draining or disk low
  4. refresh throughput; publish status.json to `ops-status`

The loop body is wrapped so a bad job, a transient git error, or a malformed
config can never kill the daemon -- in normal operation the box has no SSH
backstop, so staying up is the first requirement. Launched jobs are detached and
self-terminate via SNEK_MAX_STEPS, so the daemon never kills anything; a daemon
restart re-adopts still-running jobs by pid, while a *machine* reboot (a boot-id
mismatch) marks them `interrupted` so they relaunch and resume -- see `_reattach`.
"""
import json
import os
import signal
import subprocess
import sys
import time

from . import config as cfgmod
from . import gitbus
from . import launch
from .job import parse_job, Job, JobError

TERMINAL = ('done', 'failed')

# `interrupted` is deliberately NOT terminal. A job the machine killed has not finished, so it
# must stay launchable: `_scan_pending` re-launches it from its still-present spec (a training
# resumes from its checkpoint, since SNEK_MAX_STEPS is absolute), and `_auto_closeout_jobs`
# re-synthesizes a closeout whose previous attempt was cut short. Both are the recovery.

# A synthesized closeout eval outranks any pending training (default priority 100) so, when a
# training wave drains, its closeouts form the next wave before any new training starts. That
# is the whole point of auto-closeout: never train the next thing before evaluating the last.
AUTO_CLOSEOUT_PRIORITY = 10


def wants_closeout(job_type, ok, auto_closeout_enabled):
    """Whether a just-finished job should get an automatic closeout eval. Only a *training*
    that finished successfully -- smoke and benchmark runs share the 'trainer' category but
    are throwaway, and a failed run has no checkpoint worth measuring."""
    return bool(auto_closeout_enabled) and ok and job_type == 'train'


class _StubJob:
    """Stands in for a Job re-adopted from the ledger after a restart, where the
    original spec is not in hand -- only what the ledger recorded."""
    def __init__(self, jid, rec):
        self.id = jid
        self.type = rec.get('type', 'train')
        self.policy = rec.get('policy')

    @property
    def category(self):
        return 'eval' if self.type == 'eval' else 'trainer'


class Runner:
    def __init__(self, host):
        self.host = host
        self.runtime = dict(cfgmod.RUNTIME_DEFAULTS)
        cfgmod.clamp_runtime(self.runtime, host)
        self.config_notes = []
        self.ledger = self._load_ledger()
        self.running = {}          # job_id -> RunningJob
        self._viewer_pngs = None   # the PNG set the live viewer was launched with, or None
        self._wave_pngs = []       # sticky panel set for the current wave (grows, never shrinks)
        self._wave_category = None # the wave's category, so a train->eval flip resets the set
        self._queued = []          # launchable jobs waiting for the next wave, priority order
        self.stop = False
        self._reattach()

    # ------------------------------------------------------------------ ledger
    def _load_ledger(self):
        try:
            with open(self.host['LEDGER_PATH']) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def _save_ledger(self):
        os.makedirs(os.path.dirname(self.host['LEDGER_PATH']), exist_ok=True)
        tmp = self.host['LEDGER_PATH'] + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(self.ledger, fh, indent=2)
        os.replace(tmp, self.host['LEDGER_PATH'])

    def _reattach(self):
        """Re-adopt jobs the ledger says were running, or classify the ones that are gone.

        **The boot id is what separates a daemon restart from a machine reboot**, and the two
        need opposite handling. Jobs are detached (`setsid`), so across a *daemon* restart they
        really do keep running and a dead pid means the job genuinely finished. A *reboot* kills
        every child, so the same dead pid means the job was cut off partway -- and calling that
        `done` is wrong twice over: a truncated training got published as a finished arm (with a
        closeout eval measuring the partial arm), and a truncated closeout consumed its own
        `closeout: pending` marker, so the arm was never evaluated again. Both were silent.

        Recording `/proc/sys/kernel/random/boot_id` at launch makes the distinction exact, and
        it closes a second hole: on a boot-id mismatch we never consult the pid at all, so a
        recycled pid cannot be mistaken for a live job. That mattered because pids restart low
        after a reboot, and a false match would have re-adopted a phantom that never exits --
        which the wave barrier (`if self.running: return`) turns into an idle box forever.
        """
        boot = boot_id()
        for jid, rec in self.ledger.items():
            if rec.get('state') != 'running':
                continue
            rec_boot = rec.get('boot')
            rebooted = bool(boot and rec_boot and rec_boot != boot)
            pid = rec.get('pid')
            if not rebooted and pid and launch.pid_alive(pid):
                # Still ours and still alive. Stamp the current boot id onto a record that
                # predates this field so the *next* restart can classify it properly.
                if boot and not rec_boot:
                    rec['boot'] = boot
                self.running[jid] = launch.RunningJob(
                    _StubJob(jid, rec), rec.get('policy'), pid,
                    rec.get('log', ''), popen=None, started=rec.get('started'))
            elif rebooted:
                rec['state'] = 'interrupted'
                rec['finished'] = time.time()
                rec['note'] = 'killed by a reboot, not finished'
                rec['restarts'] = int(rec.get('restarts') or 0) + 1
                # No `closeout: pending` here -- the arm is unfinished, so it earns its closeout
                # when it actually completes, not now. `interrupted` is non-terminal, so the
                # relaunch happens on the next dispatch.
            else:
                rec['state'] = 'done'
                rec['finished'] = time.time()
                rec['note'] = 'completed while runner was down'
                # Same boot, dead pid: the job ran to its own end while the daemon was down.
                # Exit code is unknown across reparenting, so success is assumed; on that same
                # assumption the arm still earns its closeout, so auto-closeout does not
                # silently skip a restart-straddling arm.
                if wants_closeout(rec.get('type'), True, self.runtime.get('auto_closeout', True)):
                    rec['closeout'] = 'pending'
        self._save_ledger()

    # -------------------------------------------------------------------- loop
    def poll_once(self):
        gitbus.fetch(self.host)
        self._apply_runtime()
        self._reap()
        # Scan every poll -- not only when dispatching -- so status.json's `queued` list stays
        # current while a wave occupies the box or the daemon is paused/draining.
        self._queued = self._scan_pending()
        if not (self.runtime['paused'] or self.runtime['drain']):
            self._dispatch()
        for rj in self.running.values():
            launch.update_throughput(rj, self.host)
        self._ensure_viewer()
        self._publish()

    def _ensure_viewer(self):
        """Best-effort: while jobs run, keep one decoupled chart viewer up on the display,
        bound to the charts the *currently running* jobs produce. It is a *separate* process
        (never a child of a job), so it can never affect a run, and it exits itself when the
        jobs stop (--watch). A launch failure is logged and ignored -- a chart is never worth
        a job.

        The viewer's arg list is fixed at launch, so when a wave flips from training to eval
        the file set changes (runs/<policy>.png -> evals/<policy>_eval_progress.png) and a
        viewer left running keeps showing the finished training charts. Seen 2026-08-10. So we
        track the set the live viewer was launched with and relaunch whenever it no longer
        matches what the running jobs need.

        The panel set is *sticky within a wave*: an arm that reaches its cap and exits keeps
        its panel until the whole wave drains, so a wave of four never collapses to the two
        still training. The viewer tags the finished ones `(completed)`. `sticky_wave_pngs`
        only unions -- it never drops -- so a finished arm does not trigger a relaunch; the
        set resets when the wave flips category (train->eval) or the box goes idle."""
        if not self.runtime.get('viewer', True) or not self.running:
            # Idle between waves: forget the wave so the next one starts from its own arms,
            # not unioned onto the previous batch's (a trainer->trainer flip keeps category).
            self._wave_pngs, self._wave_category = [], None
            return
        try:
            running_jobs = [(rj.job.category, rj.policy) for rj in self.running.values()]
            current = viewer_png_paths(running_jobs, self.host['SNEK_DIR'])
            if not current:
                return
            # Wave-barrier scheduling runs one category at a time, so the wave's category is
            # simply that of whatever is running; sorted+joined stays deterministic regardless.
            category = ','.join(sorted({c for c, _ in running_jobs}))
            desired = sticky_wave_pngs(self._wave_pngs, self._wave_category, category, current)
            self._wave_pngs, self._wave_category = desired, category
            viewer_running = subprocess.run(
                ['pgrep', '-f', 'chart_viewer.py'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if not viewer_should_relaunch(viewer_running, self._viewer_pngs, desired):
                return
            # Wrong file set (the wave flipped category, or a new arm joined it), or unknown
            # across a daemon restart: drop any existing viewer and launch one bound to the
            # current set. A finished arm does not land here -- the sticky set kept its panel.
            # The pgrep/pkill pattern is not in the daemon's own argv, so it cannot self-match.
            if viewer_running:
                subprocess.run(['pkill', '-f', 'chart_viewer.py'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pngs = desired
            os.makedirs(self.host['LOG_DIR'], exist_ok=True)
            log = open(os.path.join(self.host['LOG_DIR'], 'chart_viewer.log'), 'ab')
            # The runner is a systemd service outside the graphical session, so its own
            # environment has no DISPLAY -- the viewer only reaches the monitor if we pass
            # the session's DISPLAY + X authority, exactly as launch.build_command does for
            # jobs. Without this a daemon-launched viewer opens nothing.
            env = dict(os.environ)
            if self.host.get('DISPLAY'):
                env.setdefault('DISPLAY', self.host['DISPLAY'])
                if self.host.get('XAUTHORITY'):
                    env.setdefault('XAUTHORITY', self.host['XAUTHORITY'])
            argv = [self.host['PYTHON_BIN'], '-u', 'chart_viewer.py'] + pngs + \
                   ['--watch', 'snek2.py|eval_checkpoints.py', '--interval', '1',
                    '--scale', viewer_scale(category), '--title', 'snek desktop']
            subprocess.Popen(argv, cwd=self.host['SNEK_DIR'], env=env,
                             stdout=log, stderr=log, start_new_session=True, close_fds=True)
            self._viewer_pngs = pngs
        except Exception as e:
            sys.stderr.write('viewer launch failed: {0}\n'.format(e))

    def _apply_runtime(self):
        cfg, notes = cfgmod.parse_runtime_config(gitbus.read_runtime_text(self.host), self.host)
        if cfg is None:
            self.config_notes = ['using last-known-good config; errors: ' + '; '.join(notes)]
        else:
            self.runtime, self.config_notes = cfg, notes

    def _counts(self):
        c = {'trainer': 0, 'eval': 0}
        for rj in self.running.values():
            c[rj.job.category] += 1
        return c

    def _auto_closeout_jobs(self):
        """Closeout evals to run for trainings that finished under auto-closeout and have not
        been evaluated yet -- synthesized fresh each dispatch, never persisted as specs.

        Driven entirely off the ledger, so it survives a daemon restart: the `closeout:
        pending` marker and the training's stored `env` both persist. Idempotent because the
        synthesized job's id is `<policy>-closeout`; once that eval is running or terminal the
        guard below skips it, so the still-'pending' marker cannot launch a second one. It
        inherits the training's whole env, which carries SNEK_FC_LAYERS -- without it the
        restore silently mismatches the architecture (the FC trap) and the arm scores ~0."""
        if not self.runtime.get('auto_closeout', True):
            return []
        jobs = []
        for rec in self.ledger.values():
            if rec.get('closeout') != 'pending' or not rec.get('policy'):
                continue
            eval_id = rec['policy'] + '-closeout'
            if self.ledger.get(eval_id, {}).get('state') in TERMINAL or eval_id in self.running:
                continue
            jobs.append(Job(id=eval_id, type='eval', policy=rec['policy'],
                            env=dict(rec.get('env') or {}), eval_args=['top20'],
                            priority=AUTO_CLOSEOUT_PRIORITY))
        return jobs

    def _scan_pending(self):
        """The launchable jobs waiting for the next wave, sorted by priority -- the order
        `_dispatch` draws from and the order status.json lists them in. Parses every pending
        spec, marks a malformed one failed in the ledger (persisting that immediately, since
        this runs even when dispatch is skipped), and drops any job already terminal or
        running. Auto-closeouts join the same pool; a manual spec of the same id (should not
        happen while auto-closeout is on, but can coexist) is kept over the synthesized one."""
        desired = []
        marked = False
        for name, text in gitbus.read_pending_jobs(self.host):
            try:
                job = parse_job(text, name)
            except JobError as e:
                jid = name[:-5] if name.endswith('.json') else name
                if self.ledger.get(jid, {}).get('state') != 'failed':
                    self.ledger[jid] = {'state': 'failed', 'type': 'unknown',
                                        'error': str(e), 'finished': time.time()}
                    marked = True
                continue
            state = self.ledger.get(job.id, {}).get('state')
            if state in TERMINAL or job.id in self.running:
                continue
            desired.append(job)
        seen = {job.id for job in desired}
        desired += [j for j in self._auto_closeout_jobs() if j.id not in seen]
        desired.sort(key=lambda j: j.priority)
        if marked:
            self._save_ledger()
        return desired

    def _dispatch(self):
        free = _disk_free_gb(self.host['REPO_PATH'])
        if free is not None and free < self.runtime['disk_min_gb']:
            self.config_notes = self.config_notes + \
                ['disk low ({0} GB < {1}) -- not launching'.format(free, self.runtime['disk_min_gb'])]
            return
        limits = {'trainer': self.runtime['max_trainers'], 'eval': self.runtime['max_evals']}
        desired = self._queued
        # Wave-barrier scheduling: a "wave" is a set of same-type jobs launched together,
        # up to that type's limit, and NOTHING new starts until the whole wave finishes.
        # So trainings and evals never overlap, and a freed slot is not backfilled mid-wave
        # -- the box idles on the stragglers until the slowest job in the wave is done, then
        # the next wave opens. We still scan pending specs while a wave runs (in poll_once),
        # so a malformed one is marked failed promptly, but we launch nothing until idle.
        if self.running or not desired:
            return
        # The highest-priority pending job picks the wave's type; only jobs of that type
        # join it, up to that type's limit. `desired` is already priority-sorted.
        wave_category = desired[0].category
        launched = 0
        for job in desired:
            if job.category != wave_category or launched >= limits[wave_category]:
                continue
            self._launch(job)
            launched += 1
        self._save_ledger()

    def _launch(self, job):
        try:
            rj = launch.spawn(job, self.host, self.runtime)
        except Exception as e:  # spawn is the one place a job can fail before running
            self.ledger[job.id] = {'state': 'failed', 'type': job.type,
                                   'error': 'spawn failed: {0}'.format(e),
                                   'finished': time.time()}
            return
        prior = self.ledger.get(job.id) or {}
        self.running[job.id] = rj
        self.ledger[job.id] = {'state': 'running', 'type': job.type, 'policy': rj.policy,
                               'pid': rj.pid, 'log': rj.log_path, 'started': rj.started,
                               # The boot this pid belongs to. Without it a pid outliving a
                               # reboot reads as a live job -- see _reattach.
                               'boot': boot_id(),
                               # Kept so an auto-closeout can rebuild the exact env after the
                               # daemon restarts -- crucially SNEK_FC_LAYERS (the FC trap).
                               'env': job.env}
        # Carried across the relaunch so a box that reboots repeatedly is visible in
        # status.json as a climbing count rather than as a job that looks freshly started.
        if prior.get('restarts'):
            self.ledger[job.id]['restarts'] = prior['restarts']

    def _reap(self):
        for jid in list(self.running):
            rj = self.running[jid]
            if rj.is_alive():
                continue
            rc = rj.returncode()
            ok = (rc == 0) or (rc is None and not _log_has_traceback(rj.log_path))
            rec = self.ledger.setdefault(jid, {})
            rec.update({'state': 'done' if ok else 'failed', 'type': rj.job.type,
                        'policy': rj.policy, 'finished': time.time(), 'returncode': rc})
            if wants_closeout(rj.job.type, ok, self.runtime.get('auto_closeout', True)):
                rec['closeout'] = 'pending'   # picked up by _auto_closeout_jobs next dispatch
            del self.running[jid]
            if ok:
                self._publish_results(rj)
        self._save_ledger()

    def _publish_results(self, rj):
        snek, pol = self.host['SNEK_DIR'], str(rj.policy)
        arts, runs = [], os.path.join(self.host['SNEK_DIR'], 'runs')
        if os.path.isdir(runs):
            for f in os.listdir(runs):
                if f == pol + '.md' or f.startswith(pol + '.') or f.startswith(pol + '_'):
                    arts.append(os.path.join(runs, f))
        try:
            gitbus.publish_results(self.host, rj.job, arts)
        except Exception as e:  # best-effort, but say so in the journal, not silently
            sys.stderr.write('publish_results({0}) failed: {1}\n'.format(rj.job.id, e))

    def _ledger_view(self):
        """The id->state map published in status.json, with the pending queue folded in as
        `queued` entries **at the end** -- after the run history and the running jobs -- in the
        order the wave-barrier scheduler will actually launch them, including the closeout evals
        each queued training will spawn, which run before the next training batch (see
        `anticipated_queue`). So a glance at the tail of the ledger shows the whole expected run
        order, not only the specs that exist as files today.

        A launched job moves into `running` and drops out of `_queued` the same poll; a real
        ledger state wins over the synthetic `queued` on the rare id overlap (a job re-queued
        while its prior run is still settling) -- `setdefault` keeps the real state and its
        position, appending only ids the history does not already carry."""
        limits = {'trainer': self.runtime['max_trainers'], 'eval': self.runtime['max_evals']}
        auto = self.runtime.get('auto_closeout', True)
        queued = [{'id': j.id, 'type': j.type, 'policy': j.policy, 'priority': j.priority}
                  for j in self._queued if j.id not in self.running]
        running = [{'id': jid, 'type': rj.job.type, 'policy': rj.policy}
                   for jid, rj in self.running.items()]
        existing = set(self.ledger) | set(self.running) | {j['id'] for j in queued}
        order = anticipated_queue(queued, running, limits, auto, existing)
        view = {k: v.get('state') for k, v in self.ledger.items()}
        for job in order:                              # queued at the end, in run order
            view.setdefault(job['id'], 'queued')
        return view

    def _publish(self):
        status = {
            'iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'ts': time.time(),
            'runtime': self.runtime,
            'config_notes': self.config_notes,
            'counts': self._counts(),
            'running': [{'id': rj.job.id, 'type': rj.job.type, 'policy': rj.policy,
                         'pid': rj.pid, 'step': rj.current_step,
                         'steps_per_sec': rj.steps_per_sec,
                         'elapsed_s': round(time.time() - rj.started)}
                        for rj in self.running.values()],
            'ledger': self._ledger_view(),
            'disk_free_gb': _disk_free_gb(self.host['REPO_PATH']),
            'load_avg': list(os.getloadavg()),
        }
        try:
            gitbus.publish_status(self.host, json.dumps(status, indent=2))
        except Exception as e:
            sys.stderr.write('publish_status failed: {0}\n'.format(e))


def anticipated_queue(queued, running, limits, auto_closeout, existing_ids):
    """Simulate the wave-barrier scheduler forward over the pending queue and return the jobs
    in the order they are expected to launch -- with each queued training's closeout eval
    inserted where it will actually run, before the next training batch.

    Modelled straight from `_dispatch`: repeatedly form a wave from the highest-priority job's
    category, up to that category's limit, drain it, and let every training in it spawn a
    closeout (priority `AUTO_CLOSEOUT_PRIORITY` = 10, below any training) that competes in the
    next wave. Because a closeout outranks a queued training, a batch's closeouts always slot
    in ahead of the following batch -- the exact interleaving the box will run.

    `queued` and `running` are lists of dicts with id/type/policy (queued also has priority);
    `queued` is the real pending set (already includes closeouts synthesized for *finished*
    trainings), `running` seeds anticipated closeouts for trainings on the box now, which drain
    before any queued wave. `existing_ids` is every id already known (ledger, running, queued),
    so a closeout that already exists somewhere is never invented twice. Returns the ordered
    list of queued/anticipated job dicts (never the running jobs themselves)."""
    def category(job_type):
        return 'eval' if job_type == 'eval' else 'trainer'

    seen = set(existing_ids)

    def closeout_for(policy):
        """A fresh closeout dict for `policy`, or None if one already exists or is not wanted."""
        cid = (policy or '') + '-closeout'
        if not policy or cid in seen:
            return None
        seen.add(cid)
        return {'id': cid, 'type': 'eval', 'policy': policy, 'priority': AUTO_CLOSEOUT_PRIORITY}

    pool = [dict(j) for j in queued]
    for r in running:                                  # closeouts for trainings running now
        if wants_closeout(r.get('type'), True, auto_closeout):
            c = closeout_for(r.get('policy'))
            if c:
                pool.append(c)

    order = []
    while pool:
        pool.sort(key=lambda j: j['priority'])
        wave_type = category(pool[0]['type'])
        wave = []
        for job in pool:
            if category(job['type']) == wave_type and len(wave) < limits[wave_type]:
                wave.append(job)
        wave_ids = {job['id'] for job in wave}
        order.extend(wave)
        pool = [job for job in pool if job['id'] not in wave_ids]
        for job in wave:                               # each training in the wave spawns one
            if wants_closeout(job['type'], True, auto_closeout):
                c = closeout_for(job['policy'])
                if c:
                    pool.append(c)
    return order


def viewer_png_paths(running_jobs, snek_dir):
    """Sorted PNG paths the chart viewer should show for the given running jobs.

    Each job's chart lives at a category-specific path: an eval writes
    evals/<policy>_eval_progress.png, a trainer writes runs/<policy>.png. `running_jobs`
    is an iterable of (category, policy) pairs; policies that are falsy are skipped.
    Sorted so the result is comparable across polls -- the set changes when a wave flips
    train->eval, which is exactly when the viewer's fixed arg list has gone stale."""
    pngs = []
    for category, policy in running_jobs:
        if not policy:
            continue
        if category == 'eval':
            pngs.append(os.path.join(snek_dir, 'evals', policy + '_eval_progress.png'))
        else:
            pngs.append(os.path.join(snek_dir, 'runs', policy + '.png'))
    return sorted(pngs)


def viewer_scale(category):
    """Window-size multiplier for the desktop viewer, by wave category. Eval charts run a
    bit larger (1.95 vs 1.5, ~30% up) because their detail -- the per-checkpoint eval points
    packed along 3M steps -- is what gets read closely; the training curve is coarser and
    fine at 1.5. `category` is the comma-joined set from `_ensure_viewer`, so a pure eval wave
    is exactly 'eval'; anything with a trainer in it stays at the smaller size."""
    return '1.95' if category == 'eval' else '1.5'


def sticky_wave_pngs(prev_pngs, prev_category, category, current_pngs):
    """Union this wave's currently-running PNGs into the set already on screen, so a finished
    arm keeps its panel until the whole wave drains (the viewer marks it `(completed)`) rather
    than the window shrinking to just the arms still going.

    Only grows within a wave. Resets to the current set when the wave's category flips
    (`trainer` -> `eval`, an entirely new set of charts under evals/), and `prev_category`
    None -- a fresh daemon, or the idle gap between waves -- resets too.

    It **also** resets when the running arms are disjoint from the previous set, which is the
    signal for a new wave of the *same* category — two eval waves back to back (a batch's
    close-out, then the next batch's), or trainer->trainer. Relying on `prev_category` alone
    let those union: when one `_dispatch` reaps the finished wave and launches the next in the
    same poll, `_ensure_viewer` never sees the idle gap that would have cleared the set, so the
    old batch's now-archived PNGs stayed on screen reading `(completed) (waiting…)`. Within a
    wave the running set only shrinks as arms finish, so it always overlaps the accumulated
    set and this branch does not fire; across waves the policies never repeat (the ledger runs
    each id once), so the two sets are always disjoint."""
    if prev_category != category or not prev_pngs or set(current_pngs).isdisjoint(prev_pngs):
        return sorted(current_pngs)
    return sorted(set(prev_pngs) | set(current_pngs))


def viewer_should_relaunch(viewer_running, current_pngs, desired_pngs):
    """Whether to (re)launch the viewer. Launch when nothing is up; relaunch when the live
    viewer is bound to a different file set than the running jobs now need, because the
    viewer's arg list is fixed at launch and cannot follow a train->eval transition. Never
    launch for an empty set."""
    if not desired_pngs:
        return False
    if not viewer_running:
        return True
    return current_pngs != desired_pngs


BOOT_ID_PATH = '/proc/sys/kernel/random/boot_id'
_BOOT_ID = []   # one-element cache; the value cannot change without the process dying


def boot_id():
    """This boot's kernel id, or None where the kernel does not publish one.

    Cached, because it is read on every launch and cannot change while the daemon lives -- a
    new boot is a new process by definition.

    **None means "cannot tell", and every caller degrades to the old pid-only behaviour** rather
    than guessing. That keeps the daemon working on a box without procfs (a mac, a container),
    where the reboot case cannot arise the same way; it also means a record written before this
    field existed is classified exactly as it was before, instead of being read as a reboot
    because its `boot` key is absent."""
    if not _BOOT_ID:
        try:
            with open(BOOT_ID_PATH) as fh:
                _BOOT_ID.append(fh.read().strip() or None)
        except OSError:
            _BOOT_ID.append(None)
    return _BOOT_ID[0]


def _log_has_traceback(log_path):
    try:
        size = os.path.getsize(log_path)
        with open(log_path, 'rb') as fh:
            fh.seek(max(0, size - 4096))
            tail = fh.read().decode('utf-8', 'replace')
    except OSError:
        return False
    return 'Traceback (most recent call last)' in tail


def _disk_free_gb(path):
    try:
        st = os.statvfs(path)
        return round(st.f_bavail * st.f_frsize / 1e9, 1)
    except OSError:
        return None


def main():
    host_env = os.environ.get(
        'SNEK_RUNNER_HOST_ENV',
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'host.env'))
    host = cfgmod.load_host_config(host_env)
    runner = Runner(host)

    def handle(_sig, _frame):
        runner.stop = True  # stop the daemon; detached jobs keep running
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    while not runner.stop:
        try:
            runner.poll_once()
        except Exception as e:  # the loop must never die
            sys.stderr.write('poll error: {0}\n'.format(e))
        for _ in range(int(runner.runtime['poll_seconds'])):
            if runner.stop:
                break
            time.sleep(1)


if __name__ == '__main__':
    main()
