"""The unattended dispatcher. Runs under systemd (`Restart=always`) and drives the box from git.

One cycle:

1. fetch `origin`; re-read `runtime.json` (clamped; a bad config keeps the last good one)
2. reap finished jobs and publish their artifacts to `results`
3. reconcile: if the box is idle, open **one wave** — same-type jobs, train **or** eval, never both,
   up to that type's limit. While any job runs, launch nothing. Skipped if paused/draining or if disk
   is low
4. refresh throughput; publish `status.json` to `ops-status`

The loop body is wrapped so a bad job, a transient git error or a malformed config can never kill the
daemon — in normal operation the box has no ssh backstop, so **staying up is the first requirement.**

Launched jobs are detached and self-terminate at `SNEK_MAX_STEPS`, so the daemon never kills
anything. A daemon restart re-adopts still-running jobs by pid; a **machine reboot** (a boot-id
mismatch) marks them `interrupted` so they relaunch and resume. See `_reattach` — that distinction is
a documented incident, not a refinement.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time

from . import config as config_module
from . import gitbus
from . import launch
from .job import parse_job, Job, JobError

TERMINAL = ('done', 'failed')

# `interrupted` is deliberately NOT terminal. A job the machine killed has not finished, so it must
# stay launchable: `_scan_pending` relaunches it from its still-present spec (a training resumes from
# its checkpoint, since `SNEK_MAX_STEPS` is absolute), and `_auto_stage_b_jobs` re-synthesises a wave
# whose previous attempt was cut short. Both are the recovery.

# A synthesised stage-B wave outranks any pending training (default priority 100), so when a training
# wave drains its measurement forms the next wave before any new training starts. That is the whole
# point of the auto chain: never train the next thing before measuring the last.
AUTO_STAGE_B_PRIORITY = 10

# How long the chart window gets to exit after SIGTERM before it is killed. Its only shutdown work
# is closing a figure, so this is generous rather than tight.
VIEWER_STOP_SECONDS = 5

# `SNEK_*` that belong to a *training* and must not be inherited by its measurement. Each would
# change what the wave does rather than what it measures: a step cap is meaningless to an eval, and
# the stage-A knobs describe the screen that already happened.
#
# **This daemon carries no protocol numbers, and that is deliberate.** snek2's held five — the gate,
# the episode count, a suffix — as a second copy of what `eval_plan.py` defines, and they drifted.
# The rule is inverted here: strip what belongs to the training and let `evaluate.py`'s own defaults
# be the protocol. The daemon cannot import the original anyway; it runs on base python so it can
# start before the conda env exists, and `eval_plan` needs numpy.
TRAINING_ONLY_KEYS = ('SNEK_MAX_STEPS', 'SNEK_EVAL_INTERVAL', 'SNEK_GRAPH_EVAL_EPISODES',
                      'SNEK_MIN_CHECKPOINT_SCORE', 'SNEK_INITIAL_COLLECT_STEPS',
                      'SNEK_REPLAY_BUFFER_MAX_LENGTH', 'SNEK_TORCH_THREADS')

# The `SNEK_*` a *measurement* actually depends on, and therefore the only env a batch's arms have to
# agree about to share one wave.
#
# Grouping on the *whole* inherited env is what split snek2's `b45` close-out into **three** waves —
# `{a,c}`, `{b}`, `{d}` — because its arms differed in `SNEK_SEED`, which cannot reach a measurement
# of an already-trained checkpoint. The cost was not cosmetic: three sequential waves of 2/1/1 arms
# measure a batch at a quarter of the intended width.
#
# `SNEK_LEARNING_RATE`, `SNEK_DISCOUNT`, `SNEK_TARGET_UPDATE_PERIOD`, `SNEK_IS_WEIGHTS` and
# `SNEK_FORK_BRANCHES` are the same class: they shaped the weights, and the weights are in the
# checkpoint. `SNEK_FC_LAYERS` is excluded for a different reason — `arch.json` governs the network
# at eval time and `tools/restore.py` reads it, so a differing architecture is already handled.
#
# `SNEK_ZERO_OBS` is here because it changes the *observation*, which is an input to the policy at
# measurement time, not a property of training. `tests/test_desktop_runner.py` checks this tuple against
# `env/constants.py` so a renamed knob fails loudly instead of silently splitting waves.
EVAL_RELEVANT_ENV = ('SNEK_ZERO_OBS', 'SNEK_CHASE_SAFE_SHAPING', 'SNEK_CHASE_SAFE_GATE',
                     'SNEK_FREE_SPACE_SHAPING', 'SNEK_FREE_SPACE_GATE',
                     'SNEK_FOOD_DISTANCE_REWARD', 'SNEK_PERFECT_GAME_REWARD')


def inherited_eval_env(env):
    """A training's env, stripped of what belongs to the training rather than the measurement."""
    return {key: value for key, value in (env or {}).items() if key not in TRAINING_ONLY_KEYS}


def stage_b_group_env(env):
    """The measurement-relevant slice of a training's env — the group key for a wave.

    Two arms of a batch belong in one wave unless they disagree about something a measurement can
    see. See `EVAL_RELEVANT_ENV` for why that is a short list and what keying on the long one cost.
    """
    return {key: value for key, value in (env or {}).items() if key in EVAL_RELEVANT_ENV}


def agreed_env(envs):
    """Every key a group's arms agree on, value included. The env its wave runs under.

    A group may hold arms whose envs differ in measurement-irrelevant ways, so there is no single
    arm's env to hand the wave. The agreed subset is the honest answer — every setting the wave
    inherits is one all of its arms were trained with — and it drops exactly the keys the looser
    group key stopped caring about. Handing the first arm's env to the whole wave would quietly
    attribute one arm's seed to all four.
    """
    envs = [env or {} for env in envs]
    if not envs:
        return {}
    shared = dict(envs[0])
    for env in envs[1:]:
        for key in list(shared):
            if env.get(key) != shared[key]:
                del shared[key]
    return shared


def wants_stage_b(job_type, ok, auto_enabled):
    """Whether a just-finished job earns an automatic stage-B wave.

    Only a **training** that finished successfully: smoke and benchmark runs share the `trainer`
    category but are throwaway, and a failed run has no checkpoint worth measuring.
    """
    return bool(auto_enabled) and ok and job_type == 'train'


class _StubJob(object):
    """Stands in for a Job re-adopted from the ledger after a restart, with no spec in hand."""

    def __init__(self, job_id, record):
        self.id = job_id
        self.type = record.get('type', 'train')
        self.project = record.get('project', 'snek3')
        self.policy = record.get('policy')
        # The same `policies or [policy]` fallback `Job` does, and load-bearing rather than tidy:
        # this is the path that re-adopts a *running wave* after a daemon restart, so a missing
        # `policies` here would publish one arm of four.
        self.policies = [name for name in (record.get('policies') or [self.policy]) if name]
        self.max_steps = record.get('max_steps')
        self.label = record.get('label', '')
        self.selector = record.get('selector')
        self.episodes = record.get('episodes')
        self.eval_shards = record.get('eval_shards')
        self.eval_args = []
        self.env = record.get('env') or {}

    @property
    def category(self):
        return 'eval' if self.type == 'eval' else 'trainer'


def trigger_path(host):
    """The file that makes the daemon do a network cycle now, rather than at its next `git_seconds`.

    Derived from `LEDGER_PATH`'s directory rather than being its own `host.env` key, deliberately:
    `host.env` is not in git — only `host.env.example` is — so a new required key would have to be
    hand-edited on the box during the same deploy that started depending on it, and a deploy that
    landed first would crash `load_host_config` on every start. `TRIGGER_PATH` still overrides it.
    """
    return host.get('TRIGGER_PATH') or os.path.join(
        os.path.dirname(host['LEDGER_PATH']), 'trigger')


class Runner(object):
    def __init__(self, host):
        self.host = host
        self.runtime = dict(config_module.RUNTIME_DEFAULTS)
        config_module.clamp_runtime(self.runtime, host)
        self.config_notes = []
        self.ledger = self._load_ledger()
        self.running = {}          # job_id -> RunningJob
        self._queued = []          # launchable jobs waiting for the next wave, priority order
        self._unpushed = []        # branches with local-only commits, surfaced in status.json
        # When the loop last *attempted* its network half. 0 so the first cycle does one.
        self._last_git = 0.0
        self.stop = False
        # The chart window, and the arms it was opened for. `None` means no window is up.
        self._viewer = None
        self._viewer_policies = ()
        self._reattach()

    # ------------------------------------------------------------------ ledger

    def _load_ledger(self):
        try:
            with open(self.host['LEDGER_PATH']) as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    def _save_ledger(self):
        os.makedirs(os.path.dirname(self.host['LEDGER_PATH']), exist_ok=True)
        staging = self.host['LEDGER_PATH'] + '.tmp'
        with open(staging, 'w') as handle:
            json.dump(self.ledger, handle, indent=2)
        os.replace(staging, self.host['LEDGER_PATH'])

    def _reattach(self):
        """Re-adopt jobs the ledger says were running, or classify the ones that are gone.

        **The boot id separates a daemon restart from a machine reboot, and the two need opposite
        handling.** Jobs are detached, so across a *daemon* restart they really do keep running and a
        dead pid means the job genuinely finished. A *reboot* kills every child, so the same dead pid
        means the job was cut off partway — and calling that `done` is wrong twice over: a truncated
        training gets published as a finished arm, and its measurement then measures the partial arm.
        Both were silent.

        Recording the boot id at launch makes the distinction exact, and closes a second hole: on a
        mismatch the pid is never consulted at all, so a **recycled** pid cannot be mistaken for a
        live job. That matters because pids restart low after a reboot, and a false match would
        re-adopt a phantom that never exits — which the wave barrier turns into an idle box forever.
        """
        boot = boot_id()
        for job_id, record in self.ledger.items():
            if record.get('state') != 'running':
                continue
            recorded_boot = record.get('boot')
            rebooted = bool(boot and recorded_boot and recorded_boot != boot)
            pid = record.get('pid')
            if not rebooted and pid and launch.pid_alive(pid):
                # Still ours and still alive. Stamp the current boot id onto a record that predates
                # the field, so the *next* restart can classify it properly.
                if boot and not recorded_boot:
                    record['boot'] = boot
                self.running[job_id] = launch.RunningJob(
                    _StubJob(job_id, record), record.get('policy'), pid,
                    record.get('log', ''), popen=None, started=record.get('started'))
            elif rebooted:
                record['state'] = 'interrupted'
                record['finished'] = time.time()
                record['note'] = 'killed by a reboot, not finished'
                record['restarts'] = int(record.get('restarts') or 0) + 1
                # No `stage_b: pending` here — the arm is unfinished, so it earns its measurement
                # when it actually completes. `interrupted` is non-terminal, so the relaunch happens
                # on the next dispatch.
            else:
                record['state'] = 'done'
                record['finished'] = time.time()
                record['note'] = 'completed while runner was down'
                # Same boot, dead pid: the job ran to its own end while the daemon was down. The
                # exit code is unknown across reparenting, so success is assumed — and on that same
                # assumption the arm still earns its measurement, so the auto chain does not
                # silently skip a restart-straddling arm.
                if wants_stage_b(record.get('type'), True,
                                 self.runtime.get('auto_stage_b', True)):
                    record['stage_b'] = 'pending'
        self._save_ledger()

    # -------------------------------------------------------------------- loop

    def git_due(self):
        """Whether this cycle does its network half. `git_seconds` of 0 means every cycle."""
        interval = int(self.runtime.get('git_seconds', 0) or 0)
        return interval <= 0 or (time.time() - self._last_git) >= interval

    def take_trigger(self):
        """Consumes a manual trigger if one is waiting. Returns whether there was one.

        **The unlink *is* the test.** Checking for the file and then removing it would be two
        operations with a window between them; a single `unlink` cannot be seen by two readers.
        """
        try:
            os.unlink(trigger_path(self.host))
        except OSError:
            return False
        sys.stderr.write('manual trigger consumed; forcing a git cycle\n')
        return True

    def poll_once(self, git=None):
        """One cycle. `git=None` decides from `git_seconds`; True forces it, as a trigger does.

        **The cycle has a local half and a network half, and they run at different rates.** The local
        half — reap, re-read the already-fetched `ops` ref, dispatch — touches no network and runs
        every `poll_seconds`, so work the box generates for itself still starts within seconds. The
        network half is one fetch, one status push and one retry of any local-only commit.
        """
        if git is None:
            git = self.git_due()
        if git:
            # Stamped before the attempt, not after: an attempt is what costs traffic, so a fetch
            # that fails — this box's DNS for github flaps — must not turn into a retry every
            # `poll_seconds`, which would put the traffic straight back where it was.
            self._last_git = time.time()
            gitbus.fetch(self.host)
            self._unpushed = gitbus.push_unpushed(self.host)
        self._apply_runtime()
        self._reap()
        # Scanned every poll, not only when dispatching, so `status.json`'s queue stays current
        # while a wave occupies the box or the daemon is paused.
        self._queued = self._scan_pending()
        if not (self.runtime['paused'] or self.runtime['drain']):
            self._dispatch()
        for running_job in self.running.values():
            launch.update_throughput(running_job, self.host)
        self._ensure_viewer()
        if git:
            self._publish()

    def _apply_runtime(self):
        config, notes = config_module.parse_runtime_config(
            gitbus.read_runtime_text(self.host), self.host)
        if config is None:
            self.config_notes = ['using last-known-good config; errors: ' + '; '.join(notes)]
        else:
            self.runtime, self.config_notes = config, notes

    def _counts(self):
        counts = {'trainer': 0, 'eval': 0}
        for running_job in self.running.values():
            counts[running_job.job.category] += 1
        return counts

    # ------------------------------------------------------- the automatic chain

    def _auto_stage_b_jobs(self):
        """Stage-B waves for trainings that finished and have not been measured yet.

        Synthesised fresh each dispatch, never persisted as specs, and driven entirely off the ledger
        so it survives a daemon restart — the `stage_b: pending` markers and each training's stored
        `env` both persist.

        **One job per batch, not one per arm.** A batch's arms become a single `evaluate.py` job
        carrying every policy, so its shards move to whichever arm still has checkpoints instead of a
        finished arm's share of the box going idle. Grouping is safe because of the wave barrier:
        `_dispatch` returns early while anything runs, so by the time this is read the set of markers
        is closed. That is also why grouping happens at dispatch rather than in `_scan_pending` — an
        id that shifted as more markers appeared would let a partly-finished wave relaunch under a
        new id and redo the work.

        **What stops a re-measure is `_measured_policies`, not the job id.** A `stage_b: pending`
        marker is never cleared, so the marker set says "this arm was trained", not "this arm still
        needs measuring".
        """
        if not self.runtime.get('auto_stage_b', True):
            return []
        measured = self._measured_policies()
        groups = {}
        for record in sorted(self.ledger.values(), key=lambda rec: str(rec.get('policy'))):
            if record.get('stage_b') != 'pending' or not record.get('policy'):
                continue
            if record['policy'] in measured:
                continue
            env = inherited_eval_env(record.get('env'))
            key = (batch_of(record['policy']),
                   tuple(sorted(stage_b_group_env(env).items())))
            groups.setdefault(key, {'policies': [], 'envs': []})
            groups[key]['policies'].append(record['policy'])
            groups[key]['envs'].append(env)
        jobs, taken = [], set()
        for (batch, _), group in sorted(groups.items(), key=lambda item: item[1]['policies'][0]):
            job_id = self._stage_b_id(batch, taken)
            if job_id is None:
                continue
            taken.add(job_id)
            jobs.append(Job(id=job_id, type='eval', policies=sorted(group['policies']),
                            env=agreed_env(group['envs']),
                            priority=AUTO_STAGE_B_PRIORITY))
        return jobs

    def _measured_policies(self):
        """Every policy a stage-B wave has already measured, or is measuring right now.

        Read off the wave *records*, never off the trainings' markers: a marker is set when a
        training finishes and is never cleared, so the marker set is everything ever trained. What
        used to stop a re-measure in snek2 was incidental — the id `<policy>-closeout` was already
        taken — and grouping a batch into one wave changed the id, so on the first restart after the
        wave controller shipped the daemon forecast **thirteen** waves it had already run.

        Covering by *policy* rather than by id handles three cases at once: legacy per-arm ids, an
        arm mid-migration, and a batch measured across several waves.

        `failed` counts as measured, because the reason usually is not transient — but it is
        surfaced under `attention` in `status.json`, which is the fix for snek2 silently never
        retrying one and costing a batch its whole measurement. `interrupted` does **not** count, so
        a reboot's half-finished wave is regrouped and relaunched.
        """
        covered = set()

        def add(job_id, policies):
            if _STAGE_B_ID_RE.search(str(job_id)):
                covered.update(str(name) for name in policies if name)

        for job_id, running_job in self.running.items():
            add(job_id, running_job.job.policies or [running_job.policy])
        for job_id, record in self.ledger.items():
            if record.get('state') in TERMINAL:
                add(job_id, record.get('policies') or [record.get('policy')])
        return covered

    def _stage_b_id(self, batch, taken=()):
        """`<batch>-stageb`, or `<batch>-stageb-w<k>` for a later wave of the same batch.

        The plain id is not enough and the case is real: snek2's `b20` ran 36 arms under one prefix
        in nine waves of four, and each wave needs its own id or the second collides with the first.
        `k` counts up to the first free id, which does not churn between polls because ledger records
        are never deleted.

        Free means: not claimed in this same pass, not running, and either absent from the ledger or
        **non-terminal** — an `interrupted` wave is relaunched under its own id so its completed rows
        are still there to resume from.

        `taken` is the ids claimed in this same pass, which the ledger cannot know about yet. It
        matters whenever one batch splits into two waves for disagreeing envs: without it both groups
        get the same id and the second silently replaces the first.
        """
        base = '{0}-stageb'.format(batch)
        for index in range(1, 100):
            job_id = base if index == 1 else '{0}-w{1}'.format(base, index)
            if job_id in taken or job_id in self.running:
                continue
            if self.ledger.get(job_id, {}).get('state') in TERMINAL:
                continue
            return job_id
        return None

    # ----------------------------------------------------------- scan and dispatch

    def _scan_pending(self):
        """The launchable jobs waiting for the next wave, in priority order.

        Parses every pending spec, marks a malformed one failed in the ledger — persisting that
        immediately, since this runs even when dispatch is skipped — and drops any job already
        terminal or running. Synthesised stage-B waves join the same pool; a manual spec of the same
        id is kept over the synthesised one.
        """
        desired = []
        marked = False
        for name, text in gitbus.read_pending_jobs(self.host):
            try:
                job = parse_job(text, name)
            except JobError as error:
                job_id = name[:-5] if name.endswith('.json') else name
                if self.ledger.get(job_id, {}).get('state') != 'failed':
                    self.ledger[job_id] = {'state': 'failed', 'type': 'unknown',
                                           'error': str(error), 'finished': time.time()}
                    marked = True
                continue
            state = self.ledger.get(job.id, {}).get('state')
            if state in TERMINAL or job.id in self.running:
                continue
            desired.append(job)
        seen = {job.id for job in desired}
        desired += [job for job in self._auto_stage_b_jobs() if job.id not in seen]
        desired.sort(key=lambda job: job.priority)
        if marked:
            self._save_ledger()
        return desired

    def _dispatch(self):
        free = _disk_free_gb(self.host['REPO_PATH'])
        if free is not None and free < self.runtime['disk_min_gb']:
            self.config_notes = self.config_notes + [
                'disk low ({0} GB < {1}) — not launching'.format(
                    free, self.runtime['disk_min_gb'])]
            return
        limits = {'trainer': self.runtime['max_trainers'], 'eval': self.runtime['max_evals']}
        desired = self._queued
        # **Wave-barrier scheduling.** A wave is a set of same-type jobs launched together, up to
        # that type's limit, and nothing new starts until the whole wave finishes. So trainings and
        # evals never overlap, and a freed slot is not backfilled mid-wave — the box idles on the
        # stragglers until the slowest job is done, then the next wave opens. Pending specs are still
        # scanned while a wave runs, so a malformed one is marked failed promptly.
        if self.running or not desired:
            return
        # The highest-priority pending job picks the wave's type; only jobs of that type join it.
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
            running_job = launch.spawn(job, self.host, self.runtime)
        except Exception as error:      # spawn is the one place a job can fail before running
            self.ledger[job.id] = {'state': 'failed', 'type': job.type,
                                   'error': 'spawn failed: {0}'.format(error),
                                   'finished': time.time()}
            return
        prior = self.ledger.get(job.id) or {}
        self.running[job.id] = running_job
        self.ledger[job.id] = {
            'state': 'running', 'type': job.type, 'project': getattr(job, 'project', 'snek3'),
            'policy': running_job.policy,
            # Every arm this job owns. A wave has several; a training has one and records it both
            # ways, so a reader never has to know which.
            'policies': list(job.policies),
            'pid': running_job.pid, 'log': running_job.log_path, 'started': running_job.started,
            # Carried so `at_a_glance` can show a percentage and a label for a job re-adopted after a
            # restart, with the spec not in hand.
            'max_steps': job.max_steps, 'label': job.label,
            'selector': getattr(job, 'selector', None), 'episodes': getattr(job, 'episodes', None),
            # The boot this pid belongs to. Without it a pid outliving a reboot reads as a live job.
            'boot': boot_id(),
            # Kept so an automatic stage-B wave can rebuild the exact env after a daemon restart.
            'env': job.env}
        # Carried across the relaunch, so a box that reboots repeatedly shows a climbing count rather
        # than a job that looks freshly started.
        if prior.get('restarts'):
            self.ledger[job.id]['restarts'] = prior['restarts']

    def _reap(self):
        for job_id in list(self.running):
            running_job = self.running[job_id]
            if running_job.is_alive():
                continue
            code = running_job.returncode()
            ok = (code == 0) or (code is None and not _log_has_traceback(running_job.log_path))
            record = self.ledger.setdefault(job_id, {})
            record.update({'state': 'done' if ok else 'failed', 'type': running_job.job.type,
                           'policy': running_job.policy, 'finished': time.time(),
                           'returncode': code})
            if wants_stage_b(running_job.job.type, ok, self.runtime.get('auto_stage_b', True)):
                record['stage_b'] = 'pending'    # picked up by _auto_stage_b_jobs next dispatch
            del self.running[job_id]
            if ok:
                self._publish_results(running_job)
        self._save_ledger()

    def _ensure_viewer(self):
        """Keeps a chart window on the box's monitor for whatever is running.

        **Why this is on in snek3 and was off in snek2.** snek2's window was the *trainer's* own
        in-process cv2 canvas, and one fatal XIO error under memory pressure killed all four arms at
        once on 2026-08-09. snek3's trainer never draws: this is `tools/chart_viewer.py` in a separate
        process reading the PNGs the trainer already writes, so a display failure now costs a window
        and nothing else.

        **A window that exited cleanly stays closed until the arms change.** The tool treats the user
        closing it as an instruction rather than a failure, and respawning it every 30 s would fight
        whoever is sitting at the box. A window that *crashed* is respawned, because that is the case
        the user asked to be protected from. So the rule is: open when the set of running policies
        changes, reopen on a crash, and never re-open a deliberate close.
        """
        policies = tuple(sorted(
            {name for running_job in self.running.values() for name in running_job.job.policies}))
        alive = self._viewer is not None and self._viewer.poll() is None

        if not policies or not self.runtime.get('viewer', True):
            # Nothing to watch, or turned off on `ops`. Closed rather than left showing a batch that
            # has finished.
            self._close_viewer()
            return

        if alive:
            if policies == self._viewer_policies:
                return
            # An arm joined or finished, so the panel list is stale. The list is explicit, so a
            # restart is the only way to change it.
            self._close_viewer()
        elif self._viewer is not None:
            status = self._viewer.returncode
            unchanged = policies == self._viewer_policies
            self._viewer, self._viewer_policies = None, ()
            if status == 0 and unchanged:
                return                    # closed on purpose; leave it closed
            if status != 0:
                sys.stderr.write(
                    'chart viewer died with status {0}; reopening\n'.format(status))

        pids = sorted(running_job.pid for running_job in self.running.values())
        try:
            self._viewer = launch.spawn_viewer(policies, pids, self.host)
        except OSError as error:
            sys.stderr.write('could not open the chart viewer: {0}\n'.format(error))
            self._viewer = None
        self._viewer_policies = policies if self._viewer is not None else ()

    def _close_viewer(self):
        """Terminates the window and **reaps it**, then forgets it.

        The `wait` is the point. Dropping the handle after `terminate` leaves a zombie for as long as
        the daemon runs, and the daemon runs for weeks — so a batch that restarts the window a few
        times a day would accumulate them. The timeout is short because the viewer's only shutdown
        work is closing a figure; if it somehow ignores SIGTERM it is killed.
        """
        if self._viewer is None:
            self._viewer_policies = ()
            return
        if self._viewer.poll() is None:
            self._viewer.terminate()
            try:
                self._viewer.wait(timeout=VIEWER_STOP_SECONDS)
            except subprocess.TimeoutExpired:
                self._viewer.kill()
                try:
                    self._viewer.wait(timeout=VIEWER_STOP_SECONDS)
                except subprocess.TimeoutExpired:
                    sys.stderr.write('chart viewer would not die; leaking one process\n')
        self._viewer, self._viewer_policies = None, ()

    def _publish_results(self, running_job):
        """Publishes every arm this job owned, **one push per arm**.

        Per arm rather than one push for the whole wave, because the box's DNS for github flaps and a
        wave concentrates four arms behind one push. Errors are caught per arm so the remaining arms
        are still attempted, and a push that does not land leaves the commit local — `push_unpushed`
        carries it on the next network cycle, and until it does the branch is listed under
        `attention` in `status.json` rather than being reported as published.
        """
        runs = os.path.join(self.host['SNEK_DIR'], 'runs')
        names = sorted(os.listdir(runs)) if os.path.isdir(runs) else []
        policies = [str(name) for name in
                    (running_job.job.policies or [running_job.policy]) if name]
        for policy in policies:
            artifacts = [os.path.join(runs, name) for name in names
                         if name == policy + '.md' or name.startswith(policy + '.')
                         or name.startswith(policy + '_')]
            try:
                if not gitbus.publish_results(self.host, running_job.job, artifacts):
                    self._unpushed = sorted(set(self._unpushed) | {self.host['RESULTS_BRANCH']})
            except Exception as error:   # best-effort, but say so in the journal, not silently
                sys.stderr.write('publish_results({0}, {1}) failed: {2}\n'.format(
                    running_job.job.id, policy, error))

    # ------------------------------------------------------------------ status

    def _anticipated_order(self):
        """The pending queue folded forward through the wave scheduler.

        Each queued batch's stage-B wave is slotted where it will actually run. Shared by the ledger
        view and the at-a-glance summary so both agree.
        """
        limits = {'trainer': self.runtime['max_trainers'], 'eval': self.runtime['max_evals']}
        auto = self.runtime.get('auto_stage_b', True)
        queued = [{'id': job.id, 'type': job.type, 'policy': job.policy,
                   'policies': list(job.policies), 'priority': job.priority}
                  for job in self._queued if job.id not in self.running]
        running = [{'id': job_id, 'type': running_job.job.type, 'policy': running_job.policy,
                    'policies': list(running_job.job.policies)}
                   for job_id, running_job in self.running.items()]
        existing = set(self.ledger) | set(self.running) | {job['id'] for job in queued}
        return anticipated_queue(queued, running, limits, auto, existing)

    def _batch_labels(self):
        """batch id -> human label, from queued specs, running jobs and the ledger.

        A batch's arms share a label, so the first non-empty one found wins.
        """
        labels = {}

        def add(job_id, label):
            if label:
                labels.setdefault(batch_of(job_id), label)

        for job in self._queued:
            add(job.id, getattr(job, 'label', ''))
        for job_id, running_job in self.running.items():
            add(job_id, getattr(running_job.job, 'label', ''))
        for job_id, record in self.ledger.items():
            add(job_id, record.get('label', ''))
        return labels

    def _attention(self):
        """Lines for things that need a human. Empty is the normal state.

        **A failed stage-B wave is never retried automatically** — the reason usually is not
        transient — so if it is not surfaced it is invisible, and snek2 lost a batch's whole
        measurement that way. Same for a branch holding a local-only commit: the results exist on the
        box and nowhere else, which reads identically to a pass that found nothing.
        """
        lines = []
        for branch in self._unpushed:
            lines.append('** {0} has local-only commit(s): results are on the box but not on '
                         'github. Retried every network cycle.'.format(branch))
        failed = sorted(
            (job_id for job_id, record in self.ledger.items()
             if record.get('state') == 'failed' and record.get('type') == 'eval'),
            key=lambda job_id: self.ledger[job_id].get('finished') or 0, reverse=True)
        for job_id in failed[:5]:
            record = self.ledger[job_id]
            lines.append('** {0} failed and will NOT be retried automatically ({1}). Its arms count '
                         'as measured; delete its ledger record to re-queue.'.format(
                             job_id, record.get('error') or 'rc={0}'.format(
                                 record.get('returncode'))))
        return lines

    def _ledger_view(self, order):
        """The id -> state map published in `status.json`, ordered newest/active first.

        The pending queue at the top, in the order the wave scheduler will launch it so the next job
        to run sits highest; then the running jobs; then the finished history most-recent-first. This
        is the reverse of the on-disk ledger, whose insertion order is oldest-first — only the
        *published* view is reordered, so the authoritative ledger and every restart path are
        untouched. A real ledger state always wins over the synthetic `queued` on an id overlap.
        """
        queued_ids = [job['id'] for job in order
                      if job['id'] not in self.ledger and job['id'] not in self.running]
        running_ids = [job_id for job_id, record in self.ledger.items()
                       if record.get('state') == 'running']
        for job_id in self.running:                    # a live job the ledger somehow missed
            if job_id not in running_ids and job_id not in queued_ids:
                running_ids.append(job_id)
        finished_ids = sorted(
            (job_id for job_id, record in self.ledger.items()
             if record.get('state') != 'running'),
            key=lambda job_id: self.ledger[job_id].get('finished') or 0, reverse=True)
        view = {}
        for job_id in queued_ids:
            view[job_id] = 'queued'
        for job_id in running_ids + finished_ids:
            view[job_id] = self.ledger.get(job_id, {}).get('state')
        return view

    def _publish(self):
        order = self._anticipated_order()
        running = [{'id': running_job.job.id, 'type': running_job.job.type,
                    'policy': running_job.policy,
                    'policies': list(running_job.job.policies),
                    'pid': running_job.pid, 'step': running_job.current_step,
                    'max_steps': getattr(running_job.job, 'max_steps', None),
                    'steps_per_sec': running_job.steps_per_sec,
                    'elapsed_s': round(time.time() - running_job.started)}
                   for running_job in self.running.values()]
        status = {
            'iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'ts': time.time(),
            'project': 'snek3',
            # A human summary at the top: one line per running batch with a percentage, one per
            # queued batch-phase, and anything needing a human under `attention` — so the box's
            # state reads at a glance without parsing the ledger.
            'at_a_glance': build_at_a_glance(
                running, order, self._batch_labels(),
                [flag for flag in HOLD_FLAGS if self.runtime.get(flag)],
                self._attention()),
            'runtime': self.runtime,
            'config_notes': self.config_notes,
            'counts': self._counts(),
            'running': running,
            'ledger': self._ledger_view(order),
            'viewer': {'up': self._viewer is not None and self._viewer.poll() is None,
                       'policies': list(self._viewer_policies)},
            'disk_free_gb': _disk_free_gb(self.host['REPO_PATH']),
            'load_avg': list(os.getloadavg()),
        }
        try:
            if not gitbus.publish_status(self.host, json.dumps(status, indent=2)):
                self._unpushed = sorted(set(self._unpushed) | {self.host['STATUS_BRANCH']})
            else:
                self._unpushed = [branch for branch in self._unpushed
                                  if branch != self.host['STATUS_BRANCH']]
        except Exception as error:
            sys.stderr.write('publish_status failed: {0}\n'.format(error))


# ---------------------------------------------------------------- pure helpers

_BATCH_RE = re.compile(r'^(b\d+)')
_STAGE_B_ID_RE = re.compile(r'-stageb(-w\d+)?$')


def batch_of(job_id):
    """The batch a job id belongs to.

    The leading `b<number>` — `b1a-thing-seed1` -> `b1` — or the id up to the first `-` for anything
    that is not a b-numbered arm, so `smoke-1` -> `smoke`.
    """
    job_id = job_id or ''
    match = _BATCH_RE.match(job_id)
    return match.group(1) if match else (job_id.split('-')[0] or job_id)


def phase_of(job_id, job_type):
    """A job's phase: `'stage B'`, `'training'`, `'eval'`, `'smoke'` or `'benchmark'`.

    Read off the id suffix the daemon mints, with the job type as the fallback. **The `-w<k>` tail
    has to be part of the pattern**: a batch's second wave is `b1-stageb-w2`, and a plain
    `endswith('-stageb')` would call that a bare `'eval'` and split the at-a-glance grouping in two.
    """
    if _STAGE_B_ID_RE.search(str(job_id or '')):
        return 'stage B'
    return {'train': 'training'}.get(job_type, job_type or 'eval')


HOLD_FLAGS = ('paused', 'drain')
_HOLD_WORDS = {'paused': 'paused', 'drain': 'draining'}


def hold_notice(held_by):
    """The `at_a_glance.queued` line for a held queue, or None when nothing holds it.

    Split out so the wording is testable on its own, and ordered by `HOLD_FLAGS` rather than by the
    caller so the line is stable across polls.
    """
    held = [flag for flag in HOLD_FLAGS if flag in held_by]
    if not held:
        return None
    return '** queue {0}: nothing new will start. Set {1} in runtime.json on ops to resume'.format(
        ' and '.join(_HOLD_WORDS[flag] for flag in held),
        ' and '.join('"{0}": false'.format(flag) for flag in held))


def build_at_a_glance(running, queued_order, labels, held_by=(), attention=()):
    """The `at_a_glance` block: `{'running': [str], 'queued': [str], 'attention': [str]}`.

    One line per `(batch, phase)` group in first-seen order. A running trainer dict carries
    `step`/`max_steps`, from which each running line shows the mean percent done across that batch's
    arms. `labels` maps a batch id to its human description.

    `held_by` is whichever `HOLD_FLAGS` are set; when any are, `hold_notice` goes **first** in
    `queued`, ahead of the batch lines — unconditional on the queue being non-empty, because an empty
    queue under a hold is the case most in need of the explanation.

    Kept a pure function of plain dicts, so it is testable without a live box.
    """
    def group(jobs):
        groups, index = [], {}
        for job in jobs:
            key = (batch_of(job['id']), phase_of(job['id'], job.get('type')))
            if key not in index:
                index[key] = len(groups)
                groups.append((key[0], key[1], []))
            groups[index[key]][2].append(job)
        return groups

    def described(batch):
        label = labels.get(batch)
        if not label:
            return ''
        return ' — ' + (label if len(label) <= 80 else label[:77] + '...')

    def arms(jobs):
        """"N arms", counted in **policies**, not in jobs.

        One eval job is a wave of four arms, so counting jobs would report a batch's whole
        measurement as "1 arm" — the number a reader uses to check that nothing was dropped.
        """
        count = sum(len(job.get('policies') or [job.get('policy')] or []) or 1 for job in jobs)
        return '{0} arm{1}'.format(count, '' if count == 1 else 's')

    running_lines = []
    for batch, phase, jobs in group(running):
        percents = [100.0 * job['step'] / job['max_steps']
                    for job in jobs if job.get('step') and job.get('max_steps')]
        percent = (' {0}%'.format(int(round(sum(percents) / len(percents))))
                   if percents else '')
        running_lines.append('{0}{1} — {2}{3} ({4})'.format(
            batch, described(batch), phase, percent, arms(jobs)))

    queued_lines = []
    notice = hold_notice(held_by)
    if notice:
        queued_lines.append(notice)
    for batch, phase, jobs in group(queued_order):
        queued_lines.append('{0} {1}{2} — queued ({3})'.format(
            batch, phase, described(batch), arms(jobs)))

    return {'running': running_lines, 'queued': queued_lines, 'attention': list(attention)}


def anticipated_queue(queued, running, limits, auto_stage_b, existing_ids):
    """The pending queue in the order it is expected to launch, waves and all.

    Modelled straight from `_dispatch`: repeatedly form a wave from the highest-priority job's
    category up to that category's limit, drain it, and let the trainings in it spawn **one** stage-B
    wave per batch (priority `AUTO_STAGE_B_PRIORITY`) that competes in the next wave. A measurement
    outranks a queued training, so a batch's wave always slots in ahead of the following batch —
    the exact interleaving the box will run.

    A batch's four trainings collapsing into one stage-B row is the *reason* this is grouped: snek2's
    ungrouped forecast showed four close-out jobs and then four HOF jobs, eight rows for what is now
    one process.

    `queued` and `running` are lists of dicts with id/type/policy — `queued` also has priority, a
    wave also has `policies`. `existing_ids` is every id already known, so a job that exists
    somewhere is never invented twice.
    """
    def category(job_type):
        return 'eval' if job_type == 'eval' else 'trainer'

    seen = set(existing_ids)

    def stage_b_for(jobs):
        """One stage-B wave dict per batch for a set of just-placed trainings.

        Grouped by batch exactly as `_auto_stage_b_jobs` groups the real markers, but **not** by env:
        a forecast has no env to read, and getting the count of rows right is what this is for.
        """
        out = []
        by_batch = {}
        for job in jobs:
            policy = job.get('policy')
            if not policy or not wants_stage_b(job.get('type'), True, auto_stage_b):
                continue
            by_batch.setdefault(batch_of(policy), []).append(policy)
        for batch, policies in sorted(by_batch.items()):
            job_id = '{0}-stageb'.format(batch)
            if job_id in seen:
                continue
            seen.add(job_id)
            out.append({'id': job_id, 'type': 'eval', 'policy': policies[0],
                        'policies': sorted(policies), 'priority': AUTO_STAGE_B_PRIORITY})
        return out

    pool = [dict(job) for job in queued]
    pool += stage_b_for(running)                    # follow-ons for jobs on the box now

    order = []
    while pool:
        pool.sort(key=lambda job: job['priority'])
        wave_type = category(pool[0]['type'])
        wave = []
        for job in pool:
            if category(job['type']) == wave_type and len(wave) < limits[wave_type]:
                wave.append(job)
        wave_ids = {job['id'] for job in wave}
        order.extend(wave)
        pool = [job for job in pool if job['id'] not in wave_ids]
        pool += stage_b_for(wave)       # the wave's trainings spawn their batches' measurements
    return order


BOOT_ID_PATH = '/proc/sys/kernel/random/boot_id'
_BOOT_ID = []   # one-element cache; the value cannot change without the process dying


def boot_id():
    """This boot's kernel id, or None where the kernel does not publish one.

    Cached, because it is read on every launch and cannot change while the daemon lives — a new boot
    is a new process by definition.

    **None means "cannot tell", and every caller degrades to the old pid-only behaviour** rather than
    guessing. That keeps the daemon working on a box without procfs — a mac, a container — where the
    reboot case cannot arise the same way, and it means a record written before this field existed is
    classified exactly as it was, instead of reading as a reboot because its `boot` key is absent.
    """
    if not _BOOT_ID:
        try:
            with open(BOOT_ID_PATH) as handle:
                _BOOT_ID.append(handle.read().strip() or None)
        except OSError:
            _BOOT_ID.append(None)
    return _BOOT_ID[0]


def _log_has_traceback(log_path):
    """Whether a log's tail holds a traceback. The success test for a re-adopted job.

    Only the tail, because a training log is megabytes and a traceback that mattered is at the end.
    """
    try:
        size = os.path.getsize(log_path)
        with open(log_path, 'rb') as handle:
            handle.seek(max(0, size - 4096))
            tail = handle.read().decode('utf-8', 'replace')
    except OSError:
        return False
    return 'Traceback (most recent call last)' in tail


def _disk_free_gb(path):
    try:
        stats = os.statvfs(path)
        return round(stats.f_bavail * stats.f_frsize / 1e9, 1)
    except OSError:
        return None


def main():
    host_env = os.environ.get(
        'SNEK_RUNNER_HOST_ENV',
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'config', 'host.env'))
    host = config_module.load_host_config(host_env)
    runner = Runner(host)

    def handle(_signal, _frame):
        runner.stop = True      # stop the daemon; detached jobs keep running
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    forced = False
    while not runner.stop:
        try:
            # `True if forced else None`: a trigger forces the network half, otherwise `poll_once`
            # decides from `git_seconds`. Not `forced` alone — False would *suppress* a due cycle.
            runner.poll_once(git=True if forced else None)
        except Exception as error:      # the loop must never die
            sys.stderr.write('poll error: {0}\n'.format(error))
        forced = _wait_for_next_poll(runner)
    # The jobs are detached and keep running; the window is not, and one left behind would show a
    # batch nobody is watching any more. `KillMode=process` spares it, so closing it is on us.
    runner._close_viewer()


def _wait_for_next_poll(runner):
    """Sleeps `poll_seconds` in one-second steps, returning early on a stop or a manual trigger.

    Returns True when a trigger was consumed, so the caller forces the next cycle's network half. The
    one-second granularity is what makes an ssh trigger feel immediate; it predates the trigger and
    is why the trigger could be a plain file rather than a signal handler.
    """
    for _ in range(int(runner.runtime['poll_seconds'])):
        if runner.stop:
            return False
        if runner.take_trigger():
            return True
        time.sleep(1)
    return False


if __name__ == '__main__':
    main()
