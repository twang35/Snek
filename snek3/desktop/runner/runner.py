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
from .job import parse_job, to_ascii, Job, JobError

TERMINAL = ('done', 'failed')

# `interrupted` is deliberately NOT terminal. A job the machine killed has not finished, so it must
# stay launchable: `_scan_pending` relaunches it from its still-present spec (a training resumes from
# its checkpoint, since `SNEK_MAX_STEPS` is absolute), and `_auto_jobs` re-synthesises a wave
# whose previous attempt was cut short. Both are the recovery.

# **The automatic chain: `training -> stageb -> hof5000 -> hof30k`, over the same arms.** A job that
# finishes well earns the next pass; a synthesised pass outranks any pending training (default
# priority 100), so when a training wave drains, its stage B, then its hof5000, then its hof30k form
# the next three waves before any new training starts. That is the whole point of the chain: never
# train the next thing before measuring the last — and since 2026-09-04, measuring means all three
# passes, because every batch's hof passes had been queued by hand and the b12/b13 ones were not.
#
# The passes' selectors, depths, labels and seeds live in `tools/closeout.py`'s `PASSES`; the daemon
# dispatches `tools.closeout <arms> --pass <name>` and carries only the names and the order. The
# `runtime.json` knob is still `auto_stage_b`, and it switches the whole chain.
FOLLOW_ON = {'train': 'stageb', 'stageb': 'hof5000', 'hof5000': 'hof30k'}
PASS_ORDER = ('stageb', 'hof5000', 'hof30k')
AUTO_PRIORITY = {'stageb': 10, 'hof5000': 11, 'hof30k': 12}
AUTO_STAGE_B_PRIORITY = AUTO_PRIORITY['stageb']
PHASE_NAMES = {'stageb': 'stage B', 'hof5000': 'hof5000', 'hof30k': 'hof30k'}

# `SNEK_*` that belong to a *training* and must not be inherited by its measurement. Each would
# change what the wave does rather than what it measures: a step cap is meaningless to an eval, and
# the stage-A knobs describe the screen that already happened.
#
# **This daemon carries no protocol numbers, and that is deliberate.** snek2's held five — the gate,
# the episode count, a suffix — as a second copy of what `eval_plan.py` defines, and they drifted.
# The rule is inverted here: strip what belongs to the training and let `evaluate.py`'s own defaults
# be the protocol. The daemon cannot import the original anyway; it runs on base python so it can
# start before the conda env exists, and `eval_plan` needs numpy.
# The stage-A queue's three knobs belong here for the same reason `SNEK_EVAL_INTERVAL` does: they
# configure *how the training measures itself as it runs*, and a stage-B wave measures finished
# checkpoints with `evaluate.py`, which reads none of them. Leaving them in a wave's env was harmless
# but wrong in the way this tuple exists to prevent — it attributes a training-loop setting to a
# measurement, and `runs/<policy>.md` would report a wave as having run under a worker count that
# never applied to it.
TRAINING_ONLY_KEYS = ('SNEK_MAX_STEPS', 'SNEK_EVAL_INTERVAL', 'SNEK_GRAPH_EVAL_EPISODES',
                      'SNEK_MIN_CHECKPOINT_SCORE', 'SNEK_INITIAL_COLLECT_STEPS',
                      'SNEK_REPLAY_BUFFER_MAX_LENGTH', 'SNEK_TORCH_THREADS',
                      'SNEK_EVAL_QUEUE', 'SNEK_EVAL_QUEUE_DEPTH', 'SNEK_EVAL_WORKERS')

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
    return next_pass('', job_type, ok, auto_enabled) == 'stageb'


def next_pass(job_id, job_type, ok, auto_enabled):
    """The pass a just-finished job earns over its arms, or None.

    A training earns stage B (`wants_stage_b` says which trainings); a stage-B wave earns hof5000; a
    hof5000 pass earns hof30k; hof30k and any eval that is not one of the chain's passes — a hand
    spec with its own id, a `one` re-measure — earn nothing. Success is required at every hop, for
    the same reason at every hop: the next pass selects from the file this one wrote, and a failed
    pass wrote none, so its follow-on would fail too and hide the real failure behind it.
    """
    if not auto_enabled or not ok:
        return None
    if job_type == 'train':
        return FOLLOW_ON['train']
    if job_type == 'eval':
        return FOLLOW_ON.get(pass_of(job_id))
    return None


def pending_pass(record):
    """The pass a finished ledger record is still owed, or None.

    `next_pass` is written as the record's `next_pass` marker. `stage_b: pending` is the marker the
    daemon wrote before 2026-09-04 and the box's ledger is full of them, so it reads as stage B.
    """
    return record.get('next_pass') or ('stageb' if record.get('stage_b') == 'pending' else None)


def mint_pass_id(batch, pass_name, blocked=(), used=()):
    """`<batch>-<pass>`, or `<batch>-<pass>-w<k>` for a later wave of the same batch and pass.

    The plain id is not enough and the case is real: snek2's `b20` ran 36 arms under one prefix in
    nine waves of four, and each wave needs its own id or the second collides with the first. `k`
    counts up to the first free id, which does not churn between polls because ledger records are
    never deleted.

    `used` ids are spent — a finished wave, or one running now — and push `k` past them. `blocked`
    ids stop the mint outright (None): a pending spec of that id already stands for the wave, and
    `_scan_pending` keeps a manual spec over a synthesised one for the same reason. The forecast in
    `anticipated_queue` and the dispatcher in `_auto_jobs` mint through this one function so the
    queue shown is the queue that runs; before that the forecast minted only the bare id and hid
    every later wave of a batch whose first wave had finished (b15, 2026-09-04).
    """
    base = '{0}-{1}'.format(batch, pass_name)
    for index in range(1, 100):
        job_id = base if index == 1 else '{0}-w{1}'.format(base, index)
        if job_id in blocked:
            return None
        if job_id in used:
            continue
        return job_id
    return None


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
                owed = next_pass(job_id, record.get('type'), True,
                                 self.runtime.get('auto_stage_b', True))
                if owed:
                    record['next_pass'] = owed
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
            gitbus.fetch_laptop_status(self.host)
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

    def _auto_jobs(self):
        """The chain's next passes for jobs that finished and whose arms have not had them yet.

        Synthesised fresh each dispatch, never persisted as specs, and driven entirely off the ledger
        so it survives a daemon restart — the `next_pass` markers and each job's stored `env` both
        persist. A training's marker says stage B; a stage-B wave's says hof5000; a hof5000 pass's
        says hof30k.

        **One job per batch and pass, not one per arm.** A batch's arms become a single
        `tools/closeout.py` job carrying every policy: one ledger record, one publish, one log, and
        one window over the batch. Grouping is safe because of the wave barrier: `_dispatch` returns
        early while anything runs, so by the time this is read the set of markers is closed. That is
        also why grouping happens at dispatch rather than in `_scan_pending` — an id that shifted as
        more markers appeared would let a partly-finished wave relaunch under a new id and redo the
        work.

        **What stops a re-measure is `_measured_policies`, not the job id.** A marker is never
        cleared, so the marker set says "this job finished", not "its arms still need the pass".
        """
        if not self.runtime.get('auto_stage_b', True):
            return []
        measured = {name: self._measured_policies(name) for name in PASS_ORDER}
        groups = {}
        for job_id, record in self.ledger.items():
            pass_name = pending_pass(record)
            if pass_name not in PASS_ORDER:
                continue
            env = inherited_eval_env(record.get('env'))
            for policy in (record.get('policies') or [record.get('policy')]):
                if not policy or policy in measured[pass_name]:
                    continue
                key = (PASS_ORDER.index(pass_name), batch_of(policy),
                       tuple(sorted(stage_b_group_env(env).items())))
                groups.setdefault(key, {'policies': [], 'envs': []})
                groups[key]['policies'].append(policy)
                groups[key]['envs'].append(env)
        jobs, taken = [], set()
        for (order, batch, _), group in sorted(groups.items(),
                                               key=lambda item: (item[0][0],
                                                                 min(item[1]['policies']))):
            pass_name = PASS_ORDER[order]
            job_id = self._pass_id(batch, pass_name, taken)
            if job_id is None:
                continue
            taken.add(job_id)
            jobs.append(Job(id=job_id, type='eval', policies=sorted(set(group['policies'])),
                            env=agreed_env(group['envs']), priority=AUTO_PRIORITY[pass_name],
                            eval_args=[] if pass_name == 'stageb' else ['--pass', pass_name]))
        return jobs

    def _measured_policies(self, pass_name='stageb'):
        """Every policy a wave of `pass_name` has already measured, or is measuring right now.

        Read off the wave *records*, never off the markers: a marker is set when a job finishes and
        is never cleared, so the marker set is everything ever finished. What used to stop a
        re-measure in snek2 was incidental — the id `<policy>-closeout` was already taken — and
        grouping a batch into one wave changed the id, so on the first restart after the wave
        controller shipped the daemon forecast **thirteen** waves it had already run.

        Covering by *policy* rather than by id handles three cases at once: legacy per-arm ids, an
        arm mid-migration, and a batch measured across several waves. A hand-queued spec counts too,
        as long as it uses the pass's id suffix (`b11-hof5000`, as the hand-written ones did), which
        is what lets a batch measured by hand before this chain existed not be measured again.

        `failed` counts as measured, because the reason usually is not transient — but it is
        surfaced under `attention` in `status.json`, which is the fix for snek2 silently never
        retrying one and costing a batch its whole measurement. `interrupted` does **not** count, so
        a reboot's half-finished wave is regrouped and relaunched.
        """
        covered = set()

        def add(job_id, policies):
            if pass_of(job_id) == pass_name:
                covered.update(str(name) for name in policies if name)

        for job_id, running_job in self.running.items():
            add(job_id, running_job.job.policies or [running_job.policy])
        for job_id, record in self.ledger.items():
            if record.get('state') in TERMINAL:
                add(job_id, record.get('policies') or [record.get('policy')])
        return covered

    def _pass_id(self, batch, pass_name, taken=()):
        """`<batch>-<pass>`, or `-w<k>`, free of everything the ledger knows — see `mint_pass_id`.

        Free means: not claimed in this same pass, not running, and either absent from the ledger or
        **non-terminal** — an `interrupted` wave is relaunched under its own id so its completed rows
        are still there to resume from.

        `taken` is the ids claimed in this same pass, which the ledger cannot know about yet. It
        matters whenever one batch splits into two waves for disagreeing envs: without it both groups
        get the same id and the second silently replaces the first.
        """
        used = set(taken) | set(self.running)
        used |= {job_id for job_id, record in self.ledger.items()
                 if record.get('state') in TERMINAL}
        return mint_pass_id(batch, pass_name, used=used)

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
        desired += [job for job in self._auto_jobs() if job.id not in seen]
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
        # An eval job is a whole wave and `_dispatch` starts nothing while a wave runs, so the eval
        # limit is structurally 1. It was `runtime['max_evals']` until 2026-08-29, a knob whose only
        # legal value was the one it was hardcoded to.
        limits = {'trainer': self.runtime['max_trainers'], 'eval': 1}
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
            owed = next_pass(job_id, running_job.job.type, ok,
                             self.runtime.get('auto_stage_b', True))
            if owed:
                record['next_pass'] = owed      # picked up by _auto_jobs next dispatch
            del self.running[job_id]
            if ok:
                self._publish_results(running_job)
        self._save_ledger()

    def _publish_results(self, running_job):
        """Publishes every arm this job owned, **one push per arm**.

        Per arm rather than one push for the whole wave, because the box's DNS for github flaps and a
        wave concentrates four arms behind one push. Errors are caught per arm so the remaining arms
        are still attempted, and a push that does not land leaves the commit local — `push_unpushed`
        carries it on the next network cycle, and until it does the branch is listed under
        `attention` in `status.json` rather than being reported as published.
        """
        runs = launch.runs_dir(self.host)
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
        # An eval job is a whole wave and `_dispatch` starts nothing while a wave runs, so the eval
        # limit is structurally 1. It was `runtime['max_evals']` until 2026-08-29, a knob whose only
        # legal value was the one it was hardcoded to.
        limits = {'trainer': self.runtime['max_trainers'], 'eval': 1}
        auto = self.runtime.get('auto_stage_b', True)
        queued = [{'id': job.id, 'type': job.type, 'policy': job.policy,
                   'policies': list(job.policies), 'priority': job.priority,
                   'label': getattr(job, 'label', '')}
                  for job in self._queued if job.id not in self.running]
        running = [{'id': job_id, 'type': running_job.job.type, 'policy': running_job.policy,
                    'policies': list(running_job.job.policies),
                    'label': getattr(running_job.job, 'label', '')}
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
                    'label': getattr(running_job.job, 'label', ''),
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
            'at_a_glance': with_laptop(
                build_at_a_glance(
                    running, order, self._batch_labels(),
                    [flag for flag in HOLD_FLAGS if self.runtime.get(flag)],
                    self._attention()),
                gitbus.read_laptop_status(self.host)),
            'runtime': self.runtime,
            'config_notes': self.config_notes,
            'counts': self._counts(),
            'running': running,
            'ledger': self._ledger_view(order),
            'disk_free_gb': _disk_free_gb(self.host['REPO_PATH']),
            'load_avg': list(os.getloadavg()),
        }
        try:
            if not gitbus.publish_status(self.host, status_json(status)):
                self._unpushed = sorted(set(self._unpushed) | {self.host['STATUS_BRANCH']})
            else:
                self._unpushed = [branch for branch in self._unpushed
                                  if branch != self.host['STATUS_BRANCH']]
        except Exception as error:
            sys.stderr.write('publish_status failed: {0}\n'.format(error))


# ---------------------------------------------------------------- pure helpers

def status_json(status):
    """`status` as the text published to `ops-status`.

    **`ensure_ascii=False` because a human reads this file through `git show`.** The default escapes
    every non-ASCII character, so an em dash in a job label published as a literal `\\u2014` and
    `at_a_glance` read `"b8 \\u2014 b8: kl02, seed 1 of 4 \\u2014 wave 2 of 2"` — correct JSON,
    unreadable prose. Job text is folded to ASCII in `job.py`; this covers what is deliberately not
    folded, such as a policy name, which is a path.

    A separate function rather than an inline `json.dumps` so the rule is testable without a live
    daemon — the call site above needs a host, a ledger and a git worktree.
    """
    return json.dumps(status, indent=2, ensure_ascii=False)

# **`b<number>` is the only batch prefix, in both eras and both algorithms.** A `p` series was tried
# for PPO and renamed back on 2026-08-31 (`p0-p3` -> `b3-b6`) precisely so this pattern stays one
# character wide: the prefix was missing from it for a day, and every PPO arm became its own batch —
# `b4a-...` fell through to the `split('-')[0]` fallback and returned `b4a`. Nothing measured wrong,
# but the two things that group by batch both degraded silently: `_auto_jobs` synthesised one
# wave per arm instead of one per batch, and `at_a_glance` listed eight separate lines where it should
# show one batch with eight arms. **Do not add a second prefix — name the batch `b<n>`.**
#
# The box's ledger still holds the pre-rename ids (`p1a-...`, `p2-hof5000`), and those now take the
# fallback and read as one batch per arm. That is cosmetic and confined to the ledger's display: every
# one of those waves is `done`, and nothing re-groups a finished record.
_BATCH_RE = re.compile(r'^(b\d+)')
_PASS_ID_RE = re.compile(r'-(stageb|hof5000|hof30k)(-w\d+)?$')


def batch_of(job_id):
    """The batch a job id belongs to.

    The leading `b<number>` — `b1a-thing-seed1` -> `b1` — or the id up to the first `-` for anything
    that is not a b-numbered arm, so `smoke-1` -> `smoke`.
    """
    job_id = job_id or ''
    match = _BATCH_RE.match(job_id)
    return match.group(1) if match else (job_id.split('-')[0] or job_id)


def pass_of(job_id):
    """Which of the chain's passes a job id names — `'stageb'`, `'hof5000'`, `'hof30k'` — or None.

    Read off the id suffix the daemon mints (and hand specs copy: `b11-hof5000`). **The `-w<k>` tail
    has to be part of the pattern**: a batch's second wave is `b1-stageb-w2`, and a plain
    `endswith('-stageb')` would call that a bare eval and split the at-a-glance grouping in two.
    """
    match = _PASS_ID_RE.search(str(job_id or ''))
    return match.group(1) if match else None


def phase_of(job_id, job_type):
    """A job's phase: `'stage B'`, `'hof5000'`, `'hof30k'`, `'training'`, `'eval'`, `'smoke'` or
    `'benchmark'`. The id suffix decides for the chain's passes, the job type is the fallback."""
    pass_name = pass_of(job_id)
    if pass_name:
        return PHASE_NAMES[pass_name]
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


_SEED_PIECE_RE = re.compile(r'^seed \d+ of \d+$')
_BATCH_PREFIX_RE = re.compile(r'^b\d+:\s*')
_KNOB_TOKEN_RE = re.compile(r'^b\d+[a-z]+-(.+?)(?:-seed\d+)?$')


_WAVE_RE = re.compile(r'^wave (\d+) of (\d+)$')
MAX_CELLS_SHOWN = 8


def compress_waves(tails):
    """`['wave 1 of 5', 'wave 2 of 5', 'wave 3 of 5']` -> `['waves 1-3 of 5']`; anything else passes through.

    A whole queued batch is one group, and listing its five waves one by one ran the caption past the
    80-character cut on the first day (b12: "wave 1 of 5, wave 2 of..."). Consecutive runs only, so a
    gap -- wave 2 pulled from the queue -- still shows as two ranges rather than being papered over.
    """
    out, run = [], []

    def flush():
        if not run:
            return
        total = run[0][1]
        first, last = run[0][0], run[-1][0]
        out.append('wave {0} of {1}'.format(first, total) if first == last
                   else 'waves {0}-{1} of {2}'.format(first, last, total))
        del run[:]

    for tail in tails:
        match = _WAVE_RE.match(tail)
        if not match:
            flush()
            out.append(tail)
            continue
        wave, total = int(match.group(1)), int(match.group(2))
        if run and (run[-1][1] != total or wave != run[-1][0] + 1):
            flush()
        run.append((wave, total))
    flush()
    return out


def describe_jobs(jobs):
    """One caption for a group of jobs, from the jobs themselves.

    From their labels when they have them: `tools/sweep_specs.py` writes
    `b11: lr5e4, seed 1 of 4 -- wave 3 of 4` per arm, so the batch prefix and the per-arm seed piece
    are dropped and what the arms differ in is kept, in first-seen order --
    `lr5e4, lr8e4 -- wave 3 of 4` for a wave holding two cells. From their policy ids otherwise: an
    auto-queued stage B carries no label, and its arms' knob tokens (`b11ai-lr1.5e4-seed1` ->
    `lr1.5e4`) say what it is measuring. Empty when neither yields anything, so the caller can fall
    back to whatever it has.
    """
    heads, tails = [], []
    for job in jobs:
        label = to_ascii(str(job.get('label') or ''))
        if not label:
            continue
        head, _, tail = _BATCH_PREFIX_RE.sub('', label).partition(' -- ')
        for piece in head.split(', '):
            piece = piece.strip()
            if piece and not _SEED_PIECE_RE.match(piece) and piece not in heads:
                heads.append(piece)
        tail = tail.strip()
        if tail and tail not in tails:
            tails.append(tail)
    if heads or tails:
        if len(heads) > MAX_CELLS_SHOWN:
            heads = heads[:MAX_CELLS_SHOWN] + ['+{0} more'.format(len(heads) - MAX_CELLS_SHOWN)]
        return ', '.join(heads) + (' -- ' + ', '.join(compress_waves(tails)) if tails else '')
    tokens = []
    for job in jobs:
        for policy in (job.get('policies') or [job.get('policy')]):
            match = _KNOB_TOKEN_RE.match(str(policy or ''))
            if match and match.group(1) not in tokens:
                tokens.append(match.group(1))
    return ', '.join(tokens)


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
    def group(jobs, collapse_passes=False):
        # `collapse_passes` folds the chain's three passes into one `evals` group per batch: the
        # queue shows what is owed, and a batch owed its stage B is owed its hof passes too — three
        # rows per batch said the same thing three times (user, 2026-09-04). A *running* job is one
        # pass, so the running line still names it.
        #
        # Folded into at most **two** `evals` groups per batch, split by whether the batch's own
        # training has been passed in the order: passes for arms already trained (a hand spec for a
        # finished wave) sit where they run, ahead of the training; passes that follow a queued
        # training wave are shown after it. Folding all of them into the first-seen slot put b15's
        # 32 not-yet-trained arms' evals above b15's training on 2026-09-04, which read as the box
        # measuring arms before training them. The trainings of a batch stay one line, as before.
        groups, index = [], {}
        trained_seen = set()
        for job in jobs:
            batch = batch_of(job['id'])
            phase = phase_of(job['id'], job.get('type'))
            after_training = False
            if collapse_passes and pass_of(job['id']):
                phase = 'evals'
                after_training = batch in trained_seen
            elif phase == 'training':
                trained_seen.add(batch)
            key = (batch, phase, after_training)
            if key not in index:
                index[key] = len(groups)
                groups.append((key[0], key[1], []))
            groups[index[key]][2].append(job)
        return groups

    def described(batch, jobs):
        # The text after the batch id describes THESE jobs, never the batch as a whole. Until
        # 2026-09-03 it was one label per batch, the first found, and the first found was a queued
        # spec's -- so a running b11 wave 3 was captioned "lr1e3, seed 1 of 4 -- wave 4 of 4" from
        # the arm that would run next, and a stage B was captioned by a trainer. Read the ledger to
        # learn which arms were live, which is what the caption exists to save.
        label = describe_jobs(jobs) or labels.get(batch) or ''
        if not label:
            return ''
        # Folded here as well as in `job.py`, because a label reaches this from three places and one
        # of them is the **ledger**, which persists whatever was written when the job was first
        # recorded. Folding only at parse time left b8's stored em dashes on display after the fix.
        label = to_ascii(label)
        return ' | ' + (label if len(label) <= 80 else label[:77] + '...')

    def arms(jobs):
        """"N arms", counted in **distinct policies**, not in jobs.

        One eval job is a wave of four arms, so counting jobs would report a batch's whole
        measurement as "1 arm" — the number a reader uses to check that nothing was dropped. And
        distinct, because a collapsed `evals` group holds the same arms under three passes.
        """
        names = set()
        unnamed = 0
        for job in jobs:
            policies = [name for name in (job.get('policies') or [job.get('policy')]) if name]
            if policies:
                names.update(policies)
            else:
                unnamed += 1
        count = len(names) + unnamed
        return '{0} arm{1}'.format(count, '' if count == 1 else 's')

    running_lines = []
    for batch, phase, jobs in group(running):
        percents = [100.0 * job['step'] / job['max_steps']
                    for job in jobs if job.get('step') and job.get('max_steps')]
        percent = (' {0}%'.format(int(round(sum(percents) / len(percents))))
                   if percents else '')
        running_lines.append('{0}{1} | {2}{3} ({4})'.format(
            batch, described(batch, jobs), phase, percent, arms(jobs)))

    queued_lines = []
    notice = hold_notice(held_by)
    if notice:
        queued_lines.append(notice)
    for batch, phase, jobs in group(queued_order, collapse_passes=True):
        queued_lines.append('{0} {1}{2} | queued ({3})'.format(
            batch, phase, described(batch, jobs), arms(jobs)))

    return {'running': running_lines, 'queued': queued_lines, 'attention': list(attention)}


LAPTOP_KEYS = ('laptop_running', 'laptop_queued', 'laptop_iso')


def with_laptop(glance, laptop_status_text):
    """`glance` plus the laptop's lines: `laptop_running`, `laptop_queued` and `laptop_iso`.

    The laptop's driver publishes its own `status.json` to the `laptop-status` branch
    (`tools/laptop_status.py`), in this same `at_a_glance` shape; folding it in here is what lets one
    `git show origin/ops-status:status.json` show both boxes. **`laptop_iso` is the laptop's own
    timestamp, and it is the staleness signal**: the driver's last publish before exiting is empty, so
    empty lists mean idle, while a running line under an hours-old `laptop_iso` means the driver died
    (user, 2026-09-04). Unparseable or absent text -- nothing published yet -- gives empty lists and a
    null `laptop_iso`, and never an error: the box's own status must publish regardless.
    """
    out = dict(glance)
    try:
        status = json.loads(laptop_status_text or '')
    except ValueError:
        status = {}
    if not isinstance(status, dict):
        status = {}
    laptop = status.get('at_a_glance') or {}
    out['laptop_running'] = [str(line) for line in (laptop.get('running') or [])]
    out['laptop_queued'] = [str(line) for line in (laptop.get('queued') or [])]
    out['laptop_iso'] = status.get('iso')
    return out


def anticipated_queue(queued, running, limits, auto_stage_b, existing_ids):
    """The pending queue in the order it is expected to launch, waves and all.

    Modelled straight from `_dispatch`: repeatedly form a wave from the highest-priority job's
    category up to that category's limit, drain it, and let the jobs in it spawn their follow-ons —
    **one** pass per batch (priority `AUTO_PRIORITY[pass]`) that competes in the next wave: a wave of
    trainings spawns its stage B, the stage B spawns its hof5000, that spawns its hof30k. A
    measurement outranks a queued training, so a batch's three passes always slot in ahead of the
    following batch — the exact interleaving the box will run.

    A batch's four trainings collapsing into one stage-B row is the *reason* this is grouped: snek2's
    ungrouped forecast showed four close-out jobs and then four HOF jobs, eight rows for what is now
    one process.

    `queued` and `running` are lists of dicts with id/type/policy — `queued` also has priority, a
    wave also has `policies`. `existing_ids` is every id already known — the ledger, the running
    jobs, the queued specs. An id in it that is *queued* stands for its wave and stops the mint; any
    other (finished, running) is spent and pushes the wave number, exactly as `_pass_id` does with
    the ledger, so a batch whose first wave is done forecasts `-w2` rather than nothing.
    """
    def category(job_type):
        return 'eval' if job_type == 'eval' else 'trainer'

    queued_ids = {job['id'] for job in queued}
    used = set(existing_ids) - queued_ids
    minted = set()

    def follow_ons_for(jobs):
        """One next-pass dict per batch for a set of just-placed jobs.

        Grouped by batch exactly as `_auto_jobs` groups the real markers, but **not** by env: a
        forecast has no env to read, and getting the count of rows right is what this is for.
        """
        out = []
        by_batch = {}
        for job in jobs:
            pass_name = next_pass(job.get('id'), job.get('type'), True, auto_stage_b)
            if not pass_name:
                continue
            for policy in (job.get('policies') or [job.get('policy')]):
                if policy:
                    by_batch.setdefault((pass_name, batch_of(policy)), []).append(policy)
        for (pass_name, batch), policies in sorted(by_batch.items()):
            job_id = mint_pass_id(batch, pass_name, blocked=queued_ids, used=used | minted)
            if job_id is None:
                continue
            minted.add(job_id)
            policies = sorted(set(policies))
            out.append({'id': job_id, 'type': 'eval', 'policy': policies[0],
                        'policies': policies, 'priority': AUTO_PRIORITY[pass_name]})
        return out

    pool = [dict(job) for job in queued]
    pool += follow_ons_for(running)                 # follow-ons for jobs on the box now

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
        pool += follow_ons_for(wave)    # the wave's jobs spawn their batches' next passes
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
