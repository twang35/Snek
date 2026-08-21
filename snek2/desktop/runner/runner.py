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
import re
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

# **This daemon carries no eval-protocol numbers, and that is deliberate.** It used to hold five
# of them -- the closeout gate, the HOF gate, 500 episodes, the flat-screen flag, the `_hof500`
# suffix -- as a second copy of what `eval_plan.py` defines, plus its own copy of the
# `closeout gate < HOF gate` assert. Both copies are gone, and the daemon cannot simply import the
# originals either: it runs on **base** miniconda python with `desktop/` as its working directory
# (see systemd/snek-runner.service), because it has to be able to start before the `snek` conda env
# exists, and `eval_plan` needs numpy.
#
# So the rule is inverted instead: the daemon *removes* the protocol keys from the env it inherits
# and lets the tool's own defaults decide. One definition, no import, and nothing here to drift.
# The HOF stage is `eval_wave.py --chain`, which owns the whole recipe (eval_plan.hof_settings).
EVAL_PROTOCOL_KEYS = ('EVAL_EPISODES', 'EVAL_SCREEN_EPISODES', 'EVAL_MIN_ACHIEVABLE',
                      'EVAL_ABANDON_FLOOR', 'EVAL_CONFIRM_COUNT', 'EVAL_OUT_SUFFIX')

# The `SNEK_*` a *measurement* actually depends on, and therefore the only env a batch's arms have
# to agree about to share one closeout wave. **A copy of `eval_wave.EVAL_RELEVANT_ENV`**, which is
# the source of truth -- the runner cannot import it, because that module pulls in TensorFlow, so
# `tests/test_runner.py` reads the tuple out of `eval_wave.py` and fails if the two drift.
#
# Grouping on the *whole* inherited env instead is what split `b45`'s closeout into **three** waves
# -- `{a,c}`, `{b}`, `{d}` -- because its arms differ in `SNEK_SEED`, which cannot reach a
# measurement of an already-trained checkpoint. The cost was not only cosmetic: three sequential
# waves of 2/1/1 arms run a close-out at a quarter of the intended 4 lanes, against a standing
# instruction to measure a batch as 4 parallel arms. `SNEK_LEARNING_RATE`, `SNEK_DISCOUNT`,
# `SNEK_TARGET_UPDATE_PERIOD`, `SNEK_IS_WEIGHTS` and `SNEK_FORK_BRANCHES` are the same class: they
# shaped the weights, and the weights are in the checkpoint. `SNEK_FC_LAYERS` is excluded for a
# different reason -- `arch.json` governs the network at eval time and `eval_wave` already gives a
# differing architecture its own lane through `restore_signature`.
EVAL_RELEVANT_ENV = ('SNEK_ZERO_OBS', 'SNEK_CHASE_SAFE_SHAPING', 'SNEK_CHASE_SAFE_GATE',
                     'SNEK_FREE_SPACE_SHAPING', 'SNEK_FREE_SPACE_GATE',
                     'SNEK_FOOD_DISTANCE_REWARD', 'SNEK_PERFECT_GAME_REWARD')


def inherited_eval_env(env):
    """A training's env, stripped of anything that would override the eval protocol.

    A training arm's env is inherited so the arm's own `SNEK_*` reach its close-out -- historically
    this was about `SNEK_FC_LAYERS` (the FC trap), now handled by `arch.json`, but the shaping and
    reward knobs still matter because they change what `avg_reward` means. What must *not* come
    along is an `EVAL_*` knob: those belong to the measurement, not to the arm.
    """
    return {k: v for k, v in (env or {}).items() if k not in EVAL_PROTOCOL_KEYS}


def closeout_group_env(env):
    """The measurement-relevant slice of a training's env -- the group key for a closeout wave.

    Two arms of a batch belong in one wave unless they disagree about something a measurement can
    see. See `EVAL_RELEVANT_ENV` for why that is a short list and what it cost to key on the long
    one."""
    return {k: v for k, v in (env or {}).items() if k in EVAL_RELEVANT_ENV}


def agreed_env(envs):
    """The env to run a group's wave under: every key the group's arms agree on, value included.

    A group is now allowed to hold arms whose envs differ in measurement-irrelevant ways, so there
    is no longer one arm's env to hand the wave. Taking the agreed subset is the honest answer --
    every setting the wave inherits is one all of its arms were trained with -- and it drops exactly
    the keys the looser group key stopped caring about (`SNEK_SEED`, the rates). Handing the first
    arm's env to the whole wave instead would quietly attribute one arm's seed to all four."""
    envs = [e or {} for e in envs]
    if not envs:
        return {}
    shared = dict(envs[0])
    for env in envs[1:]:
        for key in list(shared):
            if env.get(key) != shared[key]:
                del shared[key]
    return shared


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
        # The same `policies or [policy]` fallback `Job` does, and it is load-bearing rather than
        # tidy: this is the path that re-adopts a *running* wave after a daemon restart, with no
        # spec in hand, so a missing `policies` here would publish one arm of four.
        self.policies = [p for p in (rec.get('policies') or [self.policy]) if p]
        self.max_steps = rec.get('max_steps')
        self.label = rec.get('label', '')
        self.chain = bool(rec.get('chain'))

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
                # silently skip a restart-straddling arm. There is no HOF branch any more: the
                # re-measure is a stage inside the closeout's own process (`--chain`), so a
                # closeout that straddled a restart already ran it or already failed to.
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

        The panel set is *sticky within a wave*: an arm that reaches its cap and exits keeps its
        panel until the whole wave drains, so a wave of four never collapses to the two still
        training. `sticky_wave_pngs` only unions -- it never drops -- so a finished arm does not
        trigger a relaunch; the set resets when the wave flips category (train->eval) or the box
        goes idle. An **eval** wave widens further, to every chart of the batch it is measuring
        (`eval_batch_pngs`), because a batch's measurement can arrive as several waves and
        stickiness only spans one."""
        if not self.runtime.get('viewer', True) or not self.running:
            # Idle between waves: forget the wave so the next one starts from its own arms,
            # not unioned onto the previous batch's (a trainer->trainer flip keeps category).
            self._wave_pngs, self._wave_category = [], None
            return
        try:
            running_jobs = [(rj.job.category, rj.job.policies or [rj.policy])
                            for rj in self.running.values()]
            # Wave-barrier scheduling runs one category at a time, so the wave's category is
            # simply that of whatever is running; sorted+joined stays deterministic regardless.
            category = ','.join(sorted({c for c, _ in running_jobs}))
            # An eval wave widens from its running arms to every chart of the batches it is
            # measuring, so a batch sliced into several waves still shows all four panels. Only
            # eval: a *training* wave never splits (its arms are dispatched together and run
            # concurrently), so the sticky set below already keeps a finished trainer's panel, and
            # widening it would be the one change that could over-report. The `or` covers the first
            # seconds of a fresh wave, when no chart exists yet -- fall back to the running arms'
            # paths, exactly as before, so the window still opens promptly.
            current = viewer_png_paths(running_jobs, self.host['SNEK_DIR'])
            if category == 'eval':
                try:
                    names = os.listdir(os.path.join(self.host['SNEK_DIR'], 'evals'))
                except OSError:
                    names = []
                current = eval_batch_pngs(running_jobs, self.host['SNEK_DIR'], names) or current
            if not current:
                return
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
                   ['--watch', 'snek2.py|eval_wave.py|eval_checkpoints.py',
                    '--interval', '1',
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
        """Closeout waves to run for trainings that finished under auto-closeout and have not been
        evaluated yet -- synthesized fresh each dispatch, never persisted as specs.

        **One job per batch, not one per arm.** A batch's arms are grouped into a single
        `eval_wave.py` job carrying every one of their policies, so its lanes move to whichever arm
        still has checkpoints instead of a finished arm's share of the box going idle. Grouping is
        safe here because of the wave barrier: `_dispatch` returns early while anything is running,
        so by the time this is read the set of `closeout: pending` markers is closed. That is also
        why the grouping happens at dispatch rather than in `_scan_pending` -- an id that shifted as
        more markers appeared would let a partly-finished wave relaunch under a new id and redo the
        work.

        Driven entirely off the ledger, so it survives a daemon restart: the `closeout: pending`
        markers and each training's stored `env` both persist.

        **What stops a re-measure is `_measured_policies`, not the job id.** A training's
        `closeout: pending` marker is never cleared, so the marker set says "this arm was trained",
        not "this arm still needs measuring"; the id was doing that job by accident, and a
        batch-level id broke it -- see `_measured_policies`.

        **Arms are grouped by batch and by the *measurement-relevant* half of their inherited env**,
        because a wave is one process with one environment: the shaping and reward knobs change what
        `avg_reward` means, so two arms that disagree about them must not be measured under a single
        setting. Everything else about a training -- its seed, its learning rate, its target-update
        period -- is already baked into the checkpoint and cannot reach the measurement, so it must
        *not* split the wave. Keying on the whole env did, and `b45` paid three waves of 2/1/1 arms
        for it; see `EVAL_RELEVANT_ENV`. Normally a batch's arms agree on the relevant half exactly
        and there is one group.
        """
        if not self.runtime.get('auto_closeout', True):
            return []
        measured = self._measured_policies()
        groups = {}
        for rec in sorted(self.ledger.values(), key=lambda r: str(r.get('policy'))):
            if rec.get('closeout') != 'pending' or not rec.get('policy'):
                continue
            if rec['policy'] in measured:
                continue
            env = inherited_eval_env(rec.get('env'))
            key = (batch_of(rec['policy']),
                   tuple(sorted(closeout_group_env(env).items())))
            groups.setdefault(key, {'policies': [], 'envs': [], 'resume': False})
            groups[key]['policies'].append(rec['policy'])
            groups[key]['envs'].append(env)
        jobs, taken = [], set()
        for (batch, _), group in sorted(groups.items(), key=lambda kv: kv[1]['policies'][0]):
            eval_id = self._closeout_id(batch, taken)
            if eval_id is None:
                continue
            taken.add(eval_id)
            prior = self.ledger.get(eval_id, {})
            env = agreed_env(group['envs'])
            if prior.get('state') == 'interrupted':
                env['EVAL_RESUME'] = '1'           # a reboot cut the last attempt short: keep its
                                                   # full-length rows, redo only the partial one
            jobs.append(Job(id=eval_id, type='eval', policies=group['policies'],
                            env=env, eval_args=['top50'],
                            # `--chain` is the HOF re-measure, run as stage B inside this same
                            # process. There is no `hof: pending` marker any more and nothing to
                            # forecast separately: an arm with no >=98% checkpoint contributes no
                            # work to stage B, which is the normal outcome.
                            chain=self.runtime.get('auto_hof', True),
                            priority=AUTO_CLOSEOUT_PRIORITY))
        return jobs

    def _measured_policies(self):
        """Every policy a closeout wave has already measured, or is measuring right now.

        Read off the closeout *records*, never off the trainings' markers. A training's
        `closeout: pending` marker is set when it finishes and is never cleared, so the marker set
        is a list of everything ever trained under auto-closeout -- 64 arms on this box. What used
        to stop a re-measure was incidental: the id `<policy>-closeout` was already taken, and a
        taken terminal id read as "done". Grouping a batch into one wave changes the id to
        `<batch>-closeout`, which had never been used for b20-b44, so on the first restart after the
        wave controller shipped the daemon forecast **thirteen** closeout waves it had already run,
        b20's at 12 arms.

        Covering by *policy* rather than by id fixes three things at once: the legacy per-arm ids,
        an arm mid-migration (b44's two running per-arm close-outs), and a batch measured in several
        waves -- b20's nine waves of four all wrote `<policy>-closeout` records, and without this a
        later wave's group would re-measure every earlier wave's arms under `-w<k>`.

        `failed` counts as measured, matching the id-based rule it replaces: a wave that failed is
        not retried automatically, because the reason is usually not transient. `interrupted` does
        not, so a reboot's half-finished wave is regrouped and relaunched with `EVAL_RESUME`.
        """
        covered = set()

        def add(job_id, policies):
            if _CLOSEOUT_ID_RE.search(str(job_id)):
                covered.update(str(p) for p in policies if p)

        for job_id, rj in self.running.items():
            add(job_id, rj.job.policies or [rj.policy])
        for job_id, rec in self.ledger.items():
            if rec.get('state') in TERMINAL:
                add(job_id, rec.get('policies') or [rec.get('policy')])
        return covered

    def _closeout_id(self, batch, taken=()):
        """`<batch>-closeout`, or `<batch>-closeout-w<k>` for a later wave of the same batch.

        `<batch>-closeout` alone is not enough, and the case is real: **b20 ran 36 arms under one
        prefix, in nine waves of four.** Each wave needs its own id or the second one collides with
        the first. `k` counts up from 2 to the first id that is free, which does not churn between
        polls because ledger records are never deleted.

        Free means: not claimed in this same pass, not running, and either absent from the ledger or
        **non-terminal** -- an `interrupted` wave is deliberately relaunched under its own id so its
        completed rows are still there to resume from. Deciding *whether* there is anything to
        measure is `_measured_policies`' job, not this one's; by the time a group reaches here its
        arms are known to be unmeasured.

        `taken` is the ids already claimed **in this same pass**, which the ledger cannot know about
        yet. It matters whenever one batch splits into two waves for disagreeing envs: without it
        both groups are handed `<batch>-closeout` and the second silently replaces the first.
        """
        base = '{0}-closeout'.format(batch)
        for index in range(1, 100):
            eval_id = base if index == 1 else '{0}-w{1}'.format(base, index)
            if eval_id in taken or eval_id in self.running:
                continue
            if self.ledger.get(eval_id, {}).get('state') in TERMINAL:
                continue
            return eval_id
        return None

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
                               # Every arm this job owns. An eval wave has several; a training has
                               # one and records it both ways, so a reader never has to know which.
                               'policies': list(job.policies),
                               'chain': bool(getattr(job, 'chain', False)),
                               'pid': rj.pid, 'log': rj.log_path, 'started': rj.started,
                               # Carried so status.json's at-a-glance can show a % and a
                               # label for a job re-adopted after a restart, spec not in hand.
                               'max_steps': job.max_steps, 'label': job.label,
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
        """Publishes every arm this job owned, **one push per arm**.

        Per arm rather than one push for the whole wave, and the reason is a failure this project has
        already paid for: the box's DNS for github.com flaps, `publish_results` has no retry, and a
        failed push leaves the commit local while the ledger still says `done`. On 2026-08-18 that
        hid four HOF-500 files and a whole close-out for hours -- one of them a 98.2%/500 checkpoint.
        A wave makes that worse by concentrating four arms behind one push, so the error is caught
        per arm and the remaining arms are still attempted.
        """
        runs = os.path.join(self.host['SNEK_DIR'], 'runs')
        names = sorted(os.listdir(runs)) if os.path.isdir(runs) else []
        for pol in [str(p) for p in (rj.job.policies or [rj.policy]) if p]:
            arts = [os.path.join(runs, f) for f in names
                    if f == pol + '.md' or f.startswith(pol + '.') or f.startswith(pol + '_')]
            try:
                gitbus.publish_results(self.host, rj.job, arts)
            except Exception as e:  # best-effort, but say so in the journal, not silently
                sys.stderr.write('publish_results({0}, {1}) failed: {2}\n'.format(
                    rj.job.id, pol, e))

    def _anticipated_order(self):
        """The pending queue folded forward through the wave scheduler -- each queued batch's
        closeout wave slotted where it will actually run (see `anticipated_queue`). Shared by the
        ledger view and the at-a-glance summary so both agree."""
        limits = {'trainer': self.runtime['max_trainers'], 'eval': self.runtime['max_evals']}
        auto = self.runtime.get('auto_closeout', True)
        auto_hof = self.runtime.get('auto_hof', True)
        queued = [{'id': j.id, 'type': j.type, 'policy': j.policy,
                   'policies': list(j.policies), 'priority': j.priority}
                  for j in self._queued if j.id not in self.running]
        running = [{'id': jid, 'type': rj.job.type, 'policy': rj.policy,
                    'policies': list(rj.job.policies)}
                   for jid, rj in self.running.items()]
        existing = set(self.ledger) | set(self.running) | {j['id'] for j in queued}
        return anticipated_queue(queued, running, limits, auto, existing, auto_hof)

    def _batch_labels(self):
        """Map batch id -> human label, from queued specs, running jobs and the ledger. A batch's
        arms share a label, so the first non-empty one found wins."""
        labels = {}
        def add(jid, label):
            if label:
                labels.setdefault(batch_of(jid), label)
        for j in self._queued:
            add(j.id, getattr(j, 'label', ''))
        for jid, rj in self.running.items():
            add(jid, getattr(rj.job, 'label', ''))
        for jid, rec in self.ledger.items():
            add(jid, rec.get('label', ''))
        return labels

    def _ledger_view(self, order):
        """The id->state map published in status.json, ordered newest/active first: the pending
        queue at the top (in the order the wave scheduler will launch it, so the next job to run
        sits highest), then the running jobs, then the finished history most-recent-first by its
        `finished` timestamp. This is the reverse of the on-disk `ledger.json`, whose insertion
        order is oldest-first; only the *published* view is reordered, so the authoritative ledger
        and every restart path are untouched. A real ledger state always wins over the synthetic
        `queued` on an id overlap (a job re-queued while its prior run is still settling)."""
        queued_ids = [job['id'] for job in order
                      if job['id'] not in self.ledger and job['id'] not in self.running]
        running_ids = [jid for jid, rec in self.ledger.items()
                       if rec.get('state') == 'running']
        for jid in self.running:                       # a live job the ledger somehow missed
            if jid not in running_ids and jid not in queued_ids:
                running_ids.append(jid)
        finished_ids = sorted(
            (jid for jid, rec in self.ledger.items() if rec.get('state') != 'running'),
            key=lambda jid: self.ledger[jid].get('finished') or 0, reverse=True)
        view = {}
        for jid in queued_ids:
            view[jid] = 'queued'
        for jid in running_ids + finished_ids:
            view[jid] = self.ledger.get(jid, {}).get('state')
        return view

    def _publish(self):
        order = self._anticipated_order()
        running = [{'id': rj.job.id, 'type': rj.job.type, 'policy': rj.policy,
                    'policies': list(rj.job.policies),
                    'pid': rj.pid, 'step': rj.current_step,
                    'max_steps': getattr(rj.job, 'max_steps', None),
                    'steps_per_sec': rj.steps_per_sec,
                    'elapsed_s': round(time.time() - rj.started)}
                   for rj in self.running.values()]
        status = {
            'iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'ts': time.time(),
            # A human summary at the top: one line per running batch (with % done) and one per
            # queued batch-phase, so the box's state reads at a glance without parsing the ledger.
            'at_a_glance': build_at_a_glance(
                running, order, self._batch_labels(),
                [f for f in HOLD_FLAGS if self.runtime.get(f)]),
            'runtime': self.runtime,
            'config_notes': self.config_notes,
            'counts': self._counts(),
            'running': running,
            'ledger': self._ledger_view(order),
            'disk_free_gb': _disk_free_gb(self.host['REPO_PATH']),
            'load_avg': list(os.getloadavg()),
        }
        try:
            gitbus.publish_status(self.host, json.dumps(status, indent=2))
        except Exception as e:
            sys.stderr.write('publish_status failed: {0}\n'.format(e))


_BATCH_RE = re.compile(r'^(b\d+)')


def batch_of(job_id):
    """The batch a job id belongs to: the leading b<number> ('b41a-...-seed1' -> 'b41'), or the id
    up to the first '-' for anything that is not a b-numbered arm (so 'smoke-1' -> 'smoke')."""
    jid = job_id or ''
    m = _BATCH_RE.match(jid)
    return m.group(1) if m else (jid.split('-')[0] or jid)


_CLOSEOUT_ID_RE = re.compile(r'-closeout(-w\d+)?$')


def phase_of(job_id, job_type):
    """The phase of a job, read off the id suffix the daemon mints with the job type as the
    fallback: 'closeout eval' | 'hof' | 'training' | 'eval' | 'smoke' | 'benchmark'.

    The `-w<k>` tail has to be part of the pattern: a batch's second closeout wave is
    `b20-closeout-w2`, and a plain `endswith('-closeout')` would call that an 'eval' and split the
    at-a-glance grouping in two.

    `-hof` is kept for **old ledger records only**. The daemon no longer mints that id -- the
    re-measure is stage B inside the closeout's own process -- but the ledger is append-only and
    status.json still renders records from before the change.
    """
    jid = job_id or ''
    if _CLOSEOUT_ID_RE.search(jid):
        return 'closeout eval'
    if jid.endswith('-hof'):
        return 'hof'
    return {'train': 'training'}.get(job_type, job_type or 'job')


# Shown first in `at_a_glance.queued` while the queue is held, so a queue that is not moving says
# why and how to release it -- the previous behaviour was an unchanged list of queued batches, which
# reads exactly like a queue that is about to run. Both flags skip `_dispatch` identically (see
# `poll_once`), so the notice names whichever ones are actually set: telling someone to clear
# `paused` while `drain` is the flag that is set would send them to the wrong field.
HOLD_FLAGS = ('paused', 'drain')
_HOLD_WORDS = {'paused': 'paused', 'drain': 'draining'}


def hold_notice(held_by):
    """The `at_a_glance.queued` line for a held queue, or None when nothing holds it.

    `held_by` is whichever of `HOLD_FLAGS` are set. Split out from `build_at_a_glance` so the
    wording is testable on its own, and ordered by `HOLD_FLAGS` rather than by the caller so the
    line is stable across polls."""
    held = [flag for flag in HOLD_FLAGS if flag in held_by]
    if not held:
        return None
    return '** queue {0}: nothing new will start. Set {1} in runtime.json on ops to resume'.format(
        ' and '.join(_HOLD_WORDS[f] for f in held),
        ' and '.join('"{0}": false'.format(f) for f in held))


def build_at_a_glance(running, queued_order, labels, held_by=()):
    """The `at_a_glance` block for status.json: {'running': [str], 'queued': [str]}, one line per
    (batch, phase) group in first-seen order. `running` and `queued_order` are lists of job dicts;
    a running trainer dict carries `step`/`max_steps`, from which each running line shows the mean
    percent done across that batch's arms. `labels` maps a batch id to its human description.

    `held_by` is whichever of `HOLD_FLAGS` are set; when any are, `hold_notice` goes **first** in
    `queued`, ahead of the batch lines. Unconditional on the queue being non-empty, because an
    empty `queued` list under a hold is the case most in need of the explanation.

    Kept a pure function of plain dicts so it is testable without a live box."""
    def group(jobs):
        groups, index = [], {}
        for j in jobs:
            key = (batch_of(j['id']), phase_of(j['id'], j.get('type')))
            if key not in index:
                index[key] = len(groups)
                groups.append((key[0], key[1], []))
            groups[index[key]][2].append(j)
        return groups

    def described(batch):
        lab = labels.get(batch)
        if not lab:
            return ''
        return ' -- ' + (lab if len(lab) <= 80 else lab[:77] + '...')

    def arms(jobs):
        """"N arms", counted in **policies**, not in jobs.

        One eval job is a wave of four arms, so counting jobs would report a batch's whole close-out
        as "1 arm" -- the number a reader uses to check that nothing was dropped."""
        n = sum(len(job.get('policies') or [job.get('policy')] or []) or 1 for job in jobs)
        return '{0} arm{1}'.format(n, '' if n == 1 else 's')

    running_lines = []
    for batch, phase, jobs in group(running):
        pcts = [100.0 * j['step'] / j['max_steps']
                for j in jobs if j.get('step') and j.get('max_steps')]
        pct = ' {0}%'.format(int(round(sum(pcts) / len(pcts)))) if pcts else ''
        running_lines.append('{0}{1} -- {2}{3} ({4})'.format(
            batch, described(batch), phase, pct, arms(jobs)))

    queued_lines = []
    notice = hold_notice(held_by)
    if notice:
        queued_lines.append(notice)
    for batch, phase, jobs in group(queued_order):
        queued_lines.append('{0} {1}{2} -- queued ({3})'.format(
            batch, phase, described(batch), arms(jobs)))

    return {'running': running_lines, 'queued': queued_lines}


def anticipated_queue(queued, running, limits, auto_closeout, existing_ids, auto_hof=True):
    """Simulate the wave-barrier scheduler forward over the pending queue and return the jobs
    in the order they are expected to launch -- with each queued batch's closeout **wave**
    inserted where it will actually run.

    Modelled straight from `_dispatch`: repeatedly form a wave from the highest-priority job's
    category, up to that category's limit, drain it, and let the trainings in it spawn **one**
    closeout per batch (priority `AUTO_CLOSEOUT_PRIORITY` = 10) that competes in the next wave. A
    closeout outranks a queued training (100), so a batch's closeout always slots in ahead of the
    following batch -- the exact interleaving the box will run.

    **There is no HOF hop any more.** The re-measure is stage B inside the closeout's own process
    (`eval_wave.py --chain`), so it is not a job, has nothing to forecast, and cannot be a wave of
    its own. `auto_hof` is kept in the signature and reported as part of the closeout line rather
    than dropped, because the runtime flag still exists and still decides whether stage B runs.

    A batch's four trainings collapsing into one closeout row is the *reason* this is grouped: the
    old forecast showed four closeout jobs and then four HOF jobs, which is eight rows for what is
    now one process.

    `queued` and `running` are lists of dicts with id/type/policy (queued also has priority, and a
    wave carries `policies`); `queued` is the real pending set (already includes closeouts
    synthesized for *finished* trainings), `running` seeds anticipated closeouts for trainings on the
    box now. `existing_ids` is every id already known (ledger, running, queued), so a job that
    already exists somewhere is never invented twice. Returns the ordered list of queued/anticipated
    job dicts."""
    def category(job_type):
        return 'eval' if job_type == 'eval' else 'trainer'

    seen = set(existing_ids)

    def closeout_for(jobs):
        """One closeout wave dict for a set of just-placed trainings, or None.

        Grouped by batch, exactly as `_auto_closeout_jobs` groups the real markers. Not grouped by
        env here: a forecast has no env to read, and getting the *count of rows* right is what this
        function is for.
        """
        out = []
        by_batch = {}
        for job in jobs:
            policy = job.get('policy')
            if not policy or not wants_closeout(job.get('type'), True, auto_closeout):
                continue
            by_batch.setdefault(batch_of(policy), []).append(policy)
        for batch, policies in sorted(by_batch.items()):
            cid = '{0}-closeout'.format(batch)
            if cid in seen:
                continue
            seen.add(cid)
            out.append({'id': cid, 'type': 'eval', 'policy': policies[0],
                        'policies': sorted(policies), 'chain': bool(auto_hof),
                        'priority': AUTO_CLOSEOUT_PRIORITY})
        return out

    pool = [dict(j) for j in queued]
    pool += closeout_for(running)                      # follow-ons for jobs on the box now

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
        pool += closeout_for(wave)         # the wave's trainings spawn their batches' closeouts
    return order


def viewer_png_paths(running_jobs, snek_dir):
    """Sorted PNG paths the chart viewer should show for the given running jobs.

    Each job's chart lives at a category-specific path: an eval writes
    evals/<policy>_eval_progress.png, a trainer writes runs/<policy>.png. `running_jobs`
    is an iterable of `(category, policies)` pairs -- **plural**, because one eval job is now a
    wave of four arms and each arm still writes its own chart. A bare string is accepted as a
    one-element list, so a caller with a single policy in hand needs no change. Falsy policies are
    skipped.

    Sorted so the result is comparable across polls -- the set changes when a wave flips
    train->eval, which is exactly when the viewer's fixed arg list has gone stale."""
    pngs = []
    for category, policies in running_jobs:
        if isinstance(policies, str) or policies is None:
            policies = [policies]
        for policy in policies:
            if not policy:
                continue
            if category == 'eval':
                pngs.append(os.path.join(snek_dir, 'evals', policy + '_eval_progress.png'))
            else:
                pngs.append(os.path.join(snek_dir, 'runs', policy + '.png'))
    return sorted(pngs)


MAX_VIEWER_PANELS = 8


def eval_batch_pngs(running_jobs, snek_dir, eval_dir_names):
    """Panel paths for an eval wave: **every chart in `evals/` belonging to a batch this wave is
    measuring**, not only the arms measuring right now.

    The set used to be the running jobs' own policies, made sticky within a wave. That covers a
    wave that drains and is not enough, because a batch's measurement does not arrive as one wave:
    `b45`'s closeout ran as three (`{a,c}`, `{b}`, `{d}`), so the window showed 2 panels, then 1,
    then 1, and a finished arm's chart was gone by the time anyone came back to look. Reading the
    set off the batch makes the slicing invisible -- the same four charts from the first slice to
    the last.

    **Membership is decided by the chart existing on disk**, which is the same rule the laptop's
    `--glob evals/<prefix>*_eval_progress.png` uses, and it is what keeps two hard-won properties
    intact. A panel is never blank-by-construction: `chart_viewer` deliberately has no per-panel
    title, so a path for an arm that has not started yet would be an unlabelled empty box (see the
    note above `_training_alive` in `chart_viewer.py`). And the set stays bounded without a TTL --
    only arms whose charts are in `evals/` right now can appear, so a historically wide batch like
    `b20` (36 arms over several waves) cannot open a window taller than the screen, which is the
    failure the laptop's arm registry had to solve a different way.

    It relies on the batch's earlier waves *keeping* their charts, which they now do:
    `eval_plan.archive_existing_eval_pngs` exempts the batches a wave is about to measure. Before
    that, wave 2 archived wave 1's charts on startup and there was nothing on disk to find."""
    batches = set()
    for _category, policies in running_jobs:
        if isinstance(policies, str) or policies is None:
            policies = [policies]
        for policy in policies:
            if policy:
                batches.add(batch_of(policy))
    suffix = '_eval_progress.png'
    pngs = []
    for name in eval_dir_names or []:
        if not name.endswith(suffix):
            continue
        if batch_of(name[:-len(suffix)]) in batches:
            pngs.append(os.path.join(snek_dir, 'evals', name))
    return sorted(pngs)[:MAX_VIEWER_PANELS]


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
