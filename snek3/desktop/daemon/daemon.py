"""The desktop's bus adapter. Runs under systemd (`Restart=always`) and drives the box from git.

**Since 2026-09-05 the daemon schedules nothing itself.** It feeds `tools/scheduler.py` — the same
scheduler the laptop runs — and publishes what the scheduler does. One cycle:

1. fetch `origin`; re-read `runtime.json` (clamped; a bad config keeps the last good one)
2. run any `deploy`/`restart` action on `ops`
3. relay `paused`/`drain` as the scheduler's hold marker; read the shared queue's pool
   (`tools.claims show --json`, on the env python -- the daemon imports nothing from the project); start
   the scheduler if none is running and `ops` or `claims` moved since the last start
4. publish `status.json` to `ops-status`: the scheduler's own status, the pool, the box's extras, the
   laptop's lines

**Since 2026-09-06 the daemon mirrors nothing and decides nothing about what the box runs.** Both boxes
pull waves from the one queue on `ops` by claiming them on the `claims` branch (`tools/claims.py`,
`plans/archive/shared-queue.md`); the scheduler mirrors the waves this box holds into its own queue directory,
claims the next when it has nothing left, and exits when the pool has nothing for it. The daemon's
"is there work" is therefore "did `ops` or `claims` move" -- a scheduler that finds nothing to claim
exits in seconds -- plus the box's own unfinished holdings, which the pool view names.

The loop body is wrapped so a bad job, a transient git error or a malformed config can never kill the
daemon — in normal operation the box has no ssh backstop, so **staying up is the first requirement.**

The scheduler is detached (`setsid`), so a daemon restart never kills it; the new daemon re-adopts it by
pid, or by a boot id that says the box rebooted and the scheduler is gone with everything it ran — in
which case the next spawn resumes every arm from its checkpoint and every pass from its shard files,
because the scheduler's state is the filesystem (`tools/scheduler.py`). That is the whole recovery.

What the daemon remembers is one small file, `state.json` beside the ledger: the scheduler's pid and
boot, and the heads of `ops` and `claims` it was last started on. The ledger file stays for actions and
malformed specs, and as the archive of the 400 jobs the daemon ran before 2026-09-05.
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
from .job import parse_job, to_ascii, JobError, ACTION_TYPES

TERMINAL = ('done', 'failed')
PHASE_NAMES = {'stageb': 'stage B', 'hof5000': 'hof5000', 'hof30k': 'hof30k'}
# The scheduler's status file, relative to the runs directory: `tools/live_runs.py` spells the same
# path; the daemon cannot import it (base python) and pins it here.
STATUS_RELATIVE = os.path.join('.live', '.status.json')
HOLD_RELATIVE = os.path.join('.live', '.paused')
# A scheduler that exits with work pending is not relaunched sooner than this unless a trigger asks.
RESPAWN_BACKOFF_SECONDS = 600
# `tools.claims show --json` is a few fetches and a read; well past this it is stuck, not slow.
POOL_TIMEOUT = 300

# A site build is a manifest walk and a megabyte of copies; well past this it is stuck, not slow.
SITE_BUILD_TIMEOUT = 900


def trigger_path(host):
    """The file that makes the daemon do a network cycle now, rather than at its next `git_seconds`.

    Derived from `LEDGER_PATH`'s directory rather than being its own `host.env` key, deliberately:
    `host.env` is not in git — only `host.env.example` is — so a new required key would have to be
    hand-edited on the box during the same deploy that started depending on it, and a deploy that
    landed first would crash `load_host_config` on every start. `TRIGGER_PATH` still overrides it.
    """
    return host.get('TRIGGER_PATH') or os.path.join(
        os.path.dirname(host['LEDGER_PATH']), 'trigger')


def state_path(host):
    return os.path.join(os.path.dirname(host['LEDGER_PATH']), 'state.json')


class Daemon(object):
    def __init__(self, host):
        self.host = host
        self.runtime = dict(config_module.RUNTIME_DEFAULTS)
        config_module.clamp_runtime(self.runtime, host)
        self.config_notes = []
        self.ledger = self._load_json(self.host['LEDGER_PATH'])
        self.state = self._load_json(state_path(host))
        for stale in ('published', 'running', 'specs', 'seeded'):      # the mirror's memory, gone with it
            self.state.pop(stale, None)
        self.pool = {}                 # the shared queue as `tools.claims show --json` last read it
        self.pool_error = None
        self._unpushed = []        # branches with local-only commits, surfaced in status.json
        self._last_git = 0.0       # when the loop last *attempted* its network half. 0 so the first cycle does one
        self._last_spawn_failed = 0.0
        self.stop = False
        self.restart_requested = None
        self.run_command = subprocess.run      # injectable for tests: how an action's command is run
        self.site = None                       # the last site build: iso, ok, seconds, note
        self.spawn = launch.spawn_scheduler    # injectable for tests
        self.scheduler = None                  # Popen, or None when adopted by pid / not running
        self._reattach()

    # ------------------------------------------------------------------ files

    def _load_json(self, path):
        try:
            with open(path) as handle:
                loaded = json.load(handle)
                return loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        staging = path + '.tmp'
        with open(staging, 'w') as handle:
            json.dump(payload, handle, indent=2)
        os.replace(staging, path)

    def _save_ledger(self):
        self._save_json(self.host['LEDGER_PATH'], self.ledger)

    def _save_state(self):
        self._save_json(state_path(self.host), self.state)

    def box(self):
        return self.host.get('BOX') or 'desktop'

    def runs_dir(self):
        return launch.runs_dir(self.host)

    def _scheduler_status(self):
        return self._load_json(os.path.join(self.runs_dir(), STATUS_RELATIVE))

    # ------------------------------------------------------------ the scheduler

    def scheduler_alive(self):
        if self.scheduler is not None:
            if self.scheduler.poll() is None:
                return True
            self.state['last_exit'] = self.scheduler.returncode
            self.scheduler = None
            self.state['scheduler_pid'] = None
            return False
        pid = self.state.get('scheduler_pid')
        if pid and launch.pid_alive(pid):
            return True
        if pid:
            self.state['scheduler_pid'] = None
            self.state['last_exit'] = None          # reparented: unknown
        return False

    def _reattach(self):
        """Re-adopt the scheduler by pid across a daemon restart, or forget it across a reboot.

        The boot id separates the two: detached, the scheduler really does survive a daemon restart
        and its pid is still it; after a reboot the pid may be recycled onto anything, so it is never
        consulted. The `running` jobs it was seen with are published when they are next found
        finished, whichever way it went.
        """
        boot = boot_id()
        recorded = self.state.get('scheduler_boot')
        if boot and recorded and recorded != boot:
            self.state['scheduler_pid'] = None
        if not self.scheduler_alive():
            # Gone while no daemon was watching, so how it exited is unknown; the next poll starts one
            # regardless of the queue signature. On a finished queue that costs one scheduler that
            # exits at once, on an interrupted one it is the resume.
            self.state['scheduler_pid'] = None
            self.state['spawn_signature'] = None
        self._save_state()

    def _queue_signature(self):
        """What "the queue changed" means now: the heads of `ops` and `claims`. Either moving -- a spec
        pushed, a wave claimed or released by either box -- is a reason to let a scheduler look."""
        remote = self.host['GIT_REMOTE']
        return json.dumps([gitbus.ref_head(self.host['REPO_PATH'], '{0}/{1}'.format(remote, self.host['OPS_BRANCH'])),
                           gitbus.ref_head(self.host['REPO_PATH'], '{0}/{1}'.format(remote, self._claims_branch()))])

    def _claims_branch(self):
        return self.host.get('CLAIMS_BRANCH') or 'claims'

    def _held_open(self):
        """This box's claims the pool view says are not finished: work pending here."""
        return [h for h in (self.pool.get('held') or {}).get(self.box(), []) if not h.get('done')]

    def _held(self):
        return bool(self.runtime.get('paused') or self.runtime.get('drain'))

    def _relay_hold(self):
        path = os.path.join(self.runs_dir(), HOLD_RELATIVE)
        try:
            if self._held():
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w') as handle:
                    handle.write('paused by runtime.json on ops\n')
            elif os.path.exists(path):
                os.unlink(path)
        except OSError as error:
            sys.stderr.write('could not relay the hold marker: {0}\n'.format(error))

    def _maybe_spawn(self, signature, forced=False):
        """Starts the scheduler when there is a reason to: the queue changed since it last ran, a hold
        was lifted, or a trigger asked. Never while one is alive, never under a hold, never below the
        disk floor, and after a failed exit not for `RESPAWN_BACKOFF_SECONDS`."""
        if self.scheduler_alive() or self._held():
            self.state['was_held'] = self._held()
            return False
        free = _disk_free_gb(self.host['REPO_PATH'])
        if free is not None and free < self.runtime['disk_min_gb']:
            self.config_notes = self.config_notes + [
                'disk low ({0} GB < {1}) — not starting the scheduler'.format(free, self.runtime['disk_min_gb'])]
            return False
        changed = signature != self.state.get('spawn_signature')
        lifted = self.state.get('was_held') and not self._held()
        self.state['was_held'] = False
        since_spawn = time.time() - (self.state.get('spawned') or 0)
        recently_failed = self.state.get('last_exit') not in (None, 0) and since_spawn < RESPAWN_BACKOFF_SECONDS
        # Work left and nothing running it -- a wave this box holds that is not finished, or a scheduler
        # that exited non-zero -- is a reason to start one too, not only a changed queue: before 2026-09-05
        # a scheduler that died (crash, OOM, a kill) left its batch idle until a trigger or a new spec.
        # Backed off like a failed exit, so a scheduler that cannot make progress costs one start per
        # `RESPAWN_BACKOFF_SECONDS`, never a loop.
        needed = ((bool(self._held_open()) or self.state.get('last_exit') not in (None, 0))
                  and since_spawn >= RESPAWN_BACKOFF_SECONDS)
        if not (changed or lifted or forced or needed) or (recently_failed and not forced):
            return False
        try:
            process, log_path = self.spawn(self.host, self.runtime)
        except Exception as error:      # spawning is the one place this can fail before running
            self.config_notes = self.config_notes + ['scheduler did not start: {0}'.format(error)]
            return False
        self.scheduler = process
        self.state.update({'scheduler_pid': process.pid, 'scheduler_boot': boot_id(),
                           'spawned': time.time(), 'spawn_signature': signature,
                           'scheduler_log': log_path, 'last_exit': None})
        sys.stderr.write('started the scheduler: pid {0}, log {1}\n'.format(process.pid, log_path))
        return True

    # ------------------------------------------------------------------ specs

    def _actions(self):
        """The `deploy`/`restart` actions on `ops`. Malformed specs are recorded `failed` in the ledger,
        once, so a bad commit shows under `attention` and stops nothing; work specs are the scheduler's
        (`tools/claims.py` reads them off the same ref) and are not looked at here."""
        actions, marked = [], False
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
            if job.category == 'action':
                actions.append(job)
        if marked:
            self._save_ledger()
        return actions

    def _read_pool(self):
        """The shared queue's view, from `tools.claims show --json` on the env python: unclaimed work,
        each box's holdings (and whether they are finished), the lines and attention for the status, and
        the derived `ledger` the viewer reads. It fetches `ops`, `claims`, both feeds and both statuses
        itself. A failure keeps the last view and says so in `config_notes`."""
        argv = [self.host['PYTHON_BIN'], '-m', 'tools.claims', 'show', '--json']
        env = dict(os.environ, **launch.scheduler_env(self.host, self.runtime))
        env['PYTHONPATH'] = self.host['SNEK_DIR']
        try:
            result = self.run_command(argv, cwd=self.host['SNEK_DIR'], env=env, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=POOL_TIMEOUT)
            if result.returncode != 0:
                raise RuntimeError('exit {0}: {1}'.format(result.returncode, (result.stderr or '').strip()[-300:]))
            view = json.loads(result.stdout or '{}')
            if not isinstance(view, dict):
                raise RuntimeError('not a JSON object')
        except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
            self.pool_error = 'could not read the shared queue ({0}); showing the last view'.format(error)
            return self.pool
        self.pool_error = None
        self.pool = view
        return view

    # -------------------------------------------------------------------- loop

    def git_due(self):
        """Whether this cycle does its network half. `git_seconds` of 0 means every cycle."""
        interval = int(self.runtime.get('git_seconds', 0) or 0)
        return interval <= 0 or (time.time() - self._last_git) >= interval

    def take_trigger(self):
        """Consumes a manual trigger if one is waiting. **The unlink *is* the test.**"""
        try:
            os.unlink(trigger_path(self.host))
        except OSError:
            return False
        sys.stderr.write('manual trigger consumed; forcing a git cycle\n')
        return True

    def poll_once(self, git=None, forced=False):
        """One cycle. `git=None` decides from `git_seconds`; True forces it, as a trigger does.

        The local half — mirror, hold, spawn, publish results — runs every `poll_seconds`; the network
        half is one fetch, one status push and one retry of any local-only commit.
        """
        if git is None:
            git = self.git_due()
        if git:
            self._last_git = time.time()      # stamped before the attempt: an attempt is what costs traffic
            gitbus.fetch(self.host)
            gitbus.fetch_laptop_status(self.host)
            self._unpushed = gitbus.push_unpushed(self.host)
        self._apply_runtime()
        self._run_actions(self._actions())
        if self.stop:
            return
        self._relay_hold()
        if git:
            self._read_pool()           # fetches `claims` and the feeds too; the signature reads them after
        self._maybe_spawn(self._queue_signature(), forced=forced)
        self._save_state()
        if git:
            self._build_site(forced=forced)
            self._publish()

    # ---- actions: deploy and restart, run by the daemon itself

    def _run_actions(self, actions):
        for job in actions:
            if self.ledger.get(job.id, {}).get('state') in TERMINAL:
                continue
            if job.type == 'deploy':
                self._deploy(job)
            elif job.type == 'restart':
                self._restart(job, 'restart action')
            if self.stop:
                return

    def _head(self):
        result = self.run_command(['git', 'rev-parse', 'HEAD'], cwd=self.host['REPO_PATH'], text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return (result.stdout or '').strip()[:12]

    def _daemon_changed(self, before, after):
        """Whether the merge touched the daemon's own code -- the only thing a restart is for."""
        if not before or not after or before == after:
            return False
        result = self.run_command(
            ['git', 'diff', '--name-only', '{0}..{1}'.format(before, after), '--',
             'snek3/desktop/daemon', 'snek3/desktop/systemd'],
            cwd=self.host['REPO_PATH'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return bool((result.stdout or '').strip())

    def _deploy(self, job):
        """Runs `desktop/deploy` -- fetch, settle collisions, fast-forward -- as the box would from an
        ssh, and restarts if the daemon's code changed (or the spec says so). Recorded in the ledger
        with the heads and the script's tail, so `status.json` says what happened and a failed deploy
        (exit 3: a differing JSON, nothing touched) shows under `attention` and is never retried."""
        started = time.time()
        before = self._head()
        desktop_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            result = self.run_command([sys.executable, '-m', 'daemon.deploy'], cwd=desktop_dir, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
            rc, output = result.returncode, result.stdout or ''
        except (OSError, subprocess.SubprocessError) as error:
            rc, output = -1, 'could not run deploy: {0}'.format(error)
        after = self._head()
        tail = [line for line in output.strip().splitlines() if line.strip()][-8:]
        self.ledger[job.id] = {
            'state': 'done' if rc == 0 else 'failed', 'type': 'deploy', 'started': started,
            'finished': time.time(), 'rc': rc, 'head_before': before, 'head_after': after,
            'output': tail, 'error': None if rc == 0 else 'deploy exited {0}: {1}'.format(
                rc, tail[-1] if tail else '(no output)')}
        self._save_ledger()
        if rc != 0:
            return
        restart = job.restart if job.restart is not None else self._daemon_changed(before, after)
        self.ledger[job.id]['restart'] = bool(restart)
        if restart:
            self._restart(job, 'deploy {0} -> {1} changed the daemon'.format(before, after))
        else:
            self._save_ledger()

    def _restart(self, job, why):
        """Records the action as done, publishes so the ledger on `ops-status` already shows it, and
        stops the loop. systemd (`Restart=always`, 10 s) relaunches the daemon, which re-adopts the
        scheduler by pid."""
        record = self.ledger.setdefault(job.id, {'type': job.type, 'started': time.time()})
        record.update({'state': 'done', 'finished': time.time(), 'restarted': True, 'why': why})
        self._save_ledger()
        self._save_state()
        self.restart_requested = '{0}: {1}'.format(job.id, why)
        self.stop = True
        self._publish()

    def _apply_runtime(self):
        config, notes = config_module.parse_runtime_config(
            gitbus.read_runtime_text(self.host), self.host)
        if config is None:
            self.config_notes = ['using last-known-good config; errors: ' + '; '.join(notes)]
        else:
            self.runtime, self.config_notes = config, notes

    # ------------------------------------------------------------- results

    def _build_site(self, forced=False):
        """Rebuilds the GitHub Pages viewer from both boxes' results feeds and this box's live charts
        (`tools/site_build.py`, on the env python, as the scheduler is run) and snapshot-pushes the
        `site` branch. Once per network cycle; a trigger forces a rebuild even when nothing moved. A
        failure is a line under `attention`, never a stopped daemon."""
        argv = [self.host['PYTHON_BIN'], '-m', 'tools.site_build'] + (['--force'] if forced else [])
        env = dict(os.environ, **launch.scheduler_env(self.host, self.runtime))
        env['PYTHONPATH'] = self.host['SNEK_DIR']
        env['SNEK_SITE_BRANCH'] = self.host.get('SITE_BRANCH') or 'site'
        env['SNEK_SITE_WORKTREE'] = self.host.get('SITE_WORKTREE') or os.path.join(
            os.path.dirname(self.host['STATUS_WORKTREE']), 'site')      # beside status and results
        started = time.time()
        try:
            result = self.run_command(argv, cwd=self.host['SNEK_DIR'], env=env, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=SITE_BUILD_TIMEOUT)
            rc, out = result.returncode, (result.stdout or '').strip()
        except (OSError, subprocess.SubprocessError) as error:
            rc, out = -1, str(error)
        last = out.splitlines()[-1] if out else ''
        self.site = {'iso': time.strftime('%Y-%m-%dT%H:%M:%S'), 'ok': rc == 0, 'seconds': round(time.time() - started, 1),
                     'note': last[:300], 'forced': bool(forced)}
        if rc != 0:
            sys.stderr.write('site build failed (rc {0}): {1}\n'.format(rc, out[-2000:]))

    def _attention(self, status):
        lines = []
        if self.pool_error:
            lines.append('** ' + self.pool_error)
        lines.extend(str(line) for line in self.pool.get('attention') or [])
        for branch in self._unpushed:
            lines.append('** {0} has local-only commit(s): results are on the box but not on '
                         'github. Retried every network cycle.'.format(branch))
        for job_id, record in sorted(self.ledger.items(),
                                     key=lambda item: item[1].get('finished') or 0, reverse=True)[:50]:
            if record.get('state') == 'failed' and record.get('type') in ACTION_TYPES:
                lines.append('** {0} failed: {1}. Nothing was changed; fix it and queue a new id.'.format(
                    job_id, record.get('error') or 'rc={0}'.format(record.get('rc'))))
            elif record.get('state') == 'failed' and record.get('type') == 'unknown':
                lines.append('** spec {0} is malformed and is not run: {1}'.format(
                    job_id, record.get('error')))
        if self.site and not self.site.get('ok'):
            lines.append('** the site build failed at {0}: {1}. The Pages viewer is stale; retried every '
                         'network cycle, now on trigger.'.format(self.site.get('iso'), self.site.get('note')))
        if not self.scheduler_alive() and self.state.get('last_exit') not in (None, 0):
            lines.append('** the scheduler exited {0} (log {1}); it is restarted when the queue changes '
                         'or a trigger asks'.format(self.state.get('last_exit'), self.state.get('scheduler_log')))
        for line in (status.get('at_a_glance') or {}).get('attention') or []:
            if not str(line).startswith('** paused'):
                lines.append(str(line))
        return lines

    def _publish(self):
        status = self._scheduler_status()
        alive = self.scheduler_alive()
        if alive and status.get('pid') == self.state.get('scheduler_pid'):
            glance = dict(status.get('at_a_glance') or {})
            running = list(status.get('running') or [])
        else:
            glance = build_at_a_glance([], [], {})
            running = []
        # The pool is the daemon's own read (this network cycle), fresher than the scheduler's last event.
        glance['pool'] = list(self.pool.get('lines') or glance.get('pool') or [])
        glance['attention'] = self._attention(status)
        notice = hold_notice([flag for flag in HOLD_FLAGS if self.runtime.get(flag)])
        if notice:
            glance['queued'] = [notice] + [line for line in glance.get('queued', []) if not line.startswith('** queue')]
        payload = {
            'iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'ts': time.time(),
            'project': 'snek3',
            'at_a_glance': with_laptop(glance, gitbus.read_laptop_status(self.host)),
            'scheduler': {'alive': alive, 'pid': self.state.get('scheduler_pid'),
                          'spawned': self.state.get('spawned'), 'last_exit': self.state.get('last_exit'),
                          'log': self.state.get('scheduler_log'),
                          'status_iso': status.get('iso')},
            'runtime': self.runtime,
            'config_notes': self.config_notes,
            'running': running,
            # Derived, for `tools/viewer_manifest.py`'s pass states: done from the feeds, running from the
            # statuses, queued for the rest (`tools.claims.ledger`). Actions and malformed specs from ours.
            'ledger': dict({job_id: record['state'] for job_id, record in self.ledger.items()
                            if record.get('state') in TERMINAL}, **(self.pool.get('ledger') or {})),
            'pool': {key: self.pool.get(key) for key in ('unclaimed', 'held', 'status_ages', 'heads')},
            'disk_free_gb': _disk_free_gb(self.host['REPO_PATH']),
            'head': self._head(),
            'site': self.site,
            'load_avg': list(os.getloadavg()),
        }
        try:
            if not gitbus.publish_status(self.host, status_json(payload)):
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
    """
    return json.dumps(status, indent=2, ensure_ascii=False)


# **`b<number>` is the only batch prefix, in both eras and both algorithms.** A `p` series was tried
# for PPO and renamed back on 2026-08-31 (`p0-p3` -> `b3-b6`) precisely so this pattern stays one
# character wide: the prefix was missing from it for a day, and every PPO arm became its own batch.
# **Do not add a second prefix — name the batch `b<n>`.**
_BATCH_RE = re.compile(r'^(b\d+)')
_PASS_ID_RE = re.compile(r'-(stageb|hof5000|hof30k)(-w\d+)?$')


def batch_of(job_id):
    """The batch a job id belongs to: the leading `b<number>`, or the id up to the first `-`."""
    job_id = job_id or ''
    match = _BATCH_RE.match(job_id)
    return match.group(1) if match else (job_id.split('-')[0] or job_id)


def pass_of(job_id):
    """Which of the chain's passes a job id names — `'stageb'`, `'hof5000'`, `'hof30k'` — or None.

    Read off the id suffix the scheduler mints (and hand specs copy: `b11-hof5000`). **The `-w<k>`
    tail has to be part of the pattern**: a batch's second wave is `b1-stageb-w2`.
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
    """The `at_a_glance.queued` line for a held queue, or None when nothing holds it."""
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

    Consecutive runs only, so a gap -- wave 2 pulled from the queue -- still shows as two ranges.
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
    """One caption for a group of jobs, from their labels (`b11: lr5e4, seed 1 of 4 -- wave 3 of 4`
    -> `lr5e4 -- wave 3 of 4`, cells in first-seen order) or, failing labels, from their policy ids'
    knob tokens. Empty when neither yields anything."""
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


def format_eta(seconds):
    """Seconds as a reader's estimate: `<1m`, `40m`, `1.4h`, `12h`. None for None."""
    if seconds is None:
        return None
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return '<1m'
    if seconds < 3600:
        return '{0}m'.format(int(round(seconds / 60.0)))
    hours = seconds / 3600.0
    return '{0:.1f}h'.format(hours) if hours < 10 else '{0}h'.format(int(round(hours)))


def group_eta(jobs):
    """Seconds a group of jobs takes together, or None when no job carries an `eta_seconds`.

    Training arms of one wave run at once, so a wave takes its slowest arm (`wave` groups them; arms
    without one are taken as one wave, which is what a running wave is); waves and passes run one
    after another, so they add. A job without an estimate contributes nothing, so a group with some
    unknown arms reads low rather than not at all -- `remaining_text` says when that happened.
    """
    waves, serial, known = {}, 0.0, False
    for job in jobs:
        eta = job.get('eta_seconds')
        if eta is None:
            continue
        known = True
        if job.get('type') == 'train':
            key = job.get('wave')
            waves[key] = max(waves.get(key, 0.0), float(eta))
        else:
            serial += float(eta)
    return sum(waves.values()) + serial if known else None


def remaining_text(running_seconds, queued_seconds, now, partial=False):
    """One line for the whole box: `~11.3h of work: 0.6h running + 10.7h queued; clear ~Sat 23:40`.

    None when neither part has an estimate. `partial` appends a note that some job carried no
    estimate, so the total is a floor rather than a forecast.
    """
    if running_seconds is None and queued_seconds is None:
        return None
    parts = [format_eta(seconds) + ' ' + word for seconds, word in
             ((running_seconds, 'running'), (queued_seconds, 'queued')) if seconds is not None]
    total = (running_seconds or 0.0) + (queued_seconds or 0.0)
    text = '~{0} of work: {1}; clear ~{2}'.format(
        format_eta(total), ' + '.join(parts), time.strftime('%a %H:%M', time.localtime(now + total)))
    return text + ' (some jobs have no estimate yet)' if partial else text


def build_at_a_glance(running, queued_order, labels, held_by=(), attention=(), now=None):
    """The `at_a_glance` block: `{'running': [str], 'queued': [str], 'attention': [str], 'remaining': str|None}`.

    One line per `(batch, phase)` group in first-seen order. A running trainer dict carries
    `step`/`max_steps`, from which each running line shows the mean percent done across that batch's
    arms. `labels` maps a batch id to its human description. `held_by` puts `hold_notice` first in
    `queued`. Kept a pure function of plain dicts, so it is testable without a live box — and so the
    scheduler on either box can build its own lines with it.

    **Time estimates ride on the jobs** (user, 2026-09-05: "how much time is queued up for both
    boxes, to see if anything needs to be shifted"). A job may carry `eta_seconds` -- what is left of
    a running job, the whole of a queued one -- and a queued training arm its `wave` number; the
    scheduler (`tools/eta.py`) computes both and this only adds them up: a running line ends
    `| ~40m left`, a queued line `| ~4.5h`, and `remaining` is the box's total with the clock time it
    clears. Without estimates the lines are as before and `remaining` is None -- the daemon's own
    idle-queue view, for one.
    """
    def group(jobs, collapse_passes=False):
        # `collapse_passes` folds the chain's three passes into one `evals` group per batch, split by
        # whether the batch's own training has been passed in the order.
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
        label = describe_jobs(jobs) or labels.get(batch) or ''
        if not label:
            return ''
        label = to_ascii(label)
        return ' | ' + (label if len(label) <= 80 else label[:77] + '...')

    def arms(jobs):
        """"N arms", counted in **distinct policies**, not in jobs."""
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

    def eta_of(jobs):
        """`(group seconds or None, whether any job lacked an estimate)`."""
        seconds = group_eta(jobs)
        return seconds, seconds is not None and any(job.get('eta_seconds') is None for job in jobs)

    totals = {'running': None, 'queued': None}
    partial = False

    def add(kind, seconds):
        if seconds is not None:
            totals[kind] = (totals[kind] or 0.0) + seconds

    running_lines = []
    for batch, phase, jobs in group(running):
        percents = [100.0 * job['step'] / job['max_steps']
                    for job in jobs if job.get('step') and job.get('max_steps')]
        percent = (' {0}%'.format(int(round(sum(percents) / len(percents))))
                   if percents else '')
        seconds, missing = eta_of(jobs)
        partial = partial or missing
        add('running', seconds)
        running_lines.append('{0}{1} | {2}{3} ({4}){5}'.format(
            batch, described(batch, jobs), phase, percent, arms(jobs),
            ' | ~{0} left'.format(format_eta(seconds)) if seconds is not None else ''))

    queued_lines = []
    notice = hold_notice(held_by)
    if notice:
        queued_lines.append(notice)
    for batch, phase, jobs in group(queued_order, collapse_passes=True):
        seconds, missing = eta_of(jobs)
        partial = partial or missing
        add('queued', seconds)
        queued_lines.append('{0} {1}{2} ({3}){4}'.format(
            batch, phase, described(batch, jobs), arms(jobs),
            ' | ~{0}'.format(format_eta(seconds)) if seconds is not None else ''))

    return {'running': running_lines, 'queued': queued_lines, 'attention': list(attention),
            'remaining': remaining_text(totals['running'], totals['queued'],
                                        time.time() if now is None else now, partial=partial)}


LAPTOP_KEYS = ('laptop_running', 'laptop_queued', 'laptop_remaining', 'laptop_iso')


def with_laptop(glance, laptop_status_text):
    """`glance` plus the laptop's lines: `laptop_running`, `laptop_queued`, `laptop_remaining` and `laptop_iso`.

    The laptop's scheduler publishes its own `status.json` to the `laptop-status` branch
    (`tools/laptop_status.py`), in this same `at_a_glance` shape. **`laptop_iso` is the laptop's own
    timestamp, and it is the staleness signal**: the last publish before exiting is empty, so empty
    lists mean idle, while a running line under an hours-old `laptop_iso` means the scheduler died.
    Unparseable or absent text gives empty lists and a null `laptop_iso`, never an error.
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
    remaining = laptop.get('remaining')
    out['laptop_remaining'] = str(remaining) if remaining is not None else None
    out['laptop_iso'] = status.get('iso')
    return out


BOOT_ID_PATH = '/proc/sys/kernel/random/boot_id'
_BOOT_ID = []   # one-element cache; the value cannot change without the process dying


def boot_id():
    """This boot's kernel id, or None where the kernel does not publish one (a mac, a container)."""
    if not _BOOT_ID:
        try:
            with open(BOOT_ID_PATH) as handle:
                _BOOT_ID.append(handle.read().strip() or None)
        except OSError:
            _BOOT_ID.append(None)
    return _BOOT_ID[0]


def _disk_free_gb(path):
    try:
        stats = os.statvfs(path)
        return round(stats.f_bavail * stats.f_frsize / 1e9, 1)
    except OSError:
        return None


def main():
    host_env = os.environ.get(
        'SNEK_DAEMON_HOST_ENV',
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'config', 'host.env'))
    host = config_module.load_host_config(host_env)
    daemon = Daemon(host)

    def handle(_signal, _frame):
        daemon.stop = True      # stop the daemon; the detached scheduler keeps running
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    forced = False
    while not daemon.stop:
        try:
            # `True if forced else None`: a trigger forces the network half and a spawn attempt,
            # otherwise `poll_once` decides from `git_seconds`.
            daemon.poll_once(git=True if forced else None, forced=forced)
        except Exception as error:      # the loop must never die
            sys.stderr.write('poll error: {0}\n'.format(error))
        forced = _wait_for_next_poll(daemon)
    if daemon.restart_requested:
        sys.stderr.write('restarting for {0}: exiting so systemd relaunches the daemon\n'.format(
            daemon.restart_requested))


def _wait_for_next_poll(daemon):
    """Sleeps `poll_seconds` in one-second steps, returning early on a stop or a manual trigger."""
    for _ in range(int(daemon.runtime['poll_seconds'])):
        if daemon.stop:
            return False
        if daemon.take_trigger():
            return True
        time.sleep(1)
    return False


if __name__ == '__main__':
    main()
