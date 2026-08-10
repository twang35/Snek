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
self-terminate via SNEK_MAX_STEPS, so the daemon never kills anything; a restart
re-adopts still-running jobs by pid.
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
from .job import parse_job, JobError

TERMINAL = ('done', 'failed')


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
        """Re-adopt jobs the ledger says were running whose pid is still alive;
        mark the rest done (they finished while the daemon was down -- exit code
        is unknown across reparenting, so success is assumed unless later checks
        say otherwise)."""
        for jid, rec in self.ledger.items():
            if rec.get('state') != 'running':
                continue
            pid = rec.get('pid')
            if pid and launch.pid_alive(pid):
                self.running[jid] = launch.RunningJob(
                    _StubJob(jid, rec), rec.get('policy'), pid,
                    rec.get('log', ''), popen=None, started=rec.get('started'))
            else:
                rec['state'] = 'done'
                rec['finished'] = time.time()
                rec['note'] = 'completed while runner was down'
        self._save_ledger()

    # -------------------------------------------------------------------- loop
    def poll_once(self):
        gitbus.fetch(self.host)
        self._apply_runtime()
        self._reap()
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

    def _dispatch(self):
        free = _disk_free_gb(self.host['REPO_PATH'])
        if free is not None and free < self.runtime['disk_min_gb']:
            self.config_notes = self.config_notes + \
                ['disk low ({0} GB < {1}) -- not launching'.format(free, self.runtime['disk_min_gb'])]
            return
        limits = {'trainer': self.runtime['max_trainers'], 'eval': self.runtime['max_evals']}
        desired = []
        for name, text in gitbus.read_pending_jobs(self.host):
            try:
                job = parse_job(text, name)
            except JobError as e:
                jid = name[:-5] if name.endswith('.json') else name
                if self.ledger.get(jid, {}).get('state') != 'failed':
                    self.ledger[jid] = {'state': 'failed', 'type': 'unknown',
                                        'error': str(e), 'finished': time.time()}
                continue
            state = self.ledger.get(job.id, {}).get('state')
            if state in TERMINAL or job.id in self.running:
                continue
            desired.append(job)
        # Wave-barrier scheduling: a "wave" is a set of same-type jobs launched together,
        # up to that type's limit, and NOTHING new starts until the whole wave finishes.
        # So trainings and evals never overlap, and a freed slot is not backfilled mid-wave
        # -- the box idles on the stragglers until the slowest job in the wave is done, then
        # the next wave opens. We still parse pending specs while a wave runs (above), so a
        # malformed one is marked failed promptly, but we launch nothing until idle.
        if self.running or not desired:
            self._save_ledger()
            return
        # The highest-priority pending job picks the wave's type; only jobs of that type
        # join it, up to that type's limit.
        desired.sort(key=lambda j: j.priority)
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
        self.running[job.id] = rj
        self.ledger[job.id] = {'state': 'running', 'type': job.type, 'policy': rj.policy,
                               'pid': rj.pid, 'log': rj.log_path, 'started': rj.started}

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
            'ledger': {k: v.get('state') for k, v in self.ledger.items()},
            'disk_free_gb': _disk_free_gb(self.host['REPO_PATH']),
            'load_avg': list(os.getloadavg()),
        }
        try:
            gitbus.publish_status(self.host, json.dumps(status, indent=2))
        except Exception as e:
            sys.stderr.write('publish_status failed: {0}\n'.format(e))


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
    None -- a fresh daemon, or the idle gap between waves -- resets too, so a following wave
    of the same category does not inherit the previous batch's panels."""
    if prev_category != category or not prev_pngs:
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
