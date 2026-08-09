"""The unattended dispatcher. Runs under systemd (Restart=always) and drives the
whole box from git:

  1. fetch origin; re-read runtime.json (clamped; bad config -> keep last good)
  2. reap finished jobs; publish their artifacts to `results`
  3. reconcile: launch pending jobs (by id, against a local ledger) up to the
     live concurrency limits, unless paused/draining or disk is low
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
        """Best-effort: while trainers run, keep one decoupled chart viewer up on the
        display. It is a *separate* process (never a child of a trainer), so it can
        never affect training, and it exits itself when the trainers stop (--watch).
        A launch failure is logged and ignored -- a chart is never worth a job."""
        if not self.runtime.get('viewer', True) or not self.running:
            return
        try:
            # One already up? (also catches the case across a daemon restart.)
            if subprocess.run(['pgrep', '-f', 'chart_viewer.py'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                return
            pngs = [os.path.join(self.host['SNEK_DIR'], 'runs', rj.policy + '.png')
                    for rj in self.running.values() if rj.policy]
            if not pngs:
                return
            os.makedirs(self.host['LOG_DIR'], exist_ok=True)
            log = open(os.path.join(self.host['LOG_DIR'], 'chart_viewer.log'), 'ab')
            argv = [self.host['PYTHON_BIN'], '-u', 'chart_viewer.py'] + pngs + \
                   ['--watch', 'snek2.py', '--interval', '1', '--scale', '1.5',
                    '--title', 'snek training']
            subprocess.Popen(argv, cwd=self.host['SNEK_DIR'], env=dict(os.environ),
                             stdout=log, stderr=log, start_new_session=True, close_fds=True)
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
        counts = self._counts()
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
        desired.sort(key=lambda j: j.priority)
        for job in desired:
            if counts[job.category] >= limits[job.category]:
                continue
            self._launch(job)
            counts[job.category] += 1
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
