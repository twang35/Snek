"""Run a batch of desktop specs on the laptop the way the desktop daemon would.

    PYTHONPATH=. nohup /opt/miniconda3/envs/snek3/bin/python -u -m tools.laptop_batch \\
        logs/b13specs/ > logs/b13-batch.log 2>&1 &

**The same spec files, the same waves, the same stage B.** The arguments are the daemon's own
`queue/pending/*.json` specs (files, or directories of them; `git show origin/ops:...` them onto the
laptop), read in id order and run in waves of `--wave` arms. Each arm is `train.py <policy>` with the
spec's `env` and `max_steps`, exactly as `desktop/runner/launch.py` would set them; when every arm of
a wave has exited, the wave's stage B runs as one `tools.closeout` over those arms, named
`<batch>-stageb`, `<batch>-stageb-w2`, ... as the daemon names its auto-queued waves. Then the next
wave. So a batch dequeued from the desktop -- b13 on 2026-09-03, to shorten the box's queue -- runs
here with nothing rewritten and lands in `runs/` under the names every later tool expects.

**Resumable, and it never doubles an arm.** An arm whose `_evals.json` already reports `max_steps` is
skipped; an arm that is *live on this box* -- registered in `runs/.live/` by a trainer that this
process did not start, say a previous driver that was killed -- is waited for rather than relaunched;
closeout resumes its own shard files. Rerunning the same command after a kill or a reboot is the whole
recovery procedure.

**`--after <pid>` sequences batches.** The driver starts nothing until that process has exited, so a
second batch queued behind a running one -- b14 behind b13's last stage B on 2026-09-04 -- waits for
the whole of it, stage B included, rather than launching eight trainers into a closeout. The trainer
cap alone cannot do that: a closeout is not a trainer, and a driver of its own has no cap check.

**Never more than `--max-trainers` (8) trainers on the box, counting anything else running here.**
Before each launch the driver waits until the box's live trainer count is below the cap, so a batch
started beside a user's own arm shares the box rather than exceeding it.

This is the laptop half of the "one implementation for both boxes" rule, not a second scheduler: the
desktop has a daemon because it has a git bus and a queue; the laptop has this because it has a
shell. What is shared -- the spec format, the wave barrier, the stage-B naming -- is read from the
specs and copied from the daemon's behaviour rather than reimplemented with a twist.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

from env import constants
from tools import live_runs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WAVE = 8
DEFAULT_MAX_TRAINERS = 8
# Stage B runs alone on this box -- the next wave waits for it -- so the shard count is sized to the
# laptop's 14 cores (10P + 4E), not to what is left beside eight trainers. 8 left 6 cores idle
# (66% user, 20% idle, measured 2026-09-04 during b14's close-out); 12 fills the P cores and
# leaves the driver, the viewer and the OS a core or two. The desktop's number is `eval_shards`
# in `desktop/config/runtime.json`, 16 on 16 threads.
DEFAULT_SHARDS = 12
POLL_SECONDS = 20


def _log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


# ---------------------------------------------------------------- the specs

def load_specs(paths):
    """The `train` specs under `paths` (files or directories), in id order.

    Anything that is not a `train` spec -- an eval, a stage B someone copied along -- is skipped with
    a line saying so; the driver synthesises stage B itself.
    """
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, '*.json'))))
        else:
            files.append(path)
    specs = []
    for path in files:
        with open(path) as handle:
            spec = json.load(handle)
        if spec.get('type', 'train') != 'train':
            _log('skipping {0}: type {1!r} is not a training'.format(path, spec.get('type')))
            continue
        for key in ('id', 'policy', 'max_steps'):
            if key not in spec:
                raise ValueError('{0}: spec has no {1!r}'.format(path, key))
        spec.setdefault('env', {})
        specs.append(spec)
    specs.sort(key=lambda spec: spec['id'])
    if len({spec['policy'] for spec in specs}) != len(specs):
        raise ValueError('two specs name the same policy')
    return specs


def batch_id(specs):
    """`b13` from `b13aa-mb32-seed1`: the batch part of the daemon's `b<n><letters>-...` ids."""
    prefixes = {re.match(r'^([a-z]+\d+)', spec['id']).group(1)
                for spec in specs if re.match(r'^([a-z]+\d+)', spec['id'])}
    if len(prefixes) == 1:
        return prefixes.pop()
    return 'batch'


def waves(specs, size):
    return [specs[i:i + size] for i in range(0, len(specs), size)]


def stage_b_label(batch, number):
    """`b13-stageb`, then `b13-stageb-w2`, ... -- the daemon's names for its auto-queued waves."""
    return '{0}-stageb'.format(batch) if number == 1 else '{0}-stageb-w{1}'.format(batch, number)


# ---------------------------------------------------------------- one arm

def finished(spec, runs_dir=None):
    path = os.path.join(runs_dir or constants.RUNS_DIR, '{0}_evals.json'.format(spec['policy']))
    try:
        with open(path) as handle:
            return json.load(handle)['summary']['step'] >= int(spec['max_steps'])
    except (OSError, KeyError, TypeError, ValueError):
        return False


def live_pid(spec, runs_dir=None):
    """The pid of a trainer already running this arm on this box, or None."""
    pid = live_runs.read(live_runs.path_for(spec['policy'], runs_dir))
    if pid is not None and live_runs.alive(pid) and not live_runs.zombie(pid):
        return pid
    return None


def training_env(spec, base=None):
    """What `desktop/runner/launch.py` gives a trainer: the spec's env over ours, plus the cap."""
    env = dict(os.environ if base is None else base)
    env.update({key: str(value) for key, value in spec['env'].items()})
    env['SNEK_MAX_STEPS'] = str(spec['max_steps'])
    env['PYTHONPATH'] = ROOT
    return env


def trainer_count(runs_dir=None):
    return len(live_runs.live(runs_dir))


# ---------------------------------------------------------------- the driver

class Driver(object):
    """Runs the waves. Every process call goes through an attribute so a test can stand in for it."""

    def __init__(self, specs, wave=DEFAULT_WAVE, shards=DEFAULT_SHARDS, stage_b=True,
                 max_trainers=DEFAULT_MAX_TRAINERS, python=sys.executable, runs_dir=None,
                 logs_dir=None, popen=subprocess.Popen, call=subprocess.call, sleep=time.sleep):
        if wave > max_trainers:
            raise ValueError('a wave of {0} exceeds the {1}-trainer cap'.format(wave, max_trainers))
        self.specs, self.wave, self.shards, self.stage_b = specs, int(wave), int(shards), stage_b
        self.max_trainers, self.python, self.runs_dir = int(max_trainers), python, runs_dir
        self.logs_dir = logs_dir or os.path.join(ROOT, 'logs')
        self.popen, self.call, self.sleep = popen, call, sleep
        self.batch = batch_id(specs)

    def _wait_for_slot(self):
        waited = False
        while trainer_count(self.runs_dir) >= self.max_trainers:
            if not waited:
                _log('{0} trainers on the box; waiting for one to finish'.format(self.max_trainers))
                waited = True
            self.sleep(POLL_SECONDS)

    def _launch(self, spec):
        self._wait_for_slot()
        log_path = os.path.join(self.logs_dir, '{0}.log'.format(spec['policy']))
        os.makedirs(self.logs_dir, exist_ok=True)
        out = open(log_path, 'a')
        process = self.popen([self.python, '-u', 'train.py', spec['policy']], cwd=ROOT,
                             env=training_env(spec), stdout=out, stderr=subprocess.STDOUT)
        _log('launched {0} (pid {1})'.format(spec['policy'], process.pid))
        return process

    def _wait_pid(self, pid):
        while live_runs.alive(pid) and not live_runs.zombie(pid):
            self.sleep(POLL_SECONDS)

    def run_wave(self, number, arms):
        started, adopted = [], []
        for spec in arms:
            if finished(spec, self.runs_dir):
                _log('{0} already at max_steps; skipping'.format(spec['policy']))
                continue
            pid = live_pid(spec, self.runs_dir)
            if pid is not None:
                _log('{0} already training here (pid {1}); waiting for it'.format(spec['policy'], pid))
                adopted.append((spec, pid))
                continue
            started.append((spec, self._launch(spec)))
        for spec, process in started:
            code = process.wait()
            _log('{0} exited {1}'.format(spec['policy'], code))
        for spec, pid in adopted:
            self._wait_pid(pid)
            _log('{0} (pid {1}) finished'.format(spec['policy'], pid))
        if not self.stage_b:
            return 0
        label = stage_b_label(self.batch, number)
        _log('wave {0}: stage B as {1}'.format(number, label))
        os.makedirs(self.logs_dir, exist_ok=True)
        with open(os.path.join(self.logs_dir, '{0}.log'.format(label)), 'a') as out:
            code = self.call([self.python, '-u', '-m', 'tools.closeout',
                              *[spec['policy'] for spec in arms], '--shards', str(self.shards)],
                             cwd=ROOT, env={**os.environ, 'PYTHONPATH': ROOT},
                             stdout=out, stderr=subprocess.STDOUT)
        _log('wave {0}: stage B exited {1}'.format(number, code))
        return code

    def wait_for(self, pid):
        if pid is None:
            return
        if live_runs.alive(pid) and not live_runs.zombie(pid):
            _log('waiting for pid {0} to exit before starting'.format(pid))
        self._wait_pid(pid)

    def run(self, after=None):
        self.wait_for(after)
        plan = waves(self.specs, self.wave)
        _log('{0}: {1} arms in {2} wave(s) of {3}, stage B {4}'.format(
            self.batch, len(self.specs), len(plan), self.wave, 'on' if self.stage_b else 'off'))
        worst = 0
        for number, arms in enumerate(plan, start=1):
            _log('wave {0} of {1}: {2}'.format(number, len(plan), [spec['policy'] for spec in arms]))
            worst = max(worst, self.run_wave(number, arms) or 0)
        _log('{0} done'.format(self.batch))
        return worst


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('specs', nargs='+', help='desktop spec files, or directories of them')
    parser.add_argument('--wave', type=int, default=DEFAULT_WAVE, help='arms per wave')
    parser.add_argument('--shards', type=int, default=DEFAULT_SHARDS, help='stage-B shards')
    parser.add_argument('--max-trainers', type=int, default=DEFAULT_MAX_TRAINERS,
                        help='never more trainers than this on the box, counting others')
    parser.add_argument('--no-stage-b', action='store_true', help='train only')
    parser.add_argument('--after', type=int, default=None, metavar='PID',
                        help='start only once this process (another driver, a closeout) has exited')
    args = parser.parse_args(argv)
    specs = load_specs(args.specs)
    if not specs:
        parser.error('no training specs found')
    return Driver(specs, wave=args.wave, shards=args.shards, stage_b=not args.no_stage_b,
                  max_trainers=args.max_trainers).run(after=args.after)


if __name__ == '__main__':
    sys.exit(main())
