"""Run a batch of desktop specs on the laptop the way the desktop daemon would -- or a queue of them.

    PYTHONPATH=. nohup /opt/miniconda3/envs/snek3/bin/python -u -m tools.laptop_batch \\
        logs/b13specs/ > logs/b13-batch.log 2>&1 &                       # one batch
    PYTHONPATH=. nohup /opt/miniconda3/envs/snek3/bin/python -u -m tools.laptop_batch \\
        --queue logs/laptop-queue/ > logs/laptop-queue.log 2>&1 &         # every batch dropped in there

**`--queue <dir>` is the laptop's queue, and it is not a daemon.** Each subdirectory of the queue
directory is one batch -- its desktop specs, `git show`n in from `ops` -- and the driver runs the
batches in name order, one at a time, each with its waves and its three passes. **Between batches it
rescans the directory**, so a batch dropped in while another runs is picked up next, and when nothing
in the directory has unfinished work it exits. Nothing polls while there is no work; queueing a batch
is making a directory. Rerunning the same command is still the whole recovery.

**A pass whose file every arm already has is skipped, not rerun.** `runs/<arm>_checkpoint_evals
[_<label>].json` is the merged result of a pass, and a shard resumes only from its own shard files,
which the merge deletes -- so before 2026-09-04 rerunning the driver on a finished wave would have
re-measured its stage B from scratch. Now it skips to whatever pass is missing, which is also how a
driver started before the hof passes existed fills them in: rerun it, the arms and stage B are
skipped, hof5000 and hof30k run.

**The same spec files, the same waves, the same stage B.** The arguments are the daemon's own
`queue/pending/*.json` specs (files, or directories of them; `git show origin/ops:...` them onto the
laptop), read in id order and run in waves of `--wave` arms. Each arm is `train.py <policy>` with the
spec's `env` and `max_steps`, exactly as `desktop/runner/launch.py` would set them; when every arm of
a wave has exited, the wave's stage B runs as one `tools.closeout` over those arms, named
`<batch>-stageb`, `<batch>-stageb-w2`, ... as the daemon names its auto-queued waves, and then --
since 2026-09-04, as the daemon also does -- the wave's `hof5000` and `hof30k` passes over the same
arms (`<batch>-hof5000`, `<batch>-hof30k`, `-w2`, ...), each only if the pass before it exited 0,
because each selects from the file the one before it wrote. Then the next wave. So a batch dequeued
from the desktop -- b13 on 2026-09-03, to shorten the box's queue -- runs here with nothing
rewritten and lands in `runs/` under the names every later tool expects, hof passes included.

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

**What it is doing is published to the `laptop-status` branch** (`tools/laptop_status.py`): the daemon's
own `at_a_glance` shape -- one `running` line per batch with its percent, one `queued` line per
batch-phase -- rebuilt on every launch, exit, pass start and end, every ten minutes while waiting, and
once more, empty, when the driver exits. The desktop daemon folds it into `ops-status` as
`laptop_running` / `laptop_queued` / `laptop_iso`, so one `status.json` shows both boxes. `--no-status`
turns it off, for smokes.

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
from tools import closeout
from tools import laptop_status
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


def pass_label(batch, pass_name, number):
    """`b13-stageb`, then `b13-stageb-w2`, ... -- the daemon's names for its auto-queued passes.

    The same shape for every pass: `b13-hof5000`, `b13-hof30k-w2`. `desktop/runner/runner.py`'s
    `mint_pass_id` is the desktop's spelling of this; the daemon cannot import from `tools/` and this
    module keeps the dependency one-way, so the two are pinned equal by a test rather than shared.
    """
    base = '{0}-{1}'.format(batch, pass_name)
    return base if number == 1 else '{0}-w{1}'.format(base, number)


def stage_b_label(batch, number):
    return pass_label(batch, 'stageb', number)


# ---------------------------------------------------------------- one arm

def pass_file(policy, pass_name, runs_dir=None):
    """The merged file a pass writes for an arm: `runs/<arm>_checkpoint_evals[_<label>].json`.

    `tools.results.stage_b_path` spells the same path but is pinned to `constants.RUNS_DIR`, and the
    driver takes a `runs_dir` so a test can run a batch in a temporary directory.
    """
    label = closeout.PASSES[pass_name]['label']
    name = '{0}_checkpoint_evals{1}.json'.format(policy, '_' + label if label else '')
    return os.path.join(runs_dir or constants.RUNS_DIR, name)


def pass_done(arms, pass_name, runs_dir=None):
    """Whether every arm of a wave already has the pass's merged file. An empty file counts: an arm
    with no candidates gets one, and the next pass reads it and selects nothing."""
    return all(os.path.exists(pass_file(spec['policy'], pass_name, runs_dir)) for spec in arms)


def current_step(spec, runs_dir=None):
    """The arm's last evaluated step from its `_evals.json`, or None before its first eval."""
    path = os.path.join(runs_dir or constants.RUNS_DIR, '{0}_evals.json'.format(spec['policy']))
    try:
        with open(path) as handle:
            return int(json.load(handle)['summary']['step'])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def finished(spec, runs_dir=None):
    step = current_step(spec, runs_dir)
    return step is not None and step >= int(spec['max_steps'])


def arm_job(spec, runs_dir=None):
    """A spec as the job dict the daemon's `build_at_a_glance` reads for a trainer."""
    return {'id': spec['id'], 'type': 'train', 'policy': spec['policy'], 'policies': [spec['policy']],
            'label': spec.get('label', ''), 'max_steps': int(spec['max_steps']),
            'step': current_step(spec, runs_dir)}


def pass_job(batch, pass_name, number, arms):
    """A pass over a wave as the daemon's job dict: its minted id, type eval, the wave's arms."""
    return {'id': pass_label(batch, pass_name, number), 'type': 'eval',
            'policy': arms[0]['policy'] if arms else None,
            'policies': [spec['policy'] for spec in arms]}


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
                 logs_dir=None, popen=subprocess.Popen, call=subprocess.call, sleep=time.sleep,
                 passes=closeout.CHAIN, reporter=None, clock=time.time):
        if wave > max_trainers:
            raise ValueError('a wave of {0} exceeds the {1}-trainer cap'.format(wave, max_trainers))
        self.specs, self.wave, self.shards, self.stage_b = specs, int(wave), int(shards), stage_b
        # The passes each wave gets after its arms finish, in order; `stage_b=False` turns them all
        # off. The default is the whole chain, the same one the desktop daemon runs.
        self.passes = tuple(passes)
        self.max_trainers, self.python, self.runs_dir = int(max_trainers), python, runs_dir
        self.logs_dir = logs_dir or os.path.join(ROOT, 'logs')
        self.popen, self.call, self.sleep = popen, call, sleep
        self.batch = batch_id(specs)
        # What the box is doing, for `laptop-status`: the arms live under this driver (spec, pid) and
        # the pass in flight (pass_name, number, arms). `reporter` is a `Reporter`, or None to publish
        # nothing; `clock` is injectable so a test can drive the ten-minute republish.
        self.reporter, self.clock = reporter, clock
        self.live, self.active_pass = [], None
        self._last_report = 0.0

    # ---- what is running and what is owed, in the daemon's job-dict shape

    def jobs(self):
        """`(running, queued)` for this batch: live arms and the pass in flight; then every arm short
        of its cap that is not live, and every pass not yet filed, wave by wave. The same job dicts
        the daemon's `build_at_a_glance` reads, so the two boxes' lines are built by one function."""
        running = [dict(arm_job(spec, self.runs_dir), pid=pid) for spec, pid in self.live]
        if self.active_pass is not None:
            running.append(pass_job(self.batch, *self.active_pass))
        live_policies = {spec['policy'] for spec, _ in self.live}
        queued = []
        for number, arms in enumerate(waves(self.specs, self.wave), start=1):
            for spec in arms:
                if spec['policy'] not in live_policies and not finished(spec, self.runs_dir):
                    queued.append(arm_job(spec, self.runs_dir))
            if not self.stage_b:
                continue
            for pass_name in self.passes:
                if pass_done(arms, pass_name, self.runs_dir):
                    continue
                if self.active_pass is not None and self.active_pass[:2] == (pass_name, number):
                    continue
                queued.append(pass_job(self.batch, pass_name, number, arms))
        return running, queued

    def _report(self):
        if self.reporter is not None:
            self.reporter.publish(self)
            self._last_report = self.clock()

    def _tick(self):
        """Called once per wait poll: republishes every `REPUBLISH_SECONDS` so percentages move."""
        if self.reporter is not None and self.clock() - self._last_report >= laptop_status.REPUBLISH_SECONDS:
            self._report()

    def _wait_process(self, process):
        while process.poll() is None:
            self.sleep(POLL_SECONDS)
            self._tick()
        return process.returncode

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
            self._tick()

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
        self.live = [(spec, process.pid) for spec, process in started] + list(adopted)
        self._report()
        for spec, process in started:
            code = self._wait_process(process)
            _log('{0} exited {1}'.format(spec['policy'], code))
            self.live = [item for item in self.live if item[0]['policy'] != spec['policy']]
            self._report()
        for spec, pid in adopted:
            self._wait_pid(pid)
            _log('{0} (pid {1}) finished'.format(spec['policy'], pid))
            self.live = [item for item in self.live if item[0]['policy'] != spec['policy']]
            self._report()
        if not self.stage_b:
            return 0
        code = 0
        for pass_name in self.passes:
            if pass_done(arms, pass_name, self.runs_dir):
                _log('wave {0}: {1} already has every arm\'s file; skipping'.format(number, pass_name))
                continue
            code = self.run_pass(pass_name, number, arms)
            if code:
                # The next pass selects from the file this one wrote; with the pass failed, running
                # it would fail too and bury the real failure. The daemon stops its chain the same
                # way (`next_pass` needs `ok`).
                _log('wave {0}: {1} failed, so the passes behind it are not run'.format(
                    number, pass_name))
                break
        return code

    def run_pass(self, pass_name, number, arms):
        """One pass of the chain over a wave's arms, as one `tools.closeout` -- the daemon's command."""
        label = pass_label(self.batch, pass_name, number)
        _log('wave {0}: {1} as {2}'.format(number, pass_name, label))
        os.makedirs(self.logs_dir, exist_ok=True)
        argv = [self.python, '-u', '-m', 'tools.closeout', *[spec['policy'] for spec in arms]]
        if pass_name != 'stageb':
            argv += ['--pass', pass_name]
        argv += ['--shards', str(self.shards)]
        self.active_pass = (pass_name, number, list(arms))
        self._report()
        with open(os.path.join(self.logs_dir, '{0}.log'.format(label)), 'a') as out:
            code = self.call(argv, cwd=ROOT, env={**os.environ, 'PYTHONPATH': ROOT},
                             stdout=out, stderr=subprocess.STDOUT)
        self.active_pass = None
        _log('wave {0}: {1} exited {2}'.format(number, pass_name, code))
        self._report()
        return code

    def pending(self):
        """Whether anything in this batch is still to do: an arm short of its cap, or a wave short of
        a pass's file for any arm. What the queue asks before picking a batch, so a finished batch
        left in the queue directory costs nothing."""
        if not self.stage_b:
            return any(not finished(spec, self.runs_dir) for spec in self.specs)
        for arms in waves(self.specs, self.wave):
            if any(not finished(spec, self.runs_dir) for spec in arms):
                return True
            if any(not pass_done(arms, pass_name, self.runs_dir) for pass_name in self.passes):
                return True
        return False

    def wait_for(self, pid):
        if pid is None:
            return
        if live_runs.alive(pid) and not live_runs.zombie(pid):
            _log('waiting for pid {0} to exit before starting'.format(pid))
        self._wait_pid(pid)

    def run(self, after=None):
        self.wait_for(after)
        plan = waves(self.specs, self.wave)
        _log('{0}: {1} arms in {2} wave(s) of {3}, passes after each: {4}'.format(
            self.batch, len(self.specs), len(plan), self.wave,
            ', '.join(self.passes) if self.stage_b else 'none'))
        worst = 0
        for number, arms in enumerate(plan, start=1):
            _log('wave {0} of {1}: {2}'.format(number, len(plan), [spec['policy'] for spec in arms]))
            worst = max(worst, self.run_wave(number, arms) or 0)
        _log('{0} done'.format(self.batch))
        return worst


# ---------------------------------------------------------------- the status

class Reporter(object):
    """Publishes the box's state -- this driver's jobs plus every other batch waiting in the queue
    directory -- through a `laptop_status.Publisher`. `publish(None)` is the empty status a driver
    leaves when it exits with nothing to do."""

    def __init__(self, publisher, queue_dir=None, make_driver=None):
        self.publisher, self.queue_dir, self.make_driver = publisher, queue_dir, make_driver

    def jobs(self, driver):
        running, queued = ([], []) if driver is None else driver.jobs()
        if self.queue_dir and self.make_driver:
            current = None if driver is None else driver.batch
            for name, specs in queue_batches(self.queue_dir):
                other = self.make_driver(specs)
                if other.batch == current:
                    continue
                _, owed = other.jobs()
                queued.extend(owed)
        return running, queued

    def publish(self, driver):
        running, queued = self.jobs(driver)
        return self.publisher.publish(laptop_status.build(running, queued))


# ---------------------------------------------------------------- the queue

def queue_batches(queue_dir):
    """The batch directories under `queue_dir`, in name order. A file at the top level is not a batch;
    a subdirectory with no training specs is skipped with a line saying so."""
    batches = []
    for name in sorted(os.listdir(queue_dir)):
        path = os.path.join(queue_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            specs = load_specs([path])
        except ValueError as error:
            _log('skipping {0}: {1}'.format(path, error))
            continue
        if not specs:
            _log('skipping {0}: no training specs'.format(path))
            continue
        batches.append((name, specs))
    return batches


def run_queue(queue_dir, make_driver, after=None, reporter=None):
    """Runs every batch under `queue_dir` that has work left, rescanning between batches.

    `make_driver(specs)` builds the `Driver` for one batch, so the queue carries no launch settings of
    its own. Returns the worst exit status of the batches it ran. Exits -- returns -- when a scan
    finds nothing pending, which is what makes this a queue and not a daemon: it lives exactly as
    long as there is work.

    `reporter` publishes the queue's state to `laptop-status`; its last publish, as the queue exits,
    is the empty status that says the laptop is idle.
    """
    worst = 0
    ran = set()

    def driver_for(specs):
        made = make_driver(specs)
        made.reporter = reporter
        return made

    def exit_with(code):
        if reporter is not None:
            reporter.publish(None)
        return code

    if after is not None:
        make_driver([]).wait_for(after)
    while True:
        pending = [(name, specs) for name, specs in queue_batches(queue_dir)
                   if make_driver(specs).pending()]
        if not pending:
            _log('queue {0}: nothing pending; exiting'.format(queue_dir))
            return exit_with(worst)
        name, specs = pending[0]
        if name in ran:
            # Ran it and it still reports work: a pass that failed, an arm that will not reach its
            # cap. Looping on it would spin; leave it for a human and move on to what is behind it.
            rest = [item for item in pending if item[0] not in ran]
            if not rest:
                _log('queue {0}: every pending batch has already run once and still has work left '
                     '({1}); exiting'.format(queue_dir, ', '.join(sorted(ran))))
                return exit_with(worst or 1)
            name, specs = rest[0]
        _log('queue {0}: {1} of {2} pending, starting {3}'.format(
            queue_dir, len(pending), len(queue_batches(queue_dir)), name))
        ran.add(name)
        worst = max(worst, driver_for(specs).run() or 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('specs', nargs='*', help='desktop spec files, or directories of them')
    parser.add_argument('--queue', metavar='DIR', default=None,
                        help='run every batch directory under DIR in name order, rescanning between '
                             'batches; exits when none has work left')
    parser.add_argument('--wave', type=int, default=DEFAULT_WAVE, help='arms per wave')
    parser.add_argument('--shards', type=int, default=DEFAULT_SHARDS, help='stage-B shards')
    parser.add_argument('--max-trainers', type=int, default=DEFAULT_MAX_TRAINERS,
                        help='never more trainers than this on the box, counting others')
    parser.add_argument('--no-stage-b', action='store_true', help='train only: no passes at all')
    parser.add_argument('--no-hof', action='store_true',
                        help='stage B only after each wave; no hof5000 or hof30k')
    parser.add_argument('--after', type=int, default=None, metavar='PID',
                        help='start only once this process (another driver, a closeout) has exited')
    parser.add_argument('--no-status', action='store_true',
                        help='do not publish what is running to the laptop-status branch')
    args = parser.parse_args(argv)
    passes = closeout.CHAIN[:1] if args.no_hof else closeout.CHAIN

    def make_driver(specs):
        return Driver(specs, wave=args.wave, shards=args.shards, stage_b=not args.no_stage_b,
                      max_trainers=args.max_trainers, passes=passes)

    reporter = None
    if not args.no_status:
        reporter = Reporter(laptop_status.Publisher(log=_log), queue_dir=args.queue,
                            make_driver=make_driver)
    if args.queue:
        if args.specs:
            parser.error('--queue takes the queue directory only; put the batches under it')
        return run_queue(args.queue, make_driver, after=args.after, reporter=reporter)
    specs = load_specs(args.specs)
    if not specs:
        parser.error('no training specs found')
    single = make_driver(specs)
    single.reporter = reporter
    code = single.run(after=args.after)
    if reporter is not None:
        reporter.publish(None)
    return code


if __name__ == '__main__':
    sys.exit(main())
