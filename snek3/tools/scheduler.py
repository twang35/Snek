"""The scheduler: runs batches of desktop specs on this box in waves, each followed by its three passes,
and owns the box's chart window and its shared eval workers. One implementation for both boxes.

    PYTHONPATH=. nohup /opt/miniconda3/envs/snek3/bin/python -u -m tools.scheduler \\
        --queue logs/laptop-queue/ > logs/laptop-queue.log 2>&1 &         # every batch dropped in there
    PYTHONPATH=. python -m tools.scheduler --reopen-window                 # a fresh chart window, now

**`--queue <dir>` is the box's queue, and it is not a daemon.** Each subdirectory of the queue
directory is one batch -- desktop-format specs, `git show`n in from `ops` on the laptop, materialised
from `ops` by the daemon on the desktop -- and the scheduler runs the batches in name order, one at a
time, each with its waves and its three passes. **Between batches it rescans the directory**, so a
batch dropped in while another runs is picked up next -- and so is work dropped into a batch that has
already run (a hand eval spec, an added arm): a batch runs again when it owes an id it did not owe
before, never for the same work left over. When nothing in the directory has unfinished work it exits. Nothing polls while there is no work; queueing a batch is making a
directory. Rerunning the same command is still the whole recovery.

**State is the filesystem.** An arm is finished when its `runs/<arm>_evals.json` reports `max_steps`;
a pass is done when its merged `runs/<arm>_checkpoint_evals[_<label>].json` exists for every arm of
the wave; an arm already live on this box (`runs/.live/`, written by the trainer) is waited for rather
than relaunched, and so is a pass a previous scheduler left running (`runs/.live/.pass-<label>`, written
by that scheduler); a pass that failed is marked `.failed-<label>` in the batch directory so the queue
does not loop on it (delete the marker to retry). Nothing else is remembered, which is why a kill, a
reboot or a deploy costs nothing but a rerun.

**The scheduler owns the box's one chart window** (2026-09-05, `tools/window.py`). It opens the
viewer when it launches a wave or a pass, writes the PNGs to show into its own status file
(`runs/.live/.status.json` -- every arm of the current wave while training, every arm's stage-B chart
while a pass runs), and closes the viewer when it exits. No arm and no close-out opens a window any
more; `plans/scheduler.md` §0 lists what went wrong when they did.

**The scheduler starts the box's shared stage-A eval workers before each wave** (phase 3 of the same
plan). The trainers still call `ensure_workers` themselves -- a bare `train.py` gets its workers that
way -- but under the scheduler the slots are already held, so the eight-arms-racing-for-six-slots case
that once produced seven slot-0 workers has one contender.

**The same spec files, the same waves, the same chain.** The arguments are the daemon's own
`queue/pending/*.json` specs, read in id order and run in waves of `--wave` arms. Each `train` spec is
`train.py <policy>` with the spec's `env` and `max_steps`, exactly as `desktop/runner/launch.py` set
them; when every arm of a wave has exited, the wave's stage B runs as one `tools.closeout` over those
arms, named `<batch>-stageb`, `<batch>-stageb-w2`, ... as the daemon named its auto-queued waves, and
then the wave's `hof5000` and `hof30k` passes (`<batch>-hof5000`, `<batch>-hof30k`, `-w2`, ...), each
only if the pass before it exited 0, because each selects from the file the one before it wrote. An
`eval` spec in a batch directory -- a hand hof pass, a `one` re-measure -- runs after the batch's
waves as the `tools.closeout` command it spells, once, marked `.done-<id>` beside it.

**A `runs/.live/.paused` file pauses it**: what is running finishes, and the next wave, pass or eval
waits until the file is gone. The desktop daemon writes it for `runtime.json`'s `paused`/`drain`; on
the laptop, `touch` and `rm` it.

**Never more than `--max-trainers` (8) trainers on the box, counting anything else running here.**
Before each launch the scheduler waits until the box's live trainer count is below the cap.

**What it is doing is published** through `tools/laptop_status.py`: the local status file on every
event, and the `laptop-status` branch too unless `--no-status` -- the daemon's own `at_a_glance` shape,
rebuilt on every launch, exit, pass start and end, every ten minutes while waiting, and once more,
empty, when the scheduler exits.
"""

import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

from env import constants
from tools import closeout
from tools import eval_queue
from tools import laptop_status
from tools import live_runs
from tools import results
from tools import window as window_module

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WAVE = 8
DEFAULT_MAX_TRAINERS = 8
# A pass runs alone on this box -- the next wave waits for it -- so the shard count is sized to the
# laptop's 14 cores (10P + 4E), not to what is left beside eight trainers. 8 left 6 cores idle
# (66% user, 20% idle, measured 2026-09-04 during b14's close-out); 12 fills the P cores and
# leaves the scheduler, the viewer and the OS a core or two. The desktop's number is `eval_shards`
# in `desktop/config/runtime.json`, 16 on 16 threads.
DEFAULT_SHARDS = 12
POLL_SECONDS = 20


def _log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


# ---------------------------------------------------------------- the specs

def load_specs(paths):
    """Every `train` and `eval` spec under `paths` (files or directories), in id order.

    Each spec remembers the directory it came from (`_dir`), which is where its markers live. Anything
    that is neither -- a smoke, an action -- is skipped with a line saying so.
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
        kind = spec['type'] = spec.get('type', 'train')
        if kind not in ('train', 'eval'):
            _log('skipping {0}: type {1!r} is neither a training nor an eval'.format(path, kind))
            continue
        if kind == 'train':
            for key in ('id', 'policy', 'max_steps'):
                if key not in spec:
                    raise ValueError('{0}: spec has no {1!r}'.format(path, key))
        else:
            if 'id' not in spec or not (spec.get('policies') or spec.get('policy')):
                raise ValueError('{0}: an eval spec needs an id and policies'.format(path))
            spec['policies'] = [name for name in (spec.get('policies') or [spec.get('policy')]) if name]
        spec.setdefault('env', {})
        spec['_dir'] = os.path.dirname(os.path.abspath(path))
        specs.append(spec)
    specs.sort(key=lambda spec: spec['id'])
    trained = [spec['policy'] for spec in specs if spec['type'] == 'train'] if specs else []
    if len(set(trained)) != len(trained):
        raise ValueError('two specs name the same policy')
    return specs


def train_specs(specs):
    return [spec for spec in specs if spec.get('type', 'train') == 'train']


def eval_specs(specs):
    return [spec for spec in specs if spec.get('type') == 'eval']


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


# ---------------------------------------------------------------- one arm, one pass

def pass_file(policy, pass_name, runs_dir=None):
    """The merged file a pass writes for an arm: `runs/<arm>_checkpoint_evals[_<label>].json`."""
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


def eval_job(spec):
    return {'id': spec['id'], 'type': 'eval', 'policy': spec['policies'][0],
            'policies': list(spec['policies']), 'label': spec.get('label', '')}


def live_pid(spec, runs_dir=None):
    """The pid of a trainer already running this arm on this box, or None."""
    return _live_entry(spec['policy'], runs_dir)


def live_pass_pid(label, runs_dir=None):
    """The pid of a close-out already running this pass on this box (a predecessor scheduler's), or None."""
    return _live_entry(live_runs.pass_entry(label), runs_dir)


def _live_entry(name, runs_dir=None):
    pid = live_runs.read(live_runs.path_for(name, runs_dir))
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


def wave_workers(arms):
    """How many shared eval workers a wave wants: the count its specs agree on, else the default.

    `SNEK_EVAL_WORKERS` is what the trainers would ask `ensure_workers` for, so the scheduler asks
    for the same number first. A wave whose every arm turns the queue off wants none.
    """
    if arms and all(str(spec['env'].get('SNEK_EVAL_QUEUE', '1')) == '0' for spec in arms):
        return 0
    counts = {str(spec['env'].get('SNEK_EVAL_WORKERS', '')) for spec in arms}
    counts.discard('')
    if len(counts) == 1:
        try:
            return max(0, int(counts.pop()))
        except ValueError:
            pass
    return eval_queue.DEFAULT_WORKERS


def eval_argv(spec, python, shards):
    """The `tools.closeout` command an eval spec spells -- `desktop/runner/launch.py`'s `eval_command`.

    Selector and episode count are passed only if the spec named them, so the close-out's own defaults
    stay the protocol; `--pass <name>` and anything else travel in `eval_args`.
    """
    argv = [python, '-u', '-m', 'tools.closeout'] + list(spec['policies'])
    if spec.get('selector'):
        argv += ['--selector', str(spec['selector'])]
    argv += [str(arg) for arg in spec.get('eval_args') or []]
    if spec.get('episodes'):
        argv += ['--episodes', str(spec['episodes'])]
    argv += ['--shards', str(spec.get('eval_shards') or shards)]
    return argv


def eval_label(spec):
    """The label the eval spec's pass writes under, for its chart paths. None for an unlabelled pass."""
    args = [str(arg) for arg in spec.get('eval_args') or []]
    if '--label' in args and args.index('--label') + 1 < len(args):
        return args[args.index('--label') + 1]
    if '--pass' in args and args.index('--pass') + 1 < len(args):
        return closeout.PASSES.get(args[args.index('--pass') + 1], {}).get('label')
    return None


def marker(spec_dir, name):
    return os.path.join(spec_dir, '.' + name)


def mark(spec_dir, name, note=''):
    try:
        with open(marker(spec_dir, name), 'w') as handle:
            handle.write('{0} {1}\n'.format(time.strftime('%Y-%m-%dT%H:%M:%S'), note).rstrip() + '\n')
    except OSError as error:
        _log('could not write marker {0}: {1}'.format(marker(spec_dir, name), error))


def marked(spec_dir, name):
    return os.path.exists(marker(spec_dir, name))


def training_panels(arms, runs_dir=None):
    """`runs/<arm>.png` for every arm of the wave: the window shows the whole wave, finished arms too."""
    return [os.path.join(runs_dir or constants.RUNS_DIR, results.run_name(spec['policy']) + '.png')
            for spec in arms]


def pass_panels(policies, label, runs_dir=None):
    """The stage-B PNG each arm's pass is drawn into, whether or not it exists yet."""
    out = []
    for policy in policies:
        stem = os.path.basename(results.stage_b_path(policy, label))[:-len('.json')]
        out.append(os.path.join(runs_dir or constants.RUNS_DIR, stem + '.png'))
    return out


# ---------------------------------------------------------------- the driver

class Driver(object):
    """Runs one batch's waves. Every process call goes through an attribute so a test can stand in for it."""

    def __init__(self, specs, wave=DEFAULT_WAVE, shards=DEFAULT_SHARDS, stage_b=True,
                 max_trainers=DEFAULT_MAX_TRAINERS, python=sys.executable, runs_dir=None,
                 logs_dir=None, popen=subprocess.Popen, call=subprocess.call, sleep=time.sleep,
                 passes=closeout.CHAIN, reporter=None, clock=time.time, window=None,
                 ensure_workers=eval_queue.ensure_workers):
        if wave > max_trainers:
            raise ValueError('a wave of {0} exceeds the {1}-trainer cap'.format(wave, max_trainers))
        self.specs, self.evals = train_specs(specs), eval_specs(specs)
        self.wave, self.shards, self.stage_b = int(wave), int(shards), stage_b
        # The passes each wave gets after its arms finish, in order; `stage_b=False` turns them all
        # off. The default is the whole chain, the same one the desktop daemon ran.
        self.passes = tuple(passes)
        self.max_trainers, self.python, self.runs_dir = int(max_trainers), python, runs_dir
        self.logs_dir = logs_dir or os.path.join(ROOT, 'logs')
        self.popen, self.call, self.sleep = popen, call, sleep
        self.batch = batch_id(specs)
        # What the box is doing, for the status file and the window: the arms live under this driver
        # (spec, pid), the pass in flight (pass_name, number, arms), the eval spec in flight, and the
        # PNGs the window should show. `reporter` is a `Reporter`, or None to write nothing; `clock`
        # is injectable so a test can drive the ten-minute republish.
        self.reporter, self.clock = reporter, clock
        self.live, self.active_pass, self.active_eval = [], None, None
        self.panels = []
        self._last_report = 0.0
        # The box's window and its shared eval workers, both owned here. `window` is a
        # `window_module.Window` or None (tests, `SNEK_CHART_WINDOW=0` is handled inside it).
        self.window = window
        self.ensure_workers = ensure_workers
        self.workers = []

    # ---- what is running and what is owed, in the daemon's job-dict shape

    def jobs(self):
        """`(running, queued)` for this batch: live arms and the pass in flight; then every arm short
        of its cap that is not live, every pass not yet filed, wave by wave, and every eval spec not
        yet run. The same job dicts the daemon's `build_at_a_glance` reads."""
        running = [dict(arm_job(spec, self.runs_dir), pid=pid) for spec, pid in self.live]
        if self.active_pass is not None:
            running.append(pass_job(self.batch, *self.active_pass))
        if self.active_eval is not None:
            running.append(eval_job(self.active_eval))
        live_policies = {spec['policy'] for spec, _ in self.live}
        queued = []
        for number, arms in enumerate(waves(self.specs, self.wave), start=1):
            for spec in arms:
                if spec['policy'] not in live_policies and not finished(spec, self.runs_dir):
                    queued.append(arm_job(spec, self.runs_dir))
            if not self.stage_b:
                continue
            for pass_name in self.passes:
                if pass_done(arms, pass_name, self.runs_dir) or self._pass_failed(arms, pass_name, number):
                    continue
                if self.active_pass is not None and self.active_pass[:2] == (pass_name, number):
                    continue
                queued.append(pass_job(self.batch, pass_name, number, arms))
        for spec in self.evals:
            if not self._eval_done(spec) and spec is not self.active_eval:
                queued.append(eval_job(spec))
        return running, queued

    def attention(self):
        """One line per failed pass, so a marker is never silent: the queue skips it, this names it."""
        lines = []
        if live_runs.held(self.runs_dir):
            lines.append('** paused: {0} exists; nothing new starts until it is removed'.format(
                live_runs.hold_path(self.runs_dir)))
        for number, arms in enumerate(waves(self.specs, self.wave), start=1):
            for pass_name in self.passes if self.stage_b else ():
                if self._pass_failed(arms, pass_name, number):
                    lines.append('** {0} failed and is not retried; delete {1} to retry'.format(
                        pass_label(self.batch, pass_name, number),
                        marker(arms[0]['_dir'], 'failed-' + pass_label(self.batch, pass_name, number))
                        if arms[0].get('_dir') else 'its .failed marker'))
        for spec in self.evals:
            if spec.get('_dir') and marked(spec['_dir'], 'failed-' + spec['id']):
                lines.append('** {0} failed and is not retried; delete {1} to retry'.format(
                    spec['id'], marker(spec['_dir'], 'failed-' + spec['id'])))
        return lines

    def _pass_failed(self, arms, pass_name, number):
        directory = arms[0].get('_dir') if arms else None
        return bool(directory) and marked(directory, 'failed-' + pass_label(self.batch, pass_name, number))

    def _eval_done(self, spec):
        directory = spec.get('_dir')
        return bool(directory) and (marked(directory, 'done-' + spec['id'])
                                    or marked(directory, 'failed-' + spec['id']))

    def _report(self):
        if self.reporter is not None:
            self.reporter.publish(self)
            self._last_report = self.clock()

    def _tick(self):
        """Called once per wait poll: republishes every `REPUBLISH_SECONDS` so percentages move, honours
        a window reopen request, and reaps exited workers so none stays a zombie."""
        if self.window is not None and self.window.poll():
            self._report()          # a fresh window reads its panels from the file; write it now
        self.workers = eval_queue.reap(self.workers)
        if self.reporter is not None and self.clock() - self._last_report >= laptop_status.REPUBLISH_SECONDS:
            self._report()

    def _show(self, panels):
        """Points the window at `panels` and opens it if it is not up. Called at every launch."""
        self.panels = list(panels)
        if self.window is not None:
            self.window.open()

    def _wait_process(self, process):
        while process.poll() is None:
            self.sleep(POLL_SECONDS)
            self._tick()
        return process.returncode

    def _wait_while_held(self, what):
        """Blocks while the box is paused (`live_runs.held`). Nothing running is touched; the next
        launch waits. The status keeps refreshing so the pause is visible from either box."""
        waited = False
        while live_runs.held(self.runs_dir):
            if not waited:
                _log('paused ({0} exists); {1} waits until it is removed'.format(
                    live_runs.hold_path(self.runs_dir), what))
                waited = True
            self.sleep(POLL_SECONDS)
            self._tick()
        if waited:
            _log('unpaused; {0} goes ahead'.format(what))

    def _wait_for_slot(self):
        waited = False
        while trainer_count(self.runs_dir) >= self.max_trainers:
            if not waited:
                _log('{0} trainers on the box; waiting for one to finish'.format(self.max_trainers))
                waited = True
            self.sleep(POLL_SECONDS)
            self._tick()

    def _launch(self, spec):
        self._wait_for_slot()
        log_path = os.path.join(self.logs_dir, '{0}.log'.format(spec['policy']))
        os.makedirs(self.logs_dir, exist_ok=True)
        out = open(log_path, 'a')
        # Its own session, so a signal to the scheduler -- a Ctrl-C, a deploy that restarts it -- never
        # reaches the arm, and the next scheduler adopts it through the registry instead.
        process = self.popen([self.python, '-u', 'train.py', spec['policy']], cwd=ROOT,
                             env=training_env(spec), stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True)
        _log('launched {0} (pid {1})'.format(spec['policy'], process.pid))
        return process

    def _wait_pid(self, pid):
        while live_runs.alive(pid) and not live_runs.zombie(pid):
            self.sleep(POLL_SECONDS)
            self._tick()

    def _start_workers(self, arms):
        """The wave's shared stage-A workers, started before its arms so no arm has to race for a slot."""
        wanted = wave_workers(arms)
        if wanted <= 0:
            return
        started = self.ensure_workers(wanted, self.runs_dir)
        self.workers.extend(started)
        _log('{0} eval worker(s) wanted for the wave, {1} started here'.format(wanted, len(started)))

    def run_wave(self, number, arms):
        started, adopted, to_launch = [], [], []
        # Each arm is classified once -- done, live here, or to launch -- before anything starts, so the
        # workers can be up before the first arm and the registry is asked once per arm.
        for spec in arms:
            if finished(spec, self.runs_dir):
                _log('{0} already at max_steps; skipping'.format(spec['policy']))
                continue
            pid = live_pid(spec, self.runs_dir)
            if pid is not None:
                _log('{0} already training here (pid {1}); waiting for it'.format(spec['policy'], pid))
                adopted.append((spec, pid))
                continue
            to_launch.append(spec)
        if to_launch:
            self._wait_while_held('wave {0}'.format(number))
            self._start_workers(to_launch)
        for spec in to_launch:
            started.append((spec, self._launch(spec)))
        self.live = [(spec, process.pid) for spec, process in started] + list(adopted)
        if self.live:
            self._show(training_panels(arms, self.runs_dir))
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
            if self._pass_failed(arms, pass_name, number):
                _log('wave {0}: {1} is marked failed; not retried'.format(number, pass_name))
                return 1
            self._wait_while_held('wave {0} {1}'.format(number, pass_name))
            code = self.run_pass(pass_name, number, arms)
            if code:
                # The next pass selects from the file this one wrote; with the pass failed, running
                # it would fail too and bury the real failure. Marked, so the queue does not loop on it.
                _log('wave {0}: {1} failed, so the passes behind it are not run'.format(
                    number, pass_name))
                if arms[0].get('_dir'):
                    mark(arms[0]['_dir'], 'failed-' + pass_label(self.batch, pass_name, number),
                         'exit {0}'.format(code))
                break
        return code

    def run_pass(self, pass_name, number, arms):
        """One pass of the chain over a wave's arms, as one `tools.closeout` -- the daemon's command."""
        label = pass_label(self.batch, pass_name, number)
        _log('wave {0}: {1} as {2}'.format(number, pass_name, label))
        argv = [self.python, '-u', '-m', 'tools.closeout', *[spec['policy'] for spec in arms]]
        if pass_name != 'stageb':
            argv += ['--pass', pass_name]
        argv += ['--shards', str(self.shards)]
        self.active_pass = (pass_name, number, list(arms))
        self._show(pass_panels([spec['policy'] for spec in arms], closeout.PASSES[pass_name]['label'],
                               self.runs_dir))
        code = self._run_closeout(argv, label, expected=[pass_file(spec['policy'], pass_name, self.runs_dir)
                                                          for spec in arms])
        self.active_pass = None
        _log('wave {0}: {1} exited {2}'.format(number, pass_name, code))
        self._report()
        return code

    def _run_closeout(self, argv, label, expected=None):
        """Runs one close-out as `label`, or waits for the one a predecessor scheduler left running.

        The pass is registered as `runs/.live/.pass-<label>` = pid while it runs, by this scheduler,
        which holds the Popen (the arms register themselves; a close-out does not know its pass id).
        A restarted scheduler finds the entry alive and waits, as it does for an arm -- before this
        (2026-09-05) it launched a second close-out over the same shard files. An adopted pass has no
        exit code to read, so its success is `expected`: every file it was to write exists.
        """
        os.makedirs(self.logs_dir, exist_ok=True)
        self._report()
        pid = live_pass_pid(label, self.runs_dir)
        if pid is not None:
            _log('{0} already running here (pid {1}); waiting for it'.format(label, pid))
            self._wait_pid(pid)
            live_runs.unregister(live_runs.pass_entry(label), self.runs_dir)
            return 0 if expected is None or all(os.path.exists(path) for path in expected) else 1
        with open(os.path.join(self.logs_dir, '{0}.log'.format(label)), 'a') as out:
            # `call` blocks, so the window's reopen request and the republish are checked from a
            # polling `popen` instead when the injected `call` is the real one.
            if self.call is subprocess.call:
                process = self.popen(argv, cwd=ROOT, env={**os.environ, 'PYTHONPATH': ROOT},
                                     stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                live_runs.register(live_runs.pass_entry(label), process.pid, self.runs_dir)
                try:
                    return self._wait_process(process)
                finally:
                    live_runs.unregister(live_runs.pass_entry(label), self.runs_dir)
            return self.call(argv, cwd=ROOT, env={**os.environ, 'PYTHONPATH': ROOT},
                             stdout=out, stderr=subprocess.STDOUT)

    def run_evals(self):
        """The batch's eval specs, each once, after its waves. Returns the worst exit status."""
        worst = 0
        for spec in self.evals:
            if self._eval_done(spec):
                continue
            self._wait_while_held('eval {0}'.format(spec['id']))
            _log('eval {0}: {1}'.format(spec['id'], ' '.join(spec['policies'])))
            self.active_eval = spec
            self._show(pass_panels(spec['policies'], eval_label(spec), self.runs_dir))
            code = self._run_closeout(eval_argv(spec, self.python, self.shards), spec['id'])
            self.active_eval = None
            _log('eval {0} exited {1}'.format(spec['id'], code))
            if spec.get('_dir'):
                mark(spec['_dir'], ('done-' if code == 0 else 'failed-') + spec['id'], 'exit {0}'.format(code))
            self._report()
            worst = max(worst, code or 0)
        return worst

    def pending(self):
        """Whether anything in this batch is still to do: an arm short of its cap, a wave short of a
        pass's file for any arm (unless that pass is marked failed), or an eval spec not yet run.
        What the queue asks before picking a batch, so a finished batch left in the queue costs nothing."""
        for number, arms in enumerate(waves(self.specs, self.wave), start=1):
            if any(not finished(spec, self.runs_dir) for spec in arms):
                return True
            if not self.stage_b:
                continue
            for pass_name in self.passes:
                if self._pass_failed(arms, pass_name, number):
                    break               # the chain stops here for this wave
                if not pass_done(arms, pass_name, self.runs_dir):
                    return True
        return any(not self._eval_done(spec) for spec in self.evals)

    def wait_for(self, pid):
        if pid is None:
            return
        if live_runs.alive(pid) and not live_runs.zombie(pid):
            _log('waiting for pid {0} to exit before starting'.format(pid))
        self._wait_pid(pid)

    def run(self, after=None):
        self.wait_for(after)
        plan = waves(self.specs, self.wave)
        _log('{0}: {1} arms in {2} wave(s) of {3}, passes after each: {4}; {5} eval spec(s)'.format(
            self.batch, len(self.specs), len(plan), self.wave,
            ', '.join(self.passes) if self.stage_b else 'none', len(self.evals)))
        worst = 0
        for number, arms in enumerate(plan, start=1):
            _log('wave {0} of {1}: {2}'.format(number, len(plan), [spec['policy'] for spec in arms]))
            worst = max(worst, self.run_wave(number, arms) or 0)
        worst = max(worst, self.run_evals())
        _log('{0} done'.format(self.batch))
        return worst


# ---------------------------------------------------------------- the status

class Reporter(object):
    """Builds the box's status -- this driver's jobs plus every other batch waiting in the queue
    directory -- writes it to the local status file the window follows, and publishes it through a
    `laptop_status.Publisher` if there is one. `publish(None)` is the empty status a scheduler leaves
    when it exits with nothing to do."""

    def __init__(self, publisher=None, queue_dir=None, make_driver=None, runs_dir=None, window=None):
        self.publisher, self.queue_dir, self.make_driver = publisher, queue_dir, make_driver
        self.runs_dir, self.window = runs_dir, window

    def jobs(self, driver):
        running, queued = ([], []) if driver is None else driver.jobs()
        attention = [] if driver is None else driver.attention()
        if self.queue_dir and self.make_driver:
            current = None if driver is None else driver.batch
            for name, specs in queue_batches(self.queue_dir):
                other = self.make_driver(specs)
                if other.batch == current:
                    continue
                _, owed = other.jobs()
                queued.extend(owed)
                attention.extend(other.attention())
        # Every driver names the box's hold; the box has one. Order kept, duplicates dropped.
        return running, queued, list(dict.fromkeys(attention))

    def status(self, driver):
        running, queued, attention = self.jobs(driver)
        panels = [] if driver is None else driver.panels
        window_pid = self.window.pid() if self.window is not None else None
        return laptop_status.build(running, queued, panels=panels, window_pid=window_pid,
                                   attention=attention)

    def publish(self, driver):
        status = self.status(driver)
        laptop_status.write_local(status, self.runs_dir or (driver.runs_dir if driver else None))
        if self.publisher is None:
            return True
        return self.publisher.publish(status)


# ---------------------------------------------------------------- the queue

def queue_batches(queue_dir):
    """The batch directories under `queue_dir`, in name order. A file at the top level is not a batch;
    a subdirectory with no specs is skipped with a line saying so."""
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
            _log('skipping {0}: no specs'.format(path))
            continue
        batches.append((name, specs))
    return batches


def run_queue(queue_dir, make_driver, after=None, reporter=None):
    """Runs every batch under `queue_dir` that has work left, rescanning between batches.

    `make_driver(specs)` builds the `Driver` for one batch, so the queue carries no launch settings of
    its own. Returns the worst exit status of the batches it ran. Exits -- returns -- when a scan
    finds nothing pending, which is what makes this a queue and not a daemon: it lives exactly as
    long as there is work.

    `reporter` publishes the queue's state; its last publish, as the queue exits, is the empty status
    that says the box is idle.
    """
    worst = 0
    # batch name -> the set of job ids it still owed when it last ran. A batch is run again only if
    # it now owes an id it did not owe then: work queued into it while it ran (a hand eval spec, an
    # added arm), picked up at this boundary like a new batch would be. The same set, or a smaller
    # one, is an arm that will not reach its cap, and looping on it would spin.
    ran = {}

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
        pending = []
        for name, specs in queue_batches(queue_dir):
            made = make_driver(specs)
            if made.pending():
                pending.append((name, specs, owed_ids(made)))
        if not pending:
            _log('queue {0}: nothing pending; exiting'.format(queue_dir))
            return exit_with(worst)
        fresh = [item for item in pending if item[2] - ran.get(item[0], frozenset())]
        if not fresh:
            _log('queue {0}: every pending batch has already run once with the same work left '
                 '({1}); exiting'.format(queue_dir, ', '.join(sorted(ran))))
            return exit_with(worst or 1)
        name, specs, owed = fresh[0]
        if name in ran:
            _log('queue {0}: {1} has new work since it ran ({2}); running it again'.format(
                queue_dir, name, ', '.join(sorted(owed - ran[name]))))
        _log('queue {0}: {1} of {2} pending, starting {3}'.format(
            queue_dir, len(pending), len(queue_batches(queue_dir)), name))
        ran[name] = owed
        worst = max(worst, driver_for(specs).run() or 0)


def owed_ids(driver):
    """The ids of everything a fresh driver still has to do: arms short of their cap, passes without
    every arm's file, eval specs without a marker. What `run_queue` compares between scans."""
    _, queued = driver.jobs()
    return frozenset(job['id'] for job in queued)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('specs', nargs='*', help='desktop spec files, or directories of them')
    parser.add_argument('--queue', metavar='DIR', default=None,
                        help='run every batch directory under DIR in name order, rescanning between '
                             'batches; exits when none has work left')
    parser.add_argument('--wave', type=int, default=DEFAULT_WAVE, help='arms per wave')
    parser.add_argument('--shards', type=int, default=DEFAULT_SHARDS, help='stage-B shard pool')
    parser.add_argument('--max-trainers', type=int, default=DEFAULT_MAX_TRAINERS,
                        help='never more trainers than this on the box, counting others')
    parser.add_argument('--no-stage-b', action='store_true', help='train only: no passes at all')
    parser.add_argument('--no-hof', action='store_true',
                        help='stage B only after each wave; no hof5000 or hof30k')
    parser.add_argument('--after', type=int, default=None, metavar='PID',
                        help='start only once this process (another scheduler, a closeout) has exited')
    parser.add_argument('--no-status', action='store_true',
                        help='do not publish what is running to the laptop-status branch')
    parser.add_argument('--reopen-window', action='store_true',
                        help='ask the running scheduler for a fresh chart window, then exit')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.reopen_window:
        path = window_module.request_reopen()
        print('reopen requested ({0}); the scheduler replaces its window at its next poll'.format(path))
        return 0
    # Everything the arguments can refuse is refused here, before a single side effect: nothing below
    # this line may run for an invocation that is going to exit on its arguments.
    if args.queue and args.specs:
        parser.error('--queue takes the queue directory only; put the batches under it')
    specs = None
    if not args.queue:
        specs = load_specs(args.specs)
        if not specs:
            parser.error('no specs found')
    passes = closeout.CHAIN[:1] if args.no_hof else closeout.CHAIN

    # One scheduler per box. The last status written here names the scheduler that wrote it; if that
    # process is still alive this one must not start beside it -- two schedulers would launch the same
    # waves and each would kill the other's window (2026-09-05: a stray `main()` did just that to a
    # live wave). If it is gone, its window is killed (never adopted -- one code path, one flicker per
    # restart) and this one's is closed on the way out.
    window = window_module.Window(log=_log)
    previous = laptop_status.read_local() or {}
    if previous_scheduler_alive(previous):
        _log('a scheduler is already running here (pid {0}, status {1}); not starting a second one'.format(
            previous.get('pid'), live_runs.status_path()))
        return 2
    if window_module.wanted():
        window.kill_stale(previous.get('window_pid'))

    def make_driver(specs):
        return Driver(specs, wave=args.wave, shards=args.shards, stage_b=not args.no_stage_b,
                      max_trainers=args.max_trainers, passes=passes, window=window)

    publisher = None if args.no_status else laptop_status.Publisher(log=_log)
    reporter = Reporter(publisher, queue_dir=args.queue, make_driver=make_driver, window=window)

    def stop(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)

    try:
        if args.queue:
            return run_queue(args.queue, make_driver, after=args.after, reporter=reporter)
        single = make_driver(specs)
        single.reporter = reporter
        code = single.run(after=args.after)
        reporter.publish(None)
        return code
    finally:
        window.close()


def previous_scheduler_alive(previous):
    """Whether the scheduler that wrote `previous` (a status dict) is still running. Its own pid, read
    from the file it wrote -- no pattern. This process, a missing pid and a dead pid all count as no."""
    try:
        pid = int(previous.get('pid') or 0)
    except (TypeError, ValueError):
        return False
    return pid > 0 and pid != os.getpid() and live_runs.alive(pid) and not live_runs.zombie(pid)


if __name__ == '__main__':
    sys.exit(main())
