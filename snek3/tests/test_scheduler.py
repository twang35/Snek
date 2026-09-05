"""The scheduler: desktop specs, run here in waves, each followed by its stage B and the two hof passes,
with the window and the eval workers owned by it.

Every process call is a stand-in, so a test runs a 32-arm batch in milliseconds and asserts on what
would have been launched: the wave barrier, the pass naming the desktop uses, the chain stopping at
a failed pass, the skip of a finished arm, the wait on an arm already live, and the trainer cap.
"""

import json
import os

import pytest

from tools import scheduler
from tools import live_runs


def spec(policy, **env):
    return {'project': 'snek3', 'id': policy, 'type': 'train', 'policy': policy,
            'max_steps': 50003968, 'env': {'SNEK_PPO_MINIBATCH': '32', **env}}


def write_specs(directory, policies):
    os.makedirs(directory, exist_ok=True)
    for policy in policies:
        with open(os.path.join(directory, policy + '.json'), 'w') as handle:
            json.dump(spec(policy), handle)


class FakeProcess(object):
    """Exits on the first poll unless told to take `polls` polls first (the driver polls, it does not
    block in `wait`, so it can republish its status while an arm runs). `code` is its exit status."""

    def __init__(self, pid, polls=0, code=0):
        self.pid, self.polls, self.returncode, self.code = pid, polls, None, code

    def poll(self):
        if self.polls > 0:
            self.polls -= 1
            return None
        self.returncode = self.code
        return self.code

    def wait(self):
        self.returncode = 0
        return 0


class Calls(object):
    """Records every launch and every close-out call in order, the close-outs by pass name.

    A launched arm *finishes*: its `_evals.json` lands at the cap as the fake process exits, because an
    arm that exits short of its cap is relaunched by the driver (2026-09-05) and a fixture that wants
    that case says so (`CrashingCalls`)."""

    def __init__(self, codes=None):
        self.events, self.pid, self.codes = [], 1000, codes or {}

    def record(self, argv, **kwargs):
        self.pid += 1
        self.events.append(('train', argv[-1], kwargs['env']))
        return FakeProcess(self.pid)

    def popen(self, argv, **kwargs):
        runs = kwargs['env'].get('SNEK_RUNS_DIR') or _RUNS.get('dir')
        if runs:
            with open(os.path.join(runs, argv[-1] + '_evals.json'), 'w') as handle:
                json.dump({'summary': {'step': int(kwargs['env']['SNEK_MAX_STEPS'])}}, handle)
        return self.record(argv, **kwargs)

    def call(self, argv, **kwargs):
        pass_name = argv[argv.index('--pass') + 1] if '--pass' in argv else 'stageb'
        end = argv.index('--pass') if '--pass' in argv else argv.index('--shards')
        arms = argv[argv.index('tools.closeout') + 1:end]
        self.events.append((pass_name, tuple(arms), argv[-1]))
        return self.codes.get(pass_name, 0)


PASSES = ['stageb', 'hof5000', 'hof30k']
_RUNS = {}      # the current test's runs dir, for `Calls.popen`; set by the `box` fixture


@pytest.fixture
def box(tmp_path):
    runs = tmp_path / 'runs'
    runs.mkdir()
    logs = tmp_path / 'logs'
    _RUNS['dir'] = str(runs)
    yield {'runs': str(runs), 'logs': str(logs)}
    _RUNS.pop('dir', None)


def no_workers(count, runs_dir=None):
    """Stands in for `eval_queue.ensure_workers`: a unit test must never start a real worker."""
    return []


def driver(specs, box, calls, **kwargs):
    kwargs.setdefault('ensure_workers', no_workers)
    return scheduler.Driver(specs, runs_dir=box['runs'], logs_dir=box['logs'], popen=calls.popen,
                            call=calls.call, sleep=lambda s: None, python='py', **kwargs)


# ---------------------------------------------------------------- specs

def test_specs_load_in_id_order_from_files_and_directories_evals_included(tmp_path):
    d = str(tmp_path / 'specs')
    write_specs(d, ['b13ab-mb32-seed2', 'b13aa-mb32-seed1'])
    with open(os.path.join(d, 'b13-hof5000.json'), 'w') as handle:
        json.dump({'id': 'b13-hof5000', 'type': 'eval', 'policies': ['b13aa-mb32-seed1'],
                   'eval_args': ['--pass', 'hof5000']}, handle)
    with open(os.path.join(d, 'deploy-1.json'), 'w') as handle:
        json.dump({'id': 'deploy-1', 'type': 'deploy'}, handle)
    loose = str(tmp_path / 'b13ac-mb32-seed3.json')
    with open(loose, 'w') as handle:
        json.dump(spec('b13ac-mb32-seed3'), handle)
    specs = scheduler.load_specs([d, loose])
    assert [s['id'] for s in specs] == ['b13-hof5000', 'b13aa-mb32-seed1', 'b13ab-mb32-seed2',
                                        'b13ac-mb32-seed3']
    assert [s['id'] for s in scheduler.train_specs(specs)] == ['b13aa-mb32-seed1', 'b13ab-mb32-seed2',
                                                               'b13ac-mb32-seed3']
    assert [s['id'] for s in scheduler.eval_specs(specs)] == ['b13-hof5000']
    assert specs[1]['_dir'] == d and specs[-1]['_dir'] == str(tmp_path)


def test_batch_id_and_stage_b_labels_match_the_daemon():
    specs = [spec('b13aa-mb32-seed1'), spec('b13bf-mb2048-seed4')]
    assert scheduler.batch_id(specs) == 'b13'
    assert scheduler.stage_b_label('b13', 1) == 'b13-stageb'
    assert scheduler.stage_b_label('b13', 3) == 'b13-stageb-w3'
    assert scheduler.pass_label('b13', 'hof5000', 1) == 'b13-hof5000'
    assert scheduler.pass_label('b13', 'hof30k', 2) == 'b13-hof30k-w2'
    assert scheduler.batch_id([spec('b13aa-x-seed1'), spec('b14a-y-seed1')]) == 'batch'


def test_training_env_is_the_spec_over_ours_plus_the_absolute_cap():
    env = scheduler.training_env(spec('b13aa-mb32-seed1'), base={'HOME': '/h', 'SNEK_PPO_MINIBATCH': '256'})
    assert env['SNEK_PPO_MINIBATCH'] == '32'
    assert env['SNEK_MAX_STEPS'] == '50003968'
    assert env['HOME'] == '/h'
    assert env['PYTHONPATH'] == scheduler.ROOT


# ---------------------------------------------------------------- the waves

def test_a_batch_runs_in_waves_each_followed_by_its_own_stage_b(box):
    specs = [spec('b13{0}{1}-mb-seed{2}'.format(a, b, i % 4 + 1))
             for i, (a, b) in enumerate((x, y) for x in 'ab' for y in 'abcdefghijklmnop')][:32]
    calls = Calls()
    assert driver(specs, box, calls).run() == 0
    kinds = [e[0] for e in calls.events]
    assert kinds == (['train'] * 8 + PASSES) * 4
    first_stage_b, first_hof, first_30k = calls.events[8:11]
    assert first_stage_b[1] == first_hof[1] == first_30k[1] == tuple(s['policy'] for s in specs[:8])
    assert first_stage_b[2] == str(scheduler.DEFAULT_SHARDS)   # 12 since 2026-09-04
    assert len(os.listdir(box['logs'])) == 32 + 4 * 3      # one log per arm, one per pass per wave
    assert os.path.exists(os.path.join(box['logs'], 'b13-stageb-w4.log'))
    assert os.path.exists(os.path.join(box['logs'], 'b13-hof30k-w4.log'))


def test_a_failed_pass_stops_the_chain_for_that_wave(box):
    """hof5000 selects from the file stage B wrote, so with stage B failed it would fail too and
    bury the real failure. Same rule as the daemon's `next_pass`."""
    specs = [spec('b13aa-mb32-seed1'), spec('b13ab-mb32-seed2')]
    calls = Calls(codes={'stageb': 1})
    assert driver(specs, box, calls, wave=2).run() == 1
    assert [e[0] for e in calls.events] == ['train', 'train'] + ['stageb'] * 3, 'tried three times, then marked'
    calls = Calls(codes={'hof5000': 2})
    assert driver(specs, box, calls, wave=2).run() == 2
    assert [e[0] for e in calls.events] == ['stageb'] + ['hof5000'] * 3, 'the arms are at their cap from the first run'


def test_a_hold_marker_delays_every_launch_and_lifting_it_resumes(box):
    """`runs/.live/.paused`: what is running finishes, the next wave, pass or eval waits. The desktop
    daemon writes it for `paused`/`drain`; on the laptop it is touched by hand."""
    from tools import live_runs
    specs = [spec('b13aa-mb32-seed1'), spec('b13ab-mb32-seed2')]
    calls = Calls()
    hold = live_runs.hold_path(box['runs'])
    os.makedirs(os.path.dirname(hold))
    open(hold, 'w').close()
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        if len(slept) == 3:
            os.remove(hold)                   # the pause is lifted after three polls
    d = scheduler.Driver(specs, runs_dir=box['runs'], logs_dir=box['logs'], popen=calls.popen,
                         call=calls.call, sleep=sleep, python='py', ensure_workers=no_workers, wave=2)
    assert not calls.events
    assert d.run() == 0
    assert len(slept) >= 3 and [e[0] for e in calls.events] == ['train', 'train'] + PASSES
    open(hold, 'w').close()
    assert any('paused' in line for line in d.attention())


def test_no_hof_keeps_stage_b_and_drops_the_two_re_measures(box):
    specs = [spec('b13aa-mb32-seed1')]
    calls = Calls()
    driver(specs, box, calls, wave=1, passes=('stageb',)).run()
    assert [e[0] for e in calls.events] == ['train', 'stageb']


def test_the_pass_reaches_the_close_out_as_pass_and_never_as_numbers(box):
    """`--pass hof5000` is the whole instruction; the selector, depth, label and seed are the
    close-out's presets, so this driver carries none of them -- the same rule as the daemon."""
    specs = [spec('b13aa-mb32-seed1')]
    argvs = []

    def call(argv, **kwargs):
        argvs.append(argv)
        return 0
    scheduler.Driver(specs, runs_dir=box['runs'], logs_dir=box['logs'], popen=Calls().popen,
                        call=call, sleep=lambda s: None, python='py', ensure_workers=no_workers, wave=1).run()
    assert [a[a.index('tools.closeout') + 1:] for a in argvs] == [
        ['b13aa-mb32-seed1', '--shards', '12'],
        ['b13aa-mb32-seed1', '--pass', 'hof5000', '--shards', '12'],
        ['b13aa-mb32-seed1', '--pass', 'hof30k', '--shards', '12']]
    for argv in argvs:
        assert not {'--selector', '--episodes', '--label', '--seed'} & set(argv)


def test_a_finished_arm_is_skipped_and_its_wave_still_gets_stage_b(box):
    specs = [spec('b13aa-mb32-seed1'), spec('b13ab-mb32-seed2')]
    with open(os.path.join(box['runs'], 'b13aa-mb32-seed1_evals.json'), 'w') as handle:
        json.dump({'summary': {'step': 50003968}, 'evals': []}, handle)
    calls = Calls()
    driver(specs, box, calls, wave=2).run()
    assert [e[:2] for e in calls.events][:2] == [('train', 'b13ab-mb32-seed2'),
                                                 ('stageb', ('b13aa-mb32-seed1', 'b13ab-mb32-seed2'))]


def test_an_arm_already_live_on_the_box_is_waited_for_not_relaunched(box, monkeypatch):
    """A killed driver leaves its trainers running; the rerun must adopt them, or the box gets two."""
    specs = [spec('b13aa-mb32-seed1'), spec('b13ab-mb32-seed2')]
    live_runs.register('b13aa-mb32-seed1', pid=os.getpid(), runs_dir=box['runs'])
    ticks = {'n': 0}

    def alive(pid):
        ticks['n'] += 1
        if ticks['n'] >= 4:                                   # "finishes" after a few polls, at its cap
            with open(os.path.join(box['runs'], 'b13aa-mb32-seed1_evals.json'), 'w') as handle:
                json.dump({'summary': {'step': 50003968}}, handle)
        return pid == os.getpid() and ticks['n'] < 4

    monkeypatch.setattr(live_runs, 'alive', alive)
    calls = Calls()
    driver(specs, box, calls, wave=2).run()
    assert [e[1] for e in calls.events if e[0] == 'train'] == ['b13ab-mb32-seed2']
    assert [e[0] for e in calls.events[-3:]] == PASSES


def test_no_launch_while_the_box_is_at_the_trainer_cap(box, monkeypatch):
    counts = iter([8, 8, 7])
    monkeypatch.setattr(scheduler, 'trainer_count', lambda runs_dir=None: next(counts))
    slept = []
    calls = Calls()
    d = scheduler.Driver([spec('b13aa-mb32-seed1')], runs_dir=box['runs'], logs_dir=box['logs'],
                            popen=calls.popen, call=calls.call, sleep=slept.append, python='py', ensure_workers=no_workers,
                            wave=1, stage_b=False)
    d.run()
    assert slept == [scheduler.POLL_SECONDS] * 2
    assert [e[0] for e in calls.events] == ['train']


def test_a_wave_wider_than_the_cap_is_refused():
    with pytest.raises(ValueError):
        scheduler.Driver([spec('b13aa-mb32-seed1')], wave=9, max_trainers=8)


def test_after_holds_the_whole_batch_until_that_process_has_exited(box, monkeypatch):
    """b14 behind b13's last stage B: nothing launches while the other driver is alive."""
    ticks = {'n': 0}

    def alive(pid):
        if pid == 4242:
            ticks['n'] += 1
            return ticks['n'] < 4
        return False

    monkeypatch.setattr(live_runs, 'alive', alive)
    slept, calls = [], Calls()
    d = scheduler.Driver([spec('b14a-roll32-seed1')], runs_dir=box['runs'], logs_dir=box['logs'],
                            popen=calls.popen, call=calls.call, sleep=slept.append, python='py', ensure_workers=no_workers,
                            wave=1, stage_b=False)
    d.run(after=4242)
    assert slept == [scheduler.POLL_SECONDS] * 2      # one check logs the wait, two more poll it
    assert [e[0] for e in calls.events] == ['train']


# ---------------------------------------------------------------- passes already on disk

def _pass_files(runs, policies, *pass_names):
    for policy in policies:
        for pass_name in pass_names:
            open(scheduler.pass_file(policy, pass_name, runs), 'w').write('{"rows": []}')


def test_a_pass_every_arm_already_has_is_skipped_and_the_chain_continues(box):
    """A shard resumes only from its own shard files and the merge deletes them, so rerunning a
    finished wave's stage B would re-measure it from scratch. Skipping to the missing pass is also
    how a driver that predates the hof passes fills them in on a rerun."""
    specs = [spec('b14a-roll32-seed1'), spec('b14b-roll32-seed2')]
    for s in specs:
        with open(os.path.join(box['runs'], s['policy'] + '_evals.json'), 'w') as handle:
            json.dump({'summary': {'step': 50003968}, 'evals': []}, handle)
    _pass_files(box['runs'], [s['policy'] for s in specs], 'stageb')
    calls = Calls()
    assert driver(specs, box, calls, wave=2).run() == 0
    assert [e[0] for e in calls.events] == ['hof5000', 'hof30k']


def test_a_pass_one_arm_lacks_still_runs_over_the_whole_wave(box):
    specs = [spec('b14a-roll32-seed1'), spec('b14b-roll32-seed2')]
    _pass_files(box['runs'], ['b14a-roll32-seed1'], 'stageb')
    calls = Calls()
    driver(specs, box, calls, wave=2).run()
    assert [e[0] for e in calls.events] == ['train', 'train'] + PASSES


def test_pending_is_false_only_when_every_arm_is_capped_and_every_pass_filed(box):
    specs = [spec('b14a-roll32-seed1'), spec('b14b-roll32-seed2')]
    d = driver(specs, box, Calls(), wave=2)
    assert d.pending()
    for s in specs:
        with open(os.path.join(box['runs'], s['policy'] + '_evals.json'), 'w') as handle:
            json.dump({'summary': {'step': 50003968}, 'evals': []}, handle)
    assert d.pending(), 'trained but not measured'
    _pass_files(box['runs'], [s['policy'] for s in specs], 'stageb', 'hof5000')
    assert d.pending(), 'hof30k still owed'
    _pass_files(box['runs'], [s['policy'] for s in specs], 'hof30k')
    assert not d.pending()


# ---------------------------------------------------------------- the queue

def _queue(tmp_path, batches):
    q = tmp_path / 'queue'
    for name, policies in batches.items():
        write_specs(str(q / name), policies)
    return str(q)


def test_the_queue_runs_batches_in_name_order_and_exits_when_none_has_work(tmp_path, box):
    q = _queue(tmp_path, {'b16': ['b16a-kl-seed1'], 'b14': ['b14a-roll-seed1', 'b14b-roll-seed2']})
    calls = Calls()

    def make(specs):
        return driver(specs, box, calls, wave=8)
    # every arm's `_evals.json` appears as it "trains", so the second scan sees the batch finished
    real_popen = calls.popen

    def popen(argv, **kwargs):
        with open(os.path.join(box['runs'], argv[-1] + '_evals.json'), 'w') as handle:
            json.dump({'summary': {'step': 50003968}, 'evals': []}, handle)
        return real_popen(argv, **kwargs)
    calls.popen = popen
    orig_call = calls.call

    def call(argv, **kwargs):
        code = orig_call(argv, **kwargs)
        pass_name = calls.events[-1][0]
        _pass_files(box['runs'], calls.events[-1][1], pass_name)
        return code
    calls.call = call
    assert scheduler.run_queue(q, make) == 0
    trained = [e[1] for e in calls.events if e[0] == 'train']
    assert trained == ['b14a-roll-seed1', 'b14b-roll-seed2', 'b16a-kl-seed1'], 'b14 before b16, by name'
    assert [e[0] for e in calls.events] == ['train', 'train'] + PASSES + ['train'] + PASSES


def test_a_batch_dropped_in_while_another_runs_is_picked_up_next(tmp_path, box):
    q = _queue(tmp_path, {'b14': ['b14a-roll-seed1']})
    calls = Calls()
    state = {'dropped': False}

    def popen(argv, **kwargs):
        with open(os.path.join(box['runs'], argv[-1] + '_evals.json'), 'w') as handle:
            json.dump({'summary': {'step': 50003968}, 'evals': []}, handle)
        if not state['dropped']:
            write_specs(os.path.join(q, 'b16'), ['b16a-kl-seed1'])   # arrives mid-batch
            state['dropped'] = True
        calls.pid += 1
        calls.events.append(('train', argv[-1], kwargs['env']))
        return FakeProcess(calls.pid)

    def call(argv, **kwargs):
        pass_name = argv[argv.index('--pass') + 1] if '--pass' in argv else 'stageb'
        end = argv.index('--pass') if '--pass' in argv else argv.index('--shards')
        arms = tuple(argv[argv.index('tools.closeout') + 1:end])
        calls.events.append((pass_name, arms, argv[-1]))
        _pass_files(box['runs'], arms, pass_name)
        return 0
    calls.popen, calls.call = popen, call
    scheduler.run_queue(q, lambda specs: driver(specs, box, calls, wave=8))
    assert [e[1] for e in calls.events if e[0] == 'train'] == ['b14a-roll-seed1', 'b16a-kl-seed1']


def test_a_batch_that_still_has_work_after_running_is_not_looped_on(tmp_path, box):
    """A failed pass leaves the batch pending forever; the queue must not spin on it."""
    q = _queue(tmp_path, {'b14': ['b14a-roll-seed1']})
    calls = Calls(codes={'stageb': 1})
    runs = []

    def make(specs):
        runs.append([s['policy'] for s in specs])
        return driver(specs, box, calls, wave=8)
    assert scheduler.run_queue(q, make) == 1
    assert [e[0] for e in calls.events] == ['train'] + ['stageb'] * 3, 'ran once (three tries of the pass), then left alone'


def test_a_finished_batch_left_in_the_queue_costs_nothing(tmp_path, box):
    q = _queue(tmp_path, {'b13': ['b13a-mb-seed1']})
    with open(os.path.join(box['runs'], 'b13a-mb-seed1_evals.json'), 'w') as handle:
        json.dump({'summary': {'step': 50003968}, 'evals': []}, handle)
    _pass_files(box['runs'], ['b13a-mb-seed1'], *PASSES)
    calls = Calls()
    assert scheduler.run_queue(q, lambda specs: driver(specs, box, calls, wave=8)) == 0
    assert calls.events == []


def test_queue_and_specs_are_exclusive_on_the_command_line(tmp_path):
    with pytest.raises(SystemExit):
        scheduler.main(['--queue', str(tmp_path), 'somefile.json'])


# ---------------------------------------------------------------- laptop-status

class Published(object):
    """Stands in for `laptop_status.Publisher`: keeps every status dict handed to it."""

    def __init__(self):
        self.statuses = []

    def publish(self, status):
        self.statuses.append(status)
        return True

    def glance(self, index):
        return self.statuses[index]['at_a_glance']


class FinishingCalls(Calls):
    """Launches write the arm's finished `_evals.json` and passes write their merged file, so the
    driver's view of what is owed moves the way a real wave's does."""

    def __init__(self, runs, codes=None):
        Calls.__init__(self, codes)
        self.runs = runs

    def popen(self, argv, **kwargs):
        policy = argv[-1]
        with open(os.path.join(self.runs, policy + '_evals.json'), 'w') as handle:
            json.dump({'summary': {'step': int(kwargs['env']['SNEK_MAX_STEPS'])}}, handle)
        return Calls.popen(self, argv, **kwargs)

    def call(self, argv, **kwargs):
        code = Calls.call(self, argv, **kwargs)
        pass_name, arms, _ = self.events[-1]
        for policy in arms:
            open(scheduler.pass_file(policy, pass_name, self.runs), 'w').close()
        return code


def _spec(policy, label):
    return {'id': policy, 'policy': policy, 'max_steps': 100, 'env': {}, 'label': label}


def test_the_queue_publishes_both_boxes_shape_on_every_event_and_empty_when_it_exits(tmp_path, box):
    """The desktop's own `build_at_a_glance` draws the lines, so a running laptop wave reads exactly
    as a desktop one does, and the other batch waiting in the queue directory is listed as owed."""
    queue = tmp_path / 'queue'
    for name, specs in (('b1', [_spec('b1a-x-seed1', 'b1: x, seed 1 of 2 -- wave 1 of 1'),
                                _spec('b1b-x-seed2', 'b1: x, seed 2 of 2 -- wave 1 of 1')]),
                        ('b2', [_spec('b2a-y-seed1', 'b2: y, seed 1 of 1 -- wave 1 of 1')])):
        (queue / name).mkdir(parents=True)
        for spec in specs:
            (queue / name / (spec['id'] + '.json')).write_text(json.dumps(spec))
    calls = FinishingCalls(box['runs'])
    published = Published()
    make = lambda specs: driver(specs, box, calls, wave=2)
    reporter = scheduler.Reporter(published, queue_dir=str(queue), make_driver=make)

    scheduler.run_queue(str(queue), make, reporter=reporter)

    first = published.glance(0)
    assert first['running'] == ['b1 | x -- wave 1 of 1 | training 100% (2 arms)']
    assert first['queued'] == ['b1 evals | x | queued (2 arms)',
                               'b2 training | y -- wave 1 of 1 | queued (1 arm)',
                               'b2 evals | y | queued (1 arm)']
    running_lines = [line for status in published.statuses for line in status['at_a_glance']['running']]
    assert 'b1 | x | stage B (2 arms)' in running_lines
    assert 'b1 | x | hof5000 (2 arms)' in running_lines
    assert 'b1 | x | hof30k (2 arms)' in running_lines
    assert 'b2 | y -- wave 1 of 1 | training 100% (1 arm)' in running_lines
    last = published.statuses[-1]
    assert last['at_a_glance'] == {'running': [], 'queued': [], 'attention': []}
    assert last['box'] == 'laptop' and last['iso'] and last['running'] == []


def test_a_waiting_driver_republishes_every_ten_minutes_so_the_percent_moves(box):
    clock = {'now': 0.0}

    def sleep(seconds):
        clock['now'] += 400.0

    calls = Calls()

    class Slow(FakeProcess):
        def poll(self):
            code = FakeProcess.poll(self)
            if code is not None:                                 # reaches its cap as it exits
                with open(os.path.join(box['runs'], 'b1a-x-seed1_evals.json'), 'w') as handle:
                    json.dump({'summary': {'step': 100}}, handle)
            return code
    slow = Slow(1, polls=3)
    calls.popen = lambda argv, **kwargs: slow
    published = Published()
    d = scheduler.Driver([_spec('b1a-x-seed1', 'b1: x')], runs_dir=box['runs'], logs_dir=box['logs'],
                            popen=calls.popen, call=calls.call, sleep=sleep, python='py', ensure_workers=no_workers, stage_b=False,
                            reporter=scheduler.Reporter(published), clock=lambda: clock['now'])
    d.run()
    # at launch (t=0); at the poll that crosses 600 s (t=800); at the arm's exit
    assert len(published.statuses) == 3
    assert published.glance(0)['running'] == ['b1 | x | training (1 arm)']
    assert published.glance(-1)['running'] == []


def test_a_publisher_that_fails_never_stops_the_driver(box):
    from tools import laptop_status
    logged = []
    publisher = laptop_status.Publisher(host_config={'STATUS_BRANCH': 'x'},
                                        publish_status=lambda host, text: 1 / 0,
                                        ensure=lambda host: None, log=logged.append)
    assert publisher.publish(laptop_status.build([], [])) is False
    assert 'publish failed' in logged[0]
    calls = Calls()
    d = driver([_spec('b1a-x-seed1', 'b1: x')], box, calls, stage_b=False,
               reporter=scheduler.Reporter(publisher))
    assert d.run() == 0
    assert [event[0] for event in calls.events] == ['train']


# ---------------------------------------------------------------- the window and the workers

class FakeWindow(object):
    def __init__(self):
        self.opens, self.polls, self.closed = 0, 0, False
        self._up = False

    def open(self):
        self.opens += 1
        was_up, self._up = self._up, True
        return not was_up

    def poll(self):
        self.polls += 1
        return False

    def pid(self):
        return 4242 if self._up else None

    def close(self):
        self.closed, self._up = True, False


def test_the_window_is_opened_at_each_launch_and_pointed_at_the_wave_then_the_pass(box):
    """The scheduler owns the window: it asks for it when it launches something and tells it, through
    the status file, which PNGs to draw -- the whole wave while training, the pass's charts after."""
    specs = [spec('b13aa-mb32-seed1'), spec('b13ab-mb32-seed2')]
    calls = FinishingCalls(box['runs'])
    fake = FakeWindow()
    published = Published()
    d = scheduler.Driver(specs, runs_dir=box['runs'], logs_dir=box['logs'], popen=calls.popen,
                         call=calls.call, sleep=lambda s: None, python='py', ensure_workers=no_workers, wave=2, window=fake,
                         reporter=scheduler.Reporter(published, window=fake, runs_dir=box['runs']))
    assert d.run() == 0
    assert fake.opens == 4, 'the wave, then each of the three passes asked; the first ask opened it'
    panels = [status['panels'] for status in published.statuses]
    assert panels[0] == [os.path.join(box['runs'], 'b13aa-mb32-seed1.png'),
                         os.path.join(box['runs'], 'b13ab-mb32-seed2.png')]
    stage_b = [p for p in panels if p and p[0].endswith('b13aa-mb32-seed1_checkpoint_evals.png')]
    hof = [p for p in panels if p and p[0].endswith('b13aa-mb32-seed1_checkpoint_evals_hof5000.png')]
    assert stage_b and hof
    assert all(status['window_pid'] == 4242 for status in published.statuses[:-1])
    with open(live_runs.status_path(box['runs'])) as handle:
        local = json.load(handle)
    assert local['panels'][0].endswith('_checkpoint_evals_hof30k.png'), 'between launches the last panels stay'
    d.reporter.publish(None)                       # what the queue does as it exits
    with open(live_runs.status_path(box['runs'])) as handle:
        local = json.load(handle)
    assert local['panels'] == [] and local['at_a_glance']['running'] == [], 'the exit write is idle'


def test_the_workers_are_started_before_the_wave_with_the_count_its_specs_name(box):
    asked = []
    specs = [spec('b13aa-mb32-seed1', SNEK_EVAL_WORKERS='8'), spec('b13ab-mb32-seed2', SNEK_EVAL_WORKERS='8')]
    calls = Calls()
    d = driver(specs, box, calls, wave=2, stage_b=False,
               ensure_workers=lambda n, runs_dir=None: asked.append((n, runs_dir)) or [])
    d.run()
    assert asked == [(8, box['runs'])], 'once, before the arms, with the specs\' number'
    assert calls.events[0][0] == 'train'


def test_the_worker_count_is_the_default_when_the_specs_disagree_or_say_nothing_and_zero_when_the_queue_is_off():
    from tools import eval_queue
    assert scheduler.wave_workers([spec('a'), spec('b')]) == eval_queue.DEFAULT_WORKERS
    assert scheduler.wave_workers([spec('a', SNEK_EVAL_WORKERS='8'), spec('b', SNEK_EVAL_WORKERS='6')]) == eval_queue.DEFAULT_WORKERS
    assert scheduler.wave_workers([spec('a', SNEK_EVAL_WORKERS='4'), spec('b', SNEK_EVAL_WORKERS='4')]) == 4
    assert scheduler.wave_workers([spec('a', SNEK_EVAL_QUEUE='0'), spec('b', SNEK_EVAL_QUEUE='0')]) == 0


def test_no_workers_are_started_when_every_arm_of_the_wave_is_already_done_or_live(box):
    asked = []
    specs = [spec('b13aa-mb32-seed1')]
    with open(os.path.join(box['runs'], 'b13aa-mb32-seed1_evals.json'), 'w') as handle:
        json.dump({'summary': {'step': 50003968}, 'evals': []}, handle)
    driver(specs, box, Calls(), wave=1, stage_b=False,
           ensure_workers=lambda n, runs_dir=None: asked.append(n) or []).run()
    assert asked == []


# ---------------------------------------------------------------- markers: failed passes, eval specs

def test_a_failed_pass_is_marked_and_a_rerun_skips_it_instead_of_looping(tmp_path, box):
    q = _queue(tmp_path, {'b14': ['b14a-roll-seed1']})
    calls = FinishingCalls(box['runs'], codes={'stageb': 1})
    make = lambda specs: driver(specs, box, calls, wave=8)
    assert scheduler.run_queue(q, make) == 1
    assert os.path.exists(os.path.join(q, 'b14', '.failed-b14-stageb'))
    assert [e[0] for e in calls.events] == ['train'] + ['stageb'] * 3
    # a second scheduler over the same queue: the arm is done, the pass is marked -- nothing to do
    assert scheduler.run_queue(q, make) == 0
    assert [e[0] for e in calls.events] == ['train'] + ['stageb'] * 3, 'not retried'
    _, specs = scheduler.queue_batches(q)[0]
    assert not make(specs).pending()
    assert 'b14-stageb failed' in make(specs).attention()[0]
    os.remove(os.path.join(q, 'b14', '.failed-b14-stageb'))
    assert make(specs).pending(), 'deleting the marker is the retry'


def test_an_eval_spec_runs_once_after_the_waves_as_the_command_it_spells(tmp_path, box):
    q = _queue(tmp_path, {'b14': ['b14a-roll-seed1']})
    with open(os.path.join(q, 'b14', 'b14-one.json'), 'w') as handle:
        json.dump({'id': 'b14-one', 'type': 'eval', 'policies': ['b14a-roll-seed1', 'b14b-roll-seed2'],
                   'selector': 'one', 'episodes': 3000, 'eval_args': ['--label', 'hof'],
                   'eval_shards': 3}, handle)
    argvs = []
    calls = FinishingCalls(box['runs'])
    real_call = calls.call

    def call(argv, **kwargs):
        argvs.append(argv)
        if '--label' in argv:
            return 0
        return real_call(argv, **kwargs)
    calls.call = call
    make = lambda specs: driver(specs, box, calls, wave=8)
    assert scheduler.run_queue(q, make) == 0
    assert argvs[-1][argvs[-1].index('tools.closeout') + 1:] == [
        'b14a-roll-seed1', 'b14b-roll-seed2', '--selector', 'one', '--label', 'hof',
        '--episodes', '3000', '--shards', '3']
    assert os.path.exists(os.path.join(q, 'b14', '.done-b14-one'))
    before = len(argvs)
    assert scheduler.run_queue(q, make) == 0
    assert len(argvs) == before, 'an eval spec runs once'


def test_eval_label_reads_the_pass_or_the_label_for_the_panels():
    assert scheduler.eval_label({'eval_args': ['--pass', 'hof5000']}) == 'hof5000'
    assert scheduler.eval_label({'eval_args': ['--label', 'ab']}) == 'ab'
    assert scheduler.eval_label({'eval_args': []}) is None
    assert scheduler.pass_panels(['b1a'], 'ab', '/r') == ['/r/b1a_checkpoint_evals_ab.png']
    assert scheduler.pass_panels(['b1a'], None, '/r') == ['/r/b1a_checkpoint_evals.png']


# ---------------------------------------------------------------- main(): no side effect before its arguments are accepted

def _stub_window(monkeypatch):
    """`Window` with every process call recorded and none made."""
    killed, opened = [], []
    monkeypatch.setattr(scheduler.window_module.Window, 'kill_stale', lambda self, pid: killed.append(pid))
    monkeypatch.setattr(scheduler.window_module.Window, 'open', lambda self: opened.append(1) or False)
    monkeypatch.setattr(scheduler.window_module.Window, 'close', lambda self: None)
    return killed, opened


def test_main_refused_on_its_arguments_touches_neither_the_window_nor_the_status(tmp_path, monkeypatch):
    """2026-09-05: a test called `main()` for its argument check while b16's wave 5 trained; `main()` had
    already read the live status and killed the live window before `parser.error` ran."""
    killed, _ = _stub_window(monkeypatch)
    monkeypatch.setenv('SNEK_CHART_WINDOW', '1')
    scheduler.laptop_status.write_local(scheduler.laptop_status.build([], [], window_pid=4242))
    before = open(live_runs.status_path()).read()
    with pytest.raises(SystemExit):
        scheduler.main(['--queue', str(tmp_path), 'somefile.json'])
    with pytest.raises(SystemExit):
        scheduler.main([str(tmp_path)])                          # a directory with no specs
    assert killed == [] and open(live_runs.status_path()).read() == before


def test_main_does_not_start_beside_a_live_scheduler_and_never_kills_its_window(tmp_path, monkeypatch):
    killed, _ = _stub_window(monkeypatch)
    monkeypatch.setenv('SNEK_CHART_WINDOW', '1')
    write_specs(str(tmp_path / 'b1'), ['b1a-x-seed1'])
    # the last status names this very process as its writer, so its scheduler is trivially alive
    status = scheduler.laptop_status.build([], [], window_pid=4242)
    status['pid'] = os.getppid()
    scheduler.laptop_status.write_local(status)
    assert scheduler.main(['--queue', str(tmp_path), '--no-status']) == 2
    assert killed == [], 'a live scheduler keeps its window'
    assert scheduler.previous_scheduler_alive({'pid': os.getpid()}) is False, 'this process is not a predecessor'
    assert scheduler.previous_scheduler_alive({}) is False and scheduler.previous_scheduler_alive({'pid': 'x'}) is False


def test_a_dead_predecessors_window_is_killed_only_when_windows_are_wanted(tmp_path, monkeypatch, isolated_runs_dir):
    killed, _ = _stub_window(monkeypatch)
    status = scheduler.laptop_status.build([], [], window_pid=4242)
    status['pid'] = 2 ** 22 + 7                                   # no such process
    scheduler.laptop_status.write_local(status)
    write_specs(str(tmp_path / 'b1'), ['b1a-x-seed1'])
    with open(os.path.join(isolated_runs_dir, 'b1a-x-seed1_evals.json'), 'w') as handle:
        json.dump({'summary': {'step': 50003968}}, handle)      # nothing pending: main() returns at once
    scheduler.main(['--queue', str(tmp_path), '--no-status', '--no-stage-b'])
    assert killed == [], 'SNEK_CHART_WINDOW=0 (the suite default): no window, so nothing to kill'
    monkeypatch.setenv('SNEK_CHART_WINDOW', '1')
    scheduler.laptop_status.write_local(status)
    scheduler.main(['--queue', str(tmp_path), '--no-status', '--no-stage-b'])
    assert killed == [4242]


def test_a_reopen_writes_the_status_so_the_fresh_window_has_its_panels_at_once(box):
    class Reopening(FakeWindow):
        def poll(self):
            self.polls += 1
            return self.polls == 1                                # the first tick finds a reopen request
    fake = Reopening()
    published = Published()
    calls = Calls()
    real_popen = calls.popen
    calls.popen = lambda argv, **kw: FakeProcess(1, polls=2) if real_popen(argv, **kw) else None
    d = driver([spec('b13aa-mb32-seed1')], box, calls, wave=1, stage_b=False, window=fake,
               reporter=scheduler.Reporter(published, window=fake, runs_dir=box['runs']))
    d.run()
    assert len(published.statuses) == 3, 'launch, the reopen tick, exit -- not launch and exit alone'


def test_work_queued_into_the_running_batch_runs_at_the_next_boundary(tmp_path, box):
    """2026-09-05: the queue skipped a batch it had run once whatever it now owed, so a hand hof spec
    dropped into the running batch's directory never ran and the scheduler exited 1. The set of owed
    ids is what decides now: the same set is a stuck arm, a different set is new work."""
    q = _queue(tmp_path, {'b16': ['b16a-kl-seed1'], 'b19': ['b19a-x-seed1']})
    calls = FinishingCalls(box['runs'])
    argvs, real_call, real_popen = [], calls.call, calls.popen

    def call(argv, **kwargs):
        argvs.append(argv)
        return 0 if '--label' in argv else real_call(argv, **kwargs)

    def popen(argv, **kwargs):
        if argv[-1] == 'b16a-kl-seed1':                          # a hand hof spec lands mid-b16
            with open(os.path.join(q, 'b16', 'b16-one.json'), 'w') as handle:
                json.dump({'id': 'b16-one', 'type': 'eval', 'policies': ['b16a-kl-seed1'],
                           'eval_args': ['--label', 'hof']}, handle)
        return real_popen(argv, **kwargs)
    calls.call, calls.popen = call, popen
    assert scheduler.run_queue(q, lambda specs: driver(specs, box, calls, wave=8)) == 0
    trained = [e[1] for e in calls.events if e[0] == 'train']
    assert trained == ['b16a-kl-seed1', 'b19a-x-seed1'], 'each arm once: the rerun of b16 is the eval alone'
    assert os.path.exists(os.path.join(q, 'b16', '.done-b16-one'))
    evals = [i for i, argv in enumerate(argvs) if '--label' in argv]
    assert len(evals) == 1 and evals[0] == 3, 'after b16\'s three passes, before b19 (name order), once'


def test_the_pause_is_reported_once_however_many_batches_wait(tmp_path, box):
    q = _queue(tmp_path, {'b16': ['b16a-kl-seed1'], 'b19': ['b19a-x-seed1'], 'b20': ['b20a-y-seed1']})
    os.makedirs(live_runs.directory(box['runs']), exist_ok=True)
    open(live_runs.hold_path(box['runs']), 'w').close()
    make = lambda specs: driver(specs, box, Calls(), wave=8)
    reporter = scheduler.Reporter(Published(), queue_dir=q, make_driver=make, runs_dir=box['runs'])
    _, _, attention = reporter.jobs(make(scheduler.queue_batches(q)[0][1]))
    assert len(attention) == 1 and attention[0].startswith('** paused')


# ---------------------------------------------------------------- a pass left running by a previous scheduler

def _register_pass(box, label, pid):
    live_runs.register(live_runs.pass_entry(label), pid, box['runs'])


def test_a_pass_a_previous_scheduler_left_running_is_waited_for_not_launched_twice(box, monkeypatch):
    """Restarting the scheduler mid-pass used to start a second close-out over the same shard files."""
    specs = [spec('b13aa-mb32-seed1')]
    with open(os.path.join(box['runs'], 'b13aa-mb32-seed1_evals.json'), 'w') as handle:
        json.dump({'summary': {'step': 50003968}}, handle)
    _register_pass(box, 'b13-stageb', 777)
    polls = {'n': 0}

    def alive(pid):                              # only the pass pid is ever asked: the arm is finished
        assert pid == 777
        polls['n'] += 1
        if polls['n'] == 3:                      # the third poll finds it gone, and its file written
            open(scheduler.pass_file('b13aa-mb32-seed1', 'stageb', box['runs']), 'w').close()
        return polls['n'] < 3
    monkeypatch.setattr(scheduler.live_runs, 'alive', alive)
    calls = FinishingCalls(box['runs'])
    d = driver(specs, box, calls, wave=1)
    assert d.run() == 0
    assert [e[0] for e in calls.events] == ['hof5000', 'hof30k'], 'stage B adopted, not launched; the chain went on'
    assert polls['n'] == 3
    assert not os.path.exists(live_runs.path_for(live_runs.pass_entry('b13-stageb'), box['runs'])), 'entry cleared'
    assert live_runs.live(box['runs']) == [], 'a pass entry is never a trainer'


def test_a_stale_pass_entry_is_ignored_and_the_pass_is_launched(box, monkeypatch):
    specs = [spec('b13aa-mb32-seed1')]
    with open(os.path.join(box['runs'], 'b13aa-mb32-seed1_evals.json'), 'w') as handle:
        json.dump({'summary': {'step': 50003968}}, handle)
    _register_pass(box, 'b13-stageb', 777)
    monkeypatch.setattr(scheduler.live_runs, 'alive', lambda pid: False if pid == 777 else True)
    # dead already: the entry is stale, so the pass is launched normally
    calls = FinishingCalls(box['runs'])
    assert driver(specs, box, calls, wave=1).run() == 0
    assert [e[0] for e in calls.events] == PASSES, 'a stale entry is ignored and the pass runs'


def test_a_running_pass_is_registered_under_its_label_and_cleared_after(box):
    specs = [spec('b13aa-mb32-seed1')]
    with open(os.path.join(box['runs'], 'b13aa-mb32-seed1_evals.json'), 'w') as handle:
        json.dump({'summary': {'step': 50003968}}, handle)
    seen = {}

    def popen(argv, **kwargs):
        label = argv[argv.index('--pass') + 1] if '--pass' in argv else 'stageb'
        entry = live_runs.path_for(live_runs.pass_entry('b13-' + label), box['runs'])
        seen[label] = os.path.exists(entry)   # not yet: registered right after Popen returns
        for policy in argv[argv.index('tools.closeout') + 1:argv.index('--shards') if '--pass' not in argv else argv.index('--pass')]:
            open(scheduler.pass_file(policy, label, box['runs']), 'w').close()
        return FakeProcess(4000, polls=1)
    import subprocess
    d = scheduler.Driver(specs, runs_dir=box['runs'], logs_dir=box['logs'], popen=popen, call=subprocess.call,
                         sleep=lambda s: None, python='py', ensure_workers=no_workers, wave=1)
    registered = []
    real_wait = d._wait_process

    def wait(process):
        registered.append(sorted(name for name in os.listdir(live_runs.directory(box['runs'])) if name.startswith('.pass-')))
        return real_wait(process)
    d._wait_process = wait
    assert d.run() == 0
    assert registered == [['.pass-b13-stageb'], ['.pass-b13-hof5000'], ['.pass-b13-hof30k']]
    assert not any(name.startswith('.pass-') for name in os.listdir(live_runs.directory(box['runs'])))


# ---------------------------------------------------------------- an arm that exits short of its cap

class CrashingCalls(FinishingCalls):
    """The named arm exits `crashes` times without reaching its cap (its evals file stays at a low
    step), then finishes like the others."""

    def __init__(self, runs, crashes, codes=None):
        FinishingCalls.__init__(self, runs, codes)
        self.crashes = dict(crashes)

    def popen(self, argv, **kwargs):
        policy = argv[-1]
        if self.crashes.get(policy, 0) > 0:
            self.crashes[policy] -= 1
            with open(os.path.join(self.runs, policy + '_evals.json'), 'w') as handle:
                json.dump({'summary': {'step': 1234}}, handle)
            return self.record(argv, **kwargs)
        return FinishingCalls.popen(self, argv, **kwargs)


def test_an_arm_that_exits_short_of_its_cap_is_relaunched_in_its_wave_before_the_passes(box):
    specs = [spec('b13aa-mb32-seed1'), spec('b13ab-mb32-seed2')]
    calls = CrashingCalls(box['runs'], {'b13ab-mb32-seed2': 2})
    d = driver(specs, box, calls, wave=2)
    assert d.run() == 0
    launches = [e[1] for e in calls.events if e[0] == 'train']
    assert launches == ['b13aa-mb32-seed1', 'b13ab-mb32-seed2', 'b13ab-mb32-seed2', 'b13ab-mb32-seed2']
    assert [e[0] for e in calls.events][-3:] == PASSES, 'stage B only after the arm reached its cap'
    assert d.unfinished == [] and d.attention() == []


def test_relaunches_are_bounded_and_the_wave_says_so(box):
    specs = [spec('b13aa-mb32-seed1')]
    calls = CrashingCalls(box['runs'], {'b13aa-mb32-seed1': 99})
    d = driver(specs, box, calls, wave=1)
    d.run()
    assert [e[0] for e in calls.events].count('train') == 1 + scheduler.MAX_ARM_RELAUNCHES
    assert d.unfinished == ['b13aa-mb32-seed1']
    assert 'exited short of its cap' in d.attention()[0]
    assert [e[0] for e in calls.events][-3:] == PASSES, 'the wave still gets its passes'


def test_a_relaunch_waits_out_a_pause_so_kill_deploy_unpause_resumes_the_arms(box):
    specs = [spec('b13aa-mb32-seed1')]
    calls = CrashingCalls(box['runs'], {'b13aa-mb32-seed1': 1})
    os.makedirs(live_runs.directory(box['runs']), exist_ok=True)
    hold = live_runs.hold_path(box['runs'])
    state = {'sleeps': 0}

    def sleep(seconds):
        state['sleeps'] += 1
        if state['sleeps'] == 3:
            os.remove(hold)                         # the deploy is done; unpause
    launched_while_held = []
    orig_popen = calls.popen

    def popen(argv, **kwargs):
        # the first launch happens unpaused; the crash lands while paused (as a deploy would arrange)
        launched_while_held.append(os.path.exists(hold))
        process = orig_popen(argv, **kwargs)
        if len(launched_while_held) == 1:
            open(hold, 'w').close()
        return process
    d = scheduler.Driver(specs, runs_dir=box['runs'], logs_dir=box['logs'], popen=popen,
                         call=calls.call, sleep=sleep, python='py', ensure_workers=no_workers, wave=1)
    assert d.run() == 0
    assert launched_while_held == [False, False], 'the relaunch waited for the hold to lift'
    assert state['sleeps'] >= 3


# ---------------------------------------------------------------- a close-out that dies

def _finished_arm_file(box, policy):
    with open(os.path.join(box['runs'], policy + '_evals.json'), 'w') as handle:
        json.dump({'summary': {'step': 50003968}}, handle)


def test_a_close_out_that_dies_is_relaunched_after_its_shard_group_is_ended(box):
    """A killed controller leaves its shards running in its process group; the relaunch first ends the
    group, then the new shards resume from the rows on disk. Only after two relaunches is it failed."""
    import subprocess
    _finished_arm_file(box, 'b13aa-mb32-seed1')
    launches, groups, clock = [], [], {'now': 0.0}
    alive_groups = set()

    def popen(argv, **kwargs):
        label = argv[argv.index('--pass') + 1] if '--pass' in argv else 'stageb'
        launches.append(label)
        pid = 5000 + len(launches)
        if label == 'stageb' and launches.count('stageb') == 1:      # the first stage B is killed
            alive_groups.add(pid)
            return FakeProcess(pid, code=-9)
        for policy in ['b13aa-mb32-seed1']:
            open(scheduler.pass_file(policy, label, box['runs']), 'w').close()
        return FakeProcess(pid)

    def killpg(pgid, signum):
        groups.append((pgid, signum))
        if pgid not in alive_groups:
            raise ProcessLookupError()
        if signum != 0:
            alive_groups.discard(pgid)          # the shards go on the first signal

    d = scheduler.Driver([spec('b13aa-mb32-seed1')], runs_dir=box['runs'], logs_dir=box['logs'],
                         popen=popen, call=subprocess.call, sleep=lambda s: clock.__setitem__('now', clock['now'] + s),
                         python='py', ensure_workers=no_workers, wave=1, killpg=killpg, clock=lambda: clock['now'])
    assert d.run() == 0
    assert launches == ['stageb', 'stageb', 'hof5000', 'hof30k'], 'relaunched once, then the chain went on'
    assert groups[0] == (5001, 15) and groups[1] == (5001, 0), 'SIGTERM to the group, then a liveness probe'
    assert d.attention() == []


def test_a_close_out_that_keeps_dying_is_marked_failed_after_two_relaunches(box, tmp_path):
    import subprocess
    q = _queue(tmp_path, {'b13': ['b13aa-mb32-seed1']})
    _finished_arm_file(box, 'b13aa-mb32-seed1')
    launches = []

    def popen(argv, **kwargs):
        launches.append(1)
        return FakeProcess(6000 + len(launches), code=1)
    make = lambda specs: scheduler.Driver(specs, runs_dir=box['runs'], logs_dir=box['logs'], popen=popen,
                                          call=subprocess.call, sleep=lambda s: None, python='py',
                                          ensure_workers=no_workers, wave=1,
                                          killpg=lambda pgid, signum: (_ for _ in ()).throw(ProcessLookupError()))
    assert scheduler.run_queue(q, make) == 1
    assert len(launches) == 1 + scheduler.MAX_PASS_RELAUNCHES
    assert os.path.exists(os.path.join(q, 'b13', '.failed-b13-stageb'))


def test_an_adopted_close_out_that_dies_without_its_files_is_relaunched_too(box, monkeypatch):
    import subprocess
    _finished_arm_file(box, 'b13aa-mb32-seed1')
    live_runs.register(live_runs.pass_entry('b13-stageb'), 777, box['runs'])
    monkeypatch.setattr(scheduler.live_runs, 'alive', lambda pid: False if pid == 777 else True)
    launches, groups = [], []

    def popen(argv, **kwargs):
        label = argv[argv.index('--pass') + 1] if '--pass' in argv else 'stageb'
        launches.append(label)
        open(scheduler.pass_file('b13aa-mb32-seed1', label, box['runs']), 'w').close()
        return FakeProcess(7000 + len(launches))
    # 777 is dead when first read, so the entry is stale and the pass is launched outright; make it
    # alive for exactly one poll instead, so it is adopted and then found gone with no file
    polls = {'n': 0}

    def alive(pid):
        if pid != 777:
            return True
        polls['n'] += 1
        return polls['n'] <= 2
    monkeypatch.setattr(scheduler.live_runs, 'alive', alive)

    def killpg(pgid, signum):
        groups.append((pgid, signum))
        raise ProcessLookupError()
    d = scheduler.Driver([spec('b13aa-mb32-seed1')], runs_dir=box['runs'], logs_dir=box['logs'], popen=popen,
                         call=subprocess.call, sleep=lambda s: None, python='py', ensure_workers=no_workers,
                         wave=1, killpg=killpg)
    assert d.run() == 0
    assert launches == ['stageb', 'hof5000', 'hof30k']
    assert groups == [(777, 15)], 'the dead predecessor\'s group was probed once and was already empty'


def test_a_scheduler_killed_mid_pass_leaves_the_pass_entry_for_its_successor(box):
    import subprocess
    _finished_arm_file(box, 'b13aa-mb32-seed1')

    def popen(argv, **kwargs):
        return FakeProcess(8001, polls=5)

    def sleep(seconds):
        raise SystemExit(143)                         # SIGTERM lands while waiting on the pass
    d = scheduler.Driver([spec('b13aa-mb32-seed1')], runs_dir=box['runs'], logs_dir=box['logs'], popen=popen,
                         call=subprocess.call, sleep=sleep, python='py', ensure_workers=no_workers, wave=1)
    with pytest.raises(SystemExit):
        d.run()
    assert live_runs.read(live_runs.path_for(live_runs.pass_entry('b13-stageb'), box['runs'])) == 8001
