"""The laptop batch driver: desktop specs, run here in waves, each followed by its stage B and the
two hof passes.

Every process call is a stand-in, so a test runs a 32-arm batch in milliseconds and asserts on what
would have been launched: the wave barrier, the pass naming the desktop uses, the chain stopping at
a failed pass, the skip of a finished arm, the wait on an arm already live, and the trainer cap.
"""

import json
import os

import pytest

from tools import laptop_batch
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
    def __init__(self, pid):
        self.pid = pid

    def wait(self):
        return 0


class Calls(object):
    """Records every launch and every close-out call in order, the close-outs by pass name."""

    def __init__(self, codes=None):
        self.events, self.pid, self.codes = [], 1000, codes or {}

    def popen(self, argv, **kwargs):
        self.pid += 1
        self.events.append(('train', argv[-1], kwargs['env']))
        return FakeProcess(self.pid)

    def call(self, argv, **kwargs):
        pass_name = argv[argv.index('--pass') + 1] if '--pass' in argv else 'stageb'
        end = argv.index('--pass') if '--pass' in argv else argv.index('--shards')
        arms = argv[argv.index('tools.closeout') + 1:end]
        self.events.append((pass_name, tuple(arms), argv[-1]))
        return self.codes.get(pass_name, 0)


PASSES = ['stageb', 'hof5000', 'hof30k']


@pytest.fixture
def box(tmp_path):
    runs = tmp_path / 'runs'
    runs.mkdir()
    logs = tmp_path / 'logs'
    return {'runs': str(runs), 'logs': str(logs)}


def driver(specs, box, calls, **kwargs):
    return laptop_batch.Driver(specs, runs_dir=box['runs'], logs_dir=box['logs'], popen=calls.popen,
                               call=calls.call, sleep=lambda s: None, python='py', **kwargs)


# ---------------------------------------------------------------- specs

def test_specs_load_in_id_order_from_files_and_directories_and_skip_evals(tmp_path):
    d = str(tmp_path / 'specs')
    write_specs(d, ['b13ab-mb32-seed2', 'b13aa-mb32-seed1'])
    with open(os.path.join(d, 'b13-stageb.json'), 'w') as handle:
        json.dump({'id': 'b13-stageb', 'type': 'eval', 'policies': ['b13aa-mb32-seed1']}, handle)
    loose = str(tmp_path / 'b13ac-mb32-seed3.json')
    with open(loose, 'w') as handle:
        json.dump(spec('b13ac-mb32-seed3'), handle)
    specs = laptop_batch.load_specs([d, loose])
    assert [s['id'] for s in specs] == ['b13aa-mb32-seed1', 'b13ab-mb32-seed2', 'b13ac-mb32-seed3']


def test_batch_id_and_stage_b_labels_match_the_daemon():
    specs = [spec('b13aa-mb32-seed1'), spec('b13bf-mb2048-seed4')]
    assert laptop_batch.batch_id(specs) == 'b13'
    assert laptop_batch.stage_b_label('b13', 1) == 'b13-stageb'
    assert laptop_batch.stage_b_label('b13', 3) == 'b13-stageb-w3'
    assert laptop_batch.pass_label('b13', 'hof5000', 1) == 'b13-hof5000'
    assert laptop_batch.pass_label('b13', 'hof30k', 2) == 'b13-hof30k-w2'
    assert laptop_batch.batch_id([spec('b13aa-x-seed1'), spec('b14a-y-seed1')]) == 'batch'


def test_training_env_is_the_spec_over_ours_plus_the_absolute_cap():
    env = laptop_batch.training_env(spec('b13aa-mb32-seed1'), base={'HOME': '/h', 'SNEK_PPO_MINIBATCH': '256'})
    assert env['SNEK_PPO_MINIBATCH'] == '32'
    assert env['SNEK_MAX_STEPS'] == '50003968'
    assert env['HOME'] == '/h'
    assert env['PYTHONPATH'] == laptop_batch.ROOT


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
    assert first_stage_b[2] == str(laptop_batch.DEFAULT_SHARDS)   # 12 since 2026-09-04
    assert len(os.listdir(box['logs'])) == 32 + 4 * 3      # one log per arm, one per pass per wave
    assert os.path.exists(os.path.join(box['logs'], 'b13-stageb-w4.log'))
    assert os.path.exists(os.path.join(box['logs'], 'b13-hof30k-w4.log'))


def test_a_failed_pass_stops_the_chain_for_that_wave(box):
    """hof5000 selects from the file stage B wrote, so with stage B failed it would fail too and
    bury the real failure. Same rule as the daemon's `next_pass`."""
    specs = [spec('b13aa-mb32-seed1'), spec('b13ab-mb32-seed2')]
    calls = Calls(codes={'stageb': 1})
    assert driver(specs, box, calls, wave=2).run() == 1
    assert [e[0] for e in calls.events] == ['train', 'train', 'stageb']
    calls = Calls(codes={'hof5000': 2})
    assert driver(specs, box, calls, wave=2).run() == 2
    assert [e[0] for e in calls.events] == ['train', 'train', 'stageb', 'hof5000']


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
    laptop_batch.Driver(specs, runs_dir=box['runs'], logs_dir=box['logs'], popen=Calls().popen,
                        call=call, sleep=lambda s: None, python='py', wave=1).run()
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
        return pid == os.getpid() and ticks['n'] < 4         # "finishes" after a few polls

    monkeypatch.setattr(live_runs, 'alive', alive)
    calls = Calls()
    driver(specs, box, calls, wave=2).run()
    assert [e[1] for e in calls.events if e[0] == 'train'] == ['b13ab-mb32-seed2']
    assert [e[0] for e in calls.events[-3:]] == PASSES


def test_no_launch_while_the_box_is_at_the_trainer_cap(box, monkeypatch):
    counts = iter([8, 8, 7])
    monkeypatch.setattr(laptop_batch, 'trainer_count', lambda runs_dir=None: next(counts))
    slept = []
    calls = Calls()
    d = laptop_batch.Driver([spec('b13aa-mb32-seed1')], runs_dir=box['runs'], logs_dir=box['logs'],
                            popen=calls.popen, call=calls.call, sleep=slept.append, python='py',
                            wave=1, stage_b=False)
    d.run()
    assert slept == [laptop_batch.POLL_SECONDS] * 2
    assert [e[0] for e in calls.events] == ['train']


def test_a_wave_wider_than_the_cap_is_refused():
    with pytest.raises(ValueError):
        laptop_batch.Driver([spec('b13aa-mb32-seed1')], wave=9, max_trainers=8)


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
    d = laptop_batch.Driver([spec('b14a-roll32-seed1')], runs_dir=box['runs'], logs_dir=box['logs'],
                            popen=calls.popen, call=calls.call, sleep=slept.append, python='py',
                            wave=1, stage_b=False)
    d.run(after=4242)
    assert slept == [laptop_batch.POLL_SECONDS] * 2      # one check logs the wait, two more poll it
    assert [e[0] for e in calls.events] == ['train']


# ---------------------------------------------------------------- passes already on disk

def _pass_files(runs, policies, *pass_names):
    for policy in policies:
        for pass_name in pass_names:
            open(laptop_batch.pass_file(policy, pass_name, runs), 'w').write('{"rows": []}')


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
    assert laptop_batch.run_queue(q, make) == 0
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
    laptop_batch.run_queue(q, lambda specs: driver(specs, box, calls, wave=8))
    assert [e[1] for e in calls.events if e[0] == 'train'] == ['b14a-roll-seed1', 'b16a-kl-seed1']


def test_a_batch_that_still_has_work_after_running_is_not_looped_on(tmp_path, box):
    """A failed pass leaves the batch pending forever; the queue must not spin on it."""
    q = _queue(tmp_path, {'b14': ['b14a-roll-seed1']})
    calls = Calls(codes={'stageb': 1})
    runs = []

    def make(specs):
        runs.append([s['policy'] for s in specs])
        return driver(specs, box, calls, wave=8)
    assert laptop_batch.run_queue(q, make) == 1
    assert [e[0] for e in calls.events] == ['train', 'stageb'], 'ran once, then left alone'


def test_a_finished_batch_left_in_the_queue_costs_nothing(tmp_path, box):
    q = _queue(tmp_path, {'b13': ['b13a-mb-seed1']})
    with open(os.path.join(box['runs'], 'b13a-mb-seed1_evals.json'), 'w') as handle:
        json.dump({'summary': {'step': 50003968}, 'evals': []}, handle)
    _pass_files(box['runs'], ['b13a-mb-seed1'], *PASSES)
    calls = Calls()
    assert laptop_batch.run_queue(q, lambda specs: driver(specs, box, calls, wave=8)) == 0
    assert calls.events == []


def test_queue_and_specs_are_exclusive_on_the_command_line(tmp_path):
    with pytest.raises(SystemExit):
        laptop_batch.main(['--queue', str(tmp_path), 'somefile.json'])
