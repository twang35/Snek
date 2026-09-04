"""The laptop batch driver: desktop specs, run here in waves, each followed by its stage B.

Every process call is a stand-in, so a test runs a 32-arm batch in milliseconds and asserts on what
would have been launched: the wave barrier, the stage-B naming the desktop uses, the skip of a
finished arm, the wait on an arm already live, and the trainer cap.
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
    """Records every launch and every stage-B call in order."""

    def __init__(self):
        self.events, self.pid = [], 1000

    def popen(self, argv, **kwargs):
        self.pid += 1
        self.events.append(('train', argv[-1], kwargs['env']))
        return FakeProcess(self.pid)

    def call(self, argv, **kwargs):
        arms = argv[argv.index('tools.closeout') + 1:argv.index('--shards')]
        self.events.append(('stageb', tuple(arms), argv[-1]))
        return 0


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
    assert kinds == (['train'] * 8 + ['stageb']) * 4
    first_stage_b = calls.events[8]
    assert first_stage_b[1] == tuple(s['policy'] for s in specs[:8])
    assert first_stage_b[2] == '8'
    assert len(os.listdir(box['logs'])) == 32 + 4          # one log per arm, one per stage-B wave
    assert os.path.exists(os.path.join(box['logs'], 'b13-stageb-w4.log'))


def test_a_finished_arm_is_skipped_and_its_wave_still_gets_stage_b(box):
    specs = [spec('b13aa-mb32-seed1'), spec('b13ab-mb32-seed2')]
    with open(os.path.join(box['runs'], 'b13aa-mb32-seed1_evals.json'), 'w') as handle:
        json.dump({'summary': {'step': 50003968}, 'evals': []}, handle)
    calls = Calls()
    driver(specs, box, calls, wave=2).run()
    assert [e[:2] for e in calls.events] == [('train', 'b13ab-mb32-seed2'),
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
    assert calls.events[-1][0] == 'stageb'


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
