"""`tools/window.py`: the scheduler's window. One per box, opened on a launch, closed by its owner.

Nothing here opens a real window: `Popen` is a stand-in, and what is pinned is the ownership rules that
replaced the arms' flock race -- open once, never a second while one is up, closed by hand stays closed
until the next launch, a reopen request replaces it, a stale predecessor is killed only if it is a viewer.
"""

import os
import subprocess

import pytest

from tools import live_runs
from tools import window


class FakeProcess(object):
    def __init__(self, pid):
        self.pid, self.returncode, self.signals = pid, None, []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.signals.append('TERM')
        self.returncode = -15

    def kill(self):
        self.signals.append('KILL')
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired('viewer', timeout)
        return self.returncode


class Spawner(object):
    def __init__(self):
        self.calls, self.pid = [], 500

    def __call__(self, argv, **kwargs):
        self.pid += 1
        self.calls.append((argv, kwargs))
        return FakeProcess(self.pid)


@pytest.fixture
def box(tmp_path):
    return str(tmp_path / 'runs')


def make(box, env=None, spawner=None):
    spawner = spawner or Spawner()
    w = window.Window(runs_dir=box, env={'SNEK_CHART_WINDOW': '1', **(env or {})}, python='py',
                      popen=spawner, log=lambda message: None)
    return w, spawner


def test_open_spawns_one_viewer_following_the_status_file(box):
    w, spawner = make(box)
    assert w.open() is True
    assert w.up() and w.pid() == 501
    argv, kwargs = spawner.calls[0]
    assert argv[:4] == ['py', '-u', '-m', 'tools.chart_viewer']
    assert argv[argv.index('--follow') + 1] == live_runs.status_path(box)
    assert kwargs['start_new_session'] is True, 'a signal to either side must not reach the other'


def test_a_second_open_while_one_is_up_does_nothing(box):
    """Every launch asks; only the first asks a window into being. The 5-window bug, inverted."""
    w, spawner = make(box)
    w.open()
    assert w.open() is False
    assert len(spawner.calls) == 1


def test_a_window_closed_by_hand_comes_back_only_at_the_next_launch(box):
    w, spawner = make(box)
    w.open()
    w.process.returncode = 0            # the user closed it
    assert not w.up() and w.pid() is None
    w.poll()
    assert len(spawner.calls) == 1, 'polling never reopens'
    assert w.open() is True, 'the next launch does'
    assert len(spawner.calls) == 2


def test_windows_off_opens_nothing(box):
    w, spawner = make(box, env={'SNEK_CHART_WINDOW': '0'})
    assert w.open() is False and spawner.calls == []


def test_close_terminates_and_escalates_to_kill(box):
    w, _ = make(box)
    w.open()
    process = w.process
    w.close()
    assert process.signals == ['TERM'] and not w.up()
    w.open()
    stubborn = w.process
    stubborn.terminate = lambda: stubborn.signals.append('TERM')      # ignores SIGTERM
    w.close()
    assert stubborn.signals == ['TERM', 'KILL']


def test_a_reopen_request_replaces_the_window_and_is_consumed(box):
    w, spawner = make(box)
    w.open()
    first = w.process
    window.request_reopen(box)
    assert w.poll() is True
    assert first.signals == ['TERM'] and w.pid() == 502
    assert not os.path.exists(live_runs.reopen_path(box)), 'the unlink is the test'
    assert w.poll() is False, 'and it fires once'


def test_a_reopen_request_with_no_window_up_opens_one(box):
    w, spawner = make(box)
    window.request_reopen(box)
    assert w.poll() is True and len(spawner.calls) == 1


def test_a_spawn_that_fails_is_not_an_error(box):
    def refuse(*args, **kwargs):
        raise OSError('no display')
    w = window.Window(runs_dir=box, env={'SNEK_CHART_WINDOW': '1'}, popen=refuse, log=lambda m: None)
    assert w.open() is False and not w.up()


def test_kill_stale_kills_only_a_viewer_and_never_itself(box):
    killed = []
    w = window.Window(runs_dir=box, env={}, kill=lambda pid, sig: killed.append((pid, sig)),
                      is_viewer=lambda pid: pid == 4242, log=lambda m: None)
    assert w.kill_stale(4242) is True and killed == [(4242, 15)]
    assert w.kill_stale(4243) is False, 'a recycled pid that is not a viewer is left alone'
    assert w.kill_stale(None) is False and w.kill_stale(os.getpid()) is False
    assert killed == [(4242, 15)]


def test_is_viewer_reads_the_pids_own_command_line_not_a_scan(monkeypatch):
    seen = []

    def run(argv, **kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=b'python -u -m tools.chart_viewer --follow x\n')
    assert window.is_viewer(77, run=run) is True
    assert seen[0][:3] == ['ps', '-o', 'command='] and seen[0][-1] == '77'

    def other(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=b'python -u train.py b1a\n')
    assert window.is_viewer(77, run=other) is False

    def gone(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout=b'')
    assert window.is_viewer(77, run=gone) is False


def test_sizing_reads_both_knobs_and_ignores_garbage(capsys):
    assert window.sizing({}) == (None, None)
    assert window.sizing({'SNEK_CHART_WINDOW_SCALE': '0.6', 'SNEK_CHART_WINDOW_MAX_PX': '1400'}) == (0.6, 1400)
    assert window.sizing({'SNEK_CHART_WINDOW_SCALE': 'big'}) == (None, None)
    assert 'not a number' in capsys.readouterr().err
    argv = window.command('s.json', env={'SNEK_CHART_WINDOW_MAX_PX': '1400'}, python='py')
    assert argv[argv.index('--max-width-px') + 1] == '1400'
    assert argv[argv.index('--scale') + 1] == '1.0'


def test_wanted_is_the_one_switch():
    for off in ('0', '', 'false', 'no'):
        assert not window.wanted({'SNEK_CHART_WINDOW': off})
    assert window.wanted({}) and window.wanted({'SNEK_CHART_WINDOW': '1'})
