"""The close-out: one process per batch, both boxes, and the window over it.

`tools/closeout.py` is what the desktop daemon dispatches and what an agent types on the laptop, so
these fixtures are about the parts that only show up when something goes wrong — a failing arm, a
raising arm, a window that cannot open — which is exactly where the two hand-written sequencers it
replaced differed from each other.
"""

import os

import pytest

from tools import chart_window
from tools import closeout
from tools import eval_window
from tools import live_runs


class Waves(object):
    """Stands in for `eval_wave.run`, recording the calls and returning canned statuses."""

    def __init__(self, codes=None):
        self.codes, self.calls = codes or {}, []

    def __call__(self, policy, *args, **kwargs):
        self.calls.append(policy)
        code = self.codes.get(policy, 0)
        if isinstance(code, Exception):
            raise code
        return code


@pytest.fixture
def waves(monkeypatch):
    def install(codes=None):
        stub = Waves(codes)
        monkeypatch.setattr(closeout.eval_wave, 'run', stub)
        monkeypatch.setattr(closeout.stage_b_chart, 'redraw', lambda *a, **k: (None, []))
        return stub
    return install


@pytest.fixture
def windows(monkeypatch):
    """Records what `eval_window.ensure` was asked for, without opening anything."""
    asked = []
    monkeypatch.setattr(closeout.eval_window, 'ensure',
                        lambda *args, **kwargs: asked.append((args, kwargs)))
    return asked


def test_every_arm_is_measured_in_the_order_the_batch_named_them(waves, windows):
    stub = waves()
    assert closeout.run(['b1a', 'b1b', 'b1c'], shards=8) == 0
    assert stub.calls == ['b1a', 'b1b', 'b1c']


def test_one_arm_failing_does_not_drop_the_arms_behind_it(waves, windows):
    """The `;`-not-`&&` rule of the shell chain this replaced. The incident is snek2's."""
    stub = waves({'b1a': 1})
    assert closeout.run(['b1a', 'b1b', 'b1c']) == 1
    assert stub.calls == ['b1a', 'b1b', 'b1c'], 'a bad first arm must not end the batch'


def test_an_arm_that_raises_does_not_drop_the_arms_behind_it_either(waves, windows):
    """A restore failure is one bad arm, not a bad batch — and it used to kill the whole pass."""
    stub = waves({'b1b': RuntimeError('no arch.json')})
    assert closeout.run(['b1a', 'b1b', 'b1c']) == 1
    assert stub.calls == ['b1a', 'b1b', 'b1c']


def test_the_status_is_a_failure_even_when_the_last_arm_succeeds(waves, windows):
    waves({'b1a': 2})
    assert closeout.run(['b1a', 'b1b']) != 0


def test_the_protocol_defaults_are_the_close_outs_own(waves, windows):
    """The daemon passes a selector and an episode count only if a spec named them."""
    seen = {}
    def record(policy, selector, episodes, shards, *args, **kwargs):
        seen.update(selector=selector, episodes=episodes, shards=shards)
        return 0
    waves()
    closeout.eval_wave.run = record
    closeout.run(['b1a'])
    assert seen == {'selector': 'screen', 'episodes': 500, 'shards': 4}


# --- the window ------------------------------------------------------------------------------------

def test_the_window_is_asked_for_once_with_a_panel_per_arm(waves, windows):
    closeout.run(['b1a', 'b1b'], label='ab')
    assert len(windows) == 1, 'one window for the batch, not one per arm'
    (paths,), kwargs = windows[0]
    assert paths == eval_window.chart_paths(['b1a', 'b1b'], 'ab')
    assert kwargs['watch_pids'] == [os.getpid()], 'so it closes when the pass ends'


def test_no_window_is_asked_for_when_the_caller_says_not_to(waves, windows):
    closeout.run(['b1a'], window=False)
    assert windows == []


def test_a_single_checkpoint_asks_for_no_window(waves, windows, monkeypatch):
    """`one` writes no pass, so there would be nothing to draw."""
    monkeypatch.setattr(closeout, 'measure_one', lambda *a, **k: None)
    closeout.run(['b1a'], selector='one')
    assert windows == []


def test_a_single_checkpoint_runs_no_wave(waves, windows, monkeypatch):
    stub = waves()
    monkeypatch.setattr(closeout, 'measure_one', lambda *a, **k: None)
    assert closeout.run(['b1a'], selector='one', episodes=3000) == 0
    assert stub.calls == [], 'the `one` selector is measured in this process, not by a wave'


def test_the_eval_window_takes_a_different_slot_from_the_training_window():
    """A box can hold both at once — nothing separates a training from an eval on the laptop."""
    assert eval_window.SLOT_NAME != live_runs.WINDOW_LOCK_NAME
    assert (live_runs.window_lock_path(None, eval_window.SLOT_NAME)
            != live_runs.window_lock_path(None))


def test_the_window_command_names_its_slot_its_panels_and_the_pid_it_watches():
    argv = eval_window.command(['runs/a.png', 'runs/b.png'], watch_pids=[41, 42])
    assert argv[argv.index('--slot') + 1] == eval_window.SLOT_NAME
    assert argv[argv.index('--watch-pid') + 1] == '41,42'
    assert argv[3] == 'tools.chart_viewer', 'the same viewer draws both windows'
    assert 'runs/a.png' in argv and 'runs/b.png' in argv


def test_a_chart_path_is_where_the_stage_b_pass_is_drawn():
    from tools import stage_b_chart
    assert eval_window.chart_paths(['b1a'], 'ab') == [stage_b_chart.chart_path('b1a', 'ab')]


def test_windows_are_one_switch_and_not_two(monkeypatch):
    """`SNEK_CHART_WINDOW=0` means "no window on this box", which is one decision."""
    calls = []
    monkeypatch.setattr(chart_window, 'spawn', lambda *a, **k: calls.append(a))
    assert eval_window.ensure(['runs/a.png'], env={'SNEK_CHART_WINDOW': '0'}) is None
    assert calls == []


def test_a_window_that_will_not_open_is_not_an_error(monkeypatch, capsys, tmp_path):
    """A box with no display must not fail a measurement over a chart.

    `runs_dir` is a tmp path deliberately: against the real registry this passed alone and failed in
    the full suite, because a *live* stage-B window on the box holds the slot, `ensure` skips the
    spawn it would obviously lose, and nothing reaches the failure path being tested.
    """
    def refuse(*args, **kwargs):
        raise OSError('no display')
    monkeypatch.setattr(chart_window.subprocess, 'Popen', refuse)
    assert eval_window.ensure(['runs/a.png'], env={}, runs_dir=str(tmp_path)) is None
    assert 'stage-B window' in capsys.readouterr().err


# --- the redraw ------------------------------------------------------------------------------------

def test_the_drawer_redraws_once_more_after_the_wave_finishes():
    """The last word is the merged file's, not the pooled shard files' a mid-pass redraw saw."""
    drawn = []
    drawer = closeout.Drawer('b1a', interval=60.0)
    drawer.draw_once = lambda: drawn.append('b1a')
    with drawer:
        pass
    assert drawn == ['b1a'], 'no periodic redraw fired in the interval, and exactly one at the end'


def test_the_drawer_thread_is_a_daemon_so_it_cannot_outlive_the_pass():
    """A `stage_b_chart --watch` subprocess could be orphaned; a daemon thread cannot."""
    drawer = closeout.Drawer('b1a', interval=60.0)
    drawer.draw_once = lambda: None
    with drawer:
        assert drawer.thread.daemon and drawer.thread.is_alive()
    assert not drawer.thread.is_alive()


def test_a_redraw_failure_is_swallowed(monkeypatch, capsys):
    """A chart is a readout. A pass that died because a PNG could not be written would be backwards.

    The redraw is made to raise rather than merely fail: an arm with no rows yet returns `(None, [])`
    quite legitimately, so a fixture built on a missing policy would pass with the `except` deleted.
    """
    def explode(*args, **kwargs):
        raise IOError('disk full')
    monkeypatch.setattr(closeout.stage_b_chart, 'redraw', explode)
    assert closeout.Drawer('b1a').draw_once() is None
    assert 'chart redraw failed for b1a' in capsys.readouterr().err


def test_a_redraw_failure_does_not_fail_the_arm(monkeypatch, waves, windows):
    def explode(*args, **kwargs):
        raise IOError('disk full')
    stub = waves()
    monkeypatch.setattr(closeout.stage_b_chart, 'redraw', explode)
    assert closeout.run(['b1a', 'b1b']) == 0
    assert stub.calls == ['b1a', 'b1b']


def test_the_two_slots_do_not_exclude_each_other(tmp_path):
    """The kernel enforces it, so this asks the kernel.

    Two `flock`s on two different files coexist; two on the same one do not, even from one process,
    because the lock belongs to the open file description. That pair is the whole property: a
    stage-B window opens beside a training window, and a *second* stage-B window does not.
    """
    from tools import chart_viewer
    runs = str(tmp_path)
    assert chart_viewer.take_window_slot(runs) is not None, 'the training slot was free'
    assert chart_viewer.take_window_slot(runs, eval_window.SLOT_NAME) is not None, \
        'and the eval slot is a different lock'
    assert chart_viewer.take_window_slot(runs, eval_window.SLOT_NAME) is None, \
        'but only one window per slot'
