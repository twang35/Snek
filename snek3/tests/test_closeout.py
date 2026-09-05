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


def test_the_close_out_starts_its_viewer_even_while_a_stage_b_window_is_up(monkeypatch, tmp_path):
    """The 2026-09-04 case: b15's window was still in its closing grace when b11's pass asked.

    `chart_window.ensure` would skip the spawn on the advisory pid and the pass would never ask again.
    The eval side always spawns, and the viewer waits for the slot (`chart_viewer.stand_by_for_slot`).
    """
    runs_dir = str(tmp_path / 'runs')
    lock = live_runs.window_lock_path(runs_dir, eval_window.SLOT_NAME)
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    with open(lock, 'w') as handle:
        handle.write('{0}\n'.format(os.getpid()))          # a live pid: us
    assert eval_window.holder(runs_dir) == os.getpid(), 'the fixture reads as a live window'
    spawned = []
    monkeypatch.setattr(chart_window, 'spawn', lambda *a, **k: spawned.append(a) or object())
    assert eval_window.ensure(['runs/a.png'], watch_pids=[1], env={}, runs_dir=runs_dir) is not None
    assert len(spawned) == 1


def test_a_hand_relaunched_window_watches_the_pid_it_is_given(monkeypatch, tmp_path, capsys):
    """`python -m tools.eval_window --watch-pid <close-out>` closes with the pass, like the window
    the close-out would have opened. Before 2026-09-04 the relaunch had no such flag and never closed."""
    monkeypatch.setattr(eval_window, 'holder', lambda *a, **k: None)
    spawned = {}
    monkeypatch.setattr(chart_window, 'spawn', lambda argv, label: spawned.setdefault('argv', argv))
    assert eval_window.main(['b1a', '--label', 'ab', '--watch-pid', '41,42']) == 0
    argv = spawned['argv']
    assert argv[argv.index('--watch-pid') + 1] == '41,42'
    assert 'stay up' not in capsys.readouterr().out


def test_a_hand_relaunched_window_without_a_pid_says_it_will_not_close(monkeypatch, capsys):
    monkeypatch.setattr(eval_window, 'holder', lambda *a, **k: None)
    monkeypatch.setattr(chart_window, 'spawn', lambda argv, label: object())
    assert eval_window.main([]) == 0
    assert 'stay up until it is closed by hand' in capsys.readouterr().out


# --- the passes, by name ---------------------------------------------------------------------------

def test_the_chain_is_three_passes_each_selecting_from_the_one_before():
    assert closeout.CHAIN == ('stageb', 'hof5000', 'hof30k')
    assert closeout.FOLLOW_ON == {'stageb': 'hof5000', 'hof5000': 'hof30k'}
    assert closeout.PASSES['hof5000']['selector'] == 'above:99'
    assert closeout.PASSES['hof30k']['selector'] == 'above:99:hof5000', 'reads the hof5000 file'


def test_stage_b_is_the_default_pass_and_the_close_outs_own_defaults():
    """A command that names no pass is unchanged: `screen:97` at 500, unlabelled, seed 0."""
    assert closeout.pass_settings('stageb') == {'selector': 'screen', 'episodes': 500,
                                                'label': None, 'seed': 0}
    args = closeout.build_parser().parse_args(['b1a'])
    assert args.pass_name == 'stageb'


def test_a_hof_pass_is_labelled_so_it_never_overwrites_what_it_selected_from():
    """The output path is `runs/<arm>_checkpoint_evals[_<label>].json`; unlabelled, the 5,000-episode
    rows would replace the 500-episode file `above:99` reads. The hof-remeasure skill calls omitting
    the label 'destroying the input'; the preset makes it impossible to omit."""
    assert closeout.pass_settings('hof5000') == {'selector': 'above:99', 'episodes': 5000,
                                                 'label': 'hof5000', 'seed': 0}
    assert closeout.pass_settings('hof30k') == {'selector': 'above:99:hof5000', 'episodes': 30000,
                                                'label': 'hof30k', 'seed': 7}


def test_an_explicit_flag_wins_over_the_preset_but_none_never_unsets_it():
    chosen = closeout.pass_settings('hof5000', episodes=2000, seed=3)
    assert (chosen['episodes'], chosen['seed'], chosen['label']) == (2000, 3, 'hof5000')
    assert closeout.pass_settings('hof5000', label=None)['label'] == 'hof5000'
    with pytest.raises(ValueError):
        closeout.pass_settings('hof9000')


def test_main_hands_the_pass_to_run(monkeypatch):
    seen = {}

    def run(policies, selector, episodes, shards, label, width, seed, resume, merge, window):
        seen.update(policies=policies, selector=selector, episodes=episodes, shards=shards,
                    label=label, seed=seed)
        return 0
    monkeypatch.setattr(closeout, 'run', run)
    assert closeout.main(['b1a', 'b1b', '--pass', 'hof30k', '--shards', '12']) == 0
    assert seen == {'policies': ['b1a', 'b1b'], 'selector': 'above:99:hof5000', 'episodes': 30000,
                    'shards': 12, 'label': 'hof30k', 'seed': 7}
    closeout.main(['b1a'])
    assert (seen['selector'], seen['episodes'], seen['label'], seen['seed']) == ('screen', 500, None, 0)
