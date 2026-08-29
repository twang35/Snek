"""Fixtures for the box's one chart window: the live-arm registry, the claim, and the argv.

Nothing here opens a window. What is pinned is the machinery that decides *whether* to open one and
what it will show — plus the properties that make the window disposable, which are the reason this
design replaced a per-arm window: a separate session, a `poll` rather than a `wait`, and no path by
which a dead window can affect a run.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from tools import chart_viewer
from tools import chart_window
from tools import live_runs

DEAD_PID = 999_991


def runs(tmp_path):
    """A runs directory, which is what the registry and the panels both hang off."""
    return str(tmp_path / 'runs')


def chart(tmp_path, policy):
    directory = runs(tmp_path)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, policy + '.png')
    with open(path, 'wb') as handle:
        handle.write(b'not really a png')
    return path


# --- the registry ---------------------------------------------------------------------------------


def test_an_arm_registers_itself_and_is_live(tmp_path):
    live_runs.register('b3a-arm', runs_dir=runs(tmp_path))
    assert live_runs.live(runs(tmp_path)) == [('b3a-arm', os.getpid())]


def test_an_arm_whose_process_is_gone_is_not_live(tmp_path):
    """The `kill -9` case, and the reason no `finally` is needed to clean up after a trainer."""
    live_runs.register('b3a-arm', pid=DEAD_PID, runs_dir=runs(tmp_path))
    assert live_runs.live(runs(tmp_path)) == []


def test_a_dead_entry_is_deleted_rather_than_left_to_accumulate(tmp_path):
    live_runs.register('b3a-arm', pid=DEAD_PID, runs_dir=runs(tmp_path))
    live_runs.live(runs(tmp_path))
    assert not os.path.exists(live_runs.path_for('b3a-arm', runs_dir=runs(tmp_path)))


def test_pruning_can_be_turned_off_for_a_read_only_look(tmp_path):
    live_runs.register('b3a-arm', pid=DEAD_PID, runs_dir=runs(tmp_path))
    assert live_runs.live(runs(tmp_path), prune=False) == []
    assert os.path.exists(live_runs.path_for('b3a-arm', runs_dir=runs(tmp_path)))


def test_arms_come_back_in_a_stable_order(tmp_path, monkeypatch):
    """The window rebuilds its figure when the panel *set* changes; a permuted list must not.

    `os.listdir` is stubbed rather than trusted: on this filesystem it happened to return three
    entries already sorted, so a fixture that only registered them out of order passed with the
    `sorted()` removed — a mutation survivor, and the same class of hole as a fixture whose subject
    cannot violate it.
    """
    for policy in ('b3c-arm', 'b3a-arm', 'b3b-arm'):
        live_runs.register(policy, runs_dir=runs(tmp_path))
    monkeypatch.setattr(live_runs.os, 'listdir', lambda _path: ['b3c-arm', 'b3a-arm', 'b3b-arm'])
    assert [policy for policy, _ in live_runs.live(runs(tmp_path))] == ['b3a-arm', 'b3b-arm',
                                                                        'b3c-arm']


def test_no_registry_directory_is_empty_rather_than_an_error(tmp_path):
    assert live_runs.live(os.path.join(str(tmp_path), 'no-runs-dir')) == []


def test_unregistering_something_that_was_never_there_is_fine(tmp_path):
    live_runs.unregister('never-ran', runs_dir=runs(tmp_path))


def test_an_unreadable_entry_costs_one_panel_and_never_the_window(tmp_path):
    live_runs.register('b3a-arm', runs_dir=runs(tmp_path))
    with open(live_runs.path_for('b3a-arm', runs_dir=runs(tmp_path)), 'w') as handle:
        handle.write('not a pid\n')
    assert live_runs.live(runs(tmp_path)) == []


def test_a_live_pid_reads_as_alive_and_a_dead_one_does_not():
    assert live_runs.alive(os.getpid()) is True
    assert live_runs.alive(DEAD_PID) is False
    assert live_runs.alive(0) is False


def test_pid_1_belongs_to_another_user_and_still_counts_as_alive():
    """`os.kill` raises `PermissionError` there. Existing is what the question asks."""
    assert live_runs.alive(1) is True


def test_an_arm_with_no_chart_yet_contributes_no_panel(tmp_path):
    """It is in the registry from its first second and in the window from its first report."""
    live_runs.register('b3a-arm', runs_dir=runs(tmp_path))
    assert live_runs.chart_paths(runs(tmp_path)) == []
    path = chart(tmp_path, 'b3a-arm')
    assert live_runs.chart_paths(runs(tmp_path)) == [path]


# --- who opens the window -------------------------------------------------------------------------


def test_the_window_is_wanted_by_default():
    assert chart_window.wanted({}) is True


@pytest.mark.parametrize('value', ['0', '', 'false', 'False', 'no'])
def test_the_window_can_be_switched_off(value):
    assert chart_window.wanted({'SNEK_CHART_WINDOW': value}) is False


def test_the_first_arm_claims_the_slot_and_the_next_one_does_not(tmp_path):
    assert chart_window.claim(runs(tmp_path)) is True
    assert chart_window.claim(runs(tmp_path)) is False


def test_a_dead_windows_slot_is_claimable_again(tmp_path):
    """Otherwise closing the window once would mean the box never drew another."""
    chart_window.hold(runs(tmp_path), DEAD_PID)
    assert chart_window.holder(runs(tmp_path)) is None
    assert chart_window.claim(runs(tmp_path)) is True


def test_the_slot_records_the_window_and_not_the_trainer_that_opened_it(tmp_path, monkeypatch):
    """A trainer that finishes while other arms run must not look like a window that died."""
    monkeypatch.setattr(chart_window, 'spawn', lambda *a, **k: FakeWindow(4242))
    chart_window.ensure(runs(tmp_path), env={})
    assert live_runs.read(chart_window.lock_path(runs(tmp_path))) == 4242


def test_only_one_arm_of_a_wave_opens_a_window(tmp_path, monkeypatch):
    opened = []

    def spawn(*_args, **_kwargs):
        opened.append(1)
        return FakeWindow(os.getpid())

    monkeypatch.setattr(chart_window, 'spawn', spawn)
    handles = [chart_window.ensure(runs(tmp_path), env={}) for _ in range(4)]
    assert len(opened) == 1
    assert [handle is None for handle in handles] == [False, True, True, True]


def test_switching_the_window_off_spawns_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(chart_window, 'spawn', lambda *a, **k: pytest.fail('spawned anyway'))
    assert chart_window.ensure(runs(tmp_path), env={'SNEK_CHART_WINDOW': '0'}) is None


def test_a_window_that_will_not_start_is_not_an_error(tmp_path, monkeypatch):
    """No display, no Tk, a dead `DISPLAY`: a training must not care."""
    monkeypatch.setattr(chart_window, 'spawn', lambda *a, **k: None)
    assert chart_window.ensure(runs(tmp_path), env={}) is None


# --- the argv and the process -------------------------------------------------------------------


class FakeWindow(object):
    def __init__(self, pid, alive=True):
        self.pid = pid
        self._alive = alive
        self.waited = False

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        self.waited = True
        raise AssertionError('the trainer must never wait on the window')


def test_the_window_is_told_to_read_the_registry_rather_than_a_fixed_list():
    """Which is what lets an arm that starts later appear in a window already up."""
    argv = chart_window.command('/box/snek3/runs')
    assert '--runs-dir' in argv and argv[argv.index('--runs-dir') + 1] == '/box/snek3/runs'
    assert '--glob' not in argv


def test_the_window_refreshes_slower_than_a_report_is_written():
    argv = chart_window.command()
    assert float(argv[argv.index('--interval') + 1]) == chart_window.REFRESH_SECONDS
    assert chart_window.REFRESH_SECONDS >= 10


def test_the_one_window_fills_the_screen():
    argv = chart_window.command()
    assert float(argv[argv.index('--scale') + 1]) == 1.0


def test_the_scale_can_be_overridden_from_the_environment(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(chart_window.subprocess, 'Popen',
                        lambda argv, **kwargs: seen.update(argv=argv, kwargs=kwargs) or FakeWindow(7))
    chart_window.spawn(runs(tmp_path), env={'SNEK_CHART_WINDOW_SCALE': '0.5'})
    assert seen['argv'][seen['argv'].index('--scale') + 1] == '0.5'


def test_the_window_runs_in_its_own_session(tmp_path, monkeypatch):
    """The load-bearing property. Without it a `kill` to a trainer's group takes the window, and a
    Ctrl-C in the terminal that started the trainer takes both."""
    seen = {}
    monkeypatch.setattr(chart_window.subprocess, 'Popen',
                        lambda argv, **kwargs: seen.update(kwargs) or FakeWindow(7))
    chart_window.spawn(runs(tmp_path), env={})
    assert seen['start_new_session'] is True
    assert seen['stdin'] is subprocess.DEVNULL


def test_reaping_polls_and_never_blocks():
    window = FakeWindow(7)
    assert chart_window.reap(window) is True
    assert chart_window.reap(FakeWindow(7, alive=False)) is False
    assert window.waited is False


def test_reaping_nothing_is_not_an_error():
    assert chart_window.reap(None) is False


# --- when the window closes -----------------------------------------------------------------------


def test_a_window_stays_up_while_an_arm_runs():
    seen, since, close = chart_viewer.idle_close([('b3a', 1)], False, None, 100.0)
    assert (seen, since, close) == (True, None, False)


def test_a_window_opened_before_the_batch_never_closes_on_its_own():
    """The trap. An empty registry at startup means "not launched yet", not "the box is idle"."""
    seen, since, close = chart_viewer.idle_close([], False, None, 100.0)
    assert (seen, since, close) == (False, None, False)
    for now in (100.0, 100.0 + 10 * chart_viewer.IDLE_CLOSE_SECONDS):
        assert chart_viewer.idle_close([], seen, since, now)[2] is False


def test_a_window_survives_the_gap_between_two_waves():
    _, since, close = chart_viewer.idle_close([], True, None, 100.0)
    assert close is False
    assert chart_viewer.idle_close([], True, since, 100.0 + 30.0)[2] is False


def test_a_window_closes_once_the_box_has_stopped_training():
    _, since, _ = chart_viewer.idle_close([], True, None, 100.0)
    late = 100.0 + chart_viewer.IDLE_CLOSE_SECONDS
    assert chart_viewer.idle_close([], True, since, late)[2] is True


def test_the_idle_clock_restarts_when_a_new_arm_appears():
    _, since, _ = chart_viewer.idle_close([], True, None, 100.0)
    _, restarted, _ = chart_viewer.idle_close([('b3b', 2)], True, since, 200.0)
    assert restarted is None
    late = 200.0 + chart_viewer.IDLE_CLOSE_SECONDS - 1.0
    assert chart_viewer.idle_close([], True, restarted, late)[2] is False


def test_the_close_grace_is_minutes_not_seconds():
    assert chart_viewer.IDLE_CLOSE_SECONDS >= 60


def test_live_panels_reports_the_arms_and_the_charts_separately(tmp_path):
    """Two answers, because an arm that has not written a chart yet must not read as "no training"."""
    live_runs.register('b3a-arm', runs_dir=runs(tmp_path))
    arms, paths = chart_viewer.live_panels(runs(tmp_path))
    assert arms == [('b3a-arm', os.getpid())] and paths == []
    chart(tmp_path, 'b3a-arm')
    assert chart_viewer.live_panels(runs(tmp_path))[1] == [os.path.join(runs(tmp_path),
                                                                       'b3a-arm.png')]


def test_a_killed_window_does_not_hold_the_slot_as_a_zombie(tmp_path, monkeypatch):
    """The window is a trainer's child, and a trainer reaps on its report cadence. Between the kill
    and the reap the pid still answers `os.kill(pid, 0)`, so a hand relaunch was refused for a window
    that was visibly gone."""
    chart_window.hold(runs(tmp_path), os.getpid())
    monkeypatch.setattr(live_runs, 'zombie', lambda pid: True)
    assert chart_window.holder(runs(tmp_path)) is None
    assert chart_window.claim(runs(tmp_path)) is True


def test_an_arms_own_entry_does_not_pay_for_the_zombie_check(tmp_path, monkeypatch):
    """One `ps` per refresh per arm, to answer a question a trainer's parent settles anyway."""
    monkeypatch.setattr(live_runs, 'zombie', lambda pid: pytest.fail('scanned for an arm'))
    live_runs.register('b3a-arm', runs_dir=runs(tmp_path))
    assert live_runs.live(runs(tmp_path)) == [('b3a-arm', os.getpid())]


def test_a_ps_that_cannot_answer_is_not_a_zombie(monkeypatch):
    """Failing this way refuses to open a second window; the other way opens one over a live one."""
    def raises(*_args, **_kwargs):
        raise OSError('no ps')

    monkeypatch.setattr(live_runs.subprocess, 'run', raises)
    assert live_runs.zombie(os.getpid()) is False


def test_a_live_process_is_not_a_zombie():
    assert live_runs.zombie(os.getpid()) is False


# --- a wave is read as a wave -----------------------------------------------------------------


def test_an_arm_that_finishes_keeps_its_panel():
    """Three arms done and one still going is a batch worth glancing at, not a single chart."""
    known = chart_viewer.wave_panels([], ['runs/b3a.png', 'runs/b3b.png'], new_wave=False)
    assert chart_viewer.wave_panels(known, ['runs/b3b.png'], new_wave=False) == known


def test_an_arm_that_starts_later_joins_without_displacing_anyone():
    known = chart_viewer.wave_panels([], ['runs/b3a.png'], new_wave=False)
    assert chart_viewer.wave_panels(known, ['runs/b3a.png', 'runs/b3b.png'], new_wave=False) == [
        'runs/b3a.png', 'runs/b3b.png']


def test_a_panel_is_not_added_twice():
    known = ['runs/b3a.png']
    assert chart_viewer.wave_panels(known, ['runs/b3a.png'], new_wave=False) == known


def test_the_next_wave_starts_from_nothing():
    """Otherwise a batch launched inside the idle grace draws its predecessor's charts too — snek2
    opened a window with eight panels for four arms this way."""
    known = chart_viewer.wave_panels([], ['runs/b3a.png', 'runs/b3b.png'], new_wave=False)
    assert chart_viewer.wave_panels(known, ['runs/b4a.png'], new_wave=True) == ['runs/b4a.png']


def test_the_finished_wave_stays_up_until_the_window_closes():
    """The registry is empty for the whole idle grace. The last thing a batch shows is all of its
    arms, not none of them."""
    known = chart_viewer.wave_panels([], ['runs/b3a.png', 'runs/b3b.png'], new_wave=False)
    assert chart_viewer.wave_panels(known, [], new_wave=False) == known


# --- the loop that ties it together ---------------------------------------------------------------
#
# `run()` is a `while True` around a live figure, so a fixture cannot call it. These two drive the
# real loop in a subprocess on the Agg backend with a fake clock, because the wiring is where the
# sticky rule can be right in `wave_panels` and wrong in the caller — a `new_wave` that forgot to
# check for arms drops the finished wave's panels during the idle grace, and nothing above notices.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROGRAM = """
import os, sys
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, {root!r})
from tools import chart_viewer, live_runs

runs_dir = {runs_dir!r}
chart_viewer.IDLE_CLOSE_SECONDS = {idle}
for arm in ('armA', 'armB'):
    live_runs.register(arm, pid=os.getpid(), runs_dir=runs_dir)

clock = {{'t': 0.0}}
original = chart_viewer.Viewer.refresh


def spy(self, paths):
    print('t={{0:.0f}} {{1}}'.format(clock['t'], sorted(os.path.basename(p) for p in paths)),
          flush=True)
    return original(self, paths)


chart_viewer.Viewer.refresh = spy


def fake_now():
    clock['t'] += 1.0
    {script}
    return clock['t']


chart_viewer.run([], interval=0.01, runs_dir=runs_dir, now=fake_now, dpi=60)
"""


def run_loop(tmp_path, script, idle=3, arms=('armA', 'armB')):
    """Runs the real loop until it closes itself, and returns its panel lines."""
    for arm in arms:
        chart(tmp_path, arm)
    chart(tmp_path, 'armC')
    program = PROGRAM.format(root=ROOT, runs_dir=runs(tmp_path), idle=idle,
                             script=textwrap.indent(textwrap.dedent(script), ' ' * 4).strip())
    result = subprocess.run([sys.executable, '-c', program], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=120)
    output = result.stdout.decode('utf-8', 'replace')
    assert result.returncode == 0, output
    return [line for line in output.splitlines() if line.startswith('t=')], output


def test_the_loop_keeps_a_finished_arms_panel_until_the_window_closes(tmp_path):
    """One arm left of two, and the window still shows both — then keeps both while it idles out."""
    lines, output = run_loop(tmp_path, """
        if clock['t'] == 2.0:
            live_runs.unregister('armA', runs_dir=runs_dir)
        if clock['t'] == 4.0:
            live_runs.unregister('armB', runs_dir=runs_dir)
    """)
    assert lines, output
    assert all("['armA.png', 'armB.png']" in line for line in lines), output
    assert 'no training running' in output, output


def test_the_loop_starts_the_next_wave_from_nothing(tmp_path):
    """A batch launched inside the idle grace must not inherit its predecessor's panels."""
    lines, output = run_loop(tmp_path, """
        if clock['t'] == 2.0:
            for arm in ('armA', 'armB'):
                live_runs.unregister(arm, runs_dir=runs_dir)
        if clock['t'] == 4.0:
            live_runs.register('armC', pid=os.getpid(), runs_dir=runs_dir)
        if clock['t'] == 6.0:
            live_runs.unregister('armC', runs_dir=runs_dir)
    """, idle=2)
    assert lines[-1].endswith("['armC.png']"), output
    assert "['armA.png', 'armB.png', 'armC.png']" not in output, output
