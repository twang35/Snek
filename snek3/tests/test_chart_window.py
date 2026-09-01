"""Fixtures for the box's one chart window: the live-arm registry, the slot, and the argv.

Nothing here opens a window. What is pinned is the machinery that decides *whether* to open one and
what it will show — plus the properties that make the window disposable, which are the reason this
design replaced a per-arm window: a separate session, a `poll` rather than a `wait`, and no path by
which a dead window can affect a run.

**The one-window fixtures fork real processes, and that is not gold-plating.** The version of this
file that shipped the five-window bug asserted the guarantee by calling `ensure()` four times in a
row in one process, which cannot observe a race and passed against an implementation that opened a
mean of 6.6 windows per 8-arm batch. A lock between processes has to be tested between processes.
"""

import os
import subprocess
import sys
import textwrap
import time

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


def record_window(tmp_path, pid):
    """Writes `pid` into the slot file without holding the lock, as a dead window leaves it."""
    path = live_runs.window_lock_path(runs(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        handle.write('{0}\n'.format(pid))
    return path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# One racer: block on stdin, take the slot when the parent fires, exit 0 for won and 1 for lost.
RACER = """
import sys, time
sys.path.insert(0, {root!r})
from tools import chart_viewer
sys.stdin.read(1)
won = chart_viewer.take_window_slot({runs!r}) is not None
if won:
    time.sleep({hold!r})
sys.exit(0 if won else 1)
"""


def race_for_the_slot(runs_dir, racers=8, hold_seconds=0.5):
    """Runs `racers` real processes at `take_window_slot` at once. Returns how many won.

    Separate interpreters, not `os.fork` and not threads. An `flock` belongs to an **open file
    description**, so two threads of one process share it and would both "win" against any
    implementation; the test has to be as separate as the arms are. Fork was the first version and
    Python 3.12 is right to warn about it — this suite imports torch and matplotlib, and forking a
    multi-threaded process can deadlock the child on a lock some other thread held.

    stdin is the starting gun. Without it the racers are created one at a time and the first would
    win without ever contending, which is precisely the mistake the fixture exists to correct.
    """
    source = RACER.format(root=ROOT, runs=runs_dir, hold=hold_seconds)
    started = [subprocess.Popen([sys.executable, '-c', source], stdin=subprocess.PIPE)
               for _ in range(racers)]
    for process in started:                     # every racer is now blocked on its own stdin
        process.stdin.write(b'x')
    for process in started:                     # ... and released as close to together as we can
        process.stdin.close()
    codes = [process.wait(timeout=60) for process in started]
    assert set(codes) <= {0, 1}, 'a racer raised: {0}'.format(codes)
    return codes.count(0)


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


def test_exactly_one_of_eight_windows_launched_together_takes_the_slot(tmp_path):
    """**The regression fixture for the desktop's five windows, 2026-08-29.**

    Eight arms start in the same second and all eight ask for the window. The predecessor of this
    design answered a mean of 6.6, because its takeover path wrote a pid and read it back — which
    every racer can win. One, at eight-way contention, is the whole requirement.
    """
    assert race_for_the_slot(runs(tmp_path), racers=8) == 1


def test_the_slot_is_never_read_as_an_arm(tmp_path):
    """It lives *inside* the registry directory, so it has to be invisible to `live()` — a slot named
    without its leading dot gives the window a phantom panel named after its own lock file, and
    snek2's registry drew eight panels for four arms by admitting things that were not arms."""
    record_window(tmp_path, os.getpid())
    live_runs.register('b3a-arm', runs_dir=runs(tmp_path))
    assert live_runs.live(runs(tmp_path)) == [('b3a-arm', os.getpid())]


def test_a_slot_file_left_by_a_dead_window_is_not_a_stale_lock(tmp_path):
    """Nothing deletes the file, so **every batch after the first meets one** — and the pid in it is
    dead. That was one of the two ways into the old race; here it is not a case at all, because the
    lock is the kernel's and the file is only a place to publish a pid."""
    record_window(tmp_path, DEAD_PID)
    assert chart_viewer.take_window_slot(runs(tmp_path)) is not None


def test_a_killed_window_releases_the_slot_at_once_and_the_next_one_gets_it(tmp_path):
    """Three properties in one fixture, because they are the same property.

    A live holder turns a second window away; the slot records the *window's* own pid rather than
    that of the trainer that spawned it, so a finished trainer never looks like a dead window; and a
    `kill -9` — no unwinding, no handler, no cleanup path — releases the slot immediately.
    """
    lock = live_runs.window_lock_path(runs(tmp_path))
    source = RACER.format(root=ROOT, runs=runs(tmp_path), hold=30.0)
    window = subprocess.Popen([sys.executable, '-c', source], stdin=subprocess.PIPE)
    try:
        window.stdin.write(b'x')
        window.stdin.close()
        deadline = time.time() + 10
        while live_runs.read(lock) != window.pid and time.time() < deadline:
            time.sleep(0.05)
        assert live_runs.read(lock) == window.pid, 'the window did not publish its pid'
        assert chart_viewer.take_window_slot(runs(tmp_path)) is None
    finally:
        window.kill()
        window.wait(timeout=30)
    assert chart_viewer.take_window_slot(runs(tmp_path)) is not None


def test_every_arm_of_a_wave_may_ask_for_the_window(tmp_path, monkeypatch):
    """The launcher gates nothing, and that is the design rather than an oversight.

    `ensure` is called by every arm and holds no lock, so four simultaneous arms may start four
    viewers; the three that lose the slot exit in ~0.3 s having imported matplotlib and drawn
    nothing. Paying that once per batch is what buys a launcher with no protocol in it — and the
    protocol is what was broken.
    """
    monkeypatch.setattr(chart_window, 'spawn', lambda *a, **k: FakeWindow(os.getpid()))
    handles = [chart_window.ensure(runs(tmp_path), env={}) for _ in range(4)]
    assert [handle is None for handle in handles] == [False] * 4


def test_a_wave_joining_a_live_window_starts_no_process_at_all(tmp_path, monkeypatch):
    """The advisory read, which is the whole reason `holder` still exists: the ordinary case is an
    arm joining a window that is already up, and it should cost a file read."""
    record_window(tmp_path, os.getpid())                # a live pid: us
    monkeypatch.setattr(chart_window, 'spawn', lambda *a, **k: pytest.fail('spawned anyway'))
    assert chart_window.ensure(runs(tmp_path), env={}) is None


def test_a_dead_pid_in_the_slot_does_not_stop_the_box_drawing_again(tmp_path, monkeypatch):
    """The other half of advisory: it must never refuse on state a dead window left behind."""
    record_window(tmp_path, DEAD_PID)
    monkeypatch.setattr(chart_window, 'spawn', lambda *a, **k: FakeWindow(4242))
    assert chart_window.ensure(runs(tmp_path), env={}) is not None


def test_a_window_that_loses_the_slot_never_draws(tmp_path, monkeypatch):
    """Standing down has to happen before the figure, or the losers would flash a window each."""
    monkeypatch.setattr(chart_viewer, 'take_window_slot', lambda *a, **k: None)
    monkeypatch.setattr(chart_viewer, 'run', lambda *a, **k: pytest.fail('drew anyway'))
    assert chart_viewer.main(['--runs-dir', runs(tmp_path)]) == 0


def test_looking_at_arbitrary_charts_is_not_gated_by_the_box_window(tmp_path, monkeypatch):
    """`--glob` and a path list are somebody looking at charts by hand. Only `--runs-dir` is the
    box's window, and refusing a hand-run look because a training has one open would be absurd."""
    drew = []
    monkeypatch.setattr(chart_viewer, 'take_window_slot',
                        lambda *a, **k: pytest.fail('took the box slot'))
    monkeypatch.setattr(chart_viewer, 'run', lambda *a, **k: drew.append(1))
    assert chart_viewer.main(['--glob', str(tmp_path / '*.png')]) == 0
    assert drew == [1]


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
    record_window(tmp_path, os.getpid())
    monkeypatch.setattr(live_runs, 'zombie', lambda pid: True)
    assert chart_window.holder(runs(tmp_path)) is None


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


# --- sizing, shared by both windows ---------------------------------------------------------------
#
# The point of these is that there is *one* implementation. The laptop and the desktop differ in the
# monitor they probe, not in how they decide, and the two windows differ in what they point at and
# when they close — not in how they are sized.

def test_sizing_reads_both_knobs():
    assert chart_window.sizing({'SNEK_CHART_WINDOW_SCALE': '0.8',
                                'SNEK_CHART_WINDOW_MAX_PX': '2400'}) == (0.8, 2400)


def test_sizing_is_none_when_unset():
    assert chart_window.sizing({}) == (None, None)


def test_a_typo_in_a_size_knob_does_not_stop_a_training():
    """A window is disposable; an unparseable env var must not raise into a trainer's startup."""
    assert chart_window.sizing({'SNEK_CHART_WINDOW_MAX_PX': 'wide',
                                'SNEK_CHART_WINDOW_SCALE': 'big'}) == (None, None)


def test_max_width_is_omitted_from_argv_when_unset():
    # Absent rather than an explicit "no cap" sentinel, so the viewer's own default is the one truth.
    assert '--max-width-px' not in chart_window.size_argv(None, None)


def test_both_windows_spell_the_size_flags_identically():
    from tools import eval_window
    training = chart_window.command('runs', 0.8, 'py', 2400)
    stage_b = eval_window.command(['a.png'], None, (), 0.8, 'py', 2400)

    def flags(argv):
        return [argv[i:i + 2] for i, token in enumerate(argv)
                if token in ('--scale', '--max-width-px')]

    assert flags(training) == flags(stage_b) == [['--scale', '0.8'],
                                                ['--max-width-px', '2400']]


def test_the_stage_b_window_honours_the_env_knobs_too(monkeypatch):
    """It did not: `ensure` read the env only when a caller passed one, and `closeout.py` does not.

    So `SNEK_CHART_WINDOW_SCALE` silently applied to the training window and not to this one. This
    goes through `ensure` rather than `command` because that asymmetry lived in `ensure`.
    """
    from tools import eval_window
    captured = {}
    monkeypatch.setattr(chart_window, 'ensure',
                        lambda runs_dir, env, argv, slot, label=None: captured.setdefault('argv', argv))
    monkeypatch.setenv('SNEK_CHART_WINDOW_MAX_PX', '1800')
    monkeypatch.setenv('SNEK_CHART_WINDOW_SCALE', '0.7')
    eval_window.ensure(['a.png'])
    argv = captured['argv']
    assert argv[argv.index('--max-width-px') + 1] == '1800'
    assert argv[argv.index('--scale') + 1] == '0.7'
