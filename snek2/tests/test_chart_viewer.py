"""The viewer's launch decision: which batch a window covers, when it is enabled, and
when a second one is suppressed.

The drawing loop is not covered here — it needs a display. What is covered is everything
`snek2.main()` relies on before it spawns anything, because a wrong answer there either
opens four windows for one wave or opens none at all.
"""
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chart_viewer


def _with_env(**kv):
    """Set env vars, returning a restore callable. Keys set to None are removed."""
    saved = {k: os.environ.get(k) for k in kv}

    def restore():
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return restore


def test_batch_prefix_groups_a_wave():
    """All four arms of a wave must map to one prefix, or they get one window each."""
    for arm in ('b20a-fc50seed1', 'b20b-fc50seed2', 'b20c-fc50seed3', 'b20d-fc50seed4'):
        assert chart_viewer.batch_prefix(arm) == 'b20', arm
    assert chart_viewer.batch_prefix('b9b-disc9975b') == 'b9'
    assert chart_viewer.batch_prefix('b18b-tgt1000seed2') == 'b18'


def test_batch_prefix_keeps_batches_apart():
    """b2 must not swallow b20: the glob built from the prefix would show both."""
    assert chart_viewer.batch_prefix('b2a-thing') == 'b2'
    assert chart_viewer.batch_prefix('b20a-thing') == 'b20'
    assert chart_viewer.batch_prefix('b2a-thing') != chart_viewer.batch_prefix('b20a-thing')


def test_batch_prefix_passes_through_non_arm_names():
    """The user's own run and one-off names are their own prefix, not a shared bucket."""
    assert chart_viewer.batch_prefix('train') == 'train'
    assert chart_viewer.batch_prefix('smoke') == 'smoke'
    assert chart_viewer.batch_prefix('champion_check') == 'champion_check'


def test_viewer_enabled_defaults_per_platform():
    """Default on for the laptop, off elsewhere — the desktop daemon owns its own viewer,
    and a trainer there has no DISPLAY to open one with."""
    restore = _with_env(SNEK_CHART_VIEWER=None)
    saved_platform = sys.platform
    try:
        sys.platform = 'darwin'
        assert chart_viewer.viewer_enabled() is True
        sys.platform = 'linux'
        assert chart_viewer.viewer_enabled() is False
    finally:
        sys.platform = saved_platform
        restore()


def test_viewer_enabled_env_overrides_both_ways():
    saved_platform = sys.platform
    try:
        sys.platform = 'darwin'
        restore = _with_env(SNEK_CHART_VIEWER='0')
        assert chart_viewer.viewer_enabled() is False
        restore()
        sys.platform = 'linux'
        restore = _with_env(SNEK_CHART_VIEWER='1')
        assert chart_viewer.viewer_enabled() is True
        restore()
    finally:
        sys.platform = saved_platform


def test_viewer_enabled_accepts_the_projects_falsey_spellings():
    restore = _with_env(SNEK_CHART_VIEWER='false')
    try:
        assert chart_viewer.viewer_enabled() is False
    finally:
        restore()


def test_spawn_skipped_when_disabled_and_for_smoke():
    """Neither path may reach Popen.

    The stub *records* instead of raising: `spawn_for_policy` catches every Exception on
    purpose, so a stub that raised AssertionError would be swallowed and this test would
    pass no matter what the code did."""
    saved_run, saved_popen = chart_viewer.subprocess.run, chart_viewer.subprocess.Popen
    calls = []

    class _Res:
        stdout = b''      # no viewer running, so only the gate can stop a launch

    class _Proc:
        pid = 999

    def record(argv, **_kw):
        calls.append(argv)
        return _Proc()

    chart_viewer.subprocess.run = lambda *_a, **_kw: _Res()
    chart_viewer.subprocess.Popen = record
    restore = _with_env(SNEK_CHART_VIEWER='0')
    try:
        chart_viewer.spawn_for_policy('b20a-fc50seed1')
        assert calls == [], 'launched a viewer with SNEK_CHART_VIEWER=0'
        restore()
        restore = _with_env(SNEK_CHART_VIEWER='1')
        chart_viewer.spawn_for_policy('smoke')
        assert calls == [], 'launched a viewer for a smoke run'
        # ...and the same stubs do launch for a real arm, so the assertions above are
        # about the gate rather than about a stub that never spawns anything.
        chart_viewer.spawn_for_policy('b20a-fc50seed1')
        assert len(calls) == 1
    finally:
        restore()
        chart_viewer.subprocess.run = saved_run
        chart_viewer.subprocess.Popen = saved_popen


def test_dedupe_matches_on_the_glob_not_the_script_name():
    """A viewer showing b19 must not suppress a b20 one, and vice versa. Faked `ps`
    output stands in for a live viewer so the test needs no process."""
    saved_run = chart_viewer.subprocess.run

    class _Res:
        def __init__(self, out):
            self.stdout = out

    def fake_run(args, **_kw):
        if args[0] == 'pgrep':
            return _Res(b'4242\n')
        return _Res(b"/py -u chart_viewer.py --glob runs/b19*.png --watch snek2.py b19\n")

    chart_viewer.subprocess.run = fake_run
    try:
        assert chart_viewer.viewer_running_for('runs/b19*.png') is True
        assert chart_viewer.viewer_running_for('runs/b20*.png') is False
    finally:
        chart_viewer.subprocess.run = saved_run


def test_dedupe_reports_running_when_it_cannot_tell():
    """Fail closed: an unanswerable check must not let four arms stack four windows."""
    saved_run = chart_viewer.subprocess.run

    def fake_run(*_a, **_kw):
        raise OSError('no pgrep here')

    chart_viewer.subprocess.run = fake_run
    try:
        assert chart_viewer.viewer_running_for('runs/b20*.png') is True
    finally:
        chart_viewer.subprocess.run = saved_run


def test_dedupe_allows_the_first_launch():
    saved_run = chart_viewer.subprocess.run

    class _Res:
        stdout = b''

    chart_viewer.subprocess.run = lambda *_a, **_kw: _Res()
    try:
        assert chart_viewer.viewer_running_for('runs/b20*.png') is False
    finally:
        chart_viewer.subprocess.run = saved_run


def test_spawn_command_is_detached_and_watches_its_own_batch():
    """The argv and the Popen flags are the contract: relative glob (so cwd must be the
    snek2 dir), a watch pattern for this batch only, own session, closed fds."""
    saved_run, saved_popen = chart_viewer.subprocess.run, chart_viewer.subprocess.Popen
    captured = {}

    class _Res:
        stdout = b''

    class _Proc:
        pid = 999

    def fake_popen(argv, **kw):
        captured['argv'] = argv
        captured['kw'] = kw
        return _Proc()

    chart_viewer.subprocess.run = lambda *_a, **_kw: _Res()
    chart_viewer.subprocess.Popen = fake_popen
    restore = _with_env(SNEK_CHART_VIEWER='1')
    try:
        proc = chart_viewer.spawn_for_policy('b20c-fc50seed3')
        assert proc is not None
        argv, kw = captured['argv'], captured['kw']
        assert argv[argv.index('--glob') + 1] == 'runs/b20*.png'
        assert argv[argv.index('--watch') + 1] == 'snek2.py b20'
        assert argv[2].endswith('chart_viewer.py')
        assert kw['start_new_session'] is True
        assert kw['close_fds'] is True
        assert os.path.isfile(os.path.join(kw['cwd'], 'chart_viewer.py'))
    finally:
        restore()
        chart_viewer.subprocess.run = saved_run
        chart_viewer.subprocess.Popen = saved_popen


def test_spawn_failure_is_swallowed():
    """A chart is never worth a training run: a Popen that raises returns None, not up."""
    saved_run, saved_popen = chart_viewer.subprocess.run, chart_viewer.subprocess.Popen

    class _Res:
        stdout = b''

    chart_viewer.subprocess.run = lambda *_a, **_kw: _Res()

    def boom(*_a, **_k):
        raise OSError('exec failed')

    chart_viewer.subprocess.Popen = boom
    restore = _with_env(SNEK_CHART_VIEWER='1')
    try:
        assert chart_viewer.spawn_for_policy('b20a-fc50seed1') is None
    finally:
        restore()
        chart_viewer.subprocess.run = saved_run
        chart_viewer.subprocess.Popen = saved_popen


class _FakePlt:
    """Stands in for pyplot: records close() and can be made to fail like a dead Tk."""
    def __init__(self, raise_on_close=False):
        self.closed = []
        self.raise_on_close = raise_on_close

    def close(self, what):
        if self.raise_on_close:
            raise RuntimeError('Tk is already gone')
        self.closed.append(what)


def test_exit_now_closes_windows_before_leaving():
    """Order is the whole point: the Tk windows must be gone *before* the process exits,
    or a late Tk event calls into a finalising interpreter and aborts (the macOS
    "python quit unexpectedly" dialog)."""
    plt = _FakePlt()
    events = []
    plt_close = plt.close

    def close(what):
        events.append('close')
        plt_close(what)
    plt.close = close
    chart_viewer.exit_now(plt, exit_fn=lambda code: events.append('exit{0}'.format(code)))
    assert events == ['close', 'exit0'], events
    assert plt.closed == ['all']


def test_exit_now_still_exits_when_closing_fails():
    """A teardown that raises must not leave the viewer alive in a loop it cannot draw."""
    codes = []
    chart_viewer.exit_now(_FakePlt(raise_on_close=True), code=3, exit_fn=codes.append)
    assert codes == [3]


def test_install_signal_exit_handles_term_and_int():
    """`kill <viewer pid>` is how a window gets swapped, so SIGTERM must go through the
    clean path rather than landing mid Tk event."""
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        handler = chart_viewer.install_signal_exit(_FakePlt())
        for s in (signal.SIGINT, signal.SIGTERM):
            assert signal.getsignal(s) is handler, s
            assert signal.getsignal(s) not in (signal.SIG_DFL, signal.default_int_handler)
    finally:
        for s, h in saved.items():
            signal.signal(s, h)


def test_signal_handler_closes_windows():
    """The installed handler must run the same close-then-exit path, not just exit."""
    plt = _FakePlt()
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    real_exit = os._exit
    exits = []
    os._exit = exits.append
    try:
        handler = chart_viewer.install_signal_exit(plt)
        handler(signal.SIGTERM, None)
        assert plt.closed == ['all']
        assert exits == [0]
    finally:
        os._exit = real_exit
        for s, h in saved.items():
            signal.signal(s, h)


def test_make_figure_installs_the_handler_after_building_the_window():
    """Order, not just presence. Tk overwrites the OS-level SIGTERM handler while it creates
    the figure's window, so installing first is dead code — measured: 5 of 5 kills still
    aborted with the install before `subplots()`. This is why the two are one function."""
    events = []

    class _Grid(list):
        pass

    class _Fig:
        number = 1
        canvas = None

    class _Plt:
        def subplots(self, rows, cols, **_kw):
            events.append('subplots')
            return _Fig(), [[object() for _ in range(cols)] for _ in range(rows)]

    saved = chart_viewer.install_signal_exit
    chart_viewer.install_signal_exit = lambda _plt: events.append('install')
    try:
        fig, axes = chart_viewer.make_figure(_Plt(), 2, 2, 2.0, 'title')
        assert events == ['subplots', 'install'], events
        assert len(axes) == 4
    finally:
        chart_viewer.install_signal_exit = saved


def test_make_figure_scales_both_dimensions():
    """--scale 2 has to reach the figsize, or "double the size" silently does nothing."""
    sizes = {}

    class _Fig:
        number = 1
        canvas = None

    class _Plt:
        def subplots(self, rows, cols, figsize=None, **_kw):
            sizes['figsize'] = figsize
            return _Fig(), [[object() for _ in range(cols)] for _ in range(rows)]

    saved = chart_viewer.install_signal_exit
    chart_viewer.install_signal_exit = lambda _plt: None
    try:
        chart_viewer.make_figure(_Plt(), 2, 2, 1.0, 't')
        one = sizes['figsize']
        chart_viewer.make_figure(_Plt(), 2, 2, 2.0, 't')
        two = sizes['figsize']
        assert two == (one[0] * 2, one[1] * 2), (one, two)
    finally:
        chart_viewer.install_signal_exit = saved


def test_laptop_defaults_are_one_second_and_double_size():
    """The user's asked-for defaults, pinned so a later edit cannot quietly undo them.
    An auto-launched viewer passes neither flag, so these values are what it runs with."""
    args = chart_viewer.build_parser().parse_args([])
    assert args.interval == 1.0
    assert args.scale == 2.0
