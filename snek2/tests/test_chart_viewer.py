"""The viewer's launch decision: which batch a window covers, when it is enabled, and
when a second one is suppressed.

The drawing loop is not covered here — it needs a display. What is covered is everything
`snek2.main()` relies on before it spawns anything, because a wrong answer there either
opens four windows for one wave or opens none at all.
"""
import os
import signal
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chart_viewer

# Point the claim lock at a private directory for the whole module. These tests use real
# batch prefixes (b20), and writing the real lock could steal or suppress the window of a
# wave that is actually training on this machine.
chart_viewer.LOCK_DIR = tempfile.mkdtemp(prefix='snek-viewer-test-')


def _clear_locks():
    for name in os.listdir(chart_viewer.LOCK_DIR):
        os.remove(os.path.join(chart_viewer.LOCK_DIR, name))


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


def test_batch_prefix_groups_double_letter_arms():
    """Past 26 arms a batch rolls into double letters, which must still group under the
    batch, not be read as their own. Batch 20 has 36 arms (nine shapes x four seeds)."""
    for arm in ('b20aa-fc93x93seed1', 'b20ab-fc93x93seed2', 'b20al-fc100x50x50seed4'):
        assert chart_viewer.batch_prefix(arm) == 'b20', arm
    # the single-letter arms of the same batch still group with the double-letter ones
    assert chart_viewer.batch_prefix('b20x-fc60x30x30x30x30seed4') == \
        chart_viewer.batch_prefix('b20aa-fc93x93seed1') == 'b20'


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
    _clear_locks()
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
    """The argv and the Popen flags are the contract: the batch whose live arms to show, a
    watch pattern for this batch only, own session, closed fds, and a cwd holding the
    relative `runs/` paths the viewer builds."""
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
    _clear_locks()
    restore = _with_env(SNEK_CHART_VIEWER='1')
    try:
        proc = chart_viewer.spawn_for_policy('b20c-fc50seed3')
        assert proc is not None
        argv, kw = captured['argv'], captured['kw']
        assert argv[argv.index('--arms') + 1] == 'b20'
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
    _clear_locks()
    restore = _with_env(SNEK_CHART_VIEWER='1')
    try:
        assert chart_viewer.spawn_for_policy('b20a-fc50seed1') is None
    finally:
        restore()
        chart_viewer.subprocess.run = saved_run
        chart_viewer.subprocess.Popen = saved_popen


def _fake_ps(lines):
    """Stub subprocess.run so pgrep reports pids and ps returns `lines`."""
    class _Res:
        def __init__(self, out):
            self.stdout = out

    def run(args, **_kw):
        if args[0] == 'pgrep':
            return _Res(b'\n'.join(str(1000 + i).encode() for i in range(len(lines))) + b'\n')
        return _Res('\n'.join(lines).encode())
    return run


TRAINERS = [
    ' 1001 /opt/miniconda3/envs/snek/bin/python -u snek2.py b20i-fc200x50seed1',
    ' 1002 /opt/miniconda3/envs/snek/bin/python -u snek2.py b20j-fc200x50seed2',
]


def test_live_arms_reads_policy_names_off_the_command_line():
    """The window has to show the arms that were launched, not a batch-wide glob."""
    saved = chart_viewer.subprocess.run
    chart_viewer.subprocess.run = _fake_ps(TRAINERS)
    try:
        assert chart_viewer.live_arms('b20') == ['b20i-fc200x50seed1', 'b20j-fc200x50seed2']
    finally:
        chart_viewer.subprocess.run = saved


def test_live_arms_does_not_merge_b2_into_b20():
    """`'b20a'.startswith('b2')` is True, so a prefix test would quietly merge two batches.
    Membership goes through batch_prefix instead."""
    saved = chart_viewer.subprocess.run
    lines = TRAINERS + [' 1003 python -u snek2.py b2a-oldarm']
    chart_viewer.subprocess.run = _fake_ps(lines)
    try:
        assert chart_viewer.live_arms('b2') == ['b2a-oldarm']
        assert 'b2a-oldarm' not in chart_viewer.live_arms('b20')
    finally:
        chart_viewer.subprocess.run = saved


def test_live_arms_excludes_other_batches_and_non_trainers():
    saved = chart_viewer.subprocess.run
    lines = TRAINERS + [
        ' 1003 python -u snek2.py b21a-nextbatch',
        ' 1004 python -u eval_checkpoints.py b20i-fc200x50seed1 top20',
        ' 1005 curl -X POST https://telemetry/…snek2/snek2.py…',
    ]
    chart_viewer.subprocess.run = _fake_ps(lines)
    try:
        arms = chart_viewer.live_arms('b20')
        assert arms == ['b20i-fc200x50seed1', 'b20j-fc200x50seed2'], arms
    finally:
        chart_viewer.subprocess.run = saved


def test_live_arms_ignores_a_bare_snek2_invocation():
    """`snek2.py` with no policy argument must be skipped, not indexed past.

    The bare line is paired with a real trainer deliberately: asserting only that the bare
    line yields nothing passes even when the code raises IndexError, because live_arms
    swallows exceptions and returns []. The real arm surviving is what proves it was skipped
    rather than crashed over.
    """
    saved = chart_viewer.subprocess.run
    chart_viewer.subprocess.run = _fake_ps([' 1001 python -u snek2.py'] + TRAINERS)
    try:
        assert chart_viewer.live_arms('b20') == [
            'b20i-fc200x50seed1', 'b20j-fc200x50seed2']
    finally:
        chart_viewer.subprocess.run = saved


def test_live_arms_only_takes_a_name_from_a_python_invocation():
    """Mentioning the file is not running it.

    `pgrep -f snek2.py` matches any command line containing the string, and plenty do: git
    commands listing pathspecs, and the Airbnb git-telemetry curl whose JSON payload carries
    repo paths — that class already misread trainer counts as 6 when 4 were running. The
    fixture is the case that actually discriminates: a non-python command whose *next* token
    is `b20z-notanarm`, which `batch_prefix` does read as a b20 arm, so nothing upstream
    rejects it and only the python check can. (`b20zz-...` does *not* parse as an arm, so a
    fixture using that name tests nothing — it is rejected before this code is reached.)
    """
    saved = chart_viewer.subprocess.run
    assert chart_viewer.batch_prefix('b20z-notanarm') == 'b20', 'fixture must look like an arm'
    noise = [' 1098 git ls-files snek2/snek2.py b20z-notanarm',
             ' 1099 curl -X POST https://drnick/v1/tracking -d {"paths":["snek2/snek2.py"]}']
    chart_viewer.subprocess.run = _fake_ps(noise + TRAINERS)
    try:
        arms = chart_viewer.live_arms('b20')
        assert arms == ['b20i-fc200x50seed1', 'b20j-fc200x50seed2'], arms
        assert 'b20z-notanarm' not in arms
    finally:
        chart_viewer.subprocess.run = saved


def test_live_arms_returns_empty_when_it_cannot_look():
    """Unknown means "nothing to add" — never "drop the arms already on screen"."""
    saved = chart_viewer.subprocess.run

    def boom(*_a, **_kw):
        raise OSError('no ps here')
    chart_viewer.subprocess.run = boom
    try:
        assert chart_viewer.live_arms('b20') == []
    finally:
        chart_viewer.subprocess.run = saved


def test_wave_files_grows_as_siblings_start():
    """A wave's four arms start seconds apart, so the panel set has to fill in."""
    saved = chart_viewer.subprocess.run
    known = []
    try:
        chart_viewer.subprocess.run = _fake_ps(TRAINERS[:1])
        assert chart_viewer.wave_files('b20', known) == ['runs/b20i-fc200x50seed1.png']
        chart_viewer.subprocess.run = _fake_ps(TRAINERS)
        assert chart_viewer.wave_files('b20', known) == [
            'runs/b20i-fc200x50seed1.png', 'runs/b20j-fc200x50seed2.png']
    finally:
        chart_viewer.subprocess.run = saved


def test_wave_files_keeps_an_arm_that_has_finished():
    """The first arm to hit its cap must keep its panel — the finished curve is the reference
    the still-running siblings are read against. Also means a transient ps failure or a race
    cannot blank the window."""
    saved = chart_viewer.subprocess.run
    known = []
    try:
        chart_viewer.subprocess.run = _fake_ps(TRAINERS)
        chart_viewer.wave_files('b20', known)
        chart_viewer.subprocess.run = _fake_ps(TRAINERS[1:])   # seed1 reached 3M and exited
        assert chart_viewer.wave_files('b20', known) == [
            'runs/b20i-fc200x50seed1.png', 'runs/b20j-fc200x50seed2.png']
        chart_viewer.subprocess.run = _fake_ps([])             # whole wave gone
        assert len(chart_viewer.wave_files('b20', known)) == 2
    finally:
        chart_viewer.subprocess.run = saved


def test_wave_files_never_shows_a_finished_arm_from_an_earlier_wave():
    """The bug this replaces: globbing runs/b20*.png matched eight finished arms plus the four
    live ones, giving a window taller than the screen by wave 2."""
    saved = chart_viewer.subprocess.run
    chart_viewer.subprocess.run = _fake_ps(TRAINERS)
    try:
        files = chart_viewer.wave_files('b20', [])
        assert not any('fc50seed' in f for f in files), files
        assert len(files) == 2
    finally:
        chart_viewer.subprocess.run = saved


def test_auto_launch_scopes_the_window_to_the_running_arms():
    """The spawned argv must ask for the arms, not the batch glob, and dedupe on that."""
    saved_run, saved_popen = chart_viewer.subprocess.run, chart_viewer.subprocess.Popen
    captured = {}

    class _Res:
        stdout = b''

    class _Proc:
        pid = 1

    chart_viewer.subprocess.run = lambda *_a, **_kw: _Res()
    chart_viewer.subprocess.Popen = lambda argv, **kw: (captured.setdefault('argv', argv), _Proc())[1]
    _clear_locks()
    restore = _with_env(SNEK_CHART_VIEWER='1')
    try:
        chart_viewer.spawn_for_policy('b20i-fc200x50seed1')
        argv = captured['argv']
        assert argv[argv.index('--arms') + 1] == 'b20'
        assert '--glob' not in argv, argv
    finally:
        restore()
        chart_viewer.subprocess.run = saved_run
        chart_viewer.subprocess.Popen = saved_popen


def test_only_one_of_a_waves_arms_claims_the_viewer_slot():
    """The bug that shipped: four trainers start in the same second, all four run the pgrep
    check before any has spawned, and four windows open. The claim has to be atomic."""
    lock = chart_viewer.lock_path('b98')
    if os.path.exists(lock):
        os.remove(lock)
    try:
        claims = [chart_viewer.claim_viewer_slot('b98') for _ in range(4)]
        assert claims == [True, False, False, False], claims
    finally:
        if os.path.exists(lock):
            os.remove(lock)


def test_the_lock_ends_up_pointing_at_the_viewer_not_the_trainer():
    """Liveness has to track the *window*. Pointed at the trainer, the claim would outlive a
    killed viewer (no replacement) and be dropped when the trainer merely finished."""
    saved_run, saved_popen = chart_viewer.subprocess.run, chart_viewer.subprocess.Popen

    class _Res:
        stdout = b''

    class _Proc:
        pid = 4242

    chart_viewer.subprocess.run = lambda *_a, **_kw: _Res()
    chart_viewer.subprocess.Popen = lambda argv, **kw: _Proc()
    _clear_locks()
    restore = _with_env(SNEK_CHART_VIEWER='1')
    try:
        chart_viewer.spawn_for_policy('b20a-fc50seed1')
        with open(chart_viewer.lock_path('b20')) as fh:
            assert fh.read().strip() == '4242', 'lock should name the viewer pid'
        assert str(os.getpid()) != '4242'
    finally:
        restore()
        chart_viewer.subprocess.run = saved_run
        chart_viewer.subprocess.Popen = saved_popen
        _clear_locks()


def test_a_failed_spawn_releases_the_claim():
    """Otherwise one arm's failure silently costs the whole wave its window."""
    saved_run, saved_popen = chart_viewer.subprocess.run, chart_viewer.subprocess.Popen

    class _Res:
        stdout = b''

    def boom(*_a, **_kw):
        raise OSError('exec failed')

    chart_viewer.subprocess.run = lambda *_a, **_kw: _Res()
    chart_viewer.subprocess.Popen = boom
    _clear_locks()
    restore = _with_env(SNEK_CHART_VIEWER='1')
    try:
        assert chart_viewer.spawn_for_policy('b20a-fc50seed1') is None
        assert not os.path.exists(chart_viewer.lock_path('b20')), 'claim was not released'
        # ...so the next arm of the wave can still get a window
        assert chart_viewer.claim_viewer_slot('b20') is True
    finally:
        restore()
        chart_viewer.subprocess.run = saved_run
        chart_viewer.subprocess.Popen = saved_popen
        _clear_locks()


def test_a_dead_claimants_lock_is_taken_over():
    """A crashed or killed viewer must not suppress the window forever."""
    lock = chart_viewer.lock_path('b97')
    with open(lock, 'w') as fh:
        fh.write('999999')          # a pid that cannot be alive
    try:
        assert chart_viewer.claim_viewer_slot('b97') is True
        # ...and having taken it over, we now hold it against the next caller
        assert chart_viewer.claim_viewer_slot('b97') is False
    finally:
        if os.path.exists(lock):
            os.remove(lock)


def test_a_live_claimant_keeps_the_slot():
    """Our own pid is alive, so a second caller must be refused."""
    lock = chart_viewer.lock_path('b96')
    if os.path.exists(lock):
        os.remove(lock)
    try:
        assert chart_viewer.claim_viewer_slot('b96') is True
        assert chart_viewer.claim_viewer_slot('b96') is False
    finally:
        if os.path.exists(lock):
            os.remove(lock)


def test_two_batches_claim_independently():
    locks = [chart_viewer.lock_path(b) for b in ('b95', 'b94')]
    for p in locks:
        if os.path.exists(p):
            os.remove(p)
    try:
        assert chart_viewer.claim_viewer_slot('b95') is True
        assert chart_viewer.claim_viewer_slot('b94') is True
    finally:
        for p in locks:
            if os.path.exists(p):
                os.remove(p)


def test_dedupe_keys_on_the_arms_flag_so_two_batches_get_two_windows():
    saved = chart_viewer.subprocess.run
    chart_viewer.subprocess.run = _fake_ps(
        [' 4242 python -u chart_viewer.py --arms b19 --watch snek2.py b19'])
    try:
        # _matching_commands strips chart_viewer lines, so ask viewer_running_for directly
        # against a ps stub that keeps them.
        class _Res:
            def __init__(self, out):
                self.stdout = out

        def run(args, **_kw):
            if args[0] == 'pgrep':
                return _Res(b'4242\n')
            return _Res(b'/py -u chart_viewer.py --arms b19 --watch snek2.py b19\n')
        chart_viewer.subprocess.run = run
        assert chart_viewer.viewer_running_for('--arms b19') is True
        assert chart_viewer.viewer_running_for('--arms b20') is False
    finally:
        chart_viewer.subprocess.run = saved


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
