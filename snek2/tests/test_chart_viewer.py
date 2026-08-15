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
import time

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


def test_policy_from_png_strips_both_chart_layouts():
    """A panel is tied back to its arm through the PNG name. The trainer writes
    runs/<policy>.png; the eval writes evals/<policy>_eval_progress.png. Both must resolve to
    the bare policy so completion can be looked up against the process list."""
    assert chart_viewer.policy_from_png('runs/b20q-fc25x50x25seed1.png') == 'b20q-fc25x50x25seed1'
    assert chart_viewer.policy_from_png(
        '/snek/evals/b20q-fc25x50x25seed1_eval_progress.png') == 'b20q-fc25x50x25seed1'


def _fake_ps_by_pattern(lines):
    """Stub subprocess.run so `pgrep -f PAT` matches only lines containing PAT, the way the
    real pgrep does. `_fake_ps` ignores the pattern and returns every line, which cannot tell
    a scan of one process kind from a scan of both -- so running_policies would look correct
    even if it dropped a pattern. Here each pid maps to one line, pgrep returns the pids whose
    line contains the pattern, and ps returns exactly those pids' lines."""
    numbered = {1000 + i: line for i, line in enumerate(lines)}

    class _Res:
        def __init__(self, out):
            self.stdout = out

    def run(args, **_kw):
        if args[0] == 'pgrep':
            pat = args[2]
            hits = [pid for pid, line in numbered.items() if pat in line]
            return _Res(('\n'.join(str(p) for p in hits) + '\n').encode())
        wanted = [args[i + 1] for i, a in enumerate(args) if a == '-p']
        return _Res('\n'.join(numbered[int(p)] for p in wanted).encode())
    return run


def test_running_policies_collects_trainers_and_evals():
    """The completed tag needs to know an arm is *gone*, and an arm is present as either a
    trainer (snek2.py) or an eval (eval_checkpoints.py). Both patterns must be scanned, so a
    panel is not called completed while its eval is still measuring it. The pattern-aware fake
    is what makes this bite: were the eval scan dropped, seed2's eval-only line would vanish."""
    saved = chart_viewer.subprocess.run
    lines = [' 1001 python -u snek2.py b20q-fc25seed1',
             ' 1002 python -u eval_checkpoints.py b20r-fc25seed2 top20']
    chart_viewer.subprocess.run = _fake_ps_by_pattern(lines)
    try:
        live = chart_viewer.running_policies()
        assert 'b20q-fc25seed1' in live
        assert 'b20r-fc25seed2' in live
    finally:
        chart_viewer.subprocess.run = saved


def test_running_policies_is_token_exact_not_substring():
    """`seed1` must not read as running because `seed11` is — a token match, not `in`. A wave
    with both would otherwise leave the finished seed1 falsely un-tagged."""
    saved = chart_viewer.subprocess.run
    chart_viewer.subprocess.run = _fake_ps([' 1001 python -u snek2.py b20q-fc25seed11'])
    try:
        live = chart_viewer.running_policies()
        assert 'b20q-fc25seed11' in live
        assert 'b20q-fc25seed1' not in live      # the finished sibling stays taggable
    finally:
        chart_viewer.subprocess.run = saved


def test_running_policies_is_none_when_it_cannot_look():
    """Unreadable process list -> None, so the caller tags nothing. A false (completed) on a
    live arm is worse than a missing one, whose frozen curve already shows it stopped."""
    saved = chart_viewer.subprocess.run

    def boom(*_a, **_kw):
        raise OSError('no ps here')
    chart_viewer.subprocess.run = boom
    try:
        assert chart_viewer.running_policies() is None
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
    """--scale 2 has to reach the figsize, or "double the size" silently does nothing.

    Uses a single panel, which is well inside the screen budget, so this isolates scale
    propagation from the multi-row clamp that `figure_dims` applies (tested separately)."""
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
        chart_viewer.make_figure(_Plt(), 1, 1, 1.0, 't')
        one = sizes['figsize']
        chart_viewer.make_figure(_Plt(), 1, 1, 2.0, 't')
        two = sizes['figsize']
        assert two == (one[0] * 2, one[1] * 2), (one, two)
    finally:
        chart_viewer.install_signal_exit = saved


def test_figure_dims_clamps_a_multirow_wave_to_the_screen():
    """A 2x2 wave at the laptop default scale 2.0 is 16.8x12.0in; unclamped its 1200px height
    opens the bottom row below a laptop screen and it reads as a missing chart. figure_dims must
    shrink it to fit the height budget while preserving aspect ratio."""
    w, h = chart_viewer.figure_dims(2, 2, 2.0)
    assert h <= chart_viewer.MAX_FIG_H_IN + 1e-9
    assert w <= chart_viewer.MAX_FIG_W_IN + 1e-9
    # uniform shrink -> aspect unchanged from the requested 16.8 x 12.0
    assert abs(w / h - 16.8 / 12.0) < 1e-9


def test_figure_dims_leaves_a_fitting_grid_untouched():
    """Single panels and any grid already inside the budget must not be shrunk, so the
    clamp never makes a window that already fit smaller than asked."""
    assert chart_viewer.figure_dims(1, 1, 2.0) == (8.4, 6.0)   # under budget, unchanged
    assert chart_viewer.figure_dims(2, 2, 1.0) == (8.4, 6.0)   # 2x2 at scale 1 also fits


def test_panel_title_never_shows_completed_and_waiting_together():
    """The desktop bug: a finished arm whose eval PNG a later wave had archived read
    `(completed) (waiting…)` at once. A done arm is `(completed)` whether or not its chart
    loads; `(waiting…)` is only a live arm with no chart yet; and the two never coincide."""
    live = {'b20ah-fc100x200x100seed4'}  # only this arm still running
    pt = chart_viewer.panel_title
    # finished arm, chart archived/missing -> completed only, NOT "(completed) (waiting…)"
    assert pt('b21a-beta05seed1', live, False) == 'b21a-beta05seed1 (completed)'
    # finished arm, chart present -> completed
    assert pt('b21a-beta05seed1', live, True) == 'b21a-beta05seed1 (completed)'
    # live arm, chart not written yet -> waiting
    assert pt('b20ah-fc100x200x100seed4', live, False) == 'b20ah-fc100x200x100seed4 (waiting…)'
    # live arm, chart present -> no tag
    assert pt('b20ah-fc100x200x100seed4', live, True) == 'b20ah-fc100x200x100seed4'
    # process list unreadable (None): never tag, even with a missing chart
    assert pt('b21a-beta05seed1', None, False) == 'b21a-beta05seed1'


def test_clamp_dims_shrinks_uniformly_and_never_grows():
    """The shrink is the tighter of the two ratios and applies to both dims, so aspect holds;
    a size already inside the budget is returned untouched rather than scaled up to fill it."""
    # height is the binding constraint (12 vs 8 is tighter than 16.8 vs 14)
    w, h = chart_viewer.clamp_dims(16.8, 12.0, 14.0, 8.0)
    assert (round(w, 6), round(h, 6)) == (11.2, 8.0)
    assert abs(w / h - 16.8 / 12.0) < 1e-9          # aspect preserved
    # already fits -> unchanged, not enlarged to the budget
    assert chart_viewer.clamp_dims(8.4, 6.0, 14.0, 8.0) == (8.4, 6.0)


def test_fit_figure_to_screen_shrinks_to_a_small_display():
    """The real-screen fit must clamp a 2x2 wave to a laptop panel and be a no-op on a screen
    that already has room, reading width/height straight off Tk in pixels / dpi."""
    class _Win:
        def __init__(self, w, h):
            self._w, self._h = w, h

        def winfo_screenwidth(self):
            return self._w

        def winfo_screenheight(self):
            return self._h

    class _Canvas:
        def __init__(self, win):
            self.manager = type('M', (), {'window': win})()

    class _Fig:
        def __init__(self, win):
            self.canvas = _Canvas(win)
            self.size = None

        def get_dpi(self):
            return 100.0

        def set_size_inches(self, w, h):
            self.size = (w, h)

    # 900px-tall panel: 8.0in figure * 0.88 = 7.04in budget -> must shrink below the fallback
    small = _Fig(_Win(1440, 900))
    chart_viewer.fit_figure_to_screen(small, 11.2, 8.0)
    assert small.size is not None
    assert small.size[1] <= 900 / 100.0 * 0.88 + 1e-9
    assert abs(small.size[0] / small.size[1] - 11.2 / 8.0) < 1e-9   # aspect preserved

    # a tall external display has room -> no resize call at all
    big = _Fig(_Win(3008, 1692))
    chart_viewer.fit_figure_to_screen(big, 11.2, 8.0)
    assert big.size is None


def test_fit_figure_to_screen_survives_a_missing_backend():
    """It touches the Tk backend, which may be absent; a failure must be swallowed so the
    viewer still draws rather than dying on a headless canvas."""
    class _Fig:
        canvas = None
    chart_viewer.fit_figure_to_screen(_Fig(), 11.2, 8.0)   # must not raise


def test_laptop_defaults_are_one_second_and_scaled_up():
    """The user's asked-for defaults, pinned so a later edit cannot quietly undo them.
    An auto-launched viewer passes neither flag, so these values are what it runs with.
    Scale was raised from 2.0 to 2.6 (a 30% bigger window) on request."""
    args = chart_viewer.build_parser().parse_args([])
    assert args.interval == 1.0
    assert args.scale == 2.6


def test_grid_shape_one_column_for_one_two_otherwise():
    """The panel grid: a lone arm is one column, any wave is two, and the row count rounds up.
    A four-arm and a three-arm wave both want a 2x2, which is why a late fourth arm changes the
    set without changing the axis count -- the case the set-change rebuild exists for."""
    assert chart_viewer.grid_shape(1) == (1, 1)
    assert chart_viewer.grid_shape(2) == (1, 2)
    assert chart_viewer.grid_shape(3) == (2, 2)
    assert chart_viewer.grid_shape(4) == (2, 2)
    assert chart_viewer.grid_shape(5) == (3, 2)


def test_wave_files_accumulates_a_late_arm():
    """The recurring missing-chart bug: a fourth arm that appears after the grid is built must
    still get a panel. wave_files is sticky and re-scans every refresh, so once live_arms reports
    the late arm it joins the set and never falls back out -- proving the drop was the refresh loop
    wedging, not the arm detection. `known` is mutated in place across refreshes, as in main()."""
    saved = chart_viewer.subprocess.run
    known = []
    try:
        chart_viewer.subprocess.run = _fake_ps(TRAINERS)          # two arms
        first = chart_viewer.wave_files('b20', known)
        assert [chart_viewer.policy_from_png(f) for f in first] == [
            'b20i-fc200x50seed1', 'b20j-fc200x50seed2']
        late = TRAINERS + [' 1003 /opt/miniconda3/envs/snek/bin/python -u snek2.py b20k-fc200x50seed3']
        chart_viewer.subprocess.run = _fake_ps(late)              # a third appears
        second = chart_viewer.wave_files('b20', known)
        assert [chart_viewer.policy_from_png(f) for f in second] == [
            'b20i-fc200x50seed1', 'b20j-fc200x50seed2', 'b20k-fc200x50seed3']
    finally:
        chart_viewer.subprocess.run = saved


def test_wave_files_keeps_an_arm_when_a_scan_transiently_fails():
    """A ps hiccup returns nothing that refresh; the sticky set must not blank an arm's panel."""
    saved = chart_viewer.subprocess.run
    known = []
    try:
        chart_viewer.subprocess.run = _fake_ps(TRAINERS)
        chart_viewer.wave_files('b20', known)

        def boom(*_a, **_k):
            raise OSError('ps unavailable')
        chart_viewer.subprocess.run = boom
        held = chart_viewer.wave_files('b20', known)
        assert [chart_viewer.policy_from_png(f) for f in held] == [
            'b20i-fc200x50seed1', 'b20j-fc200x50seed2']
    finally:
        chart_viewer.subprocess.run = saved


# ---------------------------------------------------------------- the arm registry
#
# The bug these pin: on 2026-08-14 a four-arm laptop wave opened a window showing three of them.
# Discovery was a process-list snapshot only, so a panel existed if and only if some `ps` landed
# while that arm's process was visible — and nothing repaired a miss. The registry makes each
# trainer state its own name before any scan runs.


def test_register_arm_records_every_sibling_of_a_wave():
    """All four arms of a wave register, including the three that lose the viewer lock."""
    _clear_locks()
    for policy in ('b30a-chase10seed1', 'b30b-chase10seed2',
                   'b30c-chase10seed3', 'b30d-chase10seed4'):
        chart_viewer.register_arm(policy)
    assert chart_viewer.registered_arms('b30') == [
        'b30a-chase10seed1', 'b30b-chase10seed2', 'b30c-chase10seed3', 'b30d-chase10seed4']


def test_registered_arms_survives_a_scan_that_sees_nothing():
    """**The 3-of-4 bug.** Even with the process list empty every registered arm gets a panel."""
    _clear_locks()
    for policy in ('b30a-chase10seed1', 'b30b-chase10seed2',
                   'b30c-chase10seed3', 'b30d-chase10seed4'):
        chart_viewer.register_arm(policy)
    saved = chart_viewer.subprocess.run
    known = []
    try:
        chart_viewer.subprocess.run = _fake_ps([])      # no trainer visible to ps at all
        files = chart_viewer.wave_files('b30', known)
    finally:
        chart_viewer.subprocess.run = saved
    assert [chart_viewer.policy_from_png(f) for f in files] == [
        'b30a-chase10seed1', 'b30b-chase10seed2', 'b30c-chase10seed3', 'b30d-chase10seed4']


def test_wave_files_unions_the_registry_with_the_process_scan():
    """Each source covers the other's blind spot: an arm resumed by hand never registers, and an
    arm the scan misses is still in the registry."""
    _clear_locks()
    chart_viewer.register_arm('b20i-fc200x50seed1')
    saved = chart_viewer.subprocess.run
    known = []
    try:
        chart_viewer.subprocess.run = _fake_ps(TRAINERS[1:])    # only seed2 is visible
        files = chart_viewer.wave_files('b20', known)
    finally:
        chart_viewer.subprocess.run = saved
    assert [chart_viewer.policy_from_png(f) for f in files] == [
        'b20i-fc200x50seed1', 'b20j-fc200x50seed2']


def test_registered_arms_ignores_another_batch_and_junk_lines():
    """The file is keyed by prefix, but a hand-edited or half-written line must not cost the
    window — and membership is `batch_prefix(policy) == prefix`, never `startswith`.

    `b300a-` is the case that makes the difference load-bearing: `'b300a-x'.startswith('b30')` is
    True, so a prefix test would pull batch 300's arm into batch 30's window. It reaches this file
    only by hand-editing or a truncated write, which is exactly what the other lines here are."""
    _clear_locks()
    chart_viewer.register_arm('b30a-chase10seed1')
    chart_viewer.register_arm('b3a-oldarm')          # its own prefix, its own file
    with open(chart_viewer.arms_path('b30'), 'a') as handle:
        handle.write('{0:.0f}\tb300a-otherbatch\n'.format(time.time()))
        handle.write('not-a-timestamp\tb30z-junkstamp\n')
        handle.write('no-tab-at-all\n')
        handle.write('\n')
    assert chart_viewer.registered_arms('b30') == ['b30a-chase10seed1']


def test_registered_arms_ages_out_a_previous_wave():
    """A second wave under the same prefix must not inherit yesterday's arms as extra panels —
    the failure mode of the `runs/b20*.png` glob this whole mechanism replaced."""
    _clear_locks()
    now = 1_800_000_000.0
    with open(chart_viewer.arms_path('b30'), 'a') as handle:
        handle.write('{0:.0f}\tb30a-yesterday\n'.format(now - chart_viewer.ARM_REGISTRY_TTL - 60))
        handle.write('{0:.0f}\tb30a-today\n'.format(now - 30))
    assert chart_viewer.registered_arms('b30', now=now) == ['b30a-today']


def test_registered_arms_ignores_a_stamp_from_the_future():
    """A clock that jumped backwards would otherwise pin a panel for a whole TTL."""
    _clear_locks()
    now = 1_800_000_000.0
    with open(chart_viewer.arms_path('b30'), 'a') as handle:
        handle.write('{0:.0f}\tb30a-fromthefuture\n'.format(now + 3600))
        handle.write('{0:.0f}\tb30b-nowish\n'.format(now - 5))
    assert chart_viewer.registered_arms('b30', now=now) == ['b30b-nowish']


def test_registered_arms_dedupes_a_resumed_arm():
    """A resume registers the same name again; one panel, not two."""
    _clear_locks()
    chart_viewer.register_arm('b30a-chase10seed1')
    chart_viewer.register_arm('b30a-chase10seed1')
    assert chart_viewer.registered_arms('b30') == ['b30a-chase10seed1']


def test_wave_files_caps_the_panel_count():
    """A prefix that accumulates arms inside the TTL must not grow a window taller than the
    screen — the cap is what `--arms` was introduced to guarantee."""
    _clear_locks()
    for i in range(chart_viewer.MAX_WAVE_PANELS + 4):
        chart_viewer.register_arm('b30{0}-arm{1}'.format(chr(ord('a') + i), i))
    saved = chart_viewer.subprocess.run
    known = []
    try:
        chart_viewer.subprocess.run = _fake_ps([])
        files = chart_viewer.wave_files('b30', known)
    finally:
        chart_viewer.subprocess.run = saved
    assert len(files) == chart_viewer.MAX_WAVE_PANELS


def test_register_arm_never_raises_when_the_registry_is_unwritable():
    """A registry write is never worth a training run."""
    saved = chart_viewer.LOCK_DIR
    chart_viewer.LOCK_DIR = '/nonexistent-dir-for-snek-test'
    try:
        chart_viewer.register_arm('b30a-chase10seed1')       # must not raise
        assert chart_viewer.registered_arms('b30') == []
    finally:
        chart_viewer.LOCK_DIR = saved


def test_spawn_registers_the_arm_even_when_another_viewer_owns_the_window():
    """The three arms that lose the lock are exactly the ones that must still be registered."""
    _clear_locks()
    restore = _with_env(SNEK_CHART_VIEWER='1')
    saved = chart_viewer.subprocess.run

    def no_viewer_but_holder_alive(args, **_kw):
        """pgrep finds no viewer process; the lock's holder reads as a live one."""
        class _Res:
            def __init__(self, out):
                self.stdout = out
        if args[0] == 'ps' and 'stat=' in args:
            return _Res(b'S')
        return _Res(b'')
    # Popen is stubbed as well as `run`, and that is not belt-and-braces: when this assertion
    # failed during development the *real* spawn went through and opened three live b30 windows on
    # the laptop, on top of a wave that was training. A test in this file must not be able to open
    # a window even when it is wrong.
    saved_popen = chart_viewer.subprocess.Popen

    def no_popen(*_a, **_kw):
        raise AssertionError('spawn_for_policy must not launch a viewer here')
    try:
        chart_viewer.claim_viewer_slot('b30')               # someone else owns the window
        chart_viewer.subprocess.run = no_viewer_but_holder_alive
        chart_viewer.subprocess.Popen = no_popen
        assert chart_viewer.spawn_for_policy('b30b-chase10seed2') is None
    finally:
        chart_viewer.subprocess.run = saved
        chart_viewer.subprocess.Popen = saved_popen
        restore()
    assert chart_viewer.registered_arms('b30') == ['b30b-chase10seed2']


def test_smoke_runs_do_not_register():
    """Smoke output is verification, not an arm anyone watches — and `smoke` is its own prefix."""
    _clear_locks()
    restore = _with_env(SNEK_CHART_VIEWER='1')
    try:
        assert chart_viewer.spawn_for_policy('smoke') is None
    finally:
        restore()
    assert chart_viewer.registered_arms('smoke') == []


# ---------------------------------------------------------------- zombie viewers hold no claim
#
# A viewer is spawned by a trainer that never wait()s for it, so an exited viewer stays a zombie
# for as long as its parent trainer runs. `kill(pid, 0)` succeeds on a zombie, so the claim lock
# read "a live viewer owns this batch" for the rest of the wave and no trainer could reopen the
# window — the one property claim_viewer_slot promises it does not have. Found 2026-08-14.


def _fake_ps_stat(state):
    """Stub subprocess.run so `ps -o stat=` reports `state` for any pid."""
    class _Res:
        def __init__(self, out):
            self.stdout = out

    def run(args, **_kw):
        if args[0] == 'ps' and 'stat=' in args:
            return _Res(state.encode())
        return _Res(b'')
    return run


def test_pid_alive_is_false_for_a_zombie():
    """The measured case: state `ZN`, and `kill -0` says alive."""
    saved = chart_viewer.subprocess.run
    chart_viewer.subprocess.run = _fake_ps_stat('ZN')
    try:
        assert chart_viewer.pid_alive(os.getpid()) is False
    finally:
        chart_viewer.subprocess.run = saved


def test_pid_alive_is_true_for_a_running_process():
    saved = chart_viewer.subprocess.run
    chart_viewer.subprocess.run = _fake_ps_stat('S+')
    try:
        assert chart_viewer.pid_alive(os.getpid()) is True
    finally:
        chart_viewer.subprocess.run = saved


def test_pid_alive_is_false_for_a_pid_that_does_not_exist():
    assert chart_viewer.pid_alive(2 ** 30) is False


def test_pid_alive_is_false_when_ps_knows_nothing():
    """Exited between the kill test and the ps: not alive."""
    saved = chart_viewer.subprocess.run
    chart_viewer.subprocess.run = _fake_ps_stat('')
    try:
        assert chart_viewer.pid_alive(os.getpid()) is False
    finally:
        chart_viewer.subprocess.run = saved


def test_pid_alive_assumes_alive_when_ps_cannot_run():
    """A duplicate window is what the lock exists to prevent, so an unanswerable check keeps
    the claim rather than opening a second one."""
    saved = chart_viewer.subprocess.run

    def boom(*_a, **_k):
        raise OSError('ps unavailable')
    chart_viewer.subprocess.run = boom
    try:
        assert chart_viewer.pid_alive(os.getpid()) is True
    finally:
        chart_viewer.subprocess.run = saved


def test_a_zombie_viewers_lock_is_taken_over():
    """**The window must be reopenable.** A lock naming a zombie viewer is stale, so the next
    trainer of the batch wins the claim instead of being locked out for the rest of the wave."""
    _clear_locks()
    with open(chart_viewer.lock_path('b30'), 'w') as handle:
        handle.write(str(os.getpid()))       # a "viewer" pid that ps will call a zombie
    saved = chart_viewer.subprocess.run
    chart_viewer.subprocess.run = _fake_ps_stat('ZN')
    try:
        assert chart_viewer.claim_viewer_slot('b30') is True
    finally:
        chart_viewer.subprocess.run = saved


def test_a_live_viewers_lock_is_honoured():
    """The dedupe still has to hold: a running viewer keeps its claim."""
    _clear_locks()
    with open(chart_viewer.lock_path('b30'), 'w') as handle:
        handle.write(str(os.getpid()))
    saved = chart_viewer.subprocess.run
    chart_viewer.subprocess.run = _fake_ps_stat('S')
    try:
        assert chart_viewer.claim_viewer_slot('b30') is False
    finally:
        chart_viewer.subprocess.run = saved


def test_dedupe_ignores_a_zombie_viewer():
    """A viewer that exited keeps its `--arms b30` argv until its parent trainer reaps it, so the
    pgrep dedupe would report a window that closed hours ago and refuse to open a new one. This
    is the same zombie defect as the claim lock's, at the second of the two sites."""
    saved = chart_viewer.subprocess.run

    class _Res:
        def __init__(self, out):
            self.stdout = out

    def zombie_viewer(args, **_kw):
        if args[0] == 'pgrep':
            return _Res(b'4242\n')
        if args[0] == 'ps' and 'stat=' in args:
            return _Res(b'ZN')                          # the only viewer is a zombie
        return _Res(b'python chart_viewer.py --arms b30\n')
    chart_viewer.subprocess.run = zombie_viewer
    try:
        assert chart_viewer.viewer_running_for('--arms b30') is False
    finally:
        chart_viewer.subprocess.run = saved


def test_dedupe_still_sees_a_live_viewer():
    """The dedupe's whole job: one window per batch while a viewer is actually up."""
    saved = chart_viewer.subprocess.run

    class _Res:
        def __init__(self, out):
            self.stdout = out

    def live_viewer(args, **_kw):
        if args[0] == 'pgrep':
            return _Res(b'4242\n')
        if args[0] == 'ps' and 'stat=' in args:
            return _Res(b'S')
        return _Res(b'python chart_viewer.py --arms b30\n')
    chart_viewer.subprocess.run = live_viewer
    try:
        assert chart_viewer.viewer_running_for('--arms b30') is True
    finally:
        chart_viewer.subprocess.run = saved
