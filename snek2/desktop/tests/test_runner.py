"""Unit tests for the pure runner logic: config load/clamp/fallback, job parsing,
and command building. The git/subprocess glue (gitbus, spawn, the loop) needs a
real box and is not exercised here.

Run directly:  PYTHONPATH=.. python test_runner.py    (from snek2/desktop/tests)
or import and call the test_* functions like the snek2/tests suite.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runner import config as cfg
from runner import gitbus
from runner import job as jobmod
from runner import launch
from runner import runner as runnermod


def _host():
    return {'HARD_MAX_TRAINERS': 4, 'HARD_MAX_EVALS': 2, 'MIN_POLL_SECONDS': 10,
            'PYTHON_BIN': '/py', 'SNEK_DIR': '/snek', 'LOG_DIR': '/logs'}


# ---------------------------------------------------------------- host.env
def test_load_host_config_reads_and_coerces():
    with tempfile.NamedTemporaryFile('w', suffix='.env', delete=False) as fh:
        fh.write('# comment\nREPO_PATH=/r\nSNEK_DIR=/r/snek2\nPYTHON_BIN=/p\n'
                 'GIT_REMOTE=origin\nOPS_BRANCH=ops\nSTATUS_BRANCH=ops-status\n'
                 'RESULTS_BRANCH=results\nSTATUS_WORKTREE=/w/s\nRESULTS_WORKTREE=/w/r\n'
                 'LEDGER_PATH=/l\nLOG_DIR=/lg\nHARD_MAX_TRAINERS=3\nHARD_MAX_EVALS=1\n'
                 'MIN_POLL_SECONDS=15\n')
        path = fh.name
    h = cfg.load_host_config(path)
    os.unlink(path)
    assert h['REPO_PATH'] == '/r'
    assert h['HARD_MAX_TRAINERS'] == 3 and isinstance(h['HARD_MAX_TRAINERS'], int)
    assert h['MIN_POLL_SECONDS'] == 15


def test_load_host_config_missing_key_raises():
    with tempfile.NamedTemporaryFile('w', suffix='.env', delete=False) as fh:
        fh.write('REPO_PATH=/r\n')
        path = fh.name
    try:
        cfg.load_host_config(path)
        assert False, 'expected ConfigError'
    except cfg.ConfigError:
        pass
    finally:
        os.unlink(path)


# ------------------------------------------------------------- runtime.json
def test_runtime_defaults_when_valid():
    out, notes = cfg.parse_runtime_config('{"max_trainers": 3}', _host())
    assert out is not None
    assert out['max_trainers'] == 3
    assert out['poll_seconds'] == cfg.RUNTIME_DEFAULTS['poll_seconds']  # filled in


def test_runtime_clamps_to_ceilings():
    out, notes = cfg.parse_runtime_config('{"max_trainers": 50, "poll_seconds": 1}', _host())
    assert out['max_trainers'] == 4          # HARD_MAX_TRAINERS
    assert out['poll_seconds'] == 10         # MIN_POLL_SECONDS floor
    assert any('max_trainers' in n for n in notes)
    assert any('poll_seconds' in n for n in notes)


def test_runtime_bad_json_falls_back():
    out, errs = cfg.parse_runtime_config('{not json', _host())
    assert out is None and errs  # caller keeps last-known-good


def test_runtime_wrong_type_falls_back():
    out, errs = cfg.parse_runtime_config('{"max_trainers": "two"}', _host())
    assert out is None
    assert any('max_trainers' in e for e in errs)


def test_runtime_unknown_key_rejected():
    out, errs = cfg.parse_runtime_config('{"maxtrainers": 2}', _host())
    assert out is None
    assert any('unknown key' in e for e in errs)


def test_runtime_bool_must_be_bool():
    out, errs = cfg.parse_runtime_config('{"paused": 1}', _host())
    assert out is None


def test_runtime_auto_hof_defaults_on_and_takes_a_bool():
    out, _ = cfg.parse_runtime_config('{"max_trainers": 2}', _host())
    assert out['auto_hof'] is True                        # on unless disabled, like auto_closeout
    out, _ = cfg.parse_runtime_config('{"auto_hof": false}', _host())
    assert out['auto_hof'] is False
    out, errs = cfg.parse_runtime_config('{"auto_hof": 1}', _host())
    assert out is None and any('auto_hof' in e for e in errs)


# --------------------------------------------------------------------- jobs
def test_parse_train_job():
    j = jobmod.parse_job('{"id": "b20a", "type": "train", "policy": "b20a-x",'
                         ' "env": {"SNEK_SEED": 1}, "max_steps": 2000000}')
    assert j.id == 'b20a' and j.type == 'train' and j.category == 'trainer'
    assert j.env == {'SNEK_SEED': '1'}       # coerced to string
    assert j.max_steps == 2000000


def test_parse_eval_job_category():
    j = jobmod.parse_job('{"id": "e1", "type": "eval", "policy": "b19a",'
                         ' "eval_args": ["top20"], "eval_workers": 10}')
    assert j.category == 'eval' and j.eval_args == ['top20']


def test_train_requires_policy():
    try:
        jobmod.parse_job('{"id": "x", "type": "train"}')
        assert False
    except jobmod.JobError:
        pass


def test_bad_id_rejected():
    try:
        jobmod.parse_job('{"id": "has space", "type": "smoke"}')
        assert False
    except jobmod.JobError:
        pass


def test_bad_type_rejected():
    try:
        jobmod.parse_job('{"id": "x", "type": "explode"}')
        assert False
    except jobmod.JobError:
        pass


# ---------------------------------------------------------- build_command
def _runtime():
    r = dict(cfg.RUNTIME_DEFAULTS)
    return r


def test_build_train_command():
    j = jobmod.parse_job('{"id": "b20a", "type": "train", "policy": "b20a-x",'
                         ' "env": {"SNEK_DISCOUNT": "0.9975"}, "max_steps": 1500000}')
    argv, env, log, policy = launch.build_command(j, _host(), _runtime())
    assert argv == ['/py', '-u', 'snek2.py', 'b20a-x']
    assert env['SNEK_DISCOUNT'] == '0.9975'
    assert env['SNEK_MAX_STEPS'] == '1500000'
    assert policy == 'b20a-x'


def test_build_smoke_sets_zero_checkpoint():
    j = jobmod.parse_job('{"id": "s1", "type": "smoke"}')
    argv, env, log, policy = launch.build_command(j, _host(), _runtime())
    assert policy == 'smoke'
    assert env['SNEK_MIN_CHECKPOINT_SCORE'] == '0'


def test_build_benchmark_defaults_steps_and_policy():
    j = jobmod.parse_job('{"id": "cap1", "type": "benchmark"}')
    argv, env, log, policy = launch.build_command(j, _host(), _runtime())
    assert policy == 'bench-cap1'
    assert env['SNEK_MAX_STEPS'] == '20000'
    assert env['SNEK_MIN_CHECKPOINT_SCORE'] == '0'


def test_build_eval_command_uses_worker_count():
    j = jobmod.parse_job('{"id": "e1", "type": "eval", "policy": "b19a", "eval_args": ["top20"]}')
    r = _runtime(); r['eval_workers'] = 10
    argv, env, log, policy = launch.build_command(j, _host(), r)
    assert argv == ['/py', '-u', 'eval_checkpoints.py', 'b19a', 'top20']
    assert env['EVAL_WORKERS'] == '10'


def test_build_eval_command_passes_hof_env_and_args_through():
    # The synthesized HOF job's argv and its 500-episode env must reach the process untouched,
    # while EVAL_WORKERS still comes from the runtime (the "4 parallel" of a regular eval).
    j = jobmod.parse_job(json.dumps({
        'id': 'p-hof', 'type': 'eval', 'policy': 'p', 'eval_args': ['above:98'],
        'env': {'EVAL_EPISODES': '500', 'EVAL_SCREEN_EPISODES': '0',
                'EVAL_MIN_ACHIEVABLE': '98', 'EVAL_OUT_SUFFIX': '_hof500'}}))
    r = _runtime(); r['eval_workers'] = 4
    argv, env, log, policy = launch.build_command(j, _host(), r)
    assert argv == ['/py', '-u', 'eval_checkpoints.py', 'p', 'above:98']
    assert env['EVAL_EPISODES'] == '500' and env['EVAL_SCREEN_EPISODES'] == '0'
    assert env['EVAL_MIN_ACHIEVABLE'] == '98' and env['EVAL_OUT_SUFFIX'] == '_hof500'
    assert env['EVAL_WORKERS'] == '4'


def test_build_threads_injected():
    j = jobmod.parse_job('{"id": "b", "type": "train", "policy": "p"}')
    r = _runtime(); r['tf_intraop_threads'] = 4; r['omp_num_threads'] = 4
    argv, env, log, policy = launch.build_command(j, _host(), r)
    assert env['TF_NUM_INTRAOP_THREADS'] == '4'
    assert env['OMP_NUM_THREADS'] == '4'


# ------------------------------------------------------- viewer file tracking
def test_viewer_png_paths_maps_category_to_dir():
    # A trainer's chart is runs/<policy>.png; an eval's is evals/<policy>_eval_progress.png.
    paths = runnermod.viewer_png_paths([('trainer', 'b20a-x'), ('eval', 'b20b-y')], '/snek')
    assert paths == ['/snek/evals/b20b-y_eval_progress.png', '/snek/runs/b20a-x.png']


def test_viewer_png_paths_skips_empty_policy_and_sorts():
    paths = runnermod.viewer_png_paths([('trainer', None), ('trainer', 'p2'),
                                        ('trainer', 'p1')], '/snek')
    assert paths == ['/snek/runs/p1.png', '/snek/runs/p2.png']


def test_viewer_relaunches_on_train_to_eval_flip():
    # The whole bug: same four policies, but the wave went from training to eval. The file
    # set changes even though the arms are identical, so a viewer bound to the training PNGs
    # must be relaunched -- otherwise it keeps showing the finished training charts.
    pols = ['b20m', 'b20n', 'b20o', 'b20p']
    train = runnermod.viewer_png_paths([('trainer', p) for p in pols], '/snek')
    ev = runnermod.viewer_png_paths([('eval', p) for p in pols], '/snek')
    assert train != ev
    assert runnermod.viewer_should_relaunch(True, train, ev) is True
    # ...and once it is showing the eval set, it stays put.
    assert runnermod.viewer_should_relaunch(True, ev, ev) is False


def test_viewer_launches_when_none_running_and_never_for_empty():
    assert runnermod.viewer_should_relaunch(False, None, ['/snek/runs/p.png']) is True
    assert runnermod.viewer_should_relaunch(False, None, []) is False
    # Unknown current set (across a daemon restart) with a viewer up -> relaunch to be safe.
    assert runnermod.viewer_should_relaunch(True, None, ['/snek/runs/p.png']) is True


def test_sticky_wave_keeps_a_finished_arm():
    # Four arms train; one hits its cap and its trainer exits, so viewer_png_paths now lists
    # only three. The sticky set must still carry the fourth, so the window shows four panels
    # (the finished one tagged completed) instead of collapsing to the arms still running.
    pols = ['b20q', 'b20r', 'b20s', 'b20t']
    full = runnermod.viewer_png_paths([('trainer', p) for p in pols], '/snek')
    prev = runnermod.sticky_wave_pngs([], None, 'trainer', full)
    assert prev == full
    three = runnermod.viewer_png_paths([('trainer', p) for p in pols[:3]], '/snek')
    kept = runnermod.sticky_wave_pngs(prev, 'trainer', 'trainer', three)
    assert kept == full, kept          # the exited arm's panel is retained
    # ...and because the set did not shrink, no relaunch is triggered by an arm finishing.
    assert runnermod.viewer_should_relaunch(True, prev, kept) is False


def test_sticky_wave_resets_on_category_flip():
    # A train->eval flip is a wholly new set of charts (runs/ -> evals/); the finished
    # training panels must not be unioned onto the eval window.
    pols = ['b20q', 'b20r']
    train = runnermod.viewer_png_paths([('trainer', p) for p in pols], '/snek')
    ev = runnermod.viewer_png_paths([('eval', p) for p in pols], '/snek')
    got = runnermod.sticky_wave_pngs(train, 'trainer', 'eval', ev)
    assert got == ev, got
    assert all('/runs/' not in p for p in got)


def test_sticky_wave_resets_on_a_new_same_category_wave():
    # The desktop bug: batch 21's eval close-out drained and batch 20's eval close-out launched
    # in the same dispatch, so `_ensure_viewer` never saw the idle poll that clears the set.
    # Both waves are category 'eval', so the category key alone unioned b21 onto b20 and the
    # (by then archived) b21 PNGs stuck on screen reading `(completed) (waiting…)`. A running
    # set disjoint from the previous one is the new-wave signal that must reset it.
    b21 = runnermod.viewer_png_paths([('eval', p) for p in ['b21a', 'b21b', 'b21c', 'b21d']], '/snek')
    b20 = runnermod.viewer_png_paths([('eval', p) for p in ['b20ae', 'b20af', 'b20ah']], '/snek')
    got = runnermod.sticky_wave_pngs(b21, 'eval', 'eval', b20)
    assert got == b20, got                 # b21 dropped, not unioned onto the new wave
    assert all('b21' not in p for p in got)


def _runner(auto_closeout=True):
    """A Runner with a throwaway ledger and no box behind it -- enough to exercise the pure
    ledger-driven logic (_auto_closeout_jobs) without git or spawn."""
    h = _host()
    fd, path = tempfile.mkstemp(suffix='.ledger')
    os.close(fd); os.unlink(path)          # a missing ledger loads as {}
    h['LEDGER_PATH'] = path
    r = runnermod.Runner(h)
    r.runtime['auto_closeout'] = auto_closeout
    return r


def test_wants_closeout_only_for_successful_training():
    assert runnermod.wants_closeout('train', True, True) is True
    assert runnermod.wants_closeout('train', False, True) is False   # failed run: no checkpoint
    assert runnermod.wants_closeout('eval', True, True) is False      # never close out an eval
    assert runnermod.wants_closeout('smoke', True, True) is False     # throwaway
    assert runnermod.wants_closeout('benchmark', True, True) is False
    assert runnermod.wants_closeout('train', True, False) is False    # feature disabled


def test_auto_closeout_synthesizes_eval_inheriting_env():
    r = _runner()
    r.ledger['b20a-fc25seed1'] = {'state': 'done', 'type': 'train',
                                  'policy': 'b20a-fc25seed1', 'closeout': 'pending',
                                  'env': {'SNEK_FC_LAYERS': '25,50,25', 'SNEK_SEED': '1'}}
    jobs = r._auto_closeout_jobs()
    assert len(jobs) == 1
    j = jobs[0]
    assert j.id == 'b20a-fc25seed1-closeout' and j.type == 'eval'
    assert j.policy == 'b20a-fc25seed1'
    assert j.env.get('SNEK_FC_LAYERS') == '25,50,25'   # the FC trap: width must carry over
    assert j.env['EVAL_MIN_ACHIEVABLE'] == '97'        # closeout gate pinned, not inherited
    assert j.eval_args == ['top20']
    assert j.priority == runnermod.AUTO_CLOSEOUT_PRIORITY
    assert j.priority < 100                             # beats a default-priority training


def test_closeout_gate_overrides_an_inherited_min_achievable():
    # A training env that set its own gate must not leak into the closeout -- the closeout
    # recipe wins, exactly like the HOF recipe does one hop later.
    r = _runner()
    r.ledger['b40a'] = {'state': 'done', 'type': 'train', 'policy': 'b40a',
                        'closeout': 'pending',
                        'env': {'SNEK_FC_LAYERS': '320', 'EVAL_MIN_ACHIEVABLE': '80'}}
    j = r._auto_closeout_jobs()[0]
    assert j.env['EVAL_MIN_ACHIEVABLE'] == '97'
    assert runnermod.CLOSEOUT_THRESHOLD < runnermod.HOF_THRESHOLD  # HOF above:98 stays readable


def test_auto_closeout_skips_when_eval_done_running_or_unmarked():
    r = _runner()
    # unmarked done training -> nothing (no retroactive closeouts for pre-feature arms)
    r.ledger['old'] = {'state': 'done', 'type': 'train', 'policy': 'old-arm'}
    assert r._auto_closeout_jobs() == []
    # marked, but its closeout already ran -> skip (idempotent, no second eval)
    r.ledger['p1'] = {'state': 'done', 'type': 'train', 'policy': 'p1', 'closeout': 'pending'}
    r.ledger['p1-closeout'] = {'state': 'done', 'type': 'eval', 'policy': 'p1'}
    # marked, but its closeout is currently running -> skip
    r.ledger['p2'] = {'state': 'done', 'type': 'train', 'policy': 'p2', 'closeout': 'pending'}
    r.running['p2-closeout'] = object()
    ids = [j.id for j in r._auto_closeout_jobs()]
    assert ids == [], ids


def test_auto_closeout_disabled_yields_nothing():
    r = _runner(auto_closeout=False)
    r.ledger['p'] = {'state': 'done', 'type': 'train', 'policy': 'p', 'closeout': 'pending'}
    assert r._auto_closeout_jobs() == []


# ----------------------------------------------- auto-HOF: the link after the closeout
class _FakeRJ:
    """The bit of launch.RunningJob that _reap touches, with no subprocess behind it."""
    def __init__(self, jid, jtype, policy, alive=False, rc=0):
        self.job = jobmod.Job(id=jid, type=jtype, policy=policy, env={}, priority=10)
        self.policy = policy
        self.log_path = '/nonexistent.log'
        self._alive, self._rc = alive, rc

    def is_alive(self):
        return self._alive

    def returncode(self):
        return self._rc


def test_wants_hof_only_for_a_successful_closeout_eval():
    assert runnermod.wants_hof('p-closeout', 'eval', True, True) is True
    assert runnermod.wants_hof('p-closeout', 'eval', False, True) is False  # failed: no result file
    assert runnermod.wants_hof('p-hof', 'eval', True, True) is False         # no HOF-of-a-HOF
    assert runnermod.wants_hof('manual-top20', 'eval', True, True) is False  # a hand-queued eval
    assert runnermod.wants_hof('p', 'train', True, True) is False            # a training gets a closeout
    assert runnermod.wants_hof('p-closeout', 'eval', True, False) is False   # feature disabled


def test_auto_hof_synthesizes_a_500ep_flat_eval_inheriting_env():
    r = _runner()
    r.ledger['b25a-fc320seed1-closeout'] = {
        'state': 'done', 'type': 'eval', 'policy': 'b25a-fc320seed1', 'hof': 'pending',
        'env': {'SNEK_FC_LAYERS': '320', 'SNEK_SEED': '1'}}
    jobs = r._auto_hof_jobs()
    assert len(jobs) == 1
    j = jobs[0]
    assert j.id == 'b25a-fc320seed1-hof' and j.type == 'eval'
    assert j.policy == 'b25a-fc320seed1'
    assert j.eval_args == ['above:98']                 # selects the closeout's >=98% checkpoints
    assert j.env['SNEK_FC_LAYERS'] == '320'            # FC trap survives both hops (train->closeout->hof)
    assert j.env['EVAL_EPISODES'] == '500'
    assert j.env['EVAL_SCREEN_EPISODES'] == '0'        # flat, no screening
    assert j.env['EVAL_INDEPENDENT'] == '1'
    assert j.env['EVAL_MIN_ACHIEVABLE'] == '98'
    assert j.env['EVAL_OUT_SUFFIX'] == '_hof500'       # a distinct file, never clobbers the closeout
    assert j.priority == runnermod.AUTO_HOF_PRIORITY
    assert runnermod.AUTO_CLOSEOUT_PRIORITY < j.priority < 100  # after the closeout, before a training


def test_auto_hof_recipe_wins_over_an_inherited_eval_env():
    # The training env could carry stale EVAL_* knobs; the 500-episode recipe must override them.
    r = _runner()
    r.ledger['p-closeout'] = {'state': 'done', 'type': 'eval', 'policy': 'p', 'hof': 'pending',
                              'env': {'EVAL_EPISODES': '100', 'EVAL_OUT_SUFFIX': '_stale'}}
    j = r._auto_hof_jobs()[0]
    assert j.env['EVAL_EPISODES'] == '500'
    assert j.env['EVAL_OUT_SUFFIX'] == '_hof500'


def test_auto_hof_skips_when_done_running_or_unmarked():
    r = _runner()
    # a finished closeout with no hof marker (a pre-feature arm) -> nothing retroactive
    r.ledger['old-closeout'] = {'state': 'done', 'type': 'eval', 'policy': 'old'}
    assert r._auto_hof_jobs() == []
    # marked, but its hof already ran -> skip (idempotent)
    r.ledger['p1-closeout'] = {'state': 'done', 'type': 'eval', 'policy': 'p1', 'hof': 'pending'}
    r.ledger['p1-hof'] = {'state': 'done', 'type': 'eval', 'policy': 'p1'}
    # marked, but its hof is running -> skip
    r.ledger['p2-closeout'] = {'state': 'done', 'type': 'eval', 'policy': 'p2', 'hof': 'pending'}
    r.running['p2-hof'] = object()
    assert [j.id for j in r._auto_hof_jobs()] == []


def test_auto_hof_disabled_yields_nothing():
    r = _runner()
    r.runtime['auto_hof'] = False
    r.ledger['p-closeout'] = {'state': 'done', 'type': 'eval', 'policy': 'p', 'hof': 'pending'}
    assert r._auto_hof_jobs() == []


def test_reap_marks_a_finished_closeout_for_its_hof():
    r = _runner()
    r._publish_results = lambda rj: None               # no git behind this test
    r.running['p-closeout'] = _FakeRJ('p-closeout', 'eval', 'p', alive=False, rc=0)
    r._reap()
    assert r.ledger['p-closeout']['state'] == 'done'
    assert r.ledger['p-closeout'].get('hof') == 'pending'   # .get so a missing marker asserts, not KeyErrors
    assert [j.id for j in r._auto_hof_jobs()] == ['p-hof']


def test_reap_does_not_hof_a_plain_eval_or_a_failed_closeout():
    r = _runner()
    r._publish_results = lambda rj: None
    r.running['manual-top20'] = _FakeRJ('manual-top20', 'eval', 'x', alive=False, rc=0)
    r.running['q-closeout'] = _FakeRJ('q-closeout', 'eval', 'q', alive=False, rc=1)  # failed
    r._reap()
    assert 'hof' not in r.ledger['manual-top20']       # a hand-queued eval never triggers one
    assert r.ledger['q-closeout']['state'] == 'failed'
    assert 'hof' not in r.ledger['q-closeout']          # a failed closeout has no result to select from


def test_scan_pending_includes_auto_hofs_in_priority_order():
    r = _runner()
    r.ledger['p-closeout'] = {'state': 'done', 'type': 'eval', 'policy': 'p', 'hof': 'pending'}
    _stub_pending([('t.json', {'id': 't', 'type': 'train', 'policy': 't', 'priority': 100})])
    got = [j.id for j in r._scan_pending()]
    assert got == ['p-hof', 't'], got                  # the hof (priority 11) sorts ahead of the training


def test_scan_pending_queues_a_closeout_and_a_hof_together():
    r = _runner()
    r.ledger['a'] = {'state': 'done', 'type': 'train', 'policy': 'a', 'closeout': 'pending'}
    r.ledger['b-closeout'] = {'state': 'done', 'type': 'eval', 'policy': 'b', 'hof': 'pending'}
    _stub_pending([])
    got = [j.id for j in r._scan_pending()]
    assert got == ['a-closeout', 'b-hof'], got         # closeout (10) before hof (11)


def test_a_dead_closeout_across_a_daemon_restart_queues_its_hof():
    # Same boot, dead pid: the detached closeout ran to its end while the daemon was down, so it
    # earns its HOF re-measure the same way a straddling training earns its closeout.
    real_boot, real_alive = runnermod.boot_id, launch.pid_alive
    runnermod.boot_id = lambda: 'BOOT-A'
    launch.pid_alive = lambda _pid: False
    try:
        r = _runner()
        r.ledger['p-closeout'] = {'state': 'running', 'type': 'eval', 'policy': 'p',
                                  'pid': 4242, 'boot': 'BOOT-A', 'env': {'SNEK_FC_LAYERS': '320'}}
        r._reattach()
    finally:
        runnermod.boot_id, launch.pid_alive = real_boot, real_alive
    got = r.ledger['p-closeout']
    assert got['state'] == 'done' and got['hof'] == 'pending', got
    jobs = r._auto_hof_jobs()
    assert [j.id for j in jobs] == ['p-hof']
    assert jobs[0].env['SNEK_FC_LAYERS'] == '320'      # the FC trap survives the restart


def test_a_reboot_withholds_a_closeouts_hof_until_it_actually_finishes():
    # A closeout the machine killed is not done, so it must not spend its (partial) result on a
    # HOF re-measure -- it re-runs, and only the finished re-run earns the hof.
    real_boot, real_alive = runnermod.boot_id, launch.pid_alive
    runnermod.boot_id = lambda: 'BOOT-B'
    launch.pid_alive = lambda _pid: False
    try:
        r = _runner()
        r.ledger['p-closeout'] = {'state': 'running', 'type': 'eval', 'policy': 'p',
                                  'pid': 4242, 'boot': 'BOOT-A'}
        r._reattach()
    finally:
        runnermod.boot_id, launch.pid_alive = real_boot, real_alive
    got = r.ledger['p-closeout']
    assert got['state'] == 'interrupted', got
    assert 'hof' not in got, got
    assert r._auto_hof_jobs() == []


def test_viewer_scale_is_larger_for_eval_waves():
    # Eval charts get a ~30% bigger window; anything with a trainer in it stays smaller.
    # `category` is the comma-joined set _ensure_viewer builds, so only a pure eval wave
    # is 'eval' -- a mixed set (should not happen under the wave-barrier, but be safe) is not.
    assert runnermod.viewer_scale('eval') == '1.95'
    assert runnermod.viewer_scale('trainer') == '1.5'
    assert runnermod.viewer_scale('eval,trainer') == '1.5'


def test_sticky_wave_resets_when_idle_or_fresh():
    # prev_category None -- a fresh daemon, or the idle gap between waves that clears the
    # tracking -- starts the next wave from its own arms, never the previous batch's.
    new = runnermod.viewer_png_paths([('trainer', 'b21a')], '/snek')
    assert runnermod.sticky_wave_pngs(['/snek/runs/b20a.png'], None, 'trainer', new) == new


def _stub_pending(monkey_jobs):
    """Point gitbus.read_pending_jobs at a fixed list of (filename, json-text) specs."""
    texts = [(name, json.dumps(spec)) for name, spec in monkey_jobs]
    runnermod.gitbus.read_pending_jobs = lambda host: list(texts)


def test_scan_pending_orders_by_priority_and_drops_running_and_terminal():
    r = _runner(auto_closeout=False)
    _stub_pending([
        ('c.json', {'id': 'c', 'type': 'train', 'policy': 'c', 'priority': 50}),
        ('a.json', {'id': 'a', 'type': 'train', 'policy': 'a', 'priority': 10}),
        ('b.json', {'id': 'b', 'type': 'train', 'policy': 'b', 'priority': 30}),
        ('gone.json', {'id': 'gone', 'type': 'train', 'policy': 'gone', 'priority': 5}),
        ('live.json', {'id': 'live', 'type': 'train', 'policy': 'live', 'priority': 1}),
    ])
    r.ledger['gone'] = {'state': 'done'}   # terminal -> not queued
    r.running['live'] = object()            # already running -> not queued
    got = [j.id for j in r._scan_pending()]
    assert got == ['a', 'b', 'c'], got      # priority order, terminal + running excluded


def test_scan_pending_marks_a_malformed_spec_failed_and_persists():
    r = _runner(auto_closeout=False)
    _stub_pending([('bad.json', {'id': 'bad', 'type': 'not-a-type'})])
    assert r._scan_pending() == []
    assert r.ledger['bad']['state'] == 'failed'
    # persisted synchronously, since a scan can run while dispatch is skipped (paused/drain)
    with open(r.host['LEDGER_PATH']) as fh:
        assert json.load(fh)['bad']['state'] == 'failed'


def test_scan_pending_includes_auto_closeouts_in_priority_order():
    r = _runner(auto_closeout=True)
    r.ledger['p'] = {'state': 'done', 'type': 'train', 'policy': 'p', 'closeout': 'pending'}
    _stub_pending([('t.json', {'id': 't', 'type': 'train', 'policy': 't', 'priority': 100})])
    got = [j.id for j in r._scan_pending()]
    # the synthesized closeout (priority AUTO_CLOSEOUT_PRIORITY < 100) sorts ahead of the training
    assert got == ['p-closeout', 't'], got


def test_publish_status_folds_the_queue_into_the_ledger_in_order():
    r = _runner(auto_closeout=False)
    r.host['REPO_PATH'] = '/'   # _publish reads disk-free for this path
    r.ledger['done-arm'] = {'state': 'done', 'type': 'train'}   # history stays in the ledger
    captured = {}
    runnermod.gitbus.publish_status = lambda host, text: captured.setdefault('text', text)
    _stub_pending([
        ('b.json', {'id': 'b', 'type': 'train', 'policy': 'b', 'priority': 30}),
        ('a.json', {'id': 'a', 'type': 'eval', 'policy': 'a', 'priority': 10}),
    ])
    r._queued = r._scan_pending()
    r._publish()
    published = json.loads(captured['text'])
    ledger = published['ledger']
    # newest/active on top: queued jobs (launch order, a@10 before b@30), then finished history
    assert list(ledger.keys()) == ['a', 'b', 'done-arm'], list(ledger.keys())
    assert ledger['a'] == 'queued' and ledger['b'] == 'queued'
    assert ledger['done-arm'] == 'done'
    # the at-a-glance summary rides at the top of status.json, listing the queued batches
    assert list(published)[2] == 'at_a_glance', list(published)[:3]
    assert published['at_a_glance']['running'] == []
    assert any(line.split()[0] == 'a' for line in published['at_a_glance']['queued'])


AUTO = runnermod.AUTO_CLOSEOUT_PRIORITY


def test_anticipated_queue_interleaves_a_closeout_batch_between_training_batches():
    # Two training batches queued; each batch's closeouts, then its HOF re-measures, run before
    # the next batch's training -- a closeout (priority 10) and a HOF (11) both outrank a queued
    # training (200), and the HOF trails its own closeout because it is only spawned once the
    # closeout has been placed in a wave.
    queued = [{'id': p, 'type': 'train', 'policy': p, 'priority': 200}
              for p in ('a1', 'a2', 'b1', 'b2')]
    order = runnermod.anticipated_queue(
        queued, [], {'trainer': 2, 'eval': 2}, True, {'a1', 'a2', 'b1', 'b2'})
    assert [j['id'] for j in order] == [
        'a1', 'a2', 'a1-closeout', 'a2-closeout', 'a1-hof', 'a2-hof',
        'b1', 'b2', 'b1-closeout', 'b2-closeout', 'b1-hof', 'b2-hof'], [j['id'] for j in order]


def test_anticipated_queue_seeds_closeouts_for_trainings_running_now():
    # A training on the box now will spawn a closeout, which in turn spawns a HOF, all before
    # the queued batch.
    queued = [{'id': 'b1', 'type': 'train', 'policy': 'b1', 'priority': 200}]
    running = [{'id': 'r1', 'type': 'train', 'policy': 'r1'}]
    order = runnermod.anticipated_queue(
        queued, running, {'trainer': 2, 'eval': 2}, True, {'b1', 'r1'})
    assert [j['id'] for j in order] == [
        'r1-closeout', 'r1-hof', 'b1', 'b1-closeout', 'b1-hof'], [j['id'] for j in order]


def test_anticipated_queue_omits_closeouts_when_auto_closeout_off():
    queued = [{'id': 'a1', 'type': 'train', 'policy': 'a1', 'priority': 200}]
    order = runnermod.anticipated_queue(queued, [], {'trainer': 2, 'eval': 2}, False, {'a1'})
    assert [j['id'] for j in order] == ['a1']


def test_anticipated_queue_never_invents_an_existing_closeout_twice():
    # The real closeout for a finished training already sits in the queue; do not duplicate it.
    queued = [{'id': 'p', 'type': 'train', 'policy': 'p', 'priority': 200},
              {'id': 'p-closeout', 'type': 'eval', 'policy': 'p', 'priority': AUTO}]
    order = runnermod.anticipated_queue(
        queued, [], {'trainer': 2, 'eval': 2}, True, {'p', 'p-closeout'})
    ids = [j['id'] for j in order]
    assert ids.count('p-closeout') == 1, ids


def test_anticipated_queue_chains_a_hof_after_each_closeout():
    queued = [{'id': 't', 'type': 'train', 'policy': 't', 'priority': 200}]
    order = runnermod.anticipated_queue(queued, [], {'trainer': 2, 'eval': 2}, True, {'t'})
    assert [j['id'] for j in order] == ['t', 't-closeout', 't-hof'], [j['id'] for j in order]


def test_anticipated_queue_seeds_a_hof_for_a_closeout_running_now():
    running = [{'id': 'r-closeout', 'type': 'eval', 'policy': 'r'}]
    order = runnermod.anticipated_queue(
        [], running, {'trainer': 2, 'eval': 2}, True, {'r-closeout'})
    assert [j['id'] for j in order] == ['r-hof'], [j['id'] for j in order]


def test_anticipated_queue_omits_hofs_when_auto_hof_off():
    queued = [{'id': 't', 'type': 'train', 'policy': 't', 'priority': 200}]
    order = runnermod.anticipated_queue(
        queued, [], {'trainer': 2, 'eval': 2}, True, {'t'}, auto_hof=False)
    assert [j['id'] for j in order] == ['t', 't-closeout'], [j['id'] for j in order]


def test_anticipated_queue_never_invents_an_existing_hof_twice():
    queued = [{'id': 'p-closeout', 'type': 'eval', 'policy': 'p', 'priority': AUTO},
              {'id': 'p-hof', 'type': 'eval', 'policy': 'p', 'priority': runnermod.AUTO_HOF_PRIORITY}]
    order = runnermod.anticipated_queue(
        queued, [], {'trainer': 2, 'eval': 2}, True, {'p-closeout', 'p-hof'})
    ids = [j['id'] for j in order]
    assert ids.count('p-hof') == 1, ids


def test_ledger_view_anticipates_closeouts_for_a_queued_training():
    r = _runner(auto_closeout=True)
    r.runtime['max_trainers'] = 2
    r.runtime['max_evals'] = 2

    class _J:
        def __init__(self, jid, pri):
            self.id, self.policy, self.priority, self.type = jid, jid, pri, 'train'
    r._queued = [_J('t1', 200), _J('t2', 200)]
    view = r._ledger_view(r._anticipated_order())
    assert list(view.keys()) == ['t1', 't2', 't1-closeout', 't2-closeout',
                                 't1-hof', 't2-hof'], list(view.keys())
    assert all(view[k] == 'queued' for k in view)


def test_ledger_view_places_queued_first_then_running_then_finished():
    r = _runner(auto_closeout=False)
    r.ledger['hist'] = {'state': 'done'}
    r.ledger['run1'] = {'state': 'running', 'type': 'train'}

    class _J:
        def __init__(self, jid):
            self.id, self.policy, self.priority, self.type = jid, jid, 200, 'train'
    r._queued = [_J('q1')]
    view = r._ledger_view(r._anticipated_order())
    # newest/active on top: queued, then running, then the finished history
    assert list(view.keys()) == ['q1', 'run1', 'hist'], list(view.keys())
    assert view['q1'] == 'queued' and view['run1'] == 'running' and view['hist'] == 'done'


def test_ledger_view_lets_a_real_state_win_over_queued_on_overlap():
    # A job re-queued while its prior run is still in the ledger keeps its real state, not
    # the synthetic 'queued' -- the real states are applied last.
    r = _runner(auto_closeout=False)
    r.ledger['x'] = {'state': 'running', 'type': 'train'}
    class _J:  # minimal stand-in for a queued Job
        id, policy, priority, type = 'x', 'x', 100, 'train'
    r._queued = [_J()]
    assert r._ledger_view(r._anticipated_order())['x'] == 'running'


# ------------------------------------------------- boot id: reboot vs daemon restart
def _ledger_runner(**recs):
    r = _runner()
    r.ledger.update(recs)
    return r


def test_boot_id_returns_none_where_the_kernel_publishes_nothing():
    # Cached in a module-level list, so the cache has to be cleared between probes.
    real = runnermod.BOOT_ID_PATH
    runnermod._BOOT_ID[:] = []
    runnermod.BOOT_ID_PATH = '/definitely/not/a/path'
    try:
        assert runnermod.boot_id() is None
    finally:
        runnermod.BOOT_ID_PATH = real
        runnermod._BOOT_ID[:] = []


def test_boot_id_reads_and_caches_the_file():
    real = runnermod.BOOT_ID_PATH
    with tempfile.NamedTemporaryFile('w', delete=False) as fh:
        fh.write('abc-123\n')
        path = fh.name
    runnermod._BOOT_ID[:] = []
    runnermod.BOOT_ID_PATH = path
    try:
        assert runnermod.boot_id() == 'abc-123'
        os.unlink(path)                      # gone, but the cache answers anyway
        assert runnermod.boot_id() == 'abc-123'
    finally:
        runnermod.BOOT_ID_PATH = real
        runnermod._BOOT_ID[:] = []


def _reattach_with(rec, boot, pid_alive):
    """Runs _reattach over one running record under a chosen boot id and pid liveness."""
    real_boot, real_alive = runnermod.boot_id, launch.pid_alive
    runnermod.boot_id = lambda: boot
    launch.pid_alive = lambda _pid: pid_alive
    try:
        r = _runner()
        r.ledger['j'] = dict(rec)
        r._reattach()
        return r
    finally:
        runnermod.boot_id, launch.pid_alive = real_boot, real_alive


def test_a_reboot_marks_a_training_interrupted_and_withholds_the_closeout():
    # The whole point: a training the machine killed is NOT done, so it must not be published
    # as a finished arm and must not spend its closeout on a partial checkpoint set.
    rec = {'state': 'running', 'type': 'train', 'policy': 'p', 'pid': 4242, 'boot': 'BOOT-A'}
    r = _reattach_with(rec, boot='BOOT-B', pid_alive=False)
    got = r.ledger['j']
    assert got['state'] == 'interrupted', got
    assert 'closeout' not in got, got            # withheld until the arm actually finishes
    assert got['restarts'] == 1
    assert r.running == {}
    assert r._auto_closeout_jobs() == []         # nothing to evaluate yet


def test_a_reboot_ignores_a_live_pid_because_it_may_be_recycled():
    # pids restart low after a boot, so a stored pid can match an unrelated process. On a
    # boot-id mismatch the pid is not consulted at all -- otherwise the runner adopts a
    # phantom that never exits and the wave barrier idles the box forever.
    rec = {'state': 'running', 'type': 'train', 'policy': 'p', 'pid': 1, 'boot': 'BOOT-A'}
    r = _reattach_with(rec, boot='BOOT-B', pid_alive=True)
    assert r.ledger['j']['state'] == 'interrupted'
    assert r.running == {}


def test_an_interrupted_closeout_is_resynthesized_rather_than_consumed():
    # A closeout cut short by a reboot used to read `done`, which made
    # _auto_closeout_jobs skip it forever -- the arm was never evaluated again.
    r = _runner()
    r.ledger['p'] = {'state': 'done', 'type': 'train', 'policy': 'p', 'closeout': 'pending',
                     'env': {'SNEK_FC_LAYERS': '320'}}
    r.ledger['p-closeout'] = {'state': 'interrupted', 'type': 'eval', 'policy': 'p'}
    jobs = r._auto_closeout_jobs()
    assert [j.id for j in jobs] == ['p-closeout'], jobs
    assert jobs[0].env.get('SNEK_FC_LAYERS') == '320'   # the FC trap survives the retry


def test_an_interrupted_job_is_not_terminal_so_its_spec_relaunches():
    r = _runner()
    r.ledger['t1'] = {'state': 'interrupted', 'type': 'train', 'policy': 't1'}
    r.ledger['t2'] = {'state': 'done', 'type': 'train', 'policy': 't2'}
    jobs = [jobmod.Job(id='t1', type='train', policy='t1', env={}, priority=100),
            jobmod.Job(id='t2', type='train', policy='t2', env={}, priority=100)]
    real = runnermod.gitbus.read_pending_jobs
    runnermod.gitbus.read_pending_jobs = lambda host: [
        (j.id + '.json', json.dumps({'id': j.id, 'type': 'train', 'policy': j.policy}))
        for j in jobs]
    try:
        ids = [j.id for j in r._scan_pending()]
    finally:
        runnermod.gitbus.read_pending_jobs = real
    assert 't1' in ids, ids          # interrupted -> relaunched, which is the recovery
    assert 't2' not in ids, ids      # done -> terminal, never relaunched


def test_a_daemon_restart_on_the_same_boot_still_reads_a_dead_pid_as_finished():
    # The original behaviour, which must survive: detached jobs outlive a daemon restart, so
    # same boot + dead pid means the job genuinely ran to its end and earns its closeout.
    rec = {'state': 'running', 'type': 'train', 'policy': 'p', 'pid': 4242, 'boot': 'BOOT-A'}
    r = _reattach_with(rec, boot='BOOT-A', pid_alive=False)
    got = r.ledger['j']
    assert got['state'] == 'done', got
    assert got['closeout'] == 'pending'


def test_a_daemon_restart_readopts_a_live_job_and_backfills_a_missing_boot_id():
    # A record written before the boot field existed re-adopts on the pid as before, and gets
    # stamped so the *next* restart can classify it properly.
    rec = {'state': 'running', 'type': 'train', 'policy': 'p', 'pid': 4242}
    r = _reattach_with(rec, boot='BOOT-A', pid_alive=True)
    assert r.ledger['j']['state'] == 'running'
    assert r.ledger['j']['boot'] == 'BOOT-A'
    assert 'j' in r.running


def test_no_boot_id_available_falls_back_to_the_pid_only_behaviour():
    # On a box with no procfs, boot_id() is None and nothing may be inferred from it.
    rec = {'state': 'running', 'type': 'train', 'policy': 'p', 'pid': 4242, 'boot': 'BOOT-A'}
    assert _reattach_with(rec, boot=None, pid_alive=False).ledger['j']['state'] == 'done'
    assert _reattach_with(rec, boot=None, pid_alive=True).ledger['j']['state'] == 'running'


def test_restarts_carries_across_a_relaunch():
    r = _runner()
    r.ledger['j'] = {'state': 'interrupted', 'type': 'train', 'policy': 'p', 'restarts': 2}
    real = launch.spawn
    launch.spawn = lambda job, host, runtime: launch.RunningJob(job, 'p', 999, '/l')
    try:
        r._launch(jobmod.Job(id='j', type='train', policy='p', env={}, priority=100))
    finally:
        launch.spawn = real
    assert r.ledger['j']['restarts'] == 2
    assert r.ledger['j']['state'] == 'running'


def test_a_fresh_launch_records_the_boot_id():
    r = _runner()
    real_spawn, real_boot = launch.spawn, runnermod.boot_id
    launch.spawn = lambda job, host, runtime: launch.RunningJob(job, 'p', 999, '/l')
    runnermod.boot_id = lambda: 'BOOT-Z'
    try:
        r._launch(jobmod.Job(id='j', type='train', policy='p', env={}, priority=100))
    finally:
        launch.spawn, runnermod.boot_id = real_spawn, real_boot
    assert r.ledger['j']['boot'] == 'BOOT-Z'
    assert 'restarts' not in r.ledger['j']     # a first run carries no counter


# --------------------------------------------------------- stale git lock clearing
def _worktree_repo():
    """A real git repo plus a linked worktree, so --git-path resolves the way it does on the box
    (a worktree's .git is a *file*, and its locks live in .git/worktrees/<name>/)."""
    root = tempfile.mkdtemp()
    main, wt = os.path.join(root, 'main'), os.path.join(root, 'wt')
    os.makedirs(main)
    run = lambda *a: subprocess.run(list(a), cwd=main, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, check=True)
    run('git', 'init', '-q')
    run('git', 'config', 'user.email', 't@t')
    run('git', 'config', 'user.name', 't')
    open(os.path.join(main, 'f'), 'w').write('x')
    run('git', 'add', 'f')
    run('git', 'commit', '-qm', 'init')
    run('git', 'worktree', 'add', '-q', '-b', 'side', wt)
    return root, wt


def test_clear_stale_locks_removes_an_old_index_lock_in_a_worktree():
    root, wt = _worktree_repo()
    try:
        lock = gitbus._git(['rev-parse', '--git-path', 'index.lock'], cwd=wt).strip()
        if not os.path.isabs(lock):
            lock = os.path.join(wt, lock)
        open(lock, 'w').close()
        os.utime(lock, (time.time() - 3600, time.time() - 3600))
        assert os.path.exists(lock)
        assert gitbus.clear_stale_locks(wt) == [lock]
        assert not os.path.exists(lock)
        # And git works again afterwards, which is the actual point.
        open(os.path.join(wt, 'g'), 'w').write('y')
        gitbus._git(['add', 'g'], cwd=wt)
        assert gitbus._git(['status', '--porcelain'], cwd=wt).strip()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_clear_stale_locks_leaves_a_young_lock_alone():
    # A live git holds index.lock for milliseconds; deleting a fresh one would corrupt a
    # concurrent write, so the age gate is the safety property being pinned here.
    root, wt = _worktree_repo()
    try:
        lock = gitbus._git(['rev-parse', '--git-path', 'index.lock'], cwd=wt).strip()
        if not os.path.isabs(lock):
            lock = os.path.join(wt, lock)
        open(lock, 'w').close()
        assert gitbus.clear_stale_locks(wt) == []
        assert os.path.exists(lock)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_clear_stale_locks_is_a_noop_with_no_locks():
    root, wt = _worktree_repo()
    try:
        assert gitbus.clear_stale_locks(wt) == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    mod = sys.modules[__name__]
    tests = [t for t in dir(mod) if t.startswith('test')]
    fails = 0
    for name in tests:
        try:
            getattr(mod, name)()
        except Exception as e:  # noqa
            print('FAIL', name, type(e).__name__, e)
            fails += 1
    print(len(tests), 'tests,', fails, 'failed')
    sys.exit(1 if fails else 0)


# ------------------------------------------------- batch/phase parsing + at-a-glance summary

def test_batch_of_reads_the_leading_b_number():
    assert runnermod.batch_of('b41a-b29repro-seed1') == 'b41'
    assert runnermod.batch_of('b41a-b29repro-seed1-closeout') == 'b41'
    assert runnermod.batch_of('b40d-chasefree10g75seed4-hof') == 'b40'
    assert runnermod.batch_of('smoke-1') == 'smoke'   # non-b id falls back to the head


def test_phase_of_reads_the_suffix_then_the_type():
    assert runnermod.phase_of('b41a-x-closeout', 'eval') == 'closeout eval'
    assert runnermod.phase_of('b41a-x-hof', 'eval') == 'hof'
    assert runnermod.phase_of('b41a-x', 'train') == 'training'
    assert runnermod.phase_of('b41a-x', 'eval') == 'eval'   # a hand-queued eval, no suffix


def test_at_a_glance_running_line_averages_percent_across_a_batch():
    running = [
        {'id': 'b41a-x-seed1', 'type': 'train', 'policy': 'p1', 'step': 500000, 'max_steps': 2000000},
        {'id': 'b41b-x-seed2', 'type': 'train', 'policy': 'p2', 'step': 1500000, 'max_steps': 2000000},
    ]
    g = runnermod.build_at_a_glance(running, [], {'b41': 'b29 re-run (same seeds), gate=75 c=0.10'})
    assert g['queued'] == []
    assert len(g['running']) == 1, g['running']            # both arms fold into one batch line
    line = g['running'][0]
    assert line.startswith('b41 -- b29 re-run (same seeds), gate=75 c=0.10 -- training'), line
    assert '50%' in line and '(2 arms)' in line, line      # mean of 25% and 75%


def test_at_a_glance_running_line_without_steps_shows_no_percent():
    running = [{'id': 'b40a-x-seed1-closeout', 'type': 'eval', 'policy': 'p'}]
    g = runnermod.build_at_a_glance(running, [], {})
    assert g['running'] == ['b40 -- closeout eval (1 arm)'], g['running']


def test_at_a_glance_queued_lists_one_line_per_batch_phase():
    order = [
        {'id': 'b41a-x-seed1', 'type': 'train', 'policy': 'p1'},
        {'id': 'b41b-x-seed2', 'type': 'train', 'policy': 'p2'},
        {'id': 'b41a-x-seed1-closeout', 'type': 'eval', 'policy': 'p1'},
        {'id': 'b41b-x-seed2-closeout', 'type': 'eval', 'policy': 'p2'},
        {'id': 'b41a-x-seed1-hof', 'type': 'eval', 'policy': 'p1'},
    ]
    g = runnermod.build_at_a_glance([], order, {'b41': 'demo'})
    assert g['running'] == []
    assert g['queued'] == [
        'b41 training -- demo -- queued (2 arms)',
        'b41 closeout eval -- demo -- queued (2 arms)',
        'b41 hof -- demo -- queued (1 arm)',
    ], g['queued']


def test_parse_job_reads_an_optional_label():
    j = jobmod.parse_job(json.dumps(
        {'id': 'b41a', 'type': 'train', 'policy': 'b41a', 'label': 'b41: b29 re-run'}))
    assert j.label == 'b41: b29 re-run'
    j2 = jobmod.parse_job(json.dumps({'id': 'b41a', 'type': 'train', 'policy': 'b41a'}))
    assert j2.label == ''                                  # optional, defaults empty


def test_ledger_view_orders_finished_most_recent_first():
    r = _runner(auto_closeout=False)
    r.ledger['old'] = {'state': 'done', 'finished': 100.0}
    r.ledger['new'] = {'state': 'done', 'finished': 300.0}
    r.ledger['mid'] = {'state': 'failed', 'finished': 200.0}
    r._queued = []
    keys = list(r._ledger_view(r._anticipated_order()).keys())
    assert keys == ['new', 'mid', 'old'], keys
