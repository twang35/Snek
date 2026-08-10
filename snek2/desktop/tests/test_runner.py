"""Unit tests for the pure runner logic: config load/clamp/fallback, job parsing,
and command building. The git/subprocess glue (gitbus, spawn, the loop) needs a
real box and is not exercised here.

Run directly:  PYTHONPATH=.. python test_runner.py    (from snek2/desktop/tests)
or import and call the test_* functions like the snek2/tests suite.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runner import config as cfg
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
