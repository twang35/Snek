"""Unit tests for the pure runner logic: config load/clamp/fallback, job parsing,
and command building. The git/subprocess glue (gitbus, spawn, the loop) needs a
real box and is not exercised here.

Run directly:  PYTHONPATH=.. python test_runner.py    (from snek2/desktop/tests)
or import and call the test_* functions like the snek2/tests suite.
"""
import ast
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


def test_the_box_defaults_to_the_vectorised_engine():
    """The default lives in `RUNTIME_DEFAULTS`, not in `build_command`'s `.get` fallback -- so this is
    where it has to be pinned. A box with no `eval_engine` in its `runtime.json` (which is every box
    until someone adds one) reads its engine from here."""
    assert cfg.RUNTIME_DEFAULTS['eval_engine'] == 'vec'
    assert cfg.RUNTIME_DEFAULTS['vec_wave_procs'] == 0     # 0 = let vec_wave read the core count
    empty, notes = cfg.parse_runtime_config('{}', _host())
    assert empty['eval_engine'] == 'vec'


def test_an_eval_runs_the_vectorised_wave_by_default():
    """The default engine moved to `vec` on 2026-08-24. `EVAL_WORKERS`/`EVAL_LANES` size *TF worker
    processes* and this engine has none, so they must not be set -- a value silently ignored is how
    someone concludes a wave ran with four workers when it ran twelve shards."""
    j = jobmod.parse_job('{"id": "e1", "type": "eval", "policy": "b19a", "eval_args": ["top50"]}')
    argv, env, log, policy = launch.build_command(j, _host(), _runtime())
    assert argv == ['/py', '-u', 'vectorized/vec_wave.py', 'top50', 'b19a']
    assert 'EVAL_WORKERS' not in env and 'EVAL_LANES' not in env


def test_the_scalar_engine_is_still_reachable_and_still_sets_both_knobs():
    """The opt-out. It is the only way to reproduce a pre-switch measurement, and the only answer to
    a regression in the new engine that does not need a deploy."""
    j = jobmod.parse_job('{"id": "e1", "type": "eval", "policy": "b19a", "eval_args": ["top50"]}')
    r = _runtime(); r['eval_engine'] = 'scalar'; r['eval_workers'] = 4; r['eval_lanes'] = 4
    argv, env, log, policy = launch.build_command(j, _host(), r)
    assert argv == ['/py', '-u', 'eval_wave.py', 'top50', 'b19a']
    assert env['EVAL_WORKERS'] == '4' and env['EVAL_LANES'] == '4'


def test_a_job_can_pick_the_engine_without_a_runtime_edit():
    """`runtime.json` is one setting for the whole box; a single arm needing the old engine -- to
    reproduce a number, or because something looks wrong -- must not require draining it."""
    j = jobmod.parse_job(json.dumps({'id': 'e', 'type': 'eval', 'policy': 'p',
                                     'eval_args': ['top50'],
                                     'env': {'SNEK_EVAL_ENGINE': 'scalar'}}))
    argv, env, log, policy = launch.build_command(j, _host(), _runtime())
    assert argv[2] == 'eval_wave.py'
    # And the other direction, against a box configured for the scalar path.
    j = jobmod.parse_job(json.dumps({'id': 'e', 'type': 'eval', 'policy': 'p',
                                     'eval_args': ['top50'],
                                     'env': {'SNEK_EVAL_ENGINE': 'vec'}}))
    r = _runtime(); r['eval_engine'] = 'scalar'
    argv, env, log, policy = launch.build_command(j, _host(), r)
    assert argv[2] == 'vectorized/vec_wave.py'


def test_an_unknown_engine_fails_the_launch_rather_than_guessing():
    j = jobmod.parse_job(json.dumps({'id': 'e', 'type': 'eval', 'policy': 'p',
                                     'eval_args': ['top50'],
                                     'env': {'SNEK_EVAL_ENGINE': 'numpy'}}))
    try:
        launch.build_command(j, _host(), _runtime())
        assert False, 'an unknown engine was accepted'
    except ValueError:
        pass


def test_build_eval_command_puts_chain_before_the_selector():
    # Order is not cosmetic: `eval_wave.parse_options` reads leading `--chain` tokens and everything
    # after the selector is a policy name, so `top50 --chain` would be rejected as a policy. Both
    # engines share that parser -- `vec_wave` imports it -- so this holds for either.
    spec = json.dumps({
        'id': 'b45-closeout', 'type': 'eval', 'policies': ['b45a', 'b45b'],
        'eval_args': ['top50'], 'chain': True})
    for engine, script in (('vec', 'vectorized/vec_wave.py'), ('scalar', 'eval_wave.py')):
        r = _runtime(); r['eval_engine'] = engine
        argv, env, log, policy = launch.build_command(jobmod.parse_job(spec), _host(), r)
        assert argv == ['/py', '-u', script, '--chain', 'top50', 'b45a', 'b45b'], argv
        assert policy == 'b45a'                  # the throughput field: the wave's first arm


def test_build_eval_command_lets_a_spec_override_the_lane_count():
    j = jobmod.parse_job(json.dumps({'id': 'e', 'type': 'eval', 'policy': 'p',
                                     'eval_args': ['top50'], 'eval_lanes': 2,
                                     'eval_workers': 8}))
    r = _runtime(); r['eval_engine'] = 'scalar'
    argv, env, log, policy = launch.build_command(j, _host(), r)
    assert env['EVAL_LANES'] == '2' and env['EVAL_WORKERS'] == '8'


def test_a_vec_wave_takes_its_process_count_from_the_spec_or_the_runtime():
    """`eval_workers` is the spec field a job already has for "how much of the box", so it carries
    over rather than adding a second one. Unset means `vec_wave` derives it from the core count,
    which is the right answer on a box this config was not written for."""
    j = jobmod.parse_job(json.dumps({'id': 'e', 'type': 'eval', 'policy': 'p',
                                     'eval_args': ['top50'], 'eval_workers': 8}))
    argv, env, log, policy = launch.build_command(j, _host(), _runtime())
    assert env['VEC_WAVE_PROCS'] == '8'
    r = _runtime(); r['vec_wave_procs'] = 10
    j = jobmod.parse_job('{"id": "e", "type": "eval", "policy": "p", "eval_args": ["top50"]}')
    argv, env, log, policy = launch.build_command(j, _host(), r)
    assert env['VEC_WAVE_PROCS'] == '10'
    argv, env, log, policy = launch.build_command(j, _host(), _runtime())
    assert 'VEC_WAVE_PROCS' not in env, 'an unset count must be left to vec_wave, not pinned here'


def test_the_runtime_config_rejects_an_unknown_engine_and_keeps_the_last_good_one():
    """`parse_runtime_config` returning None is what makes the daemon keep its previous config, so a
    typo in `eval_engine` must land here rather than at every eval dispatch, one job at a time."""
    good, notes = cfg.parse_runtime_config('{"eval_engine": "scalar"}', _host())
    assert good is not None and good['eval_engine'] == 'scalar'
    bad, errors = cfg.parse_runtime_config('{"eval_engine": "numpy"}', _host())
    assert bad is None, bad
    assert any('eval_engine' in e for e in errors), errors


def test_the_viewer_watches_both_engines():
    """`chart_viewer --watch` is an ERE, and a pattern that cannot match reads as "the jobs stopped"
    -- six of those in a row close the window on a live wave. `vec_eval.py` has to be in it as much as
    `vec_wave.py`: the supervisor is one short-lived process per stage, its shards run for hours."""
    source = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'runner', 'runner.py')).read()
    line = [l for l in source.splitlines() if 'snek2.py|' in l]
    assert len(line) == 1, line
    for tool in ('snek2.py', 'eval_wave.py', 'eval_checkpoints.py', 'vec_wave.py', 'vec_eval.py'):
        assert tool in line[0], (tool, line[0])


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


def test_eval_batch_pngs_keeps_a_batchs_finished_arms():
    # b45's closeout ran as three waves ({a,c}, {b}, {d}). While wave 2 measures b45b alone, the
    # window must still show a's and c's finished charts -- they are on disk, and they belong to
    # the batch being measured.
    names = ['b45a-lowlr8_eval_progress.png', 'b45b-lowlr8_eval_progress.png',
             'b45c-lowlr8_eval_progress.png']
    got = runnermod.eval_batch_pngs([('eval', ['b45b-lowlr8'])], '/snek', names)
    assert got == ['/snek/evals/b45a-lowlr8_eval_progress.png',
                   '/snek/evals/b45b-lowlr8_eval_progress.png',
                   '/snek/evals/b45c-lowlr8_eval_progress.png'], got
    # The old rule -- the running job's own policies -- is what showed one panel.
    assert len(runnermod.viewer_png_paths([('eval', ['b45b-lowlr8'])], '/snek')) == 1


def test_eval_batch_pngs_ignores_other_batches_and_non_charts():
    names = ['b45a-x_eval_progress.png', 'b44a-x_eval_progress.png', 'notes.txt',
             'b45b-x.png', 'champion_7_eval_progress.png']
    got = runnermod.eval_batch_pngs([('eval', ['b45a-x'])], '/snek', names)
    assert got == ['/snek/evals/b45a-x_eval_progress.png'], got


def test_eval_batch_pngs_membership_is_the_chart_on_disk_not_the_arm_list():
    # The bound that replaces a TTL: only arms whose chart is in evals/ right now can get a panel,
    # so a wide batch measured over many waves cannot grow the window without limit, and an arm
    # that has not started yet never becomes an unlabelled blank box.
    got = runnermod.eval_batch_pngs([('eval', ['b45a-x', 'b45d-x'])], '/snek',
                                    ['b45a-x_eval_progress.png'])
    assert got == ['/snek/evals/b45a-x_eval_progress.png'], got


def test_eval_batch_pngs_is_capped():
    names = ['b20%s-x_eval_progress.png' % c for c in 'abcdefghijkl']
    got = runnermod.eval_batch_pngs([('eval', ['b20a-x'])], '/snek', names)
    assert len(got) == runnermod.MAX_VIEWER_PANELS, len(got)


def test_eval_batch_pngs_only_ever_names_eval_charts():
    # It is used for eval waves only -- a training wave keeps the sticky set, which cannot
    # over-report. So this must never emit a runs/ path whatever it is handed.
    got = runnermod.eval_batch_pngs([('trainer', ['b45a-x'])], '/snek',
                                    ['b45a-x_eval_progress.png'])
    assert all('/evals/' in path for path in got), got


def test_agreed_env_keeps_only_what_every_arm_shares():
    envs = [{'SNEK_CHASE_SAFE_SHAPING': '0.1', 'SNEK_SEED': '1'},
            {'SNEK_CHASE_SAFE_SHAPING': '0.1', 'SNEK_SEED': '2'},
            {'SNEK_CHASE_SAFE_SHAPING': '0.1', 'SNEK_SEED': '3', 'SNEK_EXTRA': 'x'}]
    assert runnermod.agreed_env(envs) == {'SNEK_CHASE_SAFE_SHAPING': '0.1'}
    assert runnermod.agreed_env([]) == {}
    assert runnermod.agreed_env([{'A': '1'}]) == {'A': '1'}


def test_closeout_group_env_ignores_everything_a_measurement_cannot_see():
    env = {'SNEK_SEED': '2', 'SNEK_LEARNING_RATE': '1e-8', 'SNEK_DISCOUNT': '0.9975',
           'SNEK_FC_LAYERS': '320', 'SNEK_TARGET_UPDATE_PERIOD': '1000',
           'SNEK_CHASE_SAFE_SHAPING': '0.1', 'SNEK_CHASE_SAFE_GATE': '75'}
    assert runnermod.closeout_group_env(env) == {'SNEK_CHASE_SAFE_SHAPING': '0.1',
                                                'SNEK_CHASE_SAFE_GATE': '75'}


def test_eval_relevant_env_matches_eval_wave():
    """Tripwire: the runner cannot import eval_wave (TensorFlow), so it carries a copy of
    EVAL_RELEVANT_ENV. Read the real tuple out of the source and fail if the two have drifted --
    a knob added to the eval's list and not to this one would silently stop splitting waves that
    must be split."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'eval_wave.py')
    tree = ast.parse(open(os.path.normpath(src)).read())
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, 'id', None) == 'EVAL_RELEVANT_ENV' for t in node.targets):
            found = tuple(el.value for el in node.value.elts)
    assert found is not None, 'EVAL_RELEVANT_ENV not found in eval_wave.py'
    assert found == runnermod.EVAL_RELEVANT_ENV, (found, runnermod.EVAL_RELEVANT_ENV)


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


def test_auto_closeout_groups_a_batch_into_one_wave():
    r = _runner()
    for arm in ('b20a-fc25seed1', 'b20b-fc25seed2', 'b20c-fc25seed3'):
        r.ledger[arm] = {'state': 'done', 'type': 'train', 'policy': arm,
                         'closeout': 'pending',
                         'env': {'SNEK_FC_LAYERS': '25,50,25', 'SNEK_SEED': '1'}}
    jobs = r._auto_closeout_jobs()
    assert len(jobs) == 1, [j.id for j in jobs]      # one job, not three
    j = jobs[0]
    assert j.id == 'b20-closeout' and j.type == 'eval'
    assert j.policies == ['b20a-fc25seed1', 'b20b-fc25seed2', 'b20c-fc25seed3']
    assert j.policy == 'b20a-fc25seed1'              # sugar: the first arm
    assert j.env.get('SNEK_FC_LAYERS') == '25,50,25'  # the arm's own knobs still carry over
    assert j.eval_args == ['top50']
    assert j.priority == runnermod.AUTO_CLOSEOUT_PRIORITY
    assert j.priority < 100                          # beats a default-priority training


def test_auto_closeout_groups_arms_that_differ_only_in_seed_into_one_wave():
    # The b45 regression. Its four arms are one experiment differing only in SNEK_SEED, which
    # cannot reach a measurement of an already-trained checkpoint -- but the group key was the
    # whole inherited env, so the closeout split into three waves ({a,c} seed 2, {b} seed 1,
    # {d} seed 3) and ran at a quarter of the intended 4 lanes.
    r = _runner()
    seeds = {'b45a-lowlr8-b29b': '2', 'b45b-lowlr8-b29a': '1',
             'b45c-lowlr8-b40b': '2', 'b45d-lowlr8-b29c': '3'}
    for arm, seed in seeds.items():
        r.ledger[arm] = {'state': 'done', 'type': 'train', 'policy': arm, 'closeout': 'pending',
                         'env': {'SNEK_SEED': seed, 'SNEK_LEARNING_RATE': '1e-8',
                                 'SNEK_CHASE_SAFE_SHAPING': '0.1', 'SNEK_CHASE_SAFE_GATE': '75'}}
    jobs = r._auto_closeout_jobs()
    assert len(jobs) == 1, [(j.id, j.policies) for j in jobs]
    assert jobs[0].policies == sorted(seeds), jobs[0].policies
    # The wave's env carries what the arms agree on and not one arm's seed.
    assert jobs[0].env.get('SNEK_CHASE_SAFE_SHAPING') == '0.1'
    assert 'SNEK_SEED' not in jobs[0].env, jobs[0].env


def test_auto_closeout_still_splits_arms_that_disagree_about_shaping():
    # The property the loose key must not lose: shaping and reward knobs change what avg_reward
    # means, so those arms cannot share one process.
    r = _runner()
    r.ledger['b46a-x'] = {'state': 'done', 'type': 'train', 'policy': 'b46a-x',
                          'closeout': 'pending', 'env': {'SNEK_FREE_SPACE_SHAPING': '0.1'}}
    r.ledger['b46b-x'] = {'state': 'done', 'type': 'train', 'policy': 'b46b-x',
                          'closeout': 'pending', 'env': {'SNEK_FREE_SPACE_SHAPING': '0'}}
    jobs = r._auto_closeout_jobs()
    assert len(jobs) == 2, [(j.id, j.policies) for j in jobs]
    assert {j.env.get('SNEK_FREE_SPACE_SHAPING') for j in jobs} == {'0.1', '0'}


def test_the_closeout_carries_no_eval_protocol_env_at_all():
    # The daemon used to pin the gate here, as a second copy of `eval_plan.DEFAULT_MIN_ACHIEVABLE`.
    # It now *strips* the protocol keys instead and lets the tool's own default decide, so there is
    # one definition and nothing here to drift. A stale knob in the training's env must not survive.
    r = _runner()
    r.ledger['b40a'] = {'state': 'done', 'type': 'train', 'policy': 'b40a',
                        'closeout': 'pending',
                        'env': {'SNEK_FREE_SPACE_SHAPING': '0.1', 'EVAL_MIN_ACHIEVABLE': '80',
                                'EVAL_EPISODES': '10', 'EVAL_OUT_SUFFIX': '_stale'}}
    j = r._auto_closeout_jobs()[0]
    assert j.env == {'SNEK_FREE_SPACE_SHAPING': '0.1'}, j.env
    for key in runnermod.EVAL_PROTOCOL_KEYS:
        assert key not in j.env, key


def test_arms_that_disagree_about_shaping_get_separate_waves():
    # One wave is one process with one environment, and the shaping knobs change what `avg_reward`
    # means. Two arms that disagree must not be measured under a single setting.
    r = _runner()
    r.ledger['b41a'] = {'state': 'done', 'type': 'train', 'policy': 'b41a',
                        'closeout': 'pending', 'env': {'SNEK_CHASE_SAFE_SHAPING': '0.10'}}
    r.ledger['b41b'] = {'state': 'done', 'type': 'train', 'policy': 'b41b',
                        'closeout': 'pending', 'env': {}}
    jobs = r._auto_closeout_jobs()
    assert len(jobs) == 2, [(j.id, j.policies) for j in jobs]
    assert sorted(p for j in jobs for p in j.policies) == ['b41a', 'b41b']
    assert len({j.id for j in jobs}) == 2            # and the two ids do not collide


def test_a_second_wave_of_the_same_batch_gets_its_own_id():
    # b20 ran 36 arms under one prefix, in nine waves of four. Under a single `b20-closeout` the
    # later waves would be skipped as already-terminal and never measured at all.
    r = _runner()
    r.ledger['b20-closeout'] = {'state': 'done', 'type': 'eval', 'policy': 'b20a',
                                'policies': ['b20a', 'b20b']}
    r.ledger['b20e'] = {'state': 'done', 'type': 'train', 'policy': 'b20e',
                        'closeout': 'pending', 'env': {}}
    r.ledger['b20f'] = {'state': 'done', 'type': 'train', 'policy': 'b20f',
                        'closeout': 'pending', 'env': {}}
    jobs = r._auto_closeout_jobs()
    assert [j.id for j in jobs] == ['b20-closeout-w2'], [j.id for j in jobs]
    assert jobs[0].policies == ['b20e', 'b20f']


def test_the_same_wave_is_never_measured_twice():
    # Idempotence, and it has to be by arm *set* rather than by id alone now that the id is the
    # batch: a still-'pending' marker must not launch a second copy of a finished wave.
    r = _runner()
    r.ledger['b20-closeout'] = {'state': 'done', 'type': 'eval', 'policy': 'b20a',
                                'policies': ['b20a', 'b20b']}
    for arm in ('b20a', 'b20b'):
        r.ledger[arm] = {'state': 'done', 'type': 'train', 'policy': arm,
                         'closeout': 'pending', 'env': {}}
    assert r._auto_closeout_jobs() == []


def test_interrupted_closeout_resumes_a_fresh_one_does_not():
    # A reboot marks the running closeout 'interrupted' (non-terminal), so it is re-synthesized.
    # That relaunch must carry EVAL_RESUME=1 to keep the full-length rows already on disk;
    # a first-time closeout (no prior record) must not, or it would read a stale file.
    r = _runner()
    r.ledger['b41a'] = {'state': 'done', 'type': 'train', 'policy': 'b41a',
                        'closeout': 'pending', 'env': {'SNEK_SEED': '1'}}
    assert 'EVAL_RESUME' not in r._auto_closeout_jobs()[0].env   # fresh: measure from scratch
    r.ledger['b41-closeout'] = {'state': 'interrupted', 'type': 'eval', 'policy': 'b41a'}
    assert r._auto_closeout_jobs()[0].env['EVAL_RESUME'] == '1'


def test_auto_closeout_skips_when_eval_done_running_or_unmarked():
    r = _runner()
    # unmarked done training -> nothing (no retroactive closeouts for pre-feature arms)
    r.ledger['old'] = {'state': 'done', 'type': 'train', 'policy': 'old-arm'}
    assert r._auto_closeout_jobs() == []
    # marked, but its closeout already ran over the same arm -> skip (idempotent, no second eval)
    r.ledger['p1'] = {'state': 'done', 'type': 'train', 'policy': 'p1', 'closeout': 'pending'}
    r.ledger['p1-closeout'] = {'state': 'done', 'type': 'eval', 'policy': 'p1',
                               'policies': ['p1']}
    # marked, but its closeout is currently running -> skip
    r.ledger['p2'] = {'state': 'done', 'type': 'train', 'policy': 'p2', 'closeout': 'pending'}
    r.running['p2-closeout'] = _FakeRJ('p2-closeout', 'eval', 'p2', alive=True, policies=['p2'])
    ids = [j.id for j in r._auto_closeout_jobs()]
    assert ids == [], ids


def test_a_legacy_per_arm_closeout_still_counts_as_measured():
    # The marker on a finished training is never cleared, so what stops a re-measure has to be the
    # closeout records. Before 2026-08-19 it was the *id*: `<policy>-closeout` was taken, so the
    # work read as done. Grouping the batch changed the id to `<batch>-closeout`, which no b20-b44
    # arm had ever used -- and 64 finished trainings on the desktop instantly read as unmeasured,
    # b20's forecast wave alone carrying 12 arms it had already measured one at a time.
    r = _runner()
    for arm in ('b20a-x', 'b20b-x', 'b20c-x', 'b20d-x'):
        r.ledger[arm] = {'state': 'done', 'type': 'train', 'policy': arm, 'closeout': 'pending'}
        r.ledger[arm + '-closeout'] = {'state': 'done', 'type': 'eval', 'policy': arm}
    assert r._auto_closeout_jobs() == []


def test_only_the_unmeasured_arms_of_a_part_measured_batch_form_the_wave():
    # b44's shape at the migration: two arms closed out under the old per-arm ids, two still to go.
    r = _runner()
    for arm in ('b44a-x', 'b44b-x', 'b44c-x', 'b44d-x'):
        r.ledger[arm] = {'state': 'done', 'type': 'train', 'policy': arm, 'closeout': 'pending'}
    r.ledger['b44a-x-closeout'] = {'state': 'done', 'type': 'eval', 'policy': 'b44a-x'}
    r.running['b44b-x-closeout'] = _FakeRJ('b44b-x-closeout', 'eval', 'b44b-x', alive=True)
    jobs = r._auto_closeout_jobs()
    assert [j.id for j in jobs] == ['b44-closeout'], [j.id for j in jobs]
    assert jobs[0].policies == ['b44c-x', 'b44d-x'], jobs[0].policies


def test_a_failed_closeout_is_not_retried_but_an_interrupted_one_is():
    # Same split the id-based rule had: a failure is usually not transient, while `interrupted`
    # means a reboot cut the wave short and its finished rows are still on disk to resume from.
    r = _runner()
    r.ledger['b50a-x'] = {'state': 'done', 'type': 'train', 'policy': 'b50a-x',
                          'closeout': 'pending'}
    r.ledger['b50-closeout'] = {'state': 'failed', 'type': 'eval', 'policies': ['b50a-x']}
    assert r._auto_closeout_jobs() == []
    r.ledger['b50-closeout']['state'] = 'interrupted'
    jobs = r._auto_closeout_jobs()
    assert [j.id for j in jobs] == ['b50-closeout'], [j.id for j in jobs]
    assert jobs[0].env.get('EVAL_RESUME') == '1', jobs[0].env


def test_auto_closeout_disabled_yields_nothing():
    r = _runner(auto_closeout=False)
    r.ledger['p'] = {'state': 'done', 'type': 'train', 'policy': 'p', 'closeout': 'pending'}
    assert r._auto_closeout_jobs() == []


# ----------------------------------------------- the HOF stage, now inside the closeout process
class _FakeRJ:
    """The bit of launch.RunningJob that _reap touches, with no subprocess behind it."""
    def __init__(self, jid, jtype, policy, alive=False, rc=0, policies=None):
        self.job = jobmod.Job(id=jid, type=jtype, policy=policy, policies=policies,
                              env={}, priority=10)
        self.policy = policy
        self.log_path = '/nonexistent.log'
        self._alive, self._rc = alive, rc

    def is_alive(self):
        return self._alive

    def returncode(self):
        return self._rc


def test_the_daemon_no_longer_mints_a_hof_job_at_all():
    # The re-measure is stage B of the closeout's own process (`eval_wave.py --chain`), so there is
    # no `wants_hof`, no `_auto_hof_jobs`, and no `hof: pending` marker to lose results down. Named
    # explicitly rather than left as an absence, because "the function is gone" is the behaviour
    # change -- a reintroduced second copy of the recipe is what this guards against.
    assert not hasattr(runnermod, 'wants_hof')
    assert not hasattr(runnermod, '_auto_hof_jobs')
    assert not hasattr(runnermod.Runner, '_auto_hof_jobs')
    for gone in ('HOF_THRESHOLD', 'HOF_EVAL_ENV', 'HOF_EVAL_ARGS', 'AUTO_HOF_PRIORITY',
                 'CLOSEOUT_THRESHOLD', 'CLOSEOUT_EVAL_ENV'):
        assert not hasattr(runnermod, gone), gone


def test_a_closeout_wave_carries_chain_when_auto_hof_is_on():
    r = _runner()
    r.ledger['b25a-x'] = {'state': 'done', 'type': 'train', 'policy': 'b25a-x',
                          'closeout': 'pending', 'env': {'SNEK_SEED': '1'}}
    assert r._auto_closeout_jobs()[0].chain is True
    r.runtime['auto_hof'] = False
    assert r._auto_closeout_jobs()[0].chain is False


def test_reap_does_not_mark_a_finished_closeout_for_anything():
    # The marker is what `_auto_hof_jobs` used to read, and deleting that function would have made
    # a leftover marker silently invisible. Nothing sets one now.
    r = _runner()
    r._publish_results = lambda rj: None               # no git behind this test
    r.running['p-closeout'] = _FakeRJ('p-closeout', 'eval', 'p', alive=False, rc=0)
    r._reap()
    assert r.ledger['p-closeout']['state'] == 'done'
    assert 'hof' not in r.ledger['p-closeout']


def test_a_dead_closeout_across_a_daemon_restart_marks_nothing_further():
    # Same boot, dead pid: the detached closeout ran to its end while the daemon was down. Under the
    # old chain that earned it a HOF job; now stage B either already ran inside it or already failed
    # inside it, so there is nothing for the daemon to add.
    real_boot, real_alive = runnermod.boot_id, launch.pid_alive
    runnermod.boot_id = lambda: 'BOOT-A'
    launch.pid_alive = lambda _pid: False
    try:
        r = _runner()
        r.ledger['b30-closeout'] = {'state': 'running', 'type': 'eval', 'policy': 'b30a',
                                    'policies': ['b30a', 'b30b'],
                                    'pid': 4242, 'boot': 'BOOT-A', 'env': {'SNEK_SEED': '1'}}
        r._reattach()
    finally:
        runnermod.boot_id, launch.pid_alive = real_boot, real_alive
    got = r.ledger['b30-closeout']
    assert got['state'] == 'done' and 'hof' not in got, got


def test_a_restart_readopts_a_running_wave_with_all_of_its_arms():
    # `_StubJob` rebuilds a Job from the ledger record with no spec in hand. A missing `policies`
    # fallback here would publish one arm of four when the wave finished.
    real_boot, real_alive = runnermod.boot_id, launch.pid_alive
    runnermod.boot_id = lambda: 'BOOT-A'
    launch.pid_alive = lambda _pid: True
    try:
        r = _runner()
        r.ledger['b30-closeout'] = {'state': 'running', 'type': 'eval', 'policy': 'b30a',
                                    'policies': ['b30a', 'b30b', 'b30c', 'b30d'],
                                    'chain': True, 'pid': 4242, 'boot': 'BOOT-A'}
        r._reattach()
    finally:
        runnermod.boot_id, launch.pid_alive = real_boot, real_alive
    job = r.running['b30-closeout'].job
    assert job.policies == ['b30a', 'b30b', 'b30c', 'b30d']
    assert job.chain is True
    # And a record written before `policies` existed still yields one policy, not none.
    r2 = _runner()
    assert runnermod._StubJob('t', {'type': 'train', 'policy': 'p'}).policies == ['p']


def test_publish_results_pushes_each_arm_separately():
    # One push per arm, because `publish_results` has no retry and the box's DNS for github.com
    # flaps: a wave concentrating four arms behind one push would lose all four to one failure.
    r = _runner()
    pushed = []
    seen = []

    def fake_publish(host, job, arts):
        pushed.append(sorted(os.path.basename(a) for a in arts))
        if len(pushed) == 1:
            raise RuntimeError('DNS flap')
        seen.append(job.id)

    real = runnermod.gitbus.publish_results
    runnermod.gitbus.publish_results = fake_publish
    runs = os.path.join(_host()['SNEK_DIR'], 'runs')
    real_isdir, real_listdir = os.path.isdir, os.listdir
    os.path.isdir = lambda p: True if p == runs else real_isdir(p)
    os.listdir = lambda p: (['a1.md', 'a1_evals.json', 'a2.md', 'other.md']
                            if p == runs else real_listdir(p))
    try:
        r._publish_results(_FakeRJ('b30-closeout', 'eval', 'a1', policies=['a1', 'a2']))
    finally:
        runnermod.gitbus.publish_results = real
        os.path.isdir, os.listdir = real_isdir, real_listdir
    assert len(pushed) == 2, pushed                 # the failure did not stop the second arm
    assert pushed[0] == ['a1.md', 'a1_evals.json']
    assert pushed[1] == ['a2.md']
    assert seen == ['b30-closeout']


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


def test_anticipated_queue_interleaves_a_closeout_wave_between_training_batches():
    # Two training batches queued; each batch's closeout **wave** runs before the next batch's
    # training, because a closeout (priority 10) outranks a queued training (200). One row per
    # batch, not one per arm and then one per HOF -- eight rows collapse to two.
    queued = [{'id': p, 'type': 'train', 'policy': p, 'priority': 200}
              for p in ('b1a', 'b1b', 'b2a', 'b2b')]
    order = runnermod.anticipated_queue(
        queued, [], {'trainer': 2, 'eval': 2}, True, {'b1a', 'b1b', 'b2a', 'b2b'})
    assert [j['id'] for j in order] == [
        'b1a', 'b1b', 'b1-closeout', 'b2a', 'b2b', 'b2-closeout'], [j['id'] for j in order]
    wave = [j for j in order if j['id'] == 'b1-closeout'][0]
    assert wave['policies'] == ['b1a', 'b1b']
    assert wave['chain'] is True


def test_anticipated_queue_seeds_a_closeout_wave_for_trainings_running_now():
    queued = [{'id': 'b2a', 'type': 'train', 'policy': 'b2a', 'priority': 200}]
    running = [{'id': 'b1a', 'type': 'train', 'policy': 'b1a'},
               {'id': 'b1b', 'type': 'train', 'policy': 'b1b'}]
    order = runnermod.anticipated_queue(
        queued, running, {'trainer': 2, 'eval': 2}, True, {'b2a', 'b1a', 'b1b'})
    assert [j['id'] for j in order] == [
        'b1-closeout', 'b2a', 'b2-closeout'], [j['id'] for j in order]


def test_anticipated_queue_omits_closeouts_when_auto_closeout_off():
    queued = [{'id': 'a1', 'type': 'train', 'policy': 'a1', 'priority': 200}]
    order = runnermod.anticipated_queue(queued, [], {'trainer': 2, 'eval': 2}, False, {'a1'})
    assert [j['id'] for j in order] == ['a1']


def test_anticipated_queue_never_invents_an_existing_closeout_twice():
    # The real closeout for a finished training already sits in the queue; do not duplicate it.
    queued = [{'id': 'b9a', 'type': 'train', 'policy': 'b9a', 'priority': 200},
              {'id': 'b9-closeout', 'type': 'eval', 'policy': 'b9a', 'priority': AUTO}]
    order = runnermod.anticipated_queue(
        queued, [], {'trainer': 2, 'eval': 2}, True, {'b9a', 'b9-closeout'})
    ids = [j['id'] for j in order]
    assert ids.count('b9-closeout') == 1, ids


def test_anticipated_queue_forecasts_no_hof_hop_at_all():
    # The re-measure is stage B inside the closeout's own process, so it is not a job and there is
    # nothing to place in a later wave. Under the old chain this same input produced three rows.
    queued = [{'id': 'b7a', 'type': 'train', 'policy': 'b7a', 'priority': 200}]
    order = runnermod.anticipated_queue(queued, [], {'trainer': 2, 'eval': 2}, True, {'b7a'})
    assert [j['id'] for j in order] == ['b7a', 'b7-closeout'], [j['id'] for j in order]
    assert not any('hof' in j['id'] for j in order)


def test_anticipated_queue_marks_the_closeout_unchained_when_auto_hof_is_off():
    # `auto_hof` no longer adds or removes a row; it decides whether the one closeout row runs its
    # stage B. Reported on the row so the forecast still says what the box will do.
    queued = [{'id': 'b7a', 'type': 'train', 'policy': 'b7a', 'priority': 200}]
    order = runnermod.anticipated_queue(
        queued, [], {'trainer': 2, 'eval': 2}, True, {'b7a'}, auto_hof=False)
    assert [j['id'] for j in order] == ['b7a', 'b7-closeout']
    assert [j for j in order if j['id'] == 'b7-closeout'][0]['chain'] is False


def test_anticipated_queue_puts_a_whole_batch_in_one_closeout_row():
    # Four arms, `max_evals` of 1: the old forecast needed four eval waves for the closeouts and
    # four more for the HOFs. One row now, which is also what the box actually runs.
    queued = [{'id': 'b8{0}'.format(c), 'type': 'train', 'policy': 'b8{0}'.format(c),
               'priority': 200} for c in 'abcd']
    order = runnermod.anticipated_queue(
        queued, [], {'trainer': 4, 'eval': 1}, True, {j['id'] for j in queued})
    assert [j['id'] for j in order] == ['b8a', 'b8b', 'b8c', 'b8d', 'b8-closeout']
    assert [j for j in order if j['id'] == 'b8-closeout'][0]['policies'] == \
        ['b8a', 'b8b', 'b8c', 'b8d']


def test_ledger_view_anticipates_closeouts_for_a_queued_training():
    r = _runner(auto_closeout=True)
    r.runtime['max_trainers'] = 2
    r.runtime['max_evals'] = 2

    class _J:
        def __init__(self, jid, pri):
            self.id, self.policy, self.priority, self.type = jid, jid, pri, 'train'
            self.policies = [jid]
    r._queued = [_J('b6a', 200), _J('b6b', 200)]
    view = r._ledger_view(r._anticipated_order())
    # One closeout row for the batch, and no HOF hop -- four rows became three.
    assert list(view.keys()) == ['b6a', 'b6b', 'b6-closeout'], list(view.keys())
    assert all(view[k] == 'queued' for k in view)


def test_ledger_view_places_queued_first_then_running_then_finished():
    r = _runner(auto_closeout=False)
    r.ledger['hist'] = {'state': 'done'}
    r.ledger['run1'] = {'state': 'running', 'type': 'train'}

    class _J:
        def __init__(self, jid):
            self.id, self.policy, self.priority, self.type = jid, jid, 200, 'train'
            self.policies = [jid]
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
        policies = ['x']
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


def test_hold_notice_is_none_when_nothing_holds_the_queue():
    assert runnermod.hold_notice(()) is None
    assert runnermod.hold_notice(['viewer']) is None   # not a hold flag


def test_hold_notice_names_only_the_flag_that_is_set():
    paused = runnermod.hold_notice(['paused'])
    assert '"paused": false' in paused and 'drain' not in paused, paused
    assert 'runtime.json on ops' in paused, paused
    drained = runnermod.hold_notice(['drain'])
    assert '"drain": false' in drained and 'paused' not in drained, drained
    assert 'draining' in drained, drained


def test_hold_notice_names_both_flags_and_is_order_stable():
    both = runnermod.hold_notice(['drain', 'paused'])       # caller order reversed
    assert '"paused": false' in both and '"drain": false' in both, both
    assert both == runnermod.hold_notice(['paused', 'drain']), both


def test_at_a_glance_puts_the_hold_notice_before_the_batch_lines():
    order = [{'id': 'b45a-x-seed1', 'type': 'train', 'policy': 'p1'}]
    g = runnermod.build_at_a_glance([], order, {'b45': 'demo'}, ['paused'])
    assert len(g['queued']) == 2, g['queued']
    assert g['queued'][0] == runnermod.hold_notice(['paused']), g['queued']
    assert g['queued'][1].startswith('b45 training'), g['queued']


def test_at_a_glance_shows_the_hold_notice_with_an_empty_queue():
    g = runnermod.build_at_a_glance([], [], {}, ['paused'])
    assert g['queued'] == [runnermod.hold_notice(['paused'])], g['queued']


def test_at_a_glance_has_no_notice_when_not_held():
    order = [{'id': 'b45a-x-seed1', 'type': 'train', 'policy': 'p1'}]
    g = runnermod.build_at_a_glance([], order, {'b45': 'demo'})
    assert len(g['queued']) == 1 and g['queued'][0].startswith('b45 training'), g['queued']


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


# ------------------------------------------------- the wave: specs, argv, status, capacity

def test_parse_job_accepts_a_policies_wave_and_fills_in_policy():
    j = jobmod.parse_job(json.dumps({'id': 'b45-closeout', 'type': 'eval',
                                     'policies': ['b45a', 'b45b'], 'eval_args': ['top50']}))
    assert j.policies == ['b45a', 'b45b']
    assert j.policy == 'b45a'          # sugar, so a caller that only knows `policy` still works


def test_parse_job_fills_in_policies_from_a_single_policy():
    # The other direction, and it is the common one: every training spec and every hand-written
    # eval spec names one policy, and everything downstream now reads `policies`.
    j = jobmod.parse_job('{"id": "t", "type": "train", "policy": "p"}')
    assert j.policies == ['p'] and j.policy == 'p'


def test_parse_job_rejects_a_bad_policies_field():
    for bad in ({'policies': []}, {'policies': 'b45a'}, {'policies': ['b45a', '']},
                {'policies': [1]}):
        spec = dict({'id': 'e', 'type': 'eval'}, **bad)
        try:
            jobmod.parse_job(json.dumps(spec))
        except jobmod.JobError:
            continue
        raise AssertionError('accepted {0}'.format(bad))


def test_only_an_eval_takes_policies_or_chain():
    # A training is one arm by construction, so a `policies` list there is a spec mistake worth
    # naming rather than silently training the first of them.
    for spec in ({'id': 't', 'type': 'train', 'policies': ['a', 'b']},
                 {'id': 't', 'type': 'train', 'policy': 'a', 'chain': True}):
        try:
            jobmod.parse_job(json.dumps(spec))
        except jobmod.JobError:
            continue
        raise AssertionError('accepted {0}'.format(spec))


def test_chain_defaults_off_and_must_be_a_bool():
    assert jobmod.parse_job('{"id": "e", "type": "eval", "policy": "p"}').chain is False
    try:
        jobmod.parse_job('{"id": "e", "type": "eval", "policy": "p", "chain": "yes"}')
    except jobmod.JobError:
        return
    raise AssertionError('accepted a string chain')


def test_inherited_eval_env_keeps_the_arm_and_drops_the_protocol():
    got = runnermod.inherited_eval_env({
        'SNEK_CHASE_SAFE_SHAPING': '0.1', 'SNEK_FC_LAYERS': '320',
        'EVAL_EPISODES': '10', 'EVAL_MIN_ACHIEVABLE': '80', 'EVAL_OUT_SUFFIX': '_x',
        'EVAL_SCREEN_EPISODES': '0', 'EVAL_ABANDON_FLOOR': '5', 'EVAL_CONFIRM_COUNT': '3'})
    assert got == {'SNEK_CHASE_SAFE_SHAPING': '0.1', 'SNEK_FC_LAYERS': '320'}, got
    assert runnermod.inherited_eval_env(None) == {}


def test_phase_of_reads_a_later_closeout_wave_as_a_closeout():
    # `b20-closeout-w2` does not end in '-closeout', and calling it a plain 'eval' would split the
    # at-a-glance grouping in two for what is one phase of one batch.
    assert runnermod.phase_of('b20-closeout', 'eval') == 'closeout eval'
    assert runnermod.phase_of('b20-closeout-w2', 'eval') == 'closeout eval'
    assert runnermod.phase_of('b20-closeout-w11', 'eval') == 'closeout eval'
    assert runnermod.phase_of('manual-top50', 'eval') == 'eval'
    # Old ledger records only: the daemon no longer mints this id.
    assert runnermod.phase_of('p-hof', 'eval') == 'hof'


def test_at_a_glance_counts_arms_in_policies_not_in_jobs():
    # A batch's whole close-out is one job. Counting jobs would report "1 arm", which is the
    # number a reader uses to check that nothing was dropped.
    glance = runnermod.build_at_a_glance(
        [], [{'id': 'b45-closeout', 'type': 'eval',
              'policies': ['b45a', 'b45b', 'b45c', 'b45d']}], {})
    assert glance['queued'] == ['b45 closeout eval -- queued (4 arms)'], glance['queued']


def test_at_a_glance_still_counts_a_single_policy_job_as_one_arm():
    glance = runnermod.build_at_a_glance([], [{'id': 'b45a', 'type': 'train',
                                               'policy': 'b45a'}], {})
    assert glance['queued'] == ['b45 training -- queued (1 arm)'], glance['queued']


def test_viewer_png_paths_expands_a_wave_into_one_chart_per_arm():
    got = runnermod.viewer_png_paths([('eval', ['b45a', 'b45b'])], '/s')
    assert got == ['/s/evals/b45a_eval_progress.png', '/s/evals/b45b_eval_progress.png'], got


def test_viewer_png_paths_still_takes_a_bare_policy_string():
    # A caller with one policy in hand needs no change, and a None must not become 'None.png'.
    assert runnermod.viewer_png_paths([('trainer', 'b45a')], '/s') == ['/s/runs/b45a.png']
    assert runnermod.viewer_png_paths([('eval', None)], '/s') == []


def test_eval_lanes_defaults_to_four_and_clamps_to_the_host_ceiling():
    host = _host(); host['HARD_MAX_EVALS'] = 4
    out, notes = cfg.parse_runtime_config('{}', host)
    assert out['eval_lanes'] == 4
    out, notes = cfg.parse_runtime_config('{"eval_lanes": 99, "eval_workers": 1}', host)
    assert out['eval_lanes'] == 4
    assert any('eval_lanes' in n for n in notes), notes


def test_lanes_times_workers_is_capped_and_the_workers_give_way():
    # Memory is this box's binding constraint: a spawned worker carries its own ~230 MB TensorFlow
    # arena. The lanes are kept because a lane that does not exist cannot pick up another arm's
    # work, which is the whole point of a wave; a lane with fewer workers only measures slower.
    host = _host(); host['HARD_MAX_EVALS'] = 4
    out, notes = cfg.parse_runtime_config('{"eval_lanes": 4, "eval_workers": 10}', host)
    assert out['eval_lanes'] == 4
    assert out['eval_lanes'] * out['eval_workers'] <= cfg.MAX_EVAL_WORKERS
    assert out['eval_workers'] == 8, out['eval_workers']
    assert any('exceeds' in n for n in notes), notes
    # Inside the band nothing is touched.
    out, notes = cfg.parse_runtime_config('{"eval_lanes": 4, "eval_workers": 4}', host)
    assert (out['eval_lanes'], out['eval_workers']) == (4, 4)
    assert not [n for n in notes if 'exceeds' in n]


def test_a_fresh_launch_records_every_arm_of_the_wave():
    # The record is the only thing `_StubJob` has after a daemon restart, and it is what
    # `_publish_results` iterates. A record that kept one policy of four would publish one arm of
    # four -- with the ledger still reading `done`, which is exactly the class of silence that hid
    # b40's HOF results for hours.
    r = _runner()
    real_spawn = launch.spawn
    launch.spawn = lambda job, host, runtime: launch.RunningJob(job, job.policy, 999, '/l')
    try:
        r._launch(jobmod.Job(id='b45-closeout', type='eval',
                             policies=['b45a', 'b45b', 'b45c', 'b45d'],
                             eval_args=['top50'], chain=True, env={}, priority=10))
    finally:
        launch.spawn = real_spawn
    rec = r.ledger['b45-closeout']
    assert rec['policies'] == ['b45a', 'b45b', 'b45c', 'b45d'], rec['policies']
    assert rec['policy'] == 'b45a' and rec['chain'] is True
    # And the round trip a restart actually takes.
    assert runnermod._StubJob('b45-closeout', rec).policies == ['b45a', 'b45b', 'b45c', 'b45d']


def test_max_evals_multiplies_into_the_worker_cap():
    # The way this box gets OOM-killed. `max_evals` used to mean 4 x eval_workers = 16 spawned
    # workers; with a wave behind each job it means 4 x eval_lanes x eval_workers = 64, and 40 is
    # roughly where the OOM-killer sits.
    host = _host(); host['HARD_MAX_EVALS'] = 4
    out, notes = cfg.parse_runtime_config(
        '{"max_evals": 4, "eval_lanes": 4, "eval_workers": 4}', host)
    assert out['max_evals'] == 4                 # a scheduling decision, never overridden here
    assert out['eval_lanes'] == 4                # lanes are what a wave is for
    assert out['eval_workers'] == 2
    assert out['max_evals'] * out['eval_lanes'] * out['eval_workers'] <= cfg.MAX_EVAL_WORKERS
    assert any('max_evals' in n and 'exceeds' in n for n in notes), notes


def test_lanes_give_way_only_when_workers_cannot_go_lower():
    host = _host(); host['HARD_MAX_EVALS'] = 4
    out, _ = cfg.parse_runtime_config(
        '{"max_evals": 4, "eval_lanes": 4, "eval_workers": 1}', host)
    # 4 x 4 x 1 = 16, already inside the band: nothing moves.
    assert (out['eval_lanes'], out['eval_workers']) == (4, 1)
    # A ceiling low enough that even one worker per lane does not fit forces the lanes down.
    real = cfg.MAX_EVAL_WORKERS
    cfg.MAX_EVAL_WORKERS = 8
    try:
        out, notes = cfg.parse_runtime_config(
            '{"max_evals": 4, "eval_lanes": 4, "eval_workers": 4}', host)
    finally:
        cfg.MAX_EVAL_WORKERS = real
    assert out['eval_workers'] == 1 and out['eval_lanes'] == 2, (out['eval_lanes'],
                                                                out['eval_workers'])
    assert out['max_evals'] * out['eval_lanes'] * out['eval_workers'] <= 8


def test_a_zero_max_evals_does_not_divide_by_zero():
    # `max_evals: 0` is a legal way to say "run no evals", and the cap must not explode on it.
    host = _host(); host['HARD_MAX_EVALS'] = 4
    out, notes = cfg.parse_runtime_config('{"max_evals": 0, "eval_workers": 40}', host)
    assert out['max_evals'] == 0 and out['eval_workers'] >= 1
