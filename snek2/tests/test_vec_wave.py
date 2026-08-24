"""Pins the supervisor's own logic: the shard plan, the selector round-trip, the environment it
hands a shard, and the absence of any routing that would send an arm to another engine.

Everything else `vec_wave` does is delegated -- argv parsing to `eval_wave`, the recipe to
`eval_plan.hof_settings`, the measurement to `vec_eval`. So these fixtures cover exactly the part
that is new, plus one assertion per delegation that the delegation is real rather than a copy that
happens to agree today.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_plan
import eval_wave
from vectorized import vec_wave


# ------------------------------------------------------------------ the shard plan

def test_shards_are_allocated_in_proportion_to_selection_size():
    """b45's real HOF selections. The big arm must get several processes and the small one fewer --
    equal shares would leave the 298-checkpoint arm's cores idle for most of the run."""
    plan = vec_wave.plan_shards({'a': 1568, 'b': 1264, 'c': 1173, 'd': 298}, 12)
    assert sum(plan.values()) == 12, plan
    assert plan['a'] >= plan['b'] >= plan['c'] > plan['d'], plan
    assert plan['d'] >= 1


def test_the_plan_spends_every_process_it_is_given():
    """Largest-remainder, so the rounding error is spread rather than left unspent. A flooring-only
    plan would hand out 9 of 12 here and run at three quarters throughput."""
    for procs in (4, 8, 12, 16, 32):
        plan = vec_wave.plan_shards({'a': 100, 'b': 100, 'c': 100, 'd': 100}, procs)
        assert sum(plan.values()) == procs, (procs, plan)


def test_equal_arms_get_equal_shards():
    assert vec_wave.plan_shards({'a': 50, 'b': 50, 'c': 50, 'd': 50}, 12) == {
        'a': 3, 'b': 3, 'c': 3, 'd': 3}


def test_an_arm_never_gets_more_shards_than_it_has_checkpoints():
    """An empty shard is a wasted process, and `vec_eval` exits cleanly on one -- so the loss would
    be a process doing nothing, invisibly, for the whole wave.

    The cap only bites when there are **more processes than checkpoints**, which is why the fixtures
    below are tiny rather than realistic: at 12 processes and 402 checkpoints every proportional share
    is already under its arm's count, so a plan with no cap at all passes. That is what a spot-check on
    a realistic batch misses, and it is the case a re-run of a nearly-finished wave lands in -- resume
    leaves a handful of checkpoints and the process count does not change.
    """
    assert vec_wave.plan_shards({'a': 2, 'b': 2}, 12) == {'a': 2, 'b': 2}
    plan = vec_wave.plan_shards({'a': 1, 'b': 3, 'c': 5}, 12)
    assert plan == {'a': 1, 'b': 3, 'c': 5}, plan
    # And the realistic shape still holds, which is the case the cap does *not* apply to.
    plan = vec_wave.plan_shards({'big': 400, 'tiny': 2}, 12)
    assert plan['tiny'] <= 2 and plan['big'] <= 400, plan


def test_an_arm_with_work_always_gets_a_shard_even_if_that_overshoots_the_budget():
    """The floor is deliberate: dropping an arm to hold a process budget would silently not measure
    it. The overshoot is bounded by the arm count."""
    plan = vec_wave.plan_shards({'huge': 3000, 'x': 1, 'y': 1, 'z': 1}, 4)
    assert set(plan) == {'huge', 'x', 'y', 'z'}
    assert min(plan.values()) == 1
    assert sum(plan.values()) <= 4 + 4


def test_arms_with_nothing_selected_get_no_process_at_all():
    plan = vec_wave.plan_shards({'a': 40, 'b': 0, 'c': 0}, 12)
    assert 'b' not in plan and 'c' not in plan
    assert plan == {'a': 12}


def test_an_empty_wave_plans_nothing_rather_than_dividing_by_zero():
    assert vec_wave.plan_shards({}, 12) == {}
    assert vec_wave.plan_shards({'a': 0}, 12) == {}


def test_more_arms_than_processes_still_measures_every_arm():
    """A batch can be wider than the box. `procs` is raised to the arm count rather than some arms
    being dropped."""
    sizes = {chr(ord('a') + i): 100 for i in range(20)}
    plan = vec_wave.plan_shards(sizes, 12)
    assert set(plan) == set(sizes)
    assert sum(plan.values()) == 20


# ------------------------------------------------------------------ the selector round-trip

def test_every_selector_eval_wave_parses_round_trips_into_one_argv_token():
    """`vec_eval` takes the selection as a **single** argument, so a multi-token spelling has it
    measure one checkpoint and report a short selection rather than an error -- and that is exactly
    the bug this catches: 24 steps passed as 24 argv entries measured one checkpoint.

    The two CLIs differ on purpose about *how* a step list is spelled -- `eval_wave` takes bare
    tokens, `vec_eval` a comma-joined one -- so the round-trip is checked against each parser rather
    than assuming they agree."""
    for typed in (['top50'], ['top'], ['above:98'], ['above'], ['1000', '2000', '3000']):
        kind, value, rest = eval_wave.parse_selector(list(typed) + ['b45a-x'])
        token = vec_wave.selector_string(kind, value)
        assert ' ' not in token, (typed, token)
        if kind == 'explicit':
            # `vec_eval.resolve_selection` splits on commas and spaces; this must give the same steps.
            assert [int(part) for part in token.replace(',', ' ').split()] == value, (typed, token)
            continue
        # And it must mean the same thing coming back to the parser that produced it.
        again_kind, again_value, _ = eval_wave.parse_selector([token, 'b45a-x'])
        assert (again_kind, again_value) == (kind, value), (typed, token)


def test_a_default_top_selector_carries_eval_plans_own_count():
    kind, value, _ = eval_wave.parse_selector(['top', 'b45a-x'])
    assert vec_wave.selector_string(kind, value) == 'top{0}'.format(eval_plan.DEFAULT_COUNT)


def test_the_hof_gate_reaches_the_selector_as_a_whole_number():
    """`HOF_GATE` is a float, and `'above:98.0'` would parse but reads wrong in every log line."""
    assert vec_wave.selector_string('above', eval_plan.HOF_GATE) == 'above:98'


# ------------------------------------------------------------------ the child environment

def test_the_child_writes_the_canonical_paths_not_vec_evals_probe_defaults():
    """`vec_eval`'s `_vec` / `evals/vec/` defaults protect a hand-run from overwriting a TF result.
    A wave *is* the close-out, so it has to land where `eval_progress.best_of`,
    `select_checkpoints_above`, `refresh_charts.sh` and every tuning doc already look."""
    import snake_constants
    from vectorized import vec_eval
    env = vec_wave.child_env(100, '', '')
    assert env['EVAL_OUT_SUFFIX'] == ''
    # `snake_constants.EVALS_DIR`, not the string 'evals': it is absolute, and it is what
    # `eval_checkpoints` and `eval_wave` write into. A relative copy would put the wave's charts
    # somewhere no viewer looks whenever the cwd was not `snek2/`.
    assert env['VEC_EVAL_CHART_DIR'] == snake_constants.EVALS_DIR
    assert os.path.basename(env['VEC_EVAL_CHART_DIR']) == 'evals'
    # The knob has to actually be the one vec_eval reads, and it must differ from the probe default.
    assert vec_eval.DEFAULT_SUFFIX == '_vec'
    assert env['VEC_EVAL_CHART_DIR'] != os.path.join('evals', 'vec')


def test_the_scalar_protocols_knobs_are_dropped_rather_than_silently_ignored():
    """There is no screen tier and no abandon gate in this engine, so a value left in the environment
    would read as "this wave was gated at 97" in a log nobody re-checks."""
    keys = ('EVAL_SCREEN_EPISODES', 'EVAL_MIN_ACHIEVABLE', 'EVAL_ABANDON_FLOOR',
            'EVAL_CONFIRM_COUNT', 'EVAL_WORKERS', 'EVAL_LANES')
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for key in keys:
            os.environ[key] = '7'
        env = vec_wave.child_env(100, '', '')
        for key in keys:
            assert key not in env, key
    finally:
        for key, was in saved.items():
            if was is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = was


def test_a_stale_shard_setting_cannot_leak_into_the_children():
    """The parent's own `VEC_EVAL_SHARD` -- a leftover from a hand-run, or the desktop's job env --
    would override the plan for every arm that got only one shard, silently measuring a stride of it."""
    saved = os.environ.get('VEC_EVAL_SHARD')
    try:
        os.environ['VEC_EVAL_SHARD'] = '3/4'
        assert 'VEC_EVAL_SHARD' not in vec_wave.child_env(100, '', '')
    finally:
        if saved is None:
            os.environ.pop('VEC_EVAL_SHARD', None)
        else:
            os.environ['VEC_EVAL_SHARD'] = saved


def test_no_child_opens_a_chart_window():
    """One owner per wave. Twelve claimants racing one `O_EXCL` lock inside the same second is the
    documented way to end up with two windows."""
    assert vec_wave.child_env(100, '', '')['SNEK_CHART_VIEWER'] == '0'


def test_the_episode_count_and_the_hof_source_reach_the_child():
    env = vec_wave.child_env(500, '_hof500', '')
    assert env['VEC_EVAL_EPISODES'] == '500'
    assert env['EVAL_OUT_SUFFIX'] == '_hof500'
    assert env['VEC_EVAL_SOURCE'] == ''


# ------------------------------------------------------------------ the delegations

def test_stage_b_reads_its_recipe_from_eval_plan_and_not_from_a_local_copy():
    """The desktop daemon and the laptop chain script each used to carry a copy of this recipe. The
    assertion is that the numbers the supervisor prints and passes are `eval_plan`'s."""
    recipe = eval_plan.hof_settings({'suffix': ''})
    assert recipe['num_episodes'] == eval_plan.HOF_EPISODES == 500
    assert recipe['suffix'] == eval_plan.HOF_SUFFIX
    assert recipe['source_suffix'] == ''
    # Stage B selects out of stage A's file, so a non-empty close-out suffix has to carry through.
    assert eval_plan.hof_settings({'suffix': '_x'})['source_suffix'] == '_x'


def test_the_argv_helpers_are_eval_waves_own_functions():
    """Imported, not reimplemented -- so `top50`, `above:98`, `--chain` and a bare batch id cannot
    come to mean two different things depending on which entry point was typed."""
    assert vec_wave.eval_wave is eval_wave
    for name in ('parse_options', 'parse_selector', 'resolve_policies', 'completed_policies',
                 'describe_selector'):
        assert callable(getattr(eval_wave, name)), name


# --------------------------------------------------- no arm is routed away by algorithm

# Both files, because the refusal that used to live in `vec_eval` and the split that used to live in
# `vec_wave` were two halves of one behaviour -- either one coming back reverts c51 support on its own.
VEC_SOURCES = ('vectorized/vec_eval.py', 'vectorized/vec_wave.py')


def _vec_source(name):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, name)) as handle:
        return handle.read()


def test_the_c51_split_and_its_fallback_are_gone():
    """`vec_wave` measured a batch's categorical arms by shelling out to `eval_wave.py`, because
    `vec_eval` refused them. It does not refuse them any more -- see
    `tests/test_c51_eval_path.py`, which pins *why* -- so a wave has one engine and no routing.

    Named functions rather than behaviour, deliberately: there is no output to assert about a code
    path that should not exist, and the thing a future change would reach for is the name.
    """
    assert not hasattr(vec_wave, 'split_arms'), 'the algorithm split is back'
    assert not hasattr(vec_wave, 'fallback'), 'the eval_wave fallback is back'


def test_no_vec_file_refuses_a_categorical_policy():
    """An AST scan, so a re-added `policy_arch.refuse_categorical(...)` fails here rather than turning
    every c51 close-out into a `SystemExit` that only shows up when a c51 batch is next measured --
    which could be months."""
    import ast

    for name in VEC_SOURCES:
        for node in ast.walk(ast.parse(_vec_source(name))):
            if not isinstance(node, ast.Call):
                continue
            called = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(
                node.func, 'id', None)
            assert called != 'refuse_categorical', '{0} refuses categorical policies again'.format(
                name)


def test_vec_wave_never_spawns_the_scalar_wave():
    """The fallback's other half. `vec_wave` still *imports* `eval_wave` -- for its argv parsing, which
    is the point of the delegation tests above -- so the thing to pin is that it does not run it as a
    subprocess, which is what a reintroduced fallback would do.

    Docstrings are excluded rather than the whole file grepped, because this module's own header
    discusses `eval_wave.py` at length and a substring search cannot tell prose from argv.
    """
    import ast

    tree = ast.parse(_vec_source('vectorized/vec_wave.py'))
    docstrings = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, 'body', None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            assert 'eval_wave.py' not in node.value, (
                'vec_wave names eval_wave.py outside a docstring: {0!r}'.format(node.value))


# ------------------------------------------------------------------ the payload stitch

def _shard_payload(policy, steps, episodes, index, shards):
    """A shard's result file, built through `eval_plan.build_payload` exactly as `vec_eval` does, so
    this fixture cannot drift from the thing it stands in for."""
    results = [{'step': step, 'episodes': episodes, 'perfect_games': episodes - 1,
                'perfect_percent': round(100.0 * (episodes - 1) / episodes, 1),
                'perfect_ci95': [0.0, 100.0], 'avg_score': 94.0, 'avg_reward': 1.0,
                'min_score': 90, 'max_score': 95, 'episode_scores': [95] * episodes}
               for step in steps]
    spec = eval_plan.PayloadSpec(
        policy_name=policy, num_episodes=episodes, all_steps=list(steps), num_workers=None,
        screen_episodes=None, confirm_count=None, min_achievable=None, abandon_floor=None,
        measurements_planned=len(steps), episodes_planned=len(steps) * episodes,
        full_planned=len(steps), screen_planned=0, confirm_planned=0)
    progress = {'measurements': len(steps), 'session_measurements': len(steps),
                'session_episodes': len(steps) * episodes, 'session_seconds': 12.0,
                'episodes_saved': 0, 'abandoned': 0, 'stage': 'full',
                'full_done': len(steps), 'screen_done': 0, 'confirm_done': 0,
                'wave_lanes': 1024}
    return eval_plan.build_payload(spec, progress, {}, results, complete=True)


def _write(policy, suffix, payload, root):
    path = os.path.join(root, '{0}_checkpoint_evals{1}.json'.format(policy, suffix))
    import json
    with open(path, 'w') as handle:
        json.dump(payload, handle)
    return path


def test_a_sharded_arms_file_carries_the_same_keys_as_an_unsharded_ones():
    """The point of the stitch. `merge_checkpoint_evals` writes seven keys and drops the rest, so
    without this an arm's close-out file would depend on **how many processes measured it** -- a
    one-shard arm and a twelve-shard arm in the same wave publishing different-shaped files.

    Compared against `build_payload`'s own output rather than a hardcoded key list, so a field added
    to the schema is covered here the day it is added.
    """
    import json
    import shutil
    import tempfile

    root = tempfile.mkdtemp()
    saved = vec_wave.RUNS_DIR
    saved_plan = eval_plan.RUNS_DIR
    try:
        vec_wave.RUNS_DIR = eval_plan.RUNS_DIR = root
        # Three shards of one arm, strided the way `shard_steps` strides: 1,4 / 2,5 / 3,6.
        steps = [1000, 2000, 3000, 4000, 5000, 6000]
        pieces = ['-s{0}of3'.format(i) for i in (1, 2, 3)]
        for index, piece in enumerate(pieces, start=1):
            _write('arm', piece,
                   _shard_payload('arm', steps[index - 1::3], 100, index, 3), root)
        eval_plan.merge_checkpoint_evals('arm', suffixes=pieces, out_suffix='')
        stitched = vec_wave.stitch_payload('arm', '', pieces)

        # The reference: the same arm measured by a single process.
        single = _shard_payload('arm', steps, 100, 1, 1)
        missing = set(single) - set(stitched)
        assert not missing, 'stitched file is missing {0}'.format(sorted(missing))

        # And the numbers have to be the arm's, not a shard's.
        assert stitched['requested_steps'] == steps
        assert stitched['checkpoints_requested'] == 6
        assert len(stitched['results']) == 6
        assert stitched['episodes_done'] == 600
        assert stitched['measurements_planned'] == 6
        assert stitched['episodes_planned'] == 600
        assert stitched['complete'] is True
        # Wall clock is the wave's, not the sum: the shards ran at the same time, so summing would
        # report three times the elapsed time as if it had been serial.
        assert stitched['session_seconds'] == 12.0
        # Rows survive whole, `episode_scores` included -- the shards partition the selection, so no
        # step is measured twice and no row is pooled.
        assert all(len(row['episode_scores']) == 100 for row in stitched['results'])
        # Provenance the rebuild cannot know is kept from the merge.
        assert len(stitched['merged_from']) == 3
        with open(os.path.join(root, 'arm_checkpoint_evals.json')) as handle:
            assert json.load(handle)['requested_steps'] == steps, 'the stitch was not written'
    finally:
        vec_wave.RUNS_DIR = saved
        eval_plan.RUNS_DIR = saved_plan
        shutil.rmtree(root, ignore_errors=True)


def test_the_stitch_reads_the_protocol_back_out_of_the_shards():
    """Restating "flat, no gate" here would be a second opinion that could go stale against
    `vec_eval`. The assertion is that the fields travel from the shard, so a shard measured *with* a
    gate produces a merged file that says so."""
    import shutil
    import tempfile

    root = tempfile.mkdtemp()
    saved, saved_plan = vec_wave.RUNS_DIR, eval_plan.RUNS_DIR
    try:
        vec_wave.RUNS_DIR = eval_plan.RUNS_DIR = root
        pieces = ['-s1of2', '-s2of2']
        for index, (piece, steps) in enumerate(zip(pieces, ([1000, 3000], [2000, 4000])), 1):
            payload = _shard_payload('gated', steps, 100, index, 2)
            payload['min_achievable'] = 97.0
            payload['abandon_floor'] = 20
            payload['screen_episodes'] = 20
            payload['confirm_count'] = 12
            _write('gated', piece, payload, root)
        eval_plan.merge_checkpoint_evals('gated', suffixes=pieces, out_suffix='')
        stitched = vec_wave.stitch_payload('gated', '', pieces)
        assert stitched['min_achievable'] == 97.0
        assert stitched['abandon_floor'] == 20
        assert stitched['screen_episodes'] == 20
        assert stitched['confirm_count'] == 12
    finally:
        vec_wave.RUNS_DIR, eval_plan.RUNS_DIR = saved, saved_plan
        shutil.rmtree(root, ignore_errors=True)


def test_an_incomplete_shard_makes_the_arms_file_incomplete():
    """`select_checkpoints_above` refuses to select from a file that is not `complete`, so a truncated
    shard must not be laundered into a finished-looking arm by the merge."""
    import shutil
    import tempfile

    root = tempfile.mkdtemp()
    saved, saved_plan = vec_wave.RUNS_DIR, eval_plan.RUNS_DIR
    try:
        vec_wave.RUNS_DIR = eval_plan.RUNS_DIR = root
        pieces = ['-s1of2', '-s2of2']
        good = _shard_payload('part', [1000, 3000], 100, 1, 2)
        bad = _shard_payload('part', [2000, 4000], 100, 2, 2)
        bad['complete'] = False
        _write('part', pieces[0], good, root)
        _write('part', pieces[1], bad, root)
        eval_plan.merge_checkpoint_evals('part', suffixes=pieces, out_suffix='')
        stitched = vec_wave.stitch_payload('part', '', pieces)
        assert stitched['complete'] is False
        assert stitched['incomplete_sources'], stitched['incomplete_sources']
    finally:
        vec_wave.RUNS_DIR, eval_plan.RUNS_DIR = saved, saved_plan
        shutil.rmtree(root, ignore_errors=True)


def test_run_stage_spawns_a_process_per_shard_and_leaves_a_stitched_arm_file():
    """The wiring test, and the one that catches a stitch that is written but never called.

    **`subprocess.Popen` is stubbed, not `subprocess.run`** -- the whole point is that this fixture
    must not be able to start twelve real TensorFlow processes if an assertion is wrong. A
    `tests/test_chart_viewer.py` fixture that stubbed only `run` once opened three real training
    windows on the laptop when its assertion failed.

    The stub stands in for a shard: it writes the result file that shard would have written, which is
    what lets the merge and the stitch run for real.
    """
    import shutil
    import subprocess
    import tempfile

    root = tempfile.mkdtemp()
    saved = (vec_wave.RUNS_DIR, eval_plan.RUNS_DIR, vec_wave.selection_size,
             subprocess.Popen, vec_wave.subprocess.Popen)
    spawned = []

    class FakeShard(object):
        """One `vec_eval.py` process: writes its stride's rows, then exits 0."""

        def __init__(self, argv, env=None, **_):
            spawned.append((argv, env))
            policy, selector = argv[-2], argv[-1]
            shard = env.get('VEC_EVAL_SHARD')
            steps = [int(part) for part in selector.replace(',', ' ').split()]
            if shard:
                index, count = (int(x) for x in shard.split('/'))
                steps = steps[index - 1::count]
                suffix = '{0}-s{1}of{2}'.format(env['EVAL_OUT_SUFFIX'], index, count)
            else:
                suffix = env['EVAL_OUT_SUFFIX']
            _write(policy, suffix,
                   _shard_payload(policy, steps, int(env['VEC_EVAL_EPISODES']), 1, 1), root)
            self.returncode = 0

        def wait(self):
            return 0

    try:
        vec_wave.RUNS_DIR = eval_plan.RUNS_DIR = root
        vec_wave.selection_size = lambda policy, selector, source: 6
        subprocess.Popen = vec_wave.subprocess.Popen = FakeShard
        steps = '1000,2000,3000,4000,5000,6000'
        code = vec_wave.run_stage('stage A: test', ['arm-a', 'arm-b'], steps, 100, '', '', 6)
        assert code == 0, code
        assert len(spawned) == 6, spawned
        # Three shards each, and every child was told which stride it owns.
        assert sorted(env['VEC_EVAL_SHARD'] for _, env in spawned) == [
            '1/3', '1/3', '2/3', '2/3', '3/3', '3/3']
        for policy in ('arm-a', 'arm-b'):
            path = os.path.join(root, '{0}_checkpoint_evals.json'.format(policy))
            assert os.path.exists(path), path
            import json
            with open(path) as handle:
                payload = json.load(handle)
            # Stitched, not merely merged: `requested_steps` is a `build_payload` field that
            # `merge_checkpoint_evals` does not write at all.
            assert 'requested_steps' in payload, 'run_stage merged but did not stitch'
            assert payload['requested_steps'] == [1000, 2000, 3000, 4000, 5000, 6000]
            assert len(payload['results']) == 6
            assert payload['episodes_done'] == 600
    finally:
        (vec_wave.RUNS_DIR, eval_plan.RUNS_DIR, vec_wave.selection_size,
         subprocess.Popen, vec_wave.subprocess.Popen) = saved
        shutil.rmtree(root, ignore_errors=True)


def test_a_failed_shard_is_reported_and_the_stage_fails():
    """A non-zero shard must not pass as a finished arm -- `--chain` would then select a HOF pass out
    of a partial close-out. Every sibling is still waited on: it holds a slice already paid for.

    **The stub writes a valid result file and *then* exits non-zero**, so the exit code is the only
    thing that can fail this stage. A stub that wrote nothing would fail the merge instead, and the
    test would pass while `run_stage` ignored exit codes entirely -- which is exactly what an earlier
    version of this fixture did.
    """
    import contextlib
    import io
    import shutil
    import subprocess
    import tempfile

    root = tempfile.mkdtemp()
    saved = (vec_wave.RUNS_DIR, eval_plan.RUNS_DIR, vec_wave.selection_size,
             subprocess.Popen, vec_wave.subprocess.Popen)
    waited = []

    class Failing(object):
        def __init__(self, argv, env=None, **_):
            self.returncode = 1
            self.policy = argv[-2]
            shard = env['VEC_EVAL_SHARD']
            index, count = (int(x) for x in shard.split('/'))
            steps = [int(x) for x in argv[-1].split(',')][index - 1::count]
            _write(self.policy, '{0}-s{1}of{2}'.format(env['EVAL_OUT_SUFFIX'], index, count),
                   _shard_payload(self.policy, steps, 100, index, count), root)

        def wait(self):
            waited.append(self.policy)
            return 1

    try:
        vec_wave.RUNS_DIR = eval_plan.RUNS_DIR = root
        vec_wave.selection_size = lambda policy, selector, source: 4
        subprocess.Popen = vec_wave.subprocess.Popen = Failing
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            code = vec_wave.run_stage('stage A: test', ['arm-a', 'arm-b'], '1,2,3,4', 100, '', '', 4)
        assert code == 1, code
        assert len(waited) == 4, waited     # every shard reaped, not abandoned at the first failure
        # And it has to say *which* shard, or a wave of twelve reports a number and no location.
        report = said.getvalue()
        assert 'shard 1/2' in report and 'arm-a' in report and 'arm-b' in report, report
        assert '4 failure(s)' in report, report
    finally:
        (vec_wave.RUNS_DIR, eval_plan.RUNS_DIR, vec_wave.selection_size,
         subprocess.Popen, vec_wave.subprocess.Popen) = saved
        shutil.rmtree(root, ignore_errors=True)


def test_a_stage_with_nothing_selected_succeeds_without_spawning_anything():
    """The normal outcome of a HOF pass: most arms have no checkpoint at 98%. Not an error, and it
    must not spawn a process that would exit cleanly having measured nothing."""
    import subprocess

    saved = (vec_wave.selection_size, subprocess.Popen, vec_wave.subprocess.Popen)

    def explode(*args, **kwargs):
        raise AssertionError('run_stage spawned a process for an empty selection')

    try:
        vec_wave.selection_size = lambda policy, selector, source: 0
        subprocess.Popen = vec_wave.subprocess.Popen = explode
        assert vec_wave.run_stage('stage B: test', ['arm-a'], 'above:98', 500, '', '', 12) == 0
    finally:
        (vec_wave.selection_size, subprocess.Popen, vec_wave.subprocess.Popen) = saved
