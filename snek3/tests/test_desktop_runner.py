"""The desktop daemon: the `project` guard, the config tiers, `build_command`, and the scheduler.

`desktop/` imports nothing from this project — it shells out — so it goes on `sys.path` separately
here. It also runs on **base** python on the box, before the conda env exists, which is why it is
stdlib-only and why these tests must not reach for numpy or torch either.

Three things here are silent when broken and each costs a night:

- **A stale snek2 spec dispatching.** `ops` still holds ~150 of them in `queue/pending/`. They are
  valid JSON with plausible ids.
- **A local-only results commit reported as `done`.** Indistinguishable from a pass that found
  nothing, and it once hid a 98.2%/500 checkpoint for hours.
- **A `failed` eval never retried and never surfaced.** It cost snek2's batch 46 its measurement.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'desktop'))

from runner import config as config_module
from runner import job as job_module
from runner import launch
from runner import runner as runner_module
from runner.job import JobError, parse_job


HOST = {
    'REPO_PATH': '/repo', 'SNEK_DIR': '/repo/snek3', 'PYTHON_BIN': '/py',
    'GIT_REMOTE': 'origin', 'OPS_BRANCH': 'ops', 'STATUS_BRANCH': 'ops-status',
    'RESULTS_BRANCH': 'results', 'STATUS_WORKTREE': '/wt/status',
    'RESULTS_WORKTREE': '/wt/results', 'LEDGER_PATH': '/var/snek/ledger.json',
    'LOG_DIR': '/var/snek/logs', 'QUEUE_DIR': 'snek3/desktop/queue/pending',
    'RUNTIME_PATH': 'snek3/desktop/config/runtime.json',
    'HARD_MAX_EVAL_SHARDS': 16,
    'MIN_POLL_SECONDS': 10,
}


def spec(**overrides):
    body = {'project': 'snek3', 'id': 'b1a-thing', 'type': 'train', 'policy': 'b1a-thing'}
    body.update(overrides)
    return json.dumps(body)


# --- the project guard ------------------------------------------------------------------------

def test_a_snek3_spec_parses():
    parsed = parse_job(spec())
    assert parsed.project == 'snek3' and parsed.id == 'b1a-thing'


def test_a_snek2_spec_is_refused():
    """The ~150 retired specs still sitting in `queue/pending/` on `ops`.

    Refused rather than skipped, so an accidental dispatch is impossible and the rejection is
    recorded against the spec where a human will see it.
    """
    with pytest.raises(JobError, match='project must be "snek3"'):
        parse_job(spec(project='snek2'))


def test_a_spec_with_no_project_at_all_is_refused():
    # There is deliberately no default: a missing field is the exact case being guarded against, so
    # defaulting it to snek3 would defeat the guard on precisely the specs it exists for.
    body = json.loads(spec())
    del body['project']
    with pytest.raises(JobError, match='project must be "snek3"'):
        parse_job(json.dumps(body))


def test_the_guard_is_checked_before_anything_else():
    # A snek2 spec whose *other* fields are also invalid must still be reported as a project
    # mismatch, or the message sends a reader chasing the wrong problem.
    with pytest.raises(JobError, match='project'):
        parse_job(json.dumps({'project': 'snek2', 'id': '!!bad!!', 'type': 'nonsense'}))


def test_a_future_era_inherits_the_guard():
    with pytest.raises(JobError, match='project must be "snek4"'):
        parse_job(spec(), project='snek4')


# --- the rest of spec validation ----------------------------------------------------------------

def test_a_bad_id_is_refused():
    with pytest.raises(JobError, match='id must match'):
        parse_job(spec(id='has spaces'))


def test_an_unknown_type_is_refused():
    with pytest.raises(JobError, match='type must be one of'):
        parse_job(spec(type='train-but-different'))


def test_a_training_needs_a_policy():
    body = json.loads(spec())
    del body['policy']
    with pytest.raises(JobError, match='need a "policy"'):
        parse_job(json.dumps(body))


def test_only_an_eval_takes_several_policies():
    # A training is one arm by construction; `policies` on one would silently train the first.
    with pytest.raises(JobError, match='only eval jobs take "policies"'):
        parse_job(spec(policies=['a', 'b']))


def test_an_eval_wave_carries_every_arm():
    parsed = parse_job(spec(type='eval', policies=['b1a', 'b1b', 'b1c'], policy=None))
    assert parsed.policies == ['b1a', 'b1b', 'b1c']
    assert parsed.policy == 'b1a', 'the sugar field fills in from the list'


def test_a_single_policy_fills_in_the_list():
    # Both directions, because `_StubJob` rebuilds a job from a ledger record with no spec in hand.
    parsed = parse_job(spec())
    assert parsed.policies == ['b1a-thing'] and parsed.policy == 'b1a-thing'


def test_a_boolean_is_not_an_integer():
    # `isinstance(True, int)` is True in Python, so `eval_shards: true` would become one shard.
    with pytest.raises(JobError, match='eval_shards must be an integer'):
        parse_job(spec(type='eval', eval_shards=True))


def test_env_values_are_stringified():
    # They go straight into a subprocess environment, where a JSON int and float would arrive as
    # '1' and '1.0' and change an arm's config.
    parsed = parse_job(spec(env={'SNEK_SEED': 3, 'SNEK_FORK_PROB': 0.5}))
    assert parsed.env == {'SNEK_SEED': '3', 'SNEK_FORK_PROB': '0.5'}


def test_only_an_eval_takes_a_selector():
    with pytest.raises(JobError, match='only eval jobs take "selector"'):
        parse_job(spec(selector='screen:98'))


def test_malformed_json_names_its_source():
    with pytest.raises(JobError, match='b1a.json: not valid JSON'):
        parse_job('{not json', source='b1a.json')


# --- the config tiers ---------------------------------------------------------------------------

def test_the_defaults_pass_their_own_clamps():
    config, notes = config_module.parse_runtime_config('{}', HOST)
    assert config is not None and notes == [], notes
    assert config == config_module.RUNTIME_DEFAULTS


def test_one_bad_field_rejects_the_whole_file():
    """A partially applied config is worse than a rejected one: it looks like it worked.

    The box has no ssh backstop in normal operation, so the caller keeps its last known-good config.
    """
    config, errors = config_module.parse_runtime_config('{"max_trainers": "lots"}', HOST)
    assert config is None
    assert any('max_trainers must be an integer' in error for error in errors)


def test_an_unknown_key_rejects_the_file():
    # Usually a *renamed* key, and ignoring it would run the default while the file says otherwise.
    config, errors = config_module.parse_runtime_config('{"auto_closeout": true}', HOST)
    assert config is None
    assert any('unknown key: auto_closeout' in error for error in errors)


def test_a_value_over_the_ceiling_is_clamped_and_noted_not_refused():
    # An ambitious number is honoured as the nearest legal value; refusing the file would be worse.
    config, notes = config_module.parse_runtime_config('{"eval_shards": 99}', HOST)
    assert config['eval_shards'] == HOST['HARD_MAX_EVAL_SHARDS']
    assert any('eval_shards 99 clamped to 16' in note for note in notes)


def test_wave_width_has_no_host_ceiling_because_it_is_not_a_safety_property():
    """`HARD_MAX_TRAINERS` is gone; `_dispatch` is what keeps waves from overlapping.

    The second tier was worse than redundant: `runtime.json` was clamped to it and the request merely
    *noted*, so a commit asking for 8 arms against a stale ceiling of 4 ran 4 and looked like it had
    worked. 8 is now honoured exactly, and the sanity bound is far away.
    """
    config, notes = config_module.parse_runtime_config('{"max_trainers": 8}', HOST)
    assert config['max_trainers'] == 8 and notes == []


def test_the_poll_floor_holds():
    config, _ = config_module.parse_runtime_config('{"poll_seconds": 1}', HOST)
    assert config['poll_seconds'] == HOST['MIN_POLL_SECONDS']


def test_git_seconds_may_be_zero_because_zero_is_the_opt_out():
    # 0 means a network cycle every poll, which is what the daemon did before the knob existed.
    config, notes = config_module.parse_runtime_config('{"git_seconds": 0}', HOST)
    assert config['git_seconds'] == 0 and notes == []


def test_a_stage_b_wave_does_not_inherit_the_stage_a_queue_knobs():
    """The queue configures how a *training* measures itself; a wave measures finished checkpoints.

    `evaluate.py` reads none of the three, so carrying them was harmless — but this tuple exists to
    stop a training-loop setting being attributed to a measurement, and `runs/<policy>.md` would
    otherwise report a wave as having run under a worker count that never applied to it.
    """
    inherited = runner_module.inherited_eval_env({
        'SNEK_EVAL_QUEUE': '1', 'SNEK_EVAL_QUEUE_DEPTH': '16', 'SNEK_EVAL_WORKERS': '6',
        'SNEK_CHASE_SAFE_GATE': '75', 'SNEK_SEED': '2',
    })
    assert inherited == {'SNEK_CHASE_SAFE_GATE': '75', 'SNEK_SEED': '2'}


def test_max_evals_is_gone_and_naming_it_is_an_error_rather_than_a_no_op():
    """Removed 2026-08-29: an eval job *is* a wave, so the count could only ever be 1.

    Rejected rather than ignored, because a `runtime.json` still carrying it would otherwise look
    applied. The unknown-key path already refuses the whole file and keeps the last known-good config,
    which is the right outcome for a stale ops commit.
    """
    config, errors = config_module.parse_runtime_config('{"max_evals": 2}', HOST)
    assert config is None
    assert any('unknown key: max_evals' in error for error in errors)


def test_the_shard_ceiling_is_the_thread_count_and_is_no_longer_divided_by_a_wave_count():
    """Past `cpu_count` throughput *falls*, so 16 single-threaded shards is the measured optimum.

    It used to be `HARD_MAX_EVAL_SHARDS // max_evals`, which meant asking for two waves silently
    halved the width of the one wave that could actually run.
    """
    config, notes = config_module.parse_runtime_config('{"eval_shards": 16}', HOST)
    assert config['eval_shards'] == 16 and notes == []


def test_threads_default_to_one_because_more_is_measured_slower():
    # 16 shards each taking a thread per core is a 16x oversubscription, and the nets are far too
    # small for a fork-join to pay for itself: measured 1.4x slower at torch's default.
    assert config_module.RUNTIME_DEFAULTS['torch_threads'] == 1
    assert config_module.RUNTIME_DEFAULTS['omp_num_threads'] == 1


def test_host_env_missing_a_key_is_fatal(tmp_path):
    path = tmp_path / 'host.env'
    path.write_text('REPO_PATH=/repo\n')
    with pytest.raises(config_module.ConfigError, match='missing keys'):
        config_module.load_host_config(str(path))


def test_host_env_reads_comments_and_coerces_the_ceilings(tmp_path):
    path = tmp_path / 'host.env'
    path.write_text('# a comment\n\n' + '\n'.join(
        '{0}={1}'.format(key, HOST[key]) for key in config_module._REQUIRED_HOST) + '\n')
    loaded = config_module.load_host_config(str(path))
    assert loaded['HARD_MAX_EVAL_SHARDS'] == 16 and isinstance(loaded['HARD_MAX_EVAL_SHARDS'], int)


# --- build_command ------------------------------------------------------------------------------

def runtime(**overrides):
    config = dict(config_module.RUNTIME_DEFAULTS)
    config.update(overrides)
    return config


def test_a_training_invokes_train_py_with_its_policy():
    argv, env, log, policy = launch.build_command(parse_job(spec()), HOST, runtime())
    assert argv == ['/py', '-u', 'train.py', 'b1a-thing']
    assert log == 'train-b1a-thing.log' and policy == 'b1a-thing'
    assert env['PYTHONPATH'] == '.', 'snek3 entry points need the project root on the path'


def test_a_batch_is_one_job_and_one_process():
    """One ledger record, one publish and one process per batch — `tools/closeout.py` owns the order.

    The arms are still measured one at a time (2026-08-29): a wave takes a single policy, so N arms
    are N waves. What changed on 2026-08-30 is *who* sequences them — this used to be a `sh -c` chain
    built here, which the laptop's close-out did not share.
    """
    job = parse_job(spec(type='eval', policies=['b1a', 'b1b'], policy=None))
    argv, _, log, _ = launch.build_command(job, HOST, runtime(eval_shards=16))
    assert argv[:4] == ['/py', '-u', '-m', 'tools.closeout']
    assert argv[4:6] == ['b1a', 'b1b'], 'in the order the spec named them'
    assert argv[argv.index('--shards') + 1] == '16'
    assert log == 'eval-b1a-thing.log'


def test_no_shell_is_involved_at_all():
    """No `sh -c`, so nothing here quotes anything — which is what retires the whole hazard below."""
    job = parse_job(spec(type='eval', policies=['b1a', 'b1b', 'b1c'], policy=None))
    argv, _, _, _ = launch.build_command(job, HOST, runtime())
    assert argv[0] == '/py' and '-c' not in argv


def test_a_policy_name_is_data_not_shell_syntax():
    """It reaches the close-out as one argv element, so shell metacharacters in it mean nothing.

    Kept as a fixture after the `sh -c` chain went away, because the property is the point and the
    next person to reintroduce a shell here should see it fail.
    """
    job = parse_job(spec(type='eval', policies=['b1a; rm -rf /', 'b1b'], policy=None))
    argv, _, _, _ = launch.build_command(job, HOST, runtime())
    assert argv[4] == 'b1a; rm -rf /', 'the name is passed through whole'
    assert parsed(argv).policies == ['b1a; rm -rf /', 'b1b']


def test_no_episode_count_or_gate_is_passed_unless_the_spec_asked():
    """The protocol lives in `tools/closeout.py`'s defaults, not here.

    snek2's daemon carried five protocol numbers as a second copy of `eval_plan.py`'s and they
    drifted. So an unadorned eval job must pass neither, and inherit `screen:95` at 500 episodes.
    """
    job = parse_job(spec(type='eval', policies=['b1a'], policy=None))
    argv, _, _, _ = launch.build_command(job, HOST, runtime())
    assert '--episodes' not in argv
    assert 'screen' not in ' '.join(argv)


def test_a_spec_may_still_name_a_selector_and_a_depth():
    """The selector is a `--selector` FLAG for the close-out, where `evaluate.py` takes a positional.

    It has to be: the close-out's policies are `nargs='+'`, so a positional behind them is ambiguous.
    Getting this backwards is one of the two drifts that made every dispatched wave exit 2.
    """
    job = parse_job(spec(type='eval', policies=['b1a'], policy=None,
                         selector='screen:98', episodes=1000))
    argv, _, _, _ = launch.build_command(job, HOST, runtime())
    assert argv[4] == 'b1a'
    assert argv[argv.index('--selector') + 1] == 'screen:98'
    assert argv[argv.index('--episodes') + 1] == '1000'


# --- the command must parse, asked of the real parser ---------------------------------------------
#
# This is the one place these tests deliberately cross the stdlib-only line, and the reason is the
# bug it exists for. The daemon runs on base python before the conda env exists, so it duplicates the
# close-out's command line by hand -- and got it wrong twice, several policies where one is taken and
# `--selector` where the selector is positional, so every wave the box dispatched exited 2. Four
# green fixtures asserted the argv the daemon builds; none asked the parser to accept it. A test runs
# in the env even though the daemon does not, so the constraint that forces the duplication does not
# extend to the fixture that guards it.

def parsed(argv):
    """`tools.closeout`'s own parser applied to a command the daemon built. Raises if it would exit 2.

    `argv[4:]` because the command is `python -u -m tools.closeout ...` -- one token longer than the
    `python -u evaluate.py ...` it replaced on 2026-08-30.
    """
    from tools import closeout
    return closeout.build_parser().parse_args(argv[4:])


def settings(argv):
    """What the close-out would actually measure with: the pass's preset, under any explicit flag."""
    from tools import closeout
    args = parsed(argv)
    return closeout.pass_settings(args.pass_name, args.selector, args.episodes, args.label, args.seed)


@pytest.mark.parametrize('overrides', [
    {},
    {'selector': 'screen:98'},
    {'episodes': 1000},
    {'selector': 'one', 'episodes': 3000},
    {'eval_args': ['--label', 'ab']},
    {'eval_args': ['--pass', 'hof5000']},
    {'eval_args': ['--pass', 'hof30k']},
])
def test_every_command_the_daemon_builds_parses(overrides):
    job = parse_job(spec(type='eval', policies=['b1a'], policy=None, **overrides))
    argv = launch.eval_command(job, HOST, runtime(eval_shards=16))
    parsed(argv)                  # SystemExit here is the daemon being wrong, not the test


def test_a_whole_batch_is_one_command_naming_every_arm():
    """The sequencing lives in `tools/closeout.py`, so the daemon builds one process, not a chain."""
    job = parse_job(spec(type='eval', policies=['b1a', 'b1b', 'b1c'], policy=None))
    argv = launch.eval_command(job, HOST, runtime())
    assert argv[:4] == [HOST['PYTHON_BIN'], '-u', '-m', 'tools.closeout']
    assert parsed(argv).policies == ['b1a', 'b1b', 'b1c']
    assert 'sh' not in argv[0:1] and not any(token == '-c' for token in argv)


def test_an_unadorned_wave_inherits_the_protocol_from_the_closeout():
    """What the daemon leaves unsaid has to arrive as `screen:97` at 500 episodes."""
    job = parse_job(spec(type='eval', policies=['b1a'], policy=None))
    chosen = settings(launch.eval_command(job, HOST, runtime()))
    assert chosen['selector'] == 'screen' and chosen['episodes'] == 500
    assert chosen['label'] is None and chosen['seed'] == 0


def test_a_synthesised_hof_pass_names_the_pass_and_nothing_else():
    """The daemon carries no protocol numbers: `--pass hof5000` is the whole instruction, and the
    close-out's preset turns it into `above:99` at 5,000 under the `hof5000` label -- the label being
    what keeps the pass from overwriting the 500-episode file it selects from."""
    job = runner_module.Job(id='b12-hof5000', type='eval', policies=['b12aa-x', 'b12ab-x'],
                            eval_args=['--pass', 'hof5000'], priority=11)
    argv = launch.eval_command(job, HOST, runtime(eval_shards=16))
    assert '--selector' not in argv and '--episodes' not in argv and '--label' not in argv
    assert settings(argv) == {'selector': 'above:99', 'episodes': 5000, 'label': 'hof5000', 'seed': 0}
    job = runner_module.Job(id='b12-hof30k', type='eval', policies=['b12aa-x'],
                            eval_args=['--pass', 'hof30k'], priority=12)
    assert settings(launch.eval_command(job, HOST, runtime())) == {
        'selector': 'above:99:hof5000', 'episodes': 30000, 'label': 'hof30k', 'seed': 7}


def test_the_selector_reaches_the_closeout_as_a_flag():
    """The one spelling difference from `evaluate.py`, where it is positional. See `eval_command`."""
    job = parse_job(spec(type='eval', policies=['b1a', 'b1b'], policy=None, selector='screen:98'))
    argv = launch.eval_command(job, HOST, runtime())
    assert argv[argv.index('--selector') + 1] == 'screen:98'
    assert parsed(argv).selector == 'screen:98'


def test_the_eval_job_still_forwards_the_display_so_its_window_can_open():
    """The stage-B window is opened by the close-out, so it needs the same two variables a trainer
    does -- and gets them because `build_command` forwards them for every job type."""
    host = dict(HOST, DISPLAY=':0', XAUTHORITY='/home/x/.Xauthority')
    _, env, _, _ = launch.build_command(
        parse_job(spec(type='eval', policies=['b1a'], policy=None)), host, runtime())
    assert env['DISPLAY'] == ':0' and env['XAUTHORITY'] == '/home/x/.Xauthority'


def test_the_viewer_switch_reaches_an_eval_wave_too():
    _, env, _, _ = launch.build_command(
        parse_job(spec(type='eval', policies=['b1a'], policy=None)), HOST, runtime(viewer=False))
    assert env['SNEK_CHART_WINDOW'] == '0'


def test_the_thread_knobs_reach_the_subprocess():
    _, env, _, _ = launch.build_command(parse_job(spec()), HOST,
                                        runtime(torch_threads=2, omp_num_threads=3))
    assert env['SNEK_TORCH_THREADS'] == '2' and env['OMP_NUM_THREADS'] == '3'


def test_a_jobs_own_env_wins_over_the_runtime_default():
    # Otherwise a per-arm override in a spec would be silently overruled by the box's config.
    _, env, _, _ = launch.build_command(
        parse_job(spec(env={'SNEK_TORCH_THREADS': '8'})), HOST, runtime(torch_threads=1))
    assert env['SNEK_TORCH_THREADS'] == '8'


def test_a_smoke_run_can_write_checkpoints_and_reach_an_eval():
    """A smoke run scores ~0, so the default checkpoint gate would write nothing.

    Which means it could not resume either — and "resume works" is most of what a smoke run is for.
    """
    _, env, _, policy = launch.build_command(
        parse_job(spec(type='smoke', id='smoke-1', policy=None)), HOST, runtime())
    assert policy == 'smoke'
    assert env['SNEK_MIN_CHECKPOINT_SCORE'] == '0'
    assert int(env['SNEK_EVAL_INTERVAL']) < int(env['SNEK_MAX_STEPS'])


def test_a_benchmark_turns_stage_a_off_rather_than_shrinking_it():
    # Stage A is ~90% of an arm's wall clock, so leaving it on would make it most of what the
    # benchmark measured.
    _, env, _, _ = launch.build_command(
        parse_job(spec(type='benchmark', id='bench-1', policy=None, max_steps=20000)),
        HOST, runtime())
    assert env['SNEK_EVAL_INTERVAL'] == env['SNEK_MAX_STEPS'] == '20000'


def test_max_steps_from_the_spec_reaches_the_trainer():
    _, env, _, _ = launch.build_command(parse_job(spec(max_steps=3000000)), HOST, runtime())
    assert env['SNEK_MAX_STEPS'] == '3000000'


# --- the env split ------------------------------------------------------------------------------

def test_the_eval_relevant_env_names_knobs_that_still_exist():
    """A renamed reward knob would silently stop splitting waves that should be split.

    Checked against `env/constants.py`'s own `SNEK_` reads, which is the definition — the daemon
    cannot import it (numpy), so this test is the link between the two.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, 'env', 'constants.py')).read()
    for name in runner_module.EVAL_RELEVANT_ENV:
        bare = name[len('SNEK_'):]
        assert ("_num('{0}'".format(bare) in source
                or "_flag('{0}'".format(bare) in source
                or name in source), '{0} is not read by env/constants.py any more'.format(name)


def test_the_training_only_env_is_stripped_from_an_inherited_eval_env():
    # A step cap is meaningless to a measurement, and the stage-A knobs describe a screen that
    # already happened.
    stripped = runner_module.inherited_eval_env(
        {'SNEK_MAX_STEPS': '3000000', 'SNEK_SEED': '1', 'SNEK_CHASE_SAFE_SHAPING': '0.1'})
    assert 'SNEK_MAX_STEPS' not in stripped
    assert stripped == {'SNEK_SEED': '1', 'SNEK_CHASE_SAFE_SHAPING': '0.1'}


def test_arms_differing_only_by_seed_share_one_wave():
    """The failure this exists to prevent, measured on snek2's b45.

    Keying on the whole env split its close-out into three waves of 2/1/1 arms, because its arms
    differ in `SNEK_SEED` — which cannot reach a measurement of an already-trained checkpoint.
    """
    keys = [runner_module.stage_b_group_env({'SNEK_SEED': str(seed), 'SNEK_LEARNING_RATE': '1e-5'})
            for seed in (1, 2, 3, 4)]
    assert all(key == {} for key in keys)


def test_arms_differing_in_a_reward_knob_do_not_share_a_wave():
    # A wave is one process with one environment, and shaping changes what `avg_reward` means.
    first = runner_module.stage_b_group_env({'SNEK_CHASE_SAFE_SHAPING': '0.1'})
    second = runner_module.stage_b_group_env({'SNEK_CHASE_SAFE_SHAPING': '0.2'})
    assert first != second


def test_the_agreed_env_drops_what_the_arms_disagree_about():
    # Handing the first arm's env to the whole wave would attribute one arm's seed to all four.
    agreed = runner_module.agreed_env([{'SNEK_SEED': '1', 'SNEK_DISCOUNT': '0.99'},
                                       {'SNEK_SEED': '2', 'SNEK_DISCOUNT': '0.99'}])
    assert agreed == {'SNEK_DISCOUNT': '0.99'}


# --- ids and phases -----------------------------------------------------------------------------

def test_batch_of_reads_the_leading_batch_number():
    assert runner_module.batch_of('b12c-thing-seed3') == 'b12'
    assert runner_module.batch_of('smoke-1') == 'smoke'


def test_batch_of_groups_the_ppo_batches_too():
    """A PPO batch groups by batch exactly as a DQN one does — one prefix serves both.

    `_BATCH_RE` matched only `b\\d+` while the PPO series was named `p0-p3`, so every PPO arm fell
    through to the `split('-')[0]` fallback and became its own batch. Nothing measured wrong, but both
    things that group by batch degraded silently: `_auto_stage_b_jobs` promises "one job per batch,
    not one per arm" and was synthesising one wave per arm, and `at_a_glance` listed eight lines for
    one batch. The fix on 2026-08-31 was to rename the batches to `b3-b6` rather than to widen the
    pattern, so this stays a test that one prefix covers everything.
    """
    for letter in 'abcdefgh':
        assert runner_module.batch_of('b4{0}-fc200x100ep8-seed1'.format(letter)) == 'b4'
    assert runner_module.batch_of('b5a-ep8-seed1') == 'b5'
    assert runner_module.batch_of('b10a-thing-seed1') == 'b10'
    # The synthesised stage-B id round-trips back to the same batch.
    assert runner_module.batch_of('b4-stageb') == 'b4'


def test_batch_of_falls_back_for_a_legacy_p_series_id():
    """The box's ledger still holds pre-rename ids, and they take the fallback.

    Deliberate, and cosmetic: every `p`-named wave is `done`, nothing re-groups a finished record, and
    the alternative — keeping `[bp]` in the pattern — is the second prefix the rename removed.
    """
    assert runner_module.batch_of('p1a-fc200x100ep8-seed1') == 'p1a'
    assert runner_module.batch_of('p2-hof5000') == 'p2'


def test_a_second_wave_of_a_batch_is_still_stage_b():
    # `-w<k>` has to be part of the pattern, or the second wave reads as a bare eval and splits the
    # at-a-glance grouping in two.
    assert runner_module.phase_of('b1-stageb', 'eval') == 'stage B'
    assert runner_module.phase_of('b1-stageb-w2', 'eval') == 'stage B'
    assert runner_module.phase_of('b1a-thing', 'train') == 'training'


# --- the scheduler forecast ---------------------------------------------------------------------

def test_a_batchs_measurement_slots_in_ahead_of_the_next_batch():
    """The interleaving the box actually runs, and the whole point of the auto chain.

    Never train the next thing before measuring the last.
    """
    queued = [{'id': 'b1{0}-x'.format(letter), 'type': 'train', 'policy': 'b1{0}-x'.format(letter),
               'policies': ['b1{0}-x'.format(letter)], 'priority': 100}
              for letter in 'ab']
    queued += [{'id': 'b2a-y', 'type': 'train', 'policy': 'b2a-y', 'policies': ['b2a-y'],
                'priority': 100}]
    order = runner_module.anticipated_queue(
        queued, [], {'trainer': 2, 'eval': 1}, True, {job['id'] for job in queued})
    assert [job['id'] for job in order] == [
        'b1a-x', 'b1b-x', 'b1-stageb', 'b1-hof5000', 'b1-hof30k',
        'b2a-y', 'b2-stageb', 'b2-hof5000', 'b2-hof30k']
    passes = {job['id']: job for job in order if job['type'] == 'eval'}
    assert passes['b1-hof30k']['policies'] == ['b1a-x', 'b1b-x'], 'the same arms all the way down'
    assert (passes['b1-stageb']['priority'] < passes['b1-hof5000']['priority']
            < passes['b1-hof30k']['priority'] < 100), 'each pass outranks the next training'


def test_a_batchs_four_arms_forecast_one_measurement_row_not_four():
    queued = [{'id': 'b1{0}-x'.format(letter), 'type': 'train', 'policy': 'b1{0}-x'.format(letter),
               'policies': ['b1{0}-x'.format(letter)], 'priority': 100}
              for letter in 'abcd']
    order = runner_module.anticipated_queue(
        queued, [], {'trainer': 4, 'eval': 1}, True, {job['id'] for job in queued})
    stage_b = [job for job in order if job['id'].endswith('-stageb')]
    assert len(stage_b) == 1
    assert stage_b[0]['policies'] == ['b1a-x', 'b1b-x', 'b1c-x', 'b1d-x']


def test_nothing_is_forecast_when_the_auto_chain_is_off():
    queued = [{'id': 'b1a-x', 'type': 'train', 'policy': 'b1a-x', 'policies': ['b1a-x'],
               'priority': 100}]
    order = runner_module.anticipated_queue(queued, [], {'trainer': 4, 'eval': 1}, False, {'b1a-x'})
    assert [job['id'] for job in order] == ['b1a-x']


def test_a_queued_spec_stands_for_its_wave_and_is_never_invented_twice():
    """A hand-written `b1-stageb` in the queue is the wave; the forecast must not add a second.
    The chain still continues past it -- the spec's pass earns the next one."""
    queued = [{'id': 'b1a-x', 'type': 'train', 'policy': 'b1a-x', 'policies': ['b1a-x'],
               'priority': 100},
              {'id': 'b1-stageb', 'type': 'eval', 'policy': 'b1a-x', 'policies': ['b1a-x'],
               'priority': 100}]
    order = runner_module.anticipated_queue(
        queued, [], {'trainer': 4, 'eval': 1}, True, {'b1a-x', 'b1-stageb'})
    assert [job['id'] for job in order] == ['b1a-x', 'b1-stageb', 'b1-hof5000', 'b1-hof30k']


def test_a_batch_whose_first_wave_is_measured_forecasts_its_second_as_w2_not_nothing():
    """b15, 2026-09-04: wave 1 done and `b15-stageb` in the ledger, 32 arms still queued -- and the
    forecast showed no b15 measurement at all, because it minted only the bare id and found it taken.
    The dispatcher mints `-w2`, so the forecast has to."""
    queued = [{'id': 'b15{0}-x'.format(letter), 'type': 'train', 'policy': 'b15{0}-x'.format(letter),
               'policies': ['b15{0}-x'.format(letter)], 'priority': 100} for letter in 'ab']
    existing = {job['id'] for job in queued} | {'b15-stageb', 'b15-hof5000', 'b15-hof30k'}
    order = runner_module.anticipated_queue(queued, [], {'trainer': 8, 'eval': 1}, True, existing)
    assert [job['id'] for job in order] == [
        'b15a-x', 'b15b-x', 'b15-stageb-w2', 'b15-hof5000-w2', 'b15-hof30k-w2']


def test_a_running_pass_earns_its_follow_on_in_the_forecast():
    running = [{'id': 'b11-hof5000', 'type': 'eval', 'policy': 'b11a-x', 'policies': ['b11a-x', 'b11b-x']}]
    order = runner_module.anticipated_queue([], running, {'trainer': 8, 'eval': 1}, True, {'b11-hof5000'})
    assert [(job['id'], job['policies']) for job in order] == [('b11-hof30k', ['b11a-x', 'b11b-x'])]


def test_only_a_training_earns_a_measurement():
    # Smoke and benchmark share the `trainer` category but are throwaway, and a failed run has no
    # checkpoint worth measuring.
    assert runner_module.wants_stage_b('train', True, True) is True
    assert runner_module.wants_stage_b('train', False, True) is False
    assert runner_module.wants_stage_b('smoke', True, True) is False
    assert runner_module.wants_stage_b('benchmark', True, True) is False
    assert runner_module.wants_stage_b('train', True, False) is False


def test_each_pass_earns_the_next_and_the_last_earns_nothing():
    """training -> stageb -> hof5000 -> hof30k, over the same arms, and only on success: the next
    pass selects from the file this one wrote, so a failed pass has nothing to hand on."""
    nxt = runner_module.next_pass
    assert nxt('b1a-x', 'train', True, True) == 'stageb'
    assert nxt('b1-stageb', 'eval', True, True) == 'hof5000'
    assert nxt('b1-stageb-w3', 'eval', True, True) == 'hof5000'
    assert nxt('b1-hof5000', 'eval', True, True) == 'hof30k'
    assert nxt('b1-hof30k', 'eval', True, True) is None
    assert nxt('b9-spotcheck', 'eval', True, True) is None, 'a hand spec off the chain earns nothing'
    assert nxt('b1-stageb', 'eval', False, True) is None
    assert nxt('b1-stageb', 'eval', True, False) is None, 'auto_stage_b switches the whole chain'


def test_the_old_stage_b_marker_still_reads_as_stage_b_owed():
    """The box's ledger holds `stage_b: pending` on every training the daemon reaped before
    2026-09-04; the new marker is `next_pass`. Both have to mean what they meant."""
    assert runner_module.pending_pass({'stage_b': 'pending'}) == 'stageb'
    assert runner_module.pending_pass({'next_pass': 'hof30k'}) == 'hof30k'
    assert runner_module.pending_pass({'state': 'done'}) is None


def test_pass_ids_count_past_spent_waves_and_stop_at_a_queued_spec():
    mint = runner_module.mint_pass_id
    assert mint('b13', 'hof5000') == 'b13-hof5000'
    assert mint('b13', 'hof5000', used={'b13-hof5000'}) == 'b13-hof5000-w2'
    assert mint('b13', 'stageb', used={'b13-stageb', 'b13-stageb-w2'}) == 'b13-stageb-w3'
    assert mint('b13', 'stageb', blocked={'b13-stageb'}) is None
    assert mint('b13', 'stageb', blocked={'b13-stageb-w2'}, used={'b13-stageb'}) is None


def test_the_laptop_driver_spells_pass_ids_the_way_the_daemon_mints_them():
    """Two boxes, one naming: `tools/laptop_batch.py` cannot be imported by the daemon, so the two
    spellings are pinned equal here, where both are on the path."""
    from tools import laptop_batch
    for pass_name in runner_module.PASS_ORDER:
        assert laptop_batch.pass_label('b13', pass_name, 1) == runner_module.mint_pass_id('b13', pass_name)
        assert laptop_batch.pass_label('b13', pass_name, 3) == runner_module.mint_pass_id(
            'b13', pass_name, used={'b13-' + pass_name, 'b13-' + pass_name + '-w2'})
        assert runner_module.pass_of(laptop_batch.pass_label('b13', pass_name, 2)) == pass_name


def _runner(tmp_path, ledger):
    host = dict(HOST, LEDGER_PATH=str(tmp_path / 'ledger.json'))
    with open(host['LEDGER_PATH'], 'w') as handle:
        json.dump(ledger, handle)
    return runner_module.Runner(host)


def test_a_finished_stage_b_wave_synthesises_the_hof5000_pass_over_its_arms(tmp_path):
    ledger = {
        'b12-stageb': {'state': 'done', 'type': 'eval', 'policy': 'b12aa-x',
                       'policies': ['b12aa-x', 'b12ab-x'], 'next_pass': 'hof5000',
                       'env': {'SNEK_ZERO_OBS': '1'}},
    }
    jobs = _runner(tmp_path, ledger)._auto_jobs()
    assert [(job.id, job.policies, job.eval_args, job.priority, job.env) for job in jobs] == [
        ('b12-hof5000', ['b12aa-x', 'b12ab-x'], ['--pass', 'hof5000'], 11, {'SNEK_ZERO_OBS': '1'})]


def test_arms_already_covered_by_a_hand_queued_hof_pass_are_not_measured_again(tmp_path):
    """b11-hof5000 was a hand spec; its id carries the pass suffix, so the chain sees the arms as
    measured and the box does not spend 35 minutes redoing them."""
    ledger = {
        'b11-stageb': {'state': 'done', 'type': 'eval', 'policies': ['b11a-x', 'b11b-x'],
                       'next_pass': 'hof5000', 'env': {}},
        'b11-hof5000': {'state': 'done', 'type': 'eval', 'policies': ['b11a-x', 'b11b-x'],
                        'next_pass': 'hof30k', 'env': {}},
    }
    jobs = _runner(tmp_path, ledger)._auto_jobs()
    assert [(job.id, job.policies) for job in jobs] == [('b11-hof30k', ['b11a-x', 'b11b-x'])]


def test_a_legacy_marker_and_a_new_one_group_into_one_stage_b_wave(tmp_path):
    ledger = {
        'b16a-x': {'state': 'done', 'type': 'train', 'policy': 'b16a-x', 'policies': ['b16a-x'],
                   'stage_b': 'pending', 'env': {}},
        'b16b-x': {'state': 'done', 'type': 'train', 'policy': 'b16b-x', 'policies': ['b16b-x'],
                   'next_pass': 'stageb', 'env': {}},
    }
    jobs = _runner(tmp_path, ledger)._auto_jobs()
    assert [(job.id, job.policies, job.eval_args, job.priority) for job in jobs] == [
        ('b16-stageb', ['b16a-x', 'b16b-x'], [], 10)]


def test_a_second_wave_of_a_pass_gets_its_own_id(tmp_path):
    ledger = {
        'b15-stageb': {'state': 'done', 'type': 'eval', 'policies': ['b15a-x'], 'env': {}},
        'b15-stageb-w2': {'state': 'done', 'type': 'eval', 'policies': ['b15b-x'],
                          'next_pass': 'hof5000', 'env': {}},
        'b15-hof5000': {'state': 'done', 'type': 'eval', 'policies': ['b15a-x'], 'env': {}},
    }
    jobs = _runner(tmp_path, ledger)._auto_jobs()
    assert [(job.id, job.policies) for job in jobs] == [('b15-hof5000-w2', ['b15b-x'])]


# --- at_a_glance ---------------------------------------------------------------------------------

def test_a_wave_is_reported_in_arms_not_in_jobs():
    # One eval job is four arms; counting jobs would report a whole measurement as "1 arm", which is
    # the number a reader uses to check nothing was dropped.
    glance = runner_module.build_at_a_glance(
        [], [{'id': 'b1-stageb', 'type': 'eval', 'policy': 'b1a',
              'policies': ['b1a', 'b1b', 'b1c', 'b1d'], 'priority': 10}], {})
    assert glance['queued'] == ['b1 evals | queued (4 arms)']


def test_a_batchs_three_queued_passes_take_one_line_not_three():
    """The queue says what is owed, and a batch owed its stage B is owed its hof passes too; three
    rows per batch said it three times (2026-09-04). Arms are counted once, not once per pass."""
    arms = ['b16{0}-kl003-seed1'.format(letter) for letter in 'abcd']
    queued = [{'id': 'b16-' + pass_name, 'type': 'eval', 'policy': arms[0], 'policies': arms,
               'priority': priority}
              for pass_name, priority in (('stageb', 10), ('hof5000', 11), ('hof30k', 12))]
    queued.append({'id': 'b17a-clip005-seed1', 'type': 'train', 'policy': 'b17a-clip005-seed1',
                   'policies': ['b17a-clip005-seed1'], 'priority': 100})
    glance = runner_module.build_at_a_glance([], queued, {})
    assert glance['queued'] == ['b16 evals | kl003 | queued (4 arms)',
                                'b17 training | clip005 | queued (1 arm)']


def test_a_running_pass_is_named_because_only_one_runs_at_a_time():
    running = [{'id': 'b11-hof5000', 'type': 'eval', 'policy': 'b11ae-lr1e4-seed1',
                'policies': ['b11ae-lr1e4-seed1', 'b11af-lr1e4-seed2']}]
    glance = runner_module.build_at_a_glance(running, [], {})
    assert glance['running'] == ['b11 | lr1e4 | hof5000 (2 arms)']
    assert runner_module.phase_of('b11-hof30k-w2', 'eval') == 'hof30k'


def test_a_running_batch_shows_the_mean_percent_across_its_arms():
    running = [{'id': 'b1a-x', 'type': 'train', 'policy': 'b1a-x', 'policies': ['b1a-x'],
                'step': 1000000, 'max_steps': 3000000},
               {'id': 'b1b-x', 'type': 'train', 'policy': 'b1b-x', 'policies': ['b1b-x'],
                'step': 2000000, 'max_steps': 3000000}]
    glance = runner_module.build_at_a_glance(running, [], {'b1': 'the seed-matched set'})
    assert glance['running'] == ['b1 | x | training 50% (2 arms)']   # the arms' own knob token, not the batch label


def test_a_running_wave_is_captioned_by_its_own_arms_not_by_the_next_queued_spec():
    # 2026-09-03: the caption was one label per batch, first found, and the first found was the queued
    # spec's -- so wave 3 (lr5e4, lr8e4) read "lr1e3, seed 1 of 4 -- wave 4 of 4" for two hours.
    def arm(letter, cell, seed, wave, **extra):
        job = {'id': 'b11{0}-{1}-seed{2}'.format(letter, cell, seed), 'type': 'train',
               'policy': 'b11{0}-{1}-seed{2}'.format(letter, cell, seed), 'policies': [],
               'label': 'b11: {0}, seed {1} of 4 -- wave {2} of 4'.format(cell, seed, wave), 'priority': 112}
        job.update(extra)
        return job
    running = [arm('aq', 'lr5e4', 1, 3, step=50, max_steps=100), arm('ar', 'lr5e4', 2, 3, step=50, max_steps=100),
               arm('au', 'lr8e4', 1, 3, step=50, max_steps=100)]
    queued = [arm('ay', 'lr1e3', 1, 4), arm('az', 'lr1e3', 2, 4), arm('bc', 'lr2e3', 1, 4)]
    labels = {'b11': queued[0]['label']}          # what _batch_labels would have handed over
    glance = runner_module.build_at_a_glance(running, queued, labels)
    assert glance['running'] == ['b11 | lr5e4, lr8e4 -- wave 3 of 4 | training 50% (3 arms)']
    assert glance['queued'] == ['b11 training | lr1e3, lr2e3 -- wave 4 of 4 | queued (3 arms)']


def test_an_unlabelled_stage_b_is_captioned_by_the_cells_it_measures():
    stage_b = [{'id': 'b11-stageb-w2', 'type': 'eval', 'policy': 'b11ai-lr1.5e4-seed1', 'label': '',
                'policies': ['b11ai-lr1.5e4-seed1', 'b11aj-lr1.5e4-seed2', 'b11am-lr2.5e4-seed1'],
                'priority': 10}]
    glance = runner_module.build_at_a_glance(stage_b, [], {'b11': 'b11: lr1e3, seed 1 of 4 -- wave 4 of 4'})
    assert glance['running'] == ['b11 | lr1.5e4, lr2.5e4 | stage B (3 arms)']


def test_the_batch_label_is_only_a_fallback_when_the_jobs_say_nothing():
    jobs = [{'id': 'smoke-1', 'type': 'smoke', 'policy': 'smoke', 'policies': [], 'priority': 1}]
    glance = runner_module.build_at_a_glance(jobs, [], {'smoke': 'a smoke run'})
    assert glance['running'] == ['smoke | a smoke run | smoke (1 arm)']


def test_a_hold_notice_leads_the_queue_even_when_the_queue_is_empty():
    # An empty queue under a hold is the case most in need of the explanation.
    glance = runner_module.build_at_a_glance([], [], {}, held_by=['paused'])
    assert glance['queued'] and glance['queued'][0].startswith('** queue paused')
    assert '"paused": false' in glance['queued'][0]


def test_no_hold_means_no_notice():
    assert runner_module.hold_notice([]) is None


def test_the_hold_notice_is_ordered_by_the_flags_not_by_the_caller():
    # So the line is stable across polls rather than reordering with dict iteration.
    assert runner_module.hold_notice(['drain', 'paused']) == \
        runner_module.hold_notice(['paused', 'drain'])
    assert 'paused and draining' in runner_module.hold_notice(['drain', 'paused'])


def test_attention_lines_are_carried_through():
    glance = runner_module.build_at_a_glance([], [], {}, attention=['** something broke'])
    assert glance['attention'] == ['** something broke']


def test_attention_is_empty_in_the_normal_case():
    assert runner_module.build_at_a_glance([], [], {})['attention'] == []


# --- the chart window is the trainer's, not the daemon's -------------------------------------------
#
# `train.py` opens a window for its own arm, so one appears on the laptop too and nobody launches it
# by hand. The daemon owes it two things and no more: the session's display, and the ops-level switch.
# It deliberately does not manage the window — a daemon that opened its own would put a fifth window
# on a box running four arms, and would be the one holding a handle the trainer is supposed to be
# independent of.

VIEWER_HOST = dict(HOST, DISPLAY=':0', XAUTHORITY='/run/user/1000/gdm/Xauthority')


def test_a_job_inherits_the_sessions_display():
    """Without these a job cannot reach the monitor: the daemon runs outside the graphical session."""
    _, env, _, _ = launch.build_command(parse_job(spec()), VIEWER_HOST, runtime())
    assert env['DISPLAY'] == ':0'
    assert env['XAUTHORITY'] == '/run/user/1000/gdm/Xauthority'


def test_a_headless_host_forwards_nothing_and_is_not_an_error():
    _, env, _, _ = launch.build_command(parse_job(spec()), HOST, runtime())
    assert 'DISPLAY' not in env and 'XAUTHORITY' not in env


def test_the_viewer_knob_reaches_the_trainer_rather_than_the_daemon():
    _, env, _, _ = launch.build_command(parse_job(spec()), VIEWER_HOST, runtime(viewer=False))
    assert env['SNEK_CHART_WINDOW'] == '0'


def test_the_viewer_knob_on_leaves_the_trainers_default_alone():
    """On is the trainer's own default, so the daemon says nothing — one place decides, not two."""
    _, env, _, _ = launch.build_command(parse_job(spec()), VIEWER_HOST, runtime(viewer=True))
    assert 'SNEK_CHART_WINDOW' not in env


def test_a_spec_may_still_ask_for_no_window():
    job = parse_job(spec(env={'SNEK_CHART_WINDOW': '0'}))
    _, env, _, _ = launch.build_command(job, VIEWER_HOST, runtime(viewer=True))
    assert env['SNEK_CHART_WINDOW'] == '0'


def test_a_benchmark_opens_no_window():
    """It measures the training loop, and a window would be measured along with it."""
    _, env, _, _ = launch.build_command(parse_job(spec(type='benchmark')), VIEWER_HOST, runtime())
    assert env['SNEK_CHART_WINDOW'] == '0'


def test_the_daemon_does_not_manage_a_window():
    """Pinning the absence, because re-adding it is the tempting mistake.

    Two windows for one arm, and a daemon holding a handle to a process the trainer is supposed to be
    independent of.
    """
    assert not hasattr(runner_module.Runner, '_ensure_viewer')
    assert not hasattr(launch, 'spawn_viewer')


def test_viewer_is_a_bool_knob():
    parsed, notes = config_module.parse_runtime_config('{"viewer": 1}', HOST)
    assert parsed is None and any('viewer' in note for note in notes)
    parsed, notes = config_module.parse_runtime_config('{"viewer": false}', HOST)
    assert parsed is not None and parsed['viewer'] is False


# --- job text is ASCII, because status.json is read raw ---------------------------------------

def test_an_em_dash_in_a_label_is_folded_to_ascii():
    """`status.json` is published with json.dumps, so a non-ASCII character reaches the reader as a
    `\\uXXXX` escape. b8's labels published as "b8 \\u2014 b8: kl02, seed 1 of 4 \\u2014 wave 2 of 2".
    """
    parsed = parse_job(spec(label='b8 — kl02, seed 1 of 4 — wave 2 of 2'))
    assert parsed.label == 'b8 -- kl02, seed 1 of 4 -- wave 2 of 2'
    assert parsed.label.isascii()


def test_the_punctuation_an_agent_writes_has_an_ascii_spelling():
    text = '“quoted” ≥ 98% × 4 → done… – and ± 0.1'
    assert job_module.to_ascii(text) == '"quoted" >= 98% x 4 -> done... - and +/- 0.1'


def test_an_unmapped_character_becomes_a_visible_question_mark():
    # Not dropped: a silent deletion of a character nobody mapped is worse than an obvious `?`.
    assert job_module.to_ascii('snowman ☃') == 'snowman ?'


def test_notes_are_folded_too_and_plain_ascii_is_untouched():
    plain = 'Batch b8 -- what fixes b4 collapse. 8 epochs, >=98%/500.'
    assert job_module.to_ascii(plain) == plain
    assert parse_job(spec(notes='a — b')).notes == 'a -- b'


def test_folding_an_empty_or_missing_field_is_not_an_error():
    assert job_module.to_ascii('') == ''
    parsed = parse_job(spec())
    assert parsed.label == '' and parsed.notes == ''


def test_the_published_status_text_carries_no_ascii_escapes():
    """The other half: a policy name is a path and is deliberately not folded, so the writer must
    not escape it either. Asserted through `status_json`, which is what `_publish` calls."""
    text = runner_module.status_json({'at_a_glance': {'running': ['b8 \u2014 kl02']},
                                      'running': [{'policy': 'b8i-kl02-seed1'}]})
    assert '\\u2014' not in text
    assert '\u2014' in text, 'the character itself survives; only its escaping is the bug'
    assert json.loads(text)['at_a_glance']['running'] == ['b8 \u2014 kl02'], 'still valid JSON'


def test_at_a_glance_folds_a_label_that_came_from_the_ledger():
    """A label reaches the display from a queued spec, a running job, or the **ledger** -- and the
    ledger persists what was written before any fold existed. b8's stored em dashes were still on
    display after parse_job started folding, which is why this is done in both places.
    """
    running = [{'id': 'b8i-kl02-seed1', 'type': 'train', 'policy': 'b8i-kl02-seed1',
                'label': 'b8: kl02, seed 1 of 4 — wave 2 of 2', 'step': 19, 'max_steps': 100}]
    glance = runner_module.build_at_a_glance(running, [], {})
    line = glance['running'][0]
    assert line.isascii(), line
    assert '--' in line and '—' not in line
    # and the batch-label fallback folds too, for a job that carries no label of its own
    fallback = runner_module.build_at_a_glance(
        [{'id': 'smoke-1', 'type': 'smoke', 'policy': 'smoke', 'policies': []}], [],
        {'smoke': 'a run — with a dash'})['running'][0]
    assert fallback.isascii() and '--' in fallback, fallback


def test_at_a_glance_uses_no_em_dash_of_its_own():
    # The builder's own separators were em dashes, so an all-ASCII label still published two of them.
    glance = runner_module.build_at_a_glance(
        [{'id': 'b9a-thing', 'type': 'train', 'policy': 'b9a-thing', 'step': 1, 'max_steps': 2}],
        [{'id': 'b9b-thing', 'type': 'train', 'policy': 'b9b-thing'}],
        {'b9': 'plain ascii label'})
    for line in glance['running'] + glance['queued']:
        assert line.isascii(), line


# --- the box's run directory ----------------------------------------------------------------------
#
# 2026-09-03: every job writes under `<SNEK_DIR>/desktop/runs`, gitignored, so the box's checkout holds
# nothing under a path master tracks and the deploy's fast-forward cannot collide with a committed chart.

def test_every_job_is_pointed_at_the_desktop_runs_dir():
    for job in (parse_job(spec()), parse_job(spec(type='eval', policies=['b7a-x', 'b7b-y']))):
        _, env, _, _ = launch.build_command(job, HOST, runtime())
        assert env['SNEK_RUNS_DIR'] == '/repo/snek3/desktop/runs'
    assert launch.runs_dir(HOST) == '/repo/snek3/desktop/runs'


def test_throughput_is_read_from_the_desktop_runs_dir(tmp_path):
    host = dict(HOST, SNEK_DIR=str(tmp_path))
    runs = launch.runs_dir(host)
    os.makedirs(runs)
    with open(os.path.join(runs, 'b7a-x_evals.json'), 'w') as handle:
        json.dump({'summary': {'step': 4096}}, handle)
    running = launch.RunningJob(parse_job(spec()), 'b7a-x', 1, str(tmp_path / 'log'))
    launch.update_throughput(running, host)
    assert running.current_step == 4096


def test_a_whole_queued_batch_reads_as_a_wave_range_and_a_capped_cell_list():
    # b12's ten cells over five waves printed "wave 1 of 5, wave 2 of..." past the 80-char cut on 2026-09-03.
    def arm(i, cell, wave):
        return {'id': 'b12{0}-{1}-seed1'.format(chr(97 + i), cell), 'type': 'train', 'policy': 'b12x', 'policies': [],
                'label': 'b12: {0}, seed 1 of 4 -- wave {1} of 5'.format(cell, wave), 'priority': 120}
    cells = ['ep1', 'ep2', 'ep3', 'ep5', 'ep6', 'ep7', 'ep8', 'ep10', 'ep12', 'ep16']
    queued = [arm(i, c, i // 2 + 1) for i, c in enumerate(cells)]
    assert runner_module.describe_jobs(queued) == \
        'ep1, ep2, ep3, ep5, ep6, ep7, ep8, ep10, +2 more -- waves 1-5 of 5'
    # a gap in the waves is shown, not papered over
    assert runner_module.compress_waves(['wave 1 of 4', 'wave 3 of 4', 'wave 4 of 4']) == \
        ['wave 1 of 4', 'waves 3-4 of 4']
    assert runner_module.compress_waves(['wave 2 of 2', 'final pass']) == ['wave 2 of 2', 'final pass']
