"""The desktop daemon: the `project` guard, the config tiers, the specs it hands the scheduler, and
what it publishes.

`desktop/` imports nothing from this project — it shells out — so it goes on `sys.path` separately
here. It also runs on **base** python on the box, before the conda env exists, which is why it is
stdlib-only and why these tests must not reach for numpy or torch either (the two tests that import
`tools.scheduler` and `tools.closeout` are the deliberate exception, and say why).

Since 2026-09-05 the daemon schedules nothing itself: it mirrors `ops` into a local queue for
`tools/scheduler.py`, starts that once, and publishes what it finishes (`plans/scheduler.md`).

Three things here are silent when broken and each costs a night:

- **A stale snek2 spec dispatching.** `ops` still holds ~150 of them in `queue/pending/`. They are
  valid JSON with plausible ids.
- **A local-only results commit reported as `done`.** Indistinguishable from a pass that found
  nothing, and it once hid a 98.2%/500 checkpoint for hours.
- **A scheduler started twice.** Two schedulers over one queue is two windows and sixteen trainers.
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


# --- ops specs become the scheduler's queue ---------------------------------------------------------
#
# The daemon builds no command for a trainer or a close-out any more: `tools/scheduler.py` does, on both
# boxes. What the daemon owes the scheduler is a spec in the shape it reads, with the type defaults the
# old `build_command` applied folded in, so the scheduler needs no notion of a smoke or a benchmark.

def runtime(**overrides):
    config = dict(config_module.RUNTIME_DEFAULTS)
    config.update(overrides)
    return config


def test_a_training_becomes_a_train_spec_with_its_cap():
    out = launch.materialise(parse_job(spec(max_steps=3000000, label='b1: thing', env={'A': '1'})))
    assert out['type'] == 'train' and out['policy'] == 'b1a-thing' and out['max_steps'] == 3000000
    assert out['env'] == {'A': '1'} and out['label'] == 'b1: thing' and out['id'] == 'b1a-thing'


def test_a_training_with_no_cap_is_refused_rather_than_run_forever():
    with pytest.raises(ValueError):
        launch.materialise(parse_job(spec()))


def test_an_action_materialises_to_nothing():
    assert launch.materialise(parse_job(json.dumps({'project': 'snek3', 'id': 'deploy-1', 'type': 'deploy'}))) is None


def test_an_eval_keeps_its_arms_selector_depth_and_extra_args_in_order():
    job = parse_job(spec(type='eval', policies=['b1a', 'b1b'], policy=None, selector='screen:98',
                         episodes=1000, eval_args=['--pass', 'hof5000']))
    out = launch.materialise(job)
    assert out['type'] == 'eval' and out['policies'] == ['b1a', 'b1b']
    assert out['selector'] == 'screen:98' and out['episodes'] == 1000 and out['eval_args'] == ['--pass', 'hof5000']
    assert 'eval_shards' not in out, 'the pool is the runtime knob unless the spec names one'


def test_a_policy_name_is_data_not_shell_syntax():
    """The scheduler passes each policy as one argv element; no shell is involved anywhere."""
    from tools import scheduler
    out = launch.materialise(parse_job(spec(type='eval', policies=['b1a; rm -rf /', 'b1b'], policy=None)))
    argv = scheduler.eval_argv(out, '/py', 12)
    assert argv[4] == 'b1a; rm -rf /' and '-c' not in argv
    assert parsed(argv).policies == ['b1a; rm -rf /', 'b1b']


def test_a_smoke_run_can_write_checkpoints_and_reach_an_eval():
    out = launch.materialise(parse_job(spec(type='smoke', policy=None, id='smoke-1')))
    assert out['policy'] == 'smoke' and out['max_steps'] == launch.SMOKE_STEPS
    assert out['env']['SNEK_MIN_CHECKPOINT_SCORE'] == '0' and out['env']['SNEK_EVAL_INTERVAL'] == '500'


def test_a_benchmark_turns_stage_a_off_rather_than_shrinking_it():
    out = launch.materialise(parse_job(spec(type='benchmark', policy=None, id='bench-7')))
    assert out['policy'] == 'bench-bench-7' and out['max_steps'] == launch.BENCHMARK_STEPS
    assert out['env']['SNEK_EVAL_QUEUE'] == '0' and out['env']['SNEK_EVAL_INTERVAL'] == str(launch.BENCHMARK_STEPS)


def test_a_specs_own_env_wins_over_the_type_default():
    out = launch.materialise(parse_job(spec(type='smoke', env={'SNEK_EVAL_INTERVAL': '50'})))
    assert out['env']['SNEK_EVAL_INTERVAL'] == '50'


# --- the eval command must parse, asked of the real parser -------------------------------------------
#
# The one place these tests deliberately cross the stdlib-only line. Before 2026-08-30 the daemon spelled
# the close-out's command line by hand and got it wrong twice, so every wave the box dispatched exited 2.
# The spelling now lives once, in `tools.scheduler.eval_argv`; this checks the spec the daemon hands it
# still produces a command the close-out's parser accepts.

def parsed(argv):
    from tools import closeout
    return closeout.build_parser().parse_args(argv[4:])


@pytest.mark.parametrize('overrides', [
    {},
    {'selector': 'screen:98'},
    {'episodes': 1000},
    {'selector': 'one', 'episodes': 3000},
    {'eval_args': ['--label', 'ab']},
    {'eval_args': ['--pass', 'hof5000']},
    {'eval_args': ['--pass', 'hof30k']},
])
def test_every_eval_spec_the_daemon_writes_parses(overrides):
    from tools import scheduler
    job = parse_job(spec(type='eval', policies=['b1a', 'b1b'], policy=None, **overrides))
    argv = scheduler.eval_argv(launch.materialise(job), '/py', 12)
    args = parsed(argv)
    assert args.policies == ['b1a', 'b1b'] and args.shards == 12


# --- the scheduler's command and environment ----------------------------------------------------------

def test_the_runtime_knobs_reach_the_scheduler_as_flags():
    argv = launch.scheduler_command(HOST, runtime(max_trainers=6, eval_shards=16))
    assert argv[:4] == ['/py', '-u', '-m', 'tools.scheduler']
    assert argv[argv.index('--queue') + 1] == '/repo/snek3/desktop/queue-local'
    assert argv[argv.index('--wave') + 1] == '6' and argv[argv.index('--max-trainers') + 1] == '6'
    assert argv[argv.index('--shards') + 1] == '16'
    assert '--no-status' in argv, 'the daemon publishes; the scheduler must not push laptop-status'
    assert '--no-stage-b' not in argv
    assert '--no-stage-b' in launch.scheduler_command(HOST, runtime(auto_stage_b=False))


def test_the_scheduler_parser_accepts_what_the_daemon_spells():
    from tools import scheduler
    argv = launch.scheduler_command(HOST, runtime())
    args = scheduler.build_parser().parse_args(argv[4:])
    assert args.queue == '/repo/snek3/desktop/queue-local' and args.no_status


def test_every_job_is_pointed_at_the_desktop_runs_dir():
    env = launch.scheduler_env(HOST, runtime())
    assert env['SNEK_RUNS_DIR'] == '/repo/snek3/desktop/runs' and env['PYTHONPATH'] == '.'
    assert launch.runs_dir(HOST) == '/repo/snek3/desktop/runs'


def test_the_thread_knobs_reach_the_scheduler():
    env = launch.scheduler_env(HOST, runtime(torch_threads=2, omp_num_threads=3))
    assert env['SNEK_TORCH_THREADS'] == '2' and env['OMP_NUM_THREADS'] == '3'
    assert 'SNEK_TORCH_THREADS' not in launch.scheduler_env(HOST, runtime(torch_threads=0))


# --- the chart window is the scheduler's -----------------------------------------------------------
#
# The daemon runs outside the graphical session, so it owes the scheduler the session's display and the
# ops-level switch, and nothing else: the scheduler opens the one window and closes it (`tools/window.py`).

VIEWER_HOST = dict(HOST, DISPLAY=':0', XAUTHORITY='/run/user/1000/gdm/Xauthority')


def test_the_scheduler_inherits_the_sessions_display():
    env = launch.scheduler_env(VIEWER_HOST, runtime())
    assert env['DISPLAY'] == ':0' and env['XAUTHORITY'] == '/run/user/1000/gdm/Xauthority'


def test_a_headless_host_forwards_nothing_and_is_not_an_error():
    env = launch.scheduler_env(HOST, runtime())
    assert 'DISPLAY' not in env and 'XAUTHORITY' not in env


def test_the_viewer_knob_reaches_the_scheduler_only_when_off():
    assert launch.scheduler_env(VIEWER_HOST, runtime(viewer=False))['SNEK_CHART_WINDOW'] == '0'
    assert 'SNEK_CHART_WINDOW' not in launch.scheduler_env(VIEWER_HOST, runtime(viewer=True))


def test_the_daemon_does_not_manage_a_window():
    """Pinning the absence, because re-adding it is the tempting mistake."""
    assert not hasattr(runner_module.Runner, '_ensure_viewer')
    assert not hasattr(launch, 'spawn_viewer')
    assert not hasattr(launch, 'build_command'), 'the daemon builds no trainer or close-out command'


def test_viewer_is_a_bool_knob():
    parsed, notes = config_module.parse_runtime_config('{"viewer": 1}', HOST)
    assert parsed is None and any('viewer' in note for note in notes)
    parsed, notes = config_module.parse_runtime_config('{"viewer": false}', HOST)
    assert parsed is not None and parsed['viewer'] is False


# --- batches and phases --------------------------------------------------------------------------------

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


# --- at_a_glance --------------------------------------------------------------------------------------
#
# The scheduler builds its own lines with `tools.laptop_status`; these pin the daemon's copy, which
# it uses for the queue when no scheduler is running, and every reader of status.json depends on.

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


def test_passes_owed_after_a_queued_training_are_shown_after_it():
    """b15, 2026-09-04: wave 1's hand-queued hof passes run now, the other four waves' passes run
    after each wave trains. One `evals` line for the batch put all 40 arms above the training and
    read as measuring arms that had not trained. Two lines, split at the training, is the truth."""
    def arm(letter):
        return 'b15{0}-ent-seed1'.format(letter)
    wave1 = [arm(l) for l in 'ab']
    wave2 = [arm(l) for l in 'cd']
    order = [{'id': 'b15-hof5000', 'type': 'eval', 'policies': wave1, 'priority': 11},
             {'id': 'b15-hof30k', 'type': 'eval', 'policies': wave1, 'priority': 12}]
    order += [{'id': p, 'type': 'train', 'policy': p, 'policies': [p], 'priority': 100} for p in wave2]
    order += [{'id': 'b15-{0}-w2'.format(name), 'type': 'eval', 'policies': wave2, 'priority': pr}
              for name, pr in (('stageb', 10), ('hof5000', 11), ('hof30k', 12))]
    order += [{'id': 'b16a-kl-seed1', 'type': 'train', 'policy': 'b16a-kl-seed1',
               'policies': ['b16a-kl-seed1'], 'priority': 100},
              {'id': 'b16-stageb', 'type': 'eval', 'policies': ['b16a-kl-seed1'], 'priority': 10}]
    glance = runner_module.build_at_a_glance([], order, {})
    assert glance['queued'] == ['b15 evals | ent | queued (2 arms)',
                                'b15 training | ent | queued (2 arms)',
                                'b15 evals | ent | queued (2 arms)',
                                'b16 training | kl | queued (1 arm)',
                                'b16 evals | kl | queued (1 arm)']


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


# --- the laptop's lines, folded in from laptop-status -----------------------------------------------

def test_the_laptops_status_is_folded_in_with_its_own_timestamp():
    """`tools/laptop_status.py` publishes the same at_a_glance shape to `laptop-status`; the box adds it
    as laptop_running / laptop_queued, and `laptop_iso` is the laptop's own clock -- the driver's last
    publish is empty, so lines under an old `laptop_iso` mean a dead driver, not a slow one."""
    text = json.dumps({'iso': '2026-09-04T23:14:09', 'box': 'laptop',
                       'at_a_glance': {'running': ['b16 | kl003, kl005 -- wave 1 of 5 | training 12% (8 arms)'],
                                       'queued': ['b16 evals | kl003, kl005 | queued (8 arms)',
                                                  'b19 training | noadvnorm, mse | queued (24 arms)'],
                                       'attention': []}})
    glance = runner_module.with_laptop({'running': ['b15 | ent003 | stage B (8 arms)'], 'queued': [],
                                        'attention': []}, text)
    assert glance['running'] == ['b15 | ent003 | stage B (8 arms)']          # the box's own, untouched
    assert glance['laptop_running'] == ['b16 | kl003, kl005 -- wave 1 of 5 | training 12% (8 arms)']
    assert glance['laptop_queued'] == ['b16 evals | kl003, kl005 | queued (8 arms)',
                                       'b19 training | noadvnorm, mse | queued (24 arms)']
    assert glance['laptop_iso'] == '2026-09-04T23:14:09'


def test_no_laptop_status_yet_gives_empty_lines_and_a_null_timestamp_never_an_error():
    for text in ('', 'not json', '[]', '{"at_a_glance": null}'):
        glance = runner_module.with_laptop({'running': [], 'queued': [], 'attention': []}, text)
        assert glance['laptop_running'] == [] and glance['laptop_queued'] == []
        assert glance['laptop_iso'] is None


def test_the_laptop_status_branch_is_fetched_apart_from_the_three_the_box_needs(monkeypatch):
    """One `git fetch a b c laptop-status` fails whole when laptop-status does not exist yet, which
    would have stopped the box reading `ops` until the laptop first published. So it is its own fetch,
    and the branch name defaults rather than being a required host.env key."""
    from runner import gitbus
    fetched = []
    monkeypatch.setattr(gitbus, '_git', lambda args, cwd, check=False: fetched.append(args) or '')
    host = {'GIT_REMOTE': 'origin', 'OPS_BRANCH': 'ops', 'STATUS_BRANCH': 'ops-status',
            'RESULTS_BRANCH': 'results', 'REPO_PATH': '/r'}
    gitbus.fetch(host)
    gitbus.fetch_laptop_status(host)
    assert fetched == [['fetch', 'origin', 'ops', 'ops-status', 'results'],
                       ['fetch', 'origin', 'laptop-status']]
    assert gitbus.laptop_status_branch(dict(host, LAPTOP_STATUS_BRANCH='lap')) == 'lap'


# --- actions: deploy and restart over the bus -------------------------------------------------------



# --- the daemon over the scheduler ------------------------------------------------------------------
#
# What is pinned: the specs on `ops` land in the local queue in the scheduler's shape and leave it when
# dequeued; the scheduler is started once per change of the queue and adopted, never doubled; a pause is
# a marker the scheduler sees; a job the scheduler finishes is published to `results`; status.json says
# what the scheduler says, with the daemon's own attention lines and a derived `ledger` for the tools.

class FakeScheduler(object):
    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        return self.returncode


class Spawns(object):
    def __init__(self):
        self.calls, self.pid = [], 9000

    def __call__(self, host, runtime):
        self.pid += 1
        self.calls.append(dict(runtime))
        return FakeScheduler(self.pid), '/var/snek/logs/scheduler-x.log'


class Commands(object):
    """Stands in for `subprocess.run` inside the daemon: heads in sequence, a diff, a deploy result."""

    def __init__(self, heads=('aaaaaaaaaaaa', 'bbbbbbbbbbbb'), changed='snek3/desktop/runner/runner.py',
                 deploy_rc=0, deploy_out='HEAD aaaaaaaaaaaa -> bbbbbbbbbbbb; kept 0 pictures\n'):
        self.heads, self.changed, self.deploy_rc, self.deploy_out = list(heads), changed, deploy_rc, deploy_out
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        result = type('R', (), {})()
        if argv[:3] == ['git', 'rev-parse', 'HEAD']:
            result.returncode, result.stdout = 0, (self.heads.pop(0) if len(self.heads) > 1 else self.heads[0]) + '\n'
        elif argv[:3] == ['git', 'diff', '--name-only']:
            result.returncode, result.stdout = 0, self.changed
        elif argv[-2:] == ['-m', 'runner.deploy']:
            result.returncode, result.stdout = self.deploy_rc, self.deploy_out
        else:
            raise AssertionError('unexpected command {0}'.format(argv))
        return result


class Bus(object):
    """The git bus as the daemon sees it: specs on ops, a runtime, what was published where."""

    def __init__(self, monkeypatch, specs, runtime_text='{}', laptop=''):
        self.specs, self.runtime_text, self.laptop = list(specs), runtime_text, laptop
        self.status, self.results = [], []
        g = runner_module.gitbus
        monkeypatch.setattr(g, 'fetch', lambda host: None)
        monkeypatch.setattr(g, 'fetch_laptop_status', lambda host: None)
        monkeypatch.setattr(g, 'push_unpushed', lambda host: [])
        monkeypatch.setattr(g, 'read_runtime_text', lambda host: self.runtime_text)
        monkeypatch.setattr(g, 'read_laptop_status', lambda host: self.laptop)
        monkeypatch.setattr(g, 'publish_status', lambda host, text: self.status.append(json.loads(text)) or True)
        monkeypatch.setattr(g, 'publish_results',
                            lambda host, job, paths: self.results.append((job.id, sorted(paths))) or True)
        monkeypatch.setattr(g, 'read_pending_jobs',
                            lambda host: [(s['id'] + '.json', json.dumps(s)) for s in self.specs])


def _box(tmp_path, monkeypatch, specs, ledger=None, runtime_text='{}', laptop=''):
    bus = Bus(monkeypatch, specs, runtime_text, laptop)
    host = dict(HOST, SNEK_DIR=str(tmp_path / 'snek3'), REPO_PATH=str(tmp_path),
                LEDGER_PATH=str(tmp_path / 'var' / 'ledger.json'), LOG_DIR=str(tmp_path / 'var' / 'logs'))
    os.makedirs(os.path.dirname(host['LEDGER_PATH']))
    with open(host['LEDGER_PATH'], 'w') as handle:
        json.dump(ledger or {}, handle)
    monkeypatch.setattr(runner_module, '_disk_free_gb', lambda path: 500.0)
    runner = runner_module.Runner(host)
    runner.run_command = Commands()
    runner.spawn = Spawns()
    return runner, bus


def _train_spec(policy, **extra):
    body = {'project': 'snek3', 'id': policy, 'type': 'train', 'policy': policy, 'max_steps': 10,
            'label': ''}
    body.update(extra)
    return body


def _queue_files(runner):
    root = launch.queue_dir(runner.host)
    return sorted(os.path.relpath(os.path.join(folder, name), root)
                  for folder, _, names in os.walk(root) for name in names)


def _scheduler_status(runner, running=(), queued_ids=(), attention=(), glance=None):
    """What the scheduler writes to runs/.live/.status.json while it runs."""
    path = os.path.join(runner.runs_dir(), runner_module.STATUS_RELATIVE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {'iso': '2026-09-05T12:00:00', 'pid': runner.state.get('scheduler_pid'),
               'running': list(running), 'queued_ids': list(queued_ids),
               'at_a_glance': glance or {'running': ['b1 | x | training 50% (2 arms)'],
                                         'queued': [], 'attention': list(attention)}}
    with open(path, 'w') as handle:
        json.dump(payload, handle)


def test_the_ops_queue_is_mirrored_into_the_schedulers_queue_by_batch(tmp_path, monkeypatch):
    runner, bus = _box(tmp_path, monkeypatch, [
        _train_spec('b1a-x'), _train_spec('b1b-x'), _train_spec('b2a-y'),
        {'project': 'snek3', 'id': 'b1-hof5000', 'type': 'eval', 'policies': ['b1a-x', 'b1b-x'],
         'eval_args': ['--pass', 'hof5000']}])
    runner.poll_once(git=True)
    assert _queue_files(runner) == ['b1/b1-hof5000.json', 'b1/b1a-x.json', 'b1/b1b-x.json', 'b2/b2a-y.json']
    with open(os.path.join(launch.queue_dir(runner.host), 'b1', 'b1a-x.json')) as handle:
        assert json.load(handle) == launch.materialise(parse_job(json.dumps(_train_spec('b1a-x'))))
    # dequeued on ops: gone here, and the scheduler's own markers are left alone
    open(os.path.join(launch.queue_dir(runner.host), 'b1', '.done-b1a-x'), 'w').close()
    bus.specs = [s for s in bus.specs if s['id'] != 'b2a-y']
    runner.poll_once(git=True)
    assert _queue_files(runner) == ['b1/.done-b1a-x', 'b1/b1-hof5000.json', 'b1/b1a-x.json', 'b1/b1b-x.json']


def test_a_batch_with_every_job_published_is_not_mirrored_but_a_batch_with_one_left_is_mirrored_whole(tmp_path, monkeypatch):
    """`ops` still carries every finished spec, and a finished batch whose files are not in the
    box's runs directory would be retrained. A half-done batch keeps its done arms so wave numbering
    -- and the pass ids already on `results` -- stay the same."""
    ledger = {job_id: {'state': 'done', 'type': 'train', 'finished': 1.0}
              for job_id in ('b7a-x', 'b7b-x', 'b15a-e', 'b15b-e')}
    ledger['b7-stageb'] = {'state': 'done', 'type': 'eval', 'finished': 1.0}
    runner, bus = _box(tmp_path, monkeypatch, [_train_spec('b7a-x'), _train_spec('b7b-x'),
                                               _train_spec('b15a-e'), _train_spec('b15b-e'), _train_spec('b15c-e')],
                       ledger=ledger)
    runner.poll_once(git=True)
    assert _queue_files(runner) == ['b15/b15a-e.json', 'b15/b15b-e.json', 'b15/b15c-e.json']
    assert bus.status[-1]['ledger']['b7a-x'] == 'done' and bus.status[-1]['ledger']['b7-stageb'] == 'done'


def test_a_malformed_spec_is_recorded_once_and_the_rest_of_the_queue_goes_on(tmp_path, monkeypatch):
    runner, bus = _box(tmp_path, monkeypatch, [
        {'project': 'snek2', 'id': 'b46a-old', 'type': 'train', 'policy': 'b46a-old'},   # the stale snek2 spec
        _train_spec('b1a-x', max_steps=None),                                            # no cap
        _train_spec('b1b-x')])
    runner.poll_once(git=True)
    assert _queue_files(runner) == ['b1/b1b-x.json']
    assert runner.ledger['b46a-old']['state'] == 'failed' and runner.ledger['b1a-x']['state'] == 'failed'
    assert runner.ledger['b1a-x']['error'].startswith('b1a-x: a train spec needs max_steps')
    lines = bus.status[-1]['at_a_glance']['attention']
    assert any('b46a-old is malformed' in line for line in lines)
    assert bus.status[-1]['ledger']['b1a-x'] == 'failed'


def test_the_scheduler_is_started_once_per_queue_change_and_adopted_meanwhile(tmp_path, monkeypatch):
    runner, bus = _box(tmp_path, monkeypatch, [_train_spec('b1a-x')])
    runner.poll_once(git=True)
    assert len(runner.spawn.calls) == 1 and runner.state['scheduler_pid'] == 9001
    assert bus.status[-1]['scheduler']['alive'] is True and bus.status[-1]['scheduler']['pid'] == 9001
    runner.poll_once(git=True)
    runner.poll_once(git=False)
    assert len(runner.spawn.calls) == 1, 'alive: never a second one'
    runner.scheduler.returncode = 0             # it finished the queue and exited
    runner.poll_once(git=True)
    assert len(runner.spawn.calls) == 1, 'same queue, nothing new to do: not restarted'
    assert bus.status[-1]['scheduler']['alive'] is False and bus.status[-1]['scheduler']['last_exit'] == 0
    bus.specs.append(_train_spec('b2a-y'))
    runner.poll_once(git=True)
    assert len(runner.spawn.calls) == 2 and runner.state['scheduler_pid'] == 9002, 'a new spec starts it'


def test_a_trigger_starts_the_scheduler_even_when_the_queue_did_not_change(tmp_path, monkeypatch):
    runner, _ = _box(tmp_path, monkeypatch, [_train_spec('b1a-x')])
    runner.poll_once(git=True)
    runner.scheduler.returncode = 1              # crashed
    runner.poll_once(git=True)
    assert len(runner.spawn.calls) == 1, 'a failed exit is not retried on its own inside the backoff'
    runner.poll_once(git=True, forced=True)
    assert len(runner.spawn.calls) == 2


def test_a_daemon_restart_adopts_the_scheduler_by_pid_and_a_reboot_forgets_it(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, 'boot_id', lambda: 'boot-1')
    runner, bus = _box(tmp_path, monkeypatch, [_train_spec('b1a-x')])
    runner.poll_once(git=True)
    alive = {9001}
    monkeypatch.setattr(launch, 'pid_alive', lambda pid: pid in alive)
    again = runner_module.Runner(runner.host)
    again.spawn = Spawns()
    again.poll_once(git=True)
    assert again.spawn.calls == [] and again.state['scheduler_pid'] == 9001
    monkeypatch.setattr(runner_module, 'boot_id', lambda: 'a-different-boot')
    rebooted = runner_module.Runner(runner.host)
    rebooted.spawn = Spawns()
    assert rebooted.state['scheduler_pid'] is None
    rebooted.poll_once(git=True)
    assert len(rebooted.spawn.calls) == 1, 'after a reboot the queue is resumed by a fresh scheduler'


def test_a_pause_is_a_marker_the_scheduler_reads_and_no_scheduler_is_started_under_it(tmp_path, monkeypatch):
    runner, bus = _box(tmp_path, monkeypatch, [_train_spec('b1a-x')], runtime_text='{"paused": true}')
    runner.poll_once(git=True)
    hold = os.path.join(runner.runs_dir(), runner_module.HOLD_RELATIVE)
    assert os.path.exists(hold) and runner.spawn.calls == []
    assert bus.status[-1]['at_a_glance']['queued'][0].startswith('** queue paused')
    assert bus.status[-1]['at_a_glance']['queued'][1:] == ['b1 training | x | queued (1 arm)']
    assert bus.status[-1]['ledger'] == {'b1a-x': 'queued'}
    bus.runtime_text = '{}'
    runner.poll_once(git=True)
    assert not os.path.exists(hold) and len(runner.spawn.calls) == 1, 'lifting the hold starts it'


def test_the_runtime_knobs_at_spawn_are_the_ones_the_scheduler_gets(tmp_path, monkeypatch):
    runner, _ = _box(tmp_path, monkeypatch, [_train_spec('b1a-x')],
                     runtime_text='{"max_trainers": 6, "eval_shards": 10}')
    runner.poll_once(git=True)
    assert runner.spawn.calls[0]['max_trainers'] == 6 and runner.spawn.calls[0]['eval_shards'] == 10


def test_a_job_the_scheduler_finished_is_published_with_its_arms_files(tmp_path, monkeypatch):
    runner, bus = _box(tmp_path, monkeypatch, [_train_spec('b1a-x'), _train_spec('b1b-x')])
    runner.poll_once(git=True)
    runs = runner.runs_dir()
    os.makedirs(runs)
    for name in ('b1a-x.png', 'b1a-x.md', 'b1a-x_evals.json', 'b1a-x_checkpoint_evals.json',
                 'b1b-x.png', 'b1ab-other.png'):
        open(os.path.join(runs, name), 'w').close()
    _scheduler_status(runner, running=[{'id': 'b1a-x', 'type': 'train', 'policy': 'b1a-x', 'policies': ['b1a-x']},
                                       {'id': 'b1b-x', 'type': 'train', 'policy': 'b1b-x', 'policies': ['b1b-x']}])
    runner.poll_once(git=False)
    assert bus.results == [] and set(runner.state['running']) == {'b1a-x', 'b1b-x'}
    _scheduler_status(runner, running=[{'id': 'b1b-x', 'type': 'train', 'policy': 'b1b-x', 'policies': ['b1b-x']},
                                       {'id': 'b1-stageb', 'type': 'eval', 'policies': ['b1a-x']}],
                      queued_ids=['b1-hof5000'])
    runner.poll_once(git=True)
    assert [job_id for job_id, _ in bus.results] == ['b1a-x']
    assert [os.path.basename(path) for path in bus.results[0][1]] == \
        ['b1a-x.md', 'b1a-x.png', 'b1a-x_checkpoint_evals.json', 'b1a-x_evals.json']
    assert 'b1a-x' in runner.state['published']
    assert bus.status[-1]['ledger'] == {'b1a-x': 'done', 'b1b-x': 'running', 'b1-stageb': 'running',
                                        'b1-hof5000': 'queued'}
    assert bus.status[-1]['running'][1]['id'] == 'b1-stageb'
    # a pass that finishes publishes every arm's files under its own id, as the old daemon did
    _scheduler_status(runner, running=[], queued_ids=[])
    runner.poll_once(git=False)
    assert [job_id for job_id, _ in bus.results] == ['b1a-x', 'b1b-x', 'b1-stageb']
    assert [os.path.basename(p) for p in dict(bus.results)['b1-stageb']] == \
        ['b1a-x.md', 'b1a-x.png', 'b1a-x_checkpoint_evals.json', 'b1a-x_evals.json']


def test_a_job_seen_running_before_a_restart_is_still_published_after_it(tmp_path, monkeypatch):
    runner, bus = _box(tmp_path, monkeypatch, [_train_spec('b1a-x')])
    runner.poll_once(git=True)
    _scheduler_status(runner, running=[{'id': 'b1a-x', 'type': 'train', 'policy': 'b1a-x', 'policies': ['b1a-x']}])
    runner.poll_once(git=False)
    monkeypatch.setattr(launch, 'pid_alive', lambda pid: False)     # scheduler and daemon both gone
    again = runner_module.Runner(runner.host)
    again.spawn = Spawns()
    again.poll_once(git=False)
    assert [job_id for job_id, _ in bus.results] == ['b1a-x']


def test_status_carries_the_schedulers_glance_its_attention_and_the_laptop(tmp_path, monkeypatch):
    laptop = json.dumps({'iso': '2026-09-05T11:00:00', 'at_a_glance': {'running': ['b16 | kl | training 3% (8 arms)'],
                                                                        'queued': [], 'attention': []}})
    runner, bus = _box(tmp_path, monkeypatch, [_train_spec('b1a-x')], laptop=laptop)
    runner.poll_once(git=True)
    _scheduler_status(runner, running=[{'id': 'b1a-x', 'type': 'train', 'policy': 'b1a-x', 'policies': ['b1a-x']}],
                      attention=['** b1-stageb failed (exit 1); marked, not retried'])
    runner.poll_once(git=True)
    glance = bus.status[-1]['at_a_glance']
    assert glance['running'] == ['b1 | x | training 50% (2 arms)']
    assert glance['attention'] == ['** b1-stageb failed (exit 1); marked, not retried']
    assert glance['laptop_running'] == ['b16 | kl | training 3% (8 arms)'] and glance['laptop_iso'] == '2026-09-05T11:00:00'
    assert bus.status[-1]['ledger']['b1-stageb'] == 'failed'
    assert bus.status[-1]['scheduler']['status_iso'] == '2026-09-05T12:00:00'
    assert 'counts' not in bus.status[-1] and bus.status[-1]['head'] in ('aaaaaaaaaaaa', 'bbbbbbbbbbbb')


def test_a_stale_status_file_from_a_dead_scheduler_is_not_shown_as_running(tmp_path, monkeypatch):
    runner, bus = _box(tmp_path, monkeypatch, [_train_spec('b1a-x')])
    runner.poll_once(git=True)
    _scheduler_status(runner, running=[{'id': 'b1a-x', 'type': 'train', 'policy': 'b1a-x', 'policies': ['b1a-x']}])
    runner.scheduler.returncode = -9
    runner.poll_once(git=True)
    assert bus.status[-1]['at_a_glance']['running'] == [] and bus.status[-1]['running'] == []
    assert bus.status[-1]['at_a_glance']['queued'] == ['b1 training | x | queued (1 arm)'], 'the arm is short of its cap'
    assert any('the scheduler exited -9' in line for line in bus.status[-1]['at_a_glance']['attention'])


def test_an_eval_the_old_daemon_already_ran_gets_a_done_marker_so_the_scheduler_skips_it(tmp_path, monkeypatch):
    ledger = {'b1-hof5000': {'state': 'done', 'type': 'eval', 'finished': 1.0},
              'b1a-x': {'state': 'done', 'type': 'train', 'finished': 1.0}}
    runner, bus = _box(tmp_path, monkeypatch, [
        {'project': 'snek3', 'id': 'b1-hof5000', 'type': 'eval', 'policies': ['b1a-x'], 'eval_args': ['--pass', 'hof5000']},
        _train_spec('b1a-x'), _train_spec('b1b-x')], ledger=ledger)     # b1b-x keeps the batch live
    runner.poll_once(git=True)
    assert os.path.exists(os.path.join(launch.queue_dir(runner.host), 'b1', '.done-b1-hof5000'))
    assert bus.status[-1]['ledger']['b1-hof5000'] == 'done' and bus.status[-1]['ledger']['b1a-x'] == 'done'


# --- actions: deploy and restart over the bus -------------------------------------------------------

def test_deploy_and_restart_are_job_types_that_take_no_work_fields():
    deploy = parse_job(json.dumps({'project': 'snek3', 'id': 'deploy-1', 'type': 'deploy'}))
    assert deploy.category == 'action' and deploy.restart is None
    assert parse_job(json.dumps({'project': 'snek3', 'id': 'r1', 'type': 'restart'})).category == 'action'
    assert parse_job(json.dumps({'project': 'snek3', 'id': 'd2', 'type': 'deploy', 'restart': True})).restart is True
    for bad in ({'type': 'deploy', 'restart': 'yes'}, {'type': 'restart', 'restart': True},
                {'type': 'deploy', 'policy': 'x'}, {'type': 'restart', 'env': {'A': '1'}}):
        with pytest.raises(JobError):
            parse_job(json.dumps(dict({'project': 'snek3', 'id': 'a'}, **bad)))


def test_a_deploy_that_changes_the_runner_merges_records_publishes_and_stops_for_systemd(tmp_path, monkeypatch):
    """The named action replaces `ssh ... deploy; ssh ... sudo systemctl restart`: the daemon runs the
    box's own deploy, sees the runner changed between the heads, publishes the done record, and exits
    -- `Restart=always` relaunches it on the new code. The scheduler, detached, is untouched."""
    runner, bus = _box(tmp_path, monkeypatch, [{'project': 'snek3', 'id': 'deploy-1', 'type': 'deploy'},
                                               _train_spec('b1a-x')])
    commands = runner.run_command
    runner.poll_once(git=True)
    record = runner.ledger['deploy-1']
    assert record['state'] == 'done' and record['restart'] is True and record['restarted'] is True
    assert (record['head_before'], record['head_after']) == ('aaaaaaaaaaaa', 'bbbbbbbbbbbb')
    assert record['output'] == ['HEAD aaaaaaaaaaaa -> bbbbbbbbbbbb; kept 0 pictures']
    assert runner.stop and 'deploy-1' in runner.restart_requested
    assert runner.spawn.calls == [], 'the poll ended at the restart'
    assert _queue_files(runner) == ['b1/b1a-x.json'], 'actions never reach the scheduler'
    assert bus.status[-1]['ledger']['deploy-1'] == 'done'
    assert bus.status[-1]['head'] == 'bbbbbbbbbbbb'
    assert commands.calls[1][-2:] == ['-m', 'runner.deploy']
    with open(runner.host['LEDGER_PATH']) as handle:
        assert json.load(handle)['deploy-1']['state'] == 'done'


def test_a_deploy_that_touches_only_tools_does_not_restart_and_the_scheduler_starts(tmp_path, monkeypatch):
    runner, _ = _box(tmp_path, monkeypatch, [{'project': 'snek3', 'id': 'deploy-2', 'type': 'deploy'},
                                             _train_spec('b1a-x')])
    runner.run_command = Commands(changed='')
    runner.poll_once(git=True)
    assert runner.ledger['deploy-2']['state'] == 'done' and runner.ledger['deploy-2']['restart'] is False
    assert not runner.stop and len(runner.spawn.calls) == 1
    assert any(argv[:2] == ['git', 'diff'] for argv in runner.run_command.calls)


def test_restart_true_forces_it_and_a_failed_deploy_is_attention_never_a_restart(tmp_path, monkeypatch):
    runner, _ = _box(tmp_path, monkeypatch, [{'project': 'snek3', 'id': 'deploy-3', 'type': 'deploy', 'restart': True}])
    runner.run_command = Commands(changed='')
    runner.poll_once(git=True)
    assert runner.stop and runner.ledger['deploy-3']['restarted'] is True

    runner, bus = _box(tmp_path / 'two', monkeypatch,
                       [{'project': 'snek3', 'id': 'deploy-4', 'type': 'deploy', 'restart': True}])
    runner.run_command = Commands(heads=('aaaaaaaaaaaa',), deploy_rc=3,
                                  deploy_out='snek3/runs/b1a_evals.json differs from the commit; nothing touched\n')
    runner.poll_once(git=True)
    record = runner.ledger['deploy-4']
    assert record['state'] == 'failed' and record['rc'] == 3 and 'restart' not in record
    assert not runner.stop
    assert bus.status[-1]['at_a_glance']['attention'] == [
        '** deploy-4 failed: deploy exited 3: snek3/runs/b1a_evals.json differs from the commit; '
        'nothing touched. Nothing was changed; fix it and queue a new id.']


def test_a_restart_action_runs_even_under_a_pause_and_a_done_one_is_not_repeated(tmp_path, monkeypatch):
    runner, bus = _box(tmp_path, monkeypatch, [{'project': 'snek3', 'id': 'restart-1', 'type': 'restart'}],
                       runtime_text='{"paused": true}')
    runner.poll_once(git=True)
    assert runner.stop and runner.ledger['restart-1']['state'] == 'done'
    assert bus.status[-1]['at_a_glance']['queued'][0].startswith('** queue paused')
    again = runner_module.Runner(runner.host)
    again.run_command = Commands()
    again.spawn = Spawns()
    again.poll_once(git=True)
    assert not again.stop and again.restart_requested is None
