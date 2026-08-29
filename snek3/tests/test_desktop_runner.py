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
    'HARD_MAX_TRAINERS': 4, 'HARD_MAX_EVALS': 4, 'HARD_MAX_EVAL_SHARDS': 16,
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
    # An ambitious number is honoured as the ceiling; refusing the file would be the worse failure.
    config, notes = config_module.parse_runtime_config('{"max_trainers": 99}', HOST)
    assert config['max_trainers'] == HOST['HARD_MAX_TRAINERS']
    assert any('max_trainers 99 clamped to 4' in note for note in notes)


def test_the_poll_floor_holds():
    config, _ = config_module.parse_runtime_config('{"poll_seconds": 1}', HOST)
    assert config['poll_seconds'] == HOST['MIN_POLL_SECONDS']


def test_git_seconds_may_be_zero_because_zero_is_the_opt_out():
    # 0 means a network cycle every poll, which is what the daemon did before the knob existed.
    config, notes = config_module.parse_runtime_config('{"git_seconds": 0}', HOST)
    assert config['git_seconds'] == 0 and notes == []


def test_two_waves_of_sixteen_shards_is_clamped_to_one_box_worth():
    """`max_evals x eval_shards` is the whole ceiling, and both multiply.

    Past `cpu_count` throughput *falls*, so two waves of 16 makes both slower than one.
    """
    config, notes = config_module.parse_runtime_config(
        '{"max_evals": 2, "eval_shards": 16}', HOST)
    assert config['max_evals'] == 2, 'how many waves run is a scheduling decision, never clamped'
    assert config['eval_shards'] == 8
    assert any('32 shard processes exceeds 16' in note for note in notes)


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


def test_an_eval_wave_invokes_evaluate_py_with_every_arm():
    job = parse_job(spec(type='eval', policies=['b1a', 'b1b'], policy=None))
    argv, _, log, _ = launch.build_command(job, HOST, runtime(eval_shards=16))
    assert argv[:3] == ['/py', '-u', 'evaluate.py']
    assert argv[3:5] == ['b1a', 'b1b']
    assert '--shards' in argv and argv[argv.index('--shards') + 1] == '16'
    assert log == 'eval-b1a-thing.log'


def test_no_episode_count_or_gate_is_passed_unless_the_spec_asked():
    """The protocol lives in `evaluate.py`, not here.

    snek2's daemon carried five protocol numbers as a second copy of `eval_plan.py`'s and they
    drifted. So an unadorned eval job must pass neither, and inherit `screen:95` at 500 episodes.
    """
    job = parse_job(spec(type='eval', policies=['b1a'], policy=None))
    argv, _, _, _ = launch.build_command(job, HOST, runtime())
    assert '--episodes' not in argv
    assert '--selector' not in argv


def test_a_spec_may_still_name_a_selector_and_a_depth():
    job = parse_job(spec(type='eval', policies=['b1a'], policy=None,
                         selector='screen:98', episodes=1000))
    argv, _, _, _ = launch.build_command(job, HOST, runtime())
    assert argv[argv.index('--selector') + 1] == 'screen:98'
    assert argv[argv.index('--episodes') + 1] == '1000'


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

def test_batch_of_reads_the_leading_b_number():
    assert runner_module.batch_of('b12c-thing-seed3') == 'b12'
    assert runner_module.batch_of('smoke-1') == 'smoke'


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
    assert [job['id'] for job in order] == ['b1a-x', 'b1b-x', 'b1-stageb', 'b2a-y', 'b2-stageb']


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


def test_a_job_that_already_exists_is_never_invented_twice():
    queued = [{'id': 'b1a-x', 'type': 'train', 'policy': 'b1a-x', 'policies': ['b1a-x'],
               'priority': 100}]
    order = runner_module.anticipated_queue(
        queued, [], {'trainer': 4, 'eval': 1}, True, {'b1a-x', 'b1-stageb'})
    assert [job['id'] for job in order] == ['b1a-x']


def test_only_a_training_earns_a_measurement():
    # Smoke and benchmark share the `trainer` category but are throwaway, and a failed run has no
    # checkpoint worth measuring.
    assert runner_module.wants_stage_b('train', True, True) is True
    assert runner_module.wants_stage_b('train', False, True) is False
    assert runner_module.wants_stage_b('smoke', True, True) is False
    assert runner_module.wants_stage_b('benchmark', True, True) is False
    assert runner_module.wants_stage_b('train', True, False) is False


# --- at_a_glance ---------------------------------------------------------------------------------

def test_a_wave_is_reported_in_arms_not_in_jobs():
    # One eval job is four arms; counting jobs would report a whole measurement as "1 arm", which is
    # the number a reader uses to check nothing was dropped.
    glance = runner_module.build_at_a_glance(
        [], [{'id': 'b1-stageb', 'type': 'eval', 'policy': 'b1a',
              'policies': ['b1a', 'b1b', 'b1c', 'b1d'], 'priority': 10}], {})
    assert glance['queued'] == ['b1 stage B — queued (4 arms)']


def test_a_running_batch_shows_the_mean_percent_across_its_arms():
    running = [{'id': 'b1a-x', 'type': 'train', 'policy': 'b1a-x', 'policies': ['b1a-x'],
                'step': 1000000, 'max_steps': 3000000},
               {'id': 'b1b-x', 'type': 'train', 'policy': 'b1b-x', 'policies': ['b1b-x'],
                'step': 2000000, 'max_steps': 3000000}]
    glance = runner_module.build_at_a_glance(running, [], {'b1': 'the seed-matched set'})
    assert glance['running'] == ['b1 — the seed-matched set — training 50% (2 arms)']


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


# --- the chart window -----------------------------------------------------------------------------
#
# snek3 turns this back on where snek2 kept it off, and the reason it is safe now is the one thing
# worth pinning: snek2's window was the trainer's own in-process cv2 canvas and one XIO error killed
# all four arms; this is a separate process reading PNGs. So these fixtures are about the *policy* —
# when it opens, when it closes, and the one case where it must stay shut.

VIEWER_HOST = dict(HOST, DISPLAY=':0', XAUTHORITY='/run/user/1000/gdm/Xauthority')


class _FakeViewer(object):
    """A stand-in Popen: `status=None` means still running."""

    def __init__(self, status=None):
        self.returncode = status
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode


class _FakeRunning(object):
    def __init__(self, policies, pid):
        self.job = parse_job(spec(policies=None))
        self.job.policies = list(policies)
        self.pid = pid


def _runner_with(monkeypatch, running, host=None, **runtime_overrides):
    """A Runner with its ledger, reattach and git side stubbed out — only the viewer path runs."""
    monkeypatch.setattr(runner_module.Runner, '_load_ledger', lambda self: {})
    monkeypatch.setattr(runner_module.Runner, '_reattach', lambda self: None)
    instance = runner_module.Runner(host or VIEWER_HOST)
    instance.runtime.update(runtime_overrides)
    instance.running = {'j{0}'.format(i): job for i, job in enumerate(running)}
    return instance


def _record_spawns(monkeypatch, result=None):
    calls = []

    def fake_spawn(policies, pids, host):
        calls.append({'policies': tuple(policies), 'pids': tuple(pids), 'host': host})
        return result if result is not None else _FakeViewer()
    monkeypatch.setattr(launch, 'spawn_viewer', fake_spawn)
    return calls


def test_the_viewer_opens_for_the_running_arms(monkeypatch):
    calls = _record_spawns(monkeypatch)
    instance = _runner_with(monkeypatch, [_FakeRunning(['b1b'], 11), _FakeRunning(['b1c'], 12)])
    instance._ensure_viewer()
    assert len(calls) == 1
    assert calls[0]['policies'] == ('b1b', 'b1c'), 'sorted, so the panel order is stable'
    assert calls[0]['pids'] == (11, 12), 'the pids it launched, not a pgrep pattern'


def test_an_already_open_viewer_is_left_alone(monkeypatch):
    calls = _record_spawns(monkeypatch)
    instance = _runner_with(monkeypatch, [_FakeRunning(['b1b'], 11)])
    instance._ensure_viewer()
    instance._ensure_viewer()
    instance._ensure_viewer()
    assert len(calls) == 1, 'a poll every 30 s must not restart the window every 30 s'


def test_the_viewer_restarts_when_an_arm_joins(monkeypatch):
    """The panel list is explicit, so a changed set of arms is the one case needing a restart."""
    calls = _record_spawns(monkeypatch)
    instance = _runner_with(monkeypatch, [_FakeRunning(['b1b'], 11)])
    instance._ensure_viewer()
    first = instance._viewer
    instance.running['j1'] = _FakeRunning(['b1c'], 12)
    instance._ensure_viewer()
    assert len(calls) == 2 and calls[1]['policies'] == ('b1b', 'b1c')
    assert first.terminated and first.waited, 'the old window is closed AND reaped, not orphaned'


def test_the_viewer_closes_when_the_box_goes_idle(monkeypatch):
    _record_spawns(monkeypatch)
    instance = _runner_with(monkeypatch, [_FakeRunning(['b1b'], 11)])
    instance._ensure_viewer()
    opened = instance._viewer
    instance.running = {}
    instance._ensure_viewer()
    assert opened.terminated and opened.waited
    assert instance._viewer is None and instance._viewer_policies == ()


def test_a_window_the_user_closed_stays_closed(monkeypatch):
    """`chart_viewer` exits 0 when its window is closed, and calls that an instruction.

    Respawning it on the next poll would fight whoever is sitting at the box, so a clean exit with
    the same arms still running must not reopen.
    """
    calls = _record_spawns(monkeypatch)
    instance = _runner_with(monkeypatch, [_FakeRunning(['b1b'], 11)])
    instance._ensure_viewer()
    instance._viewer.returncode = 0          # the user closed the window
    instance._ensure_viewer()
    assert len(calls) == 1, 'a deliberate close is honoured'
    assert instance._viewer is None


def test_a_crashed_window_is_reopened(monkeypatch):
    """The opposite case, and the one the request is actually about: a display failure."""
    calls = _record_spawns(monkeypatch)
    instance = _runner_with(monkeypatch, [_FakeRunning(['b1b'], 11)])
    instance._ensure_viewer()
    instance._viewer.returncode = 1          # an XIO error, an OOM kill
    instance._ensure_viewer()
    assert len(calls) == 2, 'a crash is not an instruction'


def test_a_closed_window_reopens_when_the_arms_change(monkeypatch):
    calls = _record_spawns(monkeypatch)
    instance = _runner_with(monkeypatch, [_FakeRunning(['b1b'], 11)])
    instance._ensure_viewer()
    instance._viewer.returncode = 0
    instance.running['j1'] = _FakeRunning(['b1c'], 12)
    instance._ensure_viewer()
    assert len(calls) == 2, 'new content is a new window, even after a deliberate close'


def test_the_viewer_knob_turns_it_off(monkeypatch):
    calls = _record_spawns(monkeypatch)
    instance = _runner_with(monkeypatch, [_FakeRunning(['b1b'], 11)], viewer=False)
    instance._ensure_viewer()
    assert calls == [] and instance._viewer is None


def test_turning_the_knob_off_closes_an_open_window(monkeypatch):
    _record_spawns(monkeypatch)
    instance = _runner_with(monkeypatch, [_FakeRunning(['b1b'], 11)])
    instance._ensure_viewer()
    opened = instance._viewer
    instance.runtime['viewer'] = False
    instance._ensure_viewer()
    assert opened.terminated and instance._viewer is None


def test_a_headless_host_is_not_an_error(monkeypatch):
    """No DISPLAY is a valid configuration, so `spawn_viewer` declines and the daemon carries on."""
    instance = _runner_with(monkeypatch, [_FakeRunning(['b1b'], 11)], host=HOST)
    instance._ensure_viewer()
    assert instance._viewer is None and instance._viewer_policies == ()


def test_spawn_viewer_declines_without_a_display():
    assert launch.spawn_viewer(('b1b',), (11,), HOST) is None


def test_a_viewer_that_will_not_die_is_killed(monkeypatch):
    import subprocess as sp
    _record_spawns(monkeypatch)
    instance = _runner_with(monkeypatch, [_FakeRunning(['b1b'], 11)])
    instance._ensure_viewer()
    stubborn = instance._viewer

    def refuse(timeout=None):
        raise sp.TimeoutExpired('viewer', timeout)
    stubborn.wait = refuse
    instance.running = {}
    instance._ensure_viewer()
    assert stubborn.killed, 'SIGTERM ignored means SIGKILL, not a leaked window'


# --- the viewer command ---------------------------------------------------------------------------

def test_the_viewer_command_names_each_arms_png():
    argv, env = launch.viewer_command(('b1b', 'b1c'), (11, 12), VIEWER_HOST)
    assert argv[:4] == ['/py', '-u', '-m', 'tools.chart_viewer']
    assert 'runs/b1b.png' in argv and 'runs/b1c.png' in argv
    assert '--glob' not in argv, 'an explicit list, so the window shows what is running now'


def test_the_viewer_command_passes_the_pids_it_launched():
    argv, env = launch.viewer_command(('b1b',), (11, 12), VIEWER_HOST)
    assert argv[argv.index('--watch-pid') + 1] == '11,12'


def test_the_viewer_command_carries_the_session_display():
    argv, env = launch.viewer_command(('b1b',), (11,), VIEWER_HOST)
    assert env['DISPLAY'] == ':0'
    assert env['XAUTHORITY'] == '/run/user/1000/gdm/Xauthority'
    assert env['PYTHONPATH'] == '.'


def test_the_viewer_command_omits_display_when_the_host_has_none():
    argv, env = launch.viewer_command(('b1b',), (11,), HOST)
    assert 'DISPLAY' not in env and 'XAUTHORITY' not in env


def test_viewer_is_a_bool_knob():
    parsed, notes = config_module.parse_runtime_config('{"viewer": 1}', HOST)
    assert parsed is None and any('viewer' in note for note in notes)
    parsed, notes = config_module.parse_runtime_config('{"viewer": false}', HOST)
    assert parsed is not None and parsed['viewer'] is False
