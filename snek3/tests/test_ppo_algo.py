"""PPO behind the seam: the config it accepts, the config it refuses, and what a step means.

The seam's *shape* is asserted in `tests/test_train.py`, which parametrises its contract fixtures over
`train.ALGOS` — so registering `ppo` subjected it to those automatically. What is here is what is
specific to PPO's answers, and three of them would be silent:

- **A step that is not a transition.** The whole point of PPO's step unit is that its numbers compare
  to snek2's directly. A step increment that drifted from the transition count would reintroduce the
  units confusion `docs/findings.md` exists to end.
- **A DQN knob that is ignored rather than refused.** A PPO arm launched from a copied DQN command
  line would take a `1e-5` learning rate — ~64x too little movement — and report that PPO does not
  learn.
- **A checkpoint that holds the critic.** Stage B measures the policy; a checkpoint carrying a value
  head would need a new `arch.json` field and invalidate every committed sidecar.
"""

import json
import os

import numpy as np
import pytest
import torch

import train
from env import constants
from ppo import algo as ppo_algo
from tools import arch as arch_tools
from tools import checkpoints
from tools import restore


def ppo_config(monkeypatch, **overrides):
    """A small PPO config through the real `build_config`, so the knob names are the real ones."""
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    monkeypatch.setenv('SNEK_COLLECT_ENVS', '4')
    monkeypatch.setenv('SNEK_PPO_ROLLOUT', '8')
    monkeypatch.setenv('SNEK_PPO_MINIBATCH', '8')
    config = train.build_config()
    config.update({'seed': 5, 'fc_layers': (16,), 'max_steps': 10000})
    config.update(overrides)
    return config


def arch_for(config):
    return arch_tools.build_arch(config['fc_layers'], constants.NUM_ACTIONS, constants.OBS_LEN,
                                 constants.OBS_ERA, algo=config['algo'])


def built(monkeypatch, **overrides):
    config = ppo_config(monkeypatch, **overrides)
    return ppo_algo.build(config, arch_for(config)), config


# --- a step is a transition ----------------------------------------------------------------------

def test_a_step_is_a_transition_and_a_rollout_is_the_granularity(monkeypatch):
    """**The reason a PPO step number can be read against a snek2 one.**

    `advance()` returns the same number twice, unlike DQN's `(1, width)`. Nothing in a PPO rollout is
    conditional on an episode boundary, so there is no case where the two could legitimately differ.
    """
    algo, config = built(monkeypatch)
    expected = config['collect_envs'] * config['ppo_rollout']
    assert algo.step_granularity == expected == 32
    steps, transitions = algo.advance()
    assert steps == transitions == expected


def test_the_step_count_and_the_transition_count_stay_equal_over_several_rollouts(monkeypatch):
    algo, config = built(monkeypatch)
    total = 0
    for _ in range(3):
        steps, transitions = algo.advance()
        assert steps == transitions
        total += transitions
    assert algo.step == total
    assert algo.collector.counters['transitions'] == total


def test_the_trainer_evaluates_once_per_rollout_at_the_default_interval(tmp_path, monkeypatch):
    """`_whole_steps` rounds 1,000 up to a whole rollout, so no PPO-specific interval knob is needed.

    The invariant this protects is the one `train.py` calls non-negotiable: an eval interval that is
    not a multiple of the step increment would put a checkpoint at a step no eval screens.
    """
    config = ppo_config(monkeypatch)
    monkeypatch.setattr(train.constants, 'POLICY_DIR', str(tmp_path / 'savedPolicies'))
    monkeypatch.setattr(train.constants, 'RUNS_DIR', str(tmp_path / 'runs'))
    monkeypatch.setattr(train.progress_chart, 'chart_path',
                        lambda name: str(tmp_path / 'runs' / (name + '.png')))
    trainer = train.Trainer('p-interval', config)
    # 1,000 is not a multiple of this arm's 32-transition rollout, so it rounds *up* to 1,024 rather
    # than down: an interval below one step would evaluate on every step, at any rollout size.
    assert trainer.algo.step_granularity == 32
    assert trainer.eval_interval == 1024
    assert trainer.eval_interval % trainer.algo.step_granularity == 0
    assert trainer.resume_interval % trainer.eval_interval == 0
    assert trainer.report_interval % trainer.eval_interval == 0
    assert trainer.config['eval_interval'] == trainer.eval_interval, 'the report states the real one'


def test_a_rollout_larger_than_the_eval_interval_still_evaluates_every_rollout(tmp_path, monkeypatch):
    """The production case: 16,384 transitions per rollout against a 1,000-step interval.

    The rounding has to reach the granularity, not merely a multiple of the constant — otherwise the
    first eval lands at 16,384 and the interval claims 1,000, and every later "every 1,000 steps"
    statement in a report is wrong by 16x.
    """
    config = ppo_config(monkeypatch, collect_envs=128, ppo_rollout=128, ppo_minibatch=256)
    monkeypatch.setattr(train.constants, 'POLICY_DIR', str(tmp_path / 'savedPolicies'))
    monkeypatch.setattr(train.constants, 'RUNS_DIR', str(tmp_path / 'runs'))
    monkeypatch.setattr(train.progress_chart, 'chart_path',
                        lambda name: str(tmp_path / 'runs' / (name + '.png')))
    trainer = train.Trainer('p-wide', config)
    assert trainer.algo.step_granularity == 16384
    assert trainer.eval_interval == 16384
    assert trainer.resume_interval % trainer.eval_interval == 0


def test_there_is_nothing_to_prefill(monkeypatch):
    # The first rollout is the first batch; a prefill would be experience nothing learns from.
    algo, _ = built(monkeypatch)
    assert algo.prefill() == 0


# --- the knobs PPO refuses -----------------------------------------------------------------------

@pytest.mark.parametrize('knob,value', [
    ('SNEK_FORK_BRANCHES', '4'),
    ('SNEK_INITIAL_EPSILON', '0.4'),
    ('SNEK_MIN_EPSILON', '0.002'),
    ('SNEK_GUIDED_FRACTION', '0.8'),
    ('SNEK_REPLAY_RATIO', '1.0'),
    ('SNEK_PRIORITY_EXPONENT', '0.6'),
    ('SNEK_IS_WEIGHTS', '0'),
    ('SNEK_TARGET_UPDATE_PERIOD', '1000'),
    ('SNEK_N_STEP_UPDATE', '3'),
    ('SNEK_LEARNING_RATE', '1e-5'),
    ('SNEK_BATCH_SIZE', '128'),
    ('SNEK_INITIAL_COLLECT_STEPS', '2000'),
])
def test_a_dqn_knob_is_refused_by_name(knob, value, monkeypatch):
    """Refused, not ignored — the project's rule, and here the cost of ignoring is concrete.

    `SNEK_LEARNING_RATE=1e-5` is the one that matters most: it is DQN's tuned value, it is ~64x too
    small for an algorithm taking 64x fewer gradient steps, and an arm that silently took it would
    produce a flat curve and a conclusion about PPO.
    """
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    monkeypatch.setenv(knob, value)
    with pytest.raises(ValueError, match='cannot honour'):
        train.build_config()


def test_the_error_names_every_offending_knob_at_once(monkeypatch):
    # One launch usually sets several; reporting them one at a time is three failed launches.
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    monkeypatch.setenv('SNEK_FORK_BRANCHES', '4')
    monkeypatch.setenv('SNEK_LEARNING_RATE', '1e-5')
    with pytest.raises(ValueError) as caught:
        train.build_config()
    assert 'SNEK_FORK_BRANCHES' in str(caught.value)
    assert 'SNEK_LEARNING_RATE' in str(caught.value)


@pytest.mark.parametrize('knob,value', [
    ('SNEK_COLLECT_ENVS', '64'),
    ('SNEK_DISCOUNT', '0.9975'),
    ('SNEK_SEED', '3'),
    ('SNEK_MAX_STEPS', '1000'),
    ('SNEK_FC_LAYERS', '320'),
    ('SNEK_CHASE_SAFE_SHAPING', '0.1'),
])
def test_a_shared_knob_is_accepted(knob, value, monkeypatch):
    """The other half of the rule: lanes, gamma, the seed, the cap, the shape and the reward config
    mean the same thing to both algorithms, and refusing them would make a seed-matched A/B
    unlaunchable."""
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    monkeypatch.setenv(knob, value)
    assert train.build_config()['algo'] == 'ppo'


def test_dqn_still_accepts_its_own_knobs(monkeypatch):
    # The rejection list must not leak into the other algorithm.
    monkeypatch.setenv('SNEK_ALGO', 'dqn')
    monkeypatch.setenv('SNEK_LEARNING_RATE', '1e-5')
    monkeypatch.setenv('SNEK_FORK_BRANCHES', '4')
    assert train.build_config()['learning_rate'] == 1e-5


# --- config validation ---------------------------------------------------------------------------

def test_a_minibatch_larger_than_a_rollout_is_refused(monkeypatch):
    """Otherwise "4 epochs" is 4 gradient steps rather than 4 passes, and nothing says so."""
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    monkeypatch.setenv('SNEK_COLLECT_ENVS', '4')
    monkeypatch.setenv('SNEK_PPO_ROLLOUT', '8')
    monkeypatch.setenv('SNEK_PPO_MINIBATCH', '64')
    with pytest.raises(ValueError, match='larger than a whole rollout'):
        train.build_config()


@pytest.mark.parametrize('knob,value,match', [
    ('SNEK_PPO_EPOCHS', '0', 'at least 1'),
    ('SNEK_PPO_CLIP', '0', 'outside'),
    ('SNEK_PPO_CLIP', '1.0', 'outside'),
    ('SNEK_PPO_GAE_LAMBDA', '1.5', 'not in'),
    ('SNEK_PPO_GAE_LAMBDA', '-0.1', 'not in'),
])
def test_a_nonsense_ppo_knob_is_refused(knob, value, match, monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    monkeypatch.setenv(knob, value)
    with pytest.raises(ValueError, match=match):
        train.build_config()


def test_the_defaults_are_the_ones_the_plan_and_the_docs_state(monkeypatch):
    """Hand-copied from `plans/ppo.md`'s knob table, so a drift between them fails here.

    Compared against literals rather than parsed out of the doc: a parser follows the doc wherever it
    goes, and the point is to notice when the two disagree.
    """
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    documented = {'collect_envs': 128, 'ppo_rollout': 128, 'ppo_epochs': 4, 'ppo_minibatch': 256,
                  'ppo_clip': 0.2, 'ppo_gae_lambda': 0.98, 'ppo_entropy_coef': 0.01,
                  'ppo_entropy_coef_final': None, 'ppo_vf_coef': 0.5,
                  'ppo_learning_rate': 3e-4, 'ppo_target_kl': 0.0, 'ppo_normalize_adv': True,
                  'ppo_value_loss': 'huber', 'discount': 0.99}
    config = train.build_config()
    assert {key: config[key] for key in documented} == documented


def test_the_report_states_the_horizon_and_the_rollout_size(monkeypatch):
    """Both are derived, and both are what a reader checks lambda against — see `rollout.py`."""
    config = ppo_config(monkeypatch)
    out = train.reportable(config)
    assert out['ppo_transitions_per_rollout'] == 32
    assert out['ppo_horizon'] == pytest.approx(33.6, abs=0.2)
    for value in out.values():
        assert isinstance(value, (int, float, str, tuple, bool, type(None))), value


def test_every_ppo_config_key_greps_back_to_its_knob():
    """`train.py`'s contract: a report row nobody can grep back to a variable is unreproducible."""
    import re
    source = open(ppo_algo.__file__).read()
    knobs = {name.lower() for name in re.findall(r"tuned\('([A-Z_]+)'", source)}
    keys = set(ppo_algo.build_config(lambda name, default, cast=float: default))
    assert keys <= knobs, sorted(keys - knobs)


# --- the row -------------------------------------------------------------------------------------

def test_the_row_carries_the_diagnostics_that_say_why_an_arm_failed(monkeypatch):
    algo, _ = built(monkeypatch)
    algo.advance()
    fields = algo.fields()
    assert fields['entropy_coef'] == pytest.approx(0.01)
    block = fields['ppo']
    for key in ('entropy', 'approx_kl', 'clip_fraction', 'explained_variance', 'policy_loss',
                'value_loss', 'epochs_run', 'episodes', 'transitions', 'rollouts'):
        assert key in block, key
    assert block['entropy'] is not None
    assert 0.0 <= block['entropy'] <= np.log(3.0) + 1e-6


def test_the_row_is_json_serialisable(monkeypatch):
    # It goes through `runs/<policy>_evals.json` and, with the queue on, through a queue file first.
    algo, _ = built(monkeypatch)
    algo.advance()
    json.dumps(algo.fields())


def test_a_row_built_before_the_first_update_has_no_stale_diagnostics(monkeypatch):
    """`fields()` may be called before `advance()` has ever run — the trainer's first eval can be at
    step 0 on a resume. Nulls are the honest answer; a previous arm's numbers would not be."""
    algo, _ = built(monkeypatch)
    block = algo.fields()['ppo']
    assert block['entropy'] is None
    assert block['approx_kl'] is None


def test_the_log_line_does_not_raise_on_a_row_with_no_ppo_block(monkeypatch):
    algo, _ = built(monkeypatch)
    assert algo.log_extra({'step': 1}) == []
    assert 'ent' in algo.log_note({'step': 1, 'entropy_coef': 0.01})


# --- the checkpoint holds the actor --------------------------------------------------------------

def test_a_checkpoint_holds_the_actor_and_not_the_critic(tmp_path, monkeypatch):
    """**Stage B measures the policy.** A checkpoint carrying a value head would have needed a new
    `arch.json` field, and `arch.py`'s rule is that every field is required — so it would have
    invalidated every committed sidecar."""
    algo, config = built(monkeypatch)
    policy_dir = str(tmp_path / 'b4')
    arch_tools.write_arch(policy_dir, arch_for(config))
    path = checkpoints.save(policy_dir, 32, algo.net)
    payload = torch.load(path, weights_only=True)
    assert sorted(payload['model']) == sorted(algo.agent.actor.state_dict())
    assert not any('critic' in key for key in payload['model'])


def test_a_ppo_checkpoint_restores_into_a_playable_policy(tmp_path, monkeypatch):
    """The one line in `tools/restore.py` that makes `watch.py`, `record_gif.py` and every eval shard
    work on a PPO checkpoint. Asserted end to end rather than by reading the registry."""
    algo, config = built(monkeypatch)
    policy_dir = str(tmp_path / 'b5')
    arch_tools.write_arch(policy_dir, arch_for(config))
    checkpoints.save(policy_dir, 32, algo.net)

    policy_fn, restored_arch, step = restore.restore(policy_dir)
    assert restored_arch['algo'] == 'ppo' and step == 32
    sample = np.random.default_rng(0).random((7, constants.OBS_LEN)).astype(np.float32)
    actions = policy_fn(sample)
    assert actions.shape == (7,)
    assert set(np.unique(actions)) <= set(range(constants.NUM_ACTIONS))
    # And it is the same function the trainer would have measured.
    assert np.array_equal(actions, algo.policy_fn(sample))


def test_an_algo_the_restore_path_does_not_know_names_itself(tmp_path):
    policy_dir = str(tmp_path / 'b6')
    arch = arch_tools.build_arch((16,), constants.NUM_ACTIONS, constants.OBS_LEN,
                                constants.OBS_ERA, algo='sac')
    arch_tools.write_arch(policy_dir, arch)
    with pytest.raises(arch_tools.ArchMismatch, match='unknown algo'):
        restore.policy_fn_for(arch, None)


def test_the_restore_registry_covers_every_algorithm_the_trainer_can_run():
    """Or an arm trains for hours and then cannot be measured, watched or recorded."""
    missing = sorted(set(train.ALGOS) - set(restore.ALGORITHMS))
    assert not missing, missing


# --- resume --------------------------------------------------------------------------------------

def test_a_resume_restores_the_towers_and_the_algorithms_own_step(monkeypatch):
    algo, config = built(monkeypatch)
    algo.advance()
    state = algo.state_dict()

    other = ppo_algo.build(dict(config, seed=999), arch_for(config))
    assert not torch.equal(other.net.head.weight, algo.net.head.weight)
    other.load_state_dict(state)
    assert torch.equal(other.net.head.weight, algo.net.head.weight)
    assert other.step == algo.step


def test_there_is_no_side_state_to_save(monkeypatch):
    """No replay buffer, so `load_side_state` reports False and the resume line says "buffer empty"
    rather than claiming one came back."""
    algo, _ = built(monkeypatch)
    assert algo.save_side_state('/nonexistent/path') is None
    assert algo.load_side_state('/nonexistent/path') is False


# --- the entropy schedule ------------------------------------------------------------------------

def test_the_entropy_coefficient_is_constant_when_no_final_value_is_set(monkeypatch):
    """b4's setting. A constant is one fewer moving part than DQN's two-phase epsilon."""
    algo, _ = built(monkeypatch, ppo_entropy_coef=0.02, ppo_entropy_coef_final=None)
    seen = set()
    for _ in range(3):
        algo.advance()
        seen.add(algo.fields()['entropy_coef'])
    assert seen == {0.02}


def test_the_entropy_coefficient_ramps_when_a_final_value_is_set(monkeypatch):
    algo, _ = built(monkeypatch, ppo_entropy_coef=0.02, ppo_entropy_coef_final=0.0,
                    max_steps=64)
    algo.advance()                       # 32 of 64 steps: halfway
    assert algo.fields()['entropy_coef'] == pytest.approx(0.01, abs=1e-9)
    algo.advance()                       # at the cap
    assert algo.fields()['entropy_coef'] == pytest.approx(0.0, abs=1e-9)


def test_the_ramp_is_clamped_past_the_cap(monkeypatch):
    """**A resumed arm can be past its old cap**, and an unclamped fraction would push the
    coefficient beyond `final` — a *negative* entropy bonus for a descending ramp, which is an active
    push toward determinism that nothing in the config names."""
    algo, _ = built(monkeypatch, ppo_entropy_coef=0.02, ppo_entropy_coef_final=0.0, max_steps=32)
    for _ in range(4):
        algo.advance()
    assert algo.fields()['entropy_coef'] == pytest.approx(0.0, abs=1e-9)
    assert algo.fields()['entropy_coef'] >= 0.0


def test_on_eval_reports_the_live_coefficient_in_both_queue_modes(monkeypatch):
    """PPO's schedule is a function of the step, so unlike DQN's epsilon there is no one-row shift
    between the queued and unqueued spellings — and this is what makes that true."""
    algo, _ = built(monkeypatch)
    algo.advance()
    computed = algo.on_eval([], {'avg_reward': 1.0, 'perfect': 0.0})
    assert computed['entropy_coef'] == algo.fields()['entropy_coef']
    # And calling it again does not move it, because it advances no state of its own.
    assert algo.on_eval([], {'avg_reward': 1.0, 'perfect': 0.0}) == computed


def test_the_discount_and_lambda_reach_gae_in_that_order(monkeypatch):
    """**Swapping the two is a one-token mutation and both are plausible numbers.**

    At gamma=0.5, lambda=0 the advantage is exactly the one-step TD error; swapped it would be
    `r - V(s_t)` with no bootstrap at all. Read off the arm's own rollout after a real `advance()`, so
    the assertion covers the collector's call and not a re-spelling of it.
    """
    algo, _ = built(monkeypatch, discount=0.5, ppo_gae_lambda=0.0)
    algo.advance()
    roll = algo.rollout
    alive = (~roll.dones[0]).astype(np.float64)
    expected = roll.rewards[0] + 0.5 * alive * roll.values[1] - roll.values[0]
    assert roll.advantages[0] == pytest.approx(expected, abs=1e-5)
    # Not the swapped spelling, which drops the bootstrap entirely.
    swapped = roll.rewards[0] + 0.0 * alive * roll.values[1] - roll.values[0]
    assert not np.allclose(roll.advantages[0], swapped, atol=1e-4), 'the fixture must discriminate'


# --- the clip and learning-rate anneals ------------------------------------------------------------
#
# `SNEK_PPO_CLIP_FINAL` and `SNEK_PPO_LEARNING_RATE_FINAL` (2026-09-01): the PPO paper's Atari recipe,
# both ramping linearly over `max_steps` exactly as the entropy coefficient does. Absent means
# constant, which is every arm before batch b17.

from ppo import schedules as ppo_schedules


def test_absent_finals_leave_clip_and_learning_rate_constant(monkeypatch):
    algo, config = built(monkeypatch)
    assert config['ppo_clip_final'] is None and config['ppo_learning_rate_final'] is None
    for _ in range(3):
        algo.advance()
    assert algo.agent.clip == config['ppo_clip']
    assert algo.agent.learning_rate() == config['ppo_learning_rate']


def test_the_clip_and_learning_rate_ramp_with_the_step(monkeypatch):
    monkeypatch.setenv('SNEK_PPO_CLIP_FINAL', '0.02')
    monkeypatch.setenv('SNEK_PPO_LEARNING_RATE_FINAL', '0')
    algo, config = built(monkeypatch, max_steps=4 * 32)   # 4 lanes x 8 steps = 32 a rollout
    assert config['ppo_clip_final'] == 0.02 and config['ppo_learning_rate_final'] == 0.0
    seen = []
    for _ in range(5):                                    # one rollout past the cap
        algo.advance()
        seen.append((algo.step, algo.agent.clip, algo.agent.learning_rate()))
    for step, clip, lr in seen:
        fraction = min(1.0, step / config['max_steps'])
        assert clip == pytest.approx(0.2 + fraction * (0.02 - 0.2))
        assert lr == pytest.approx(3e-4 * (1.0 - fraction))
    # Past the cap both sit at their floors rather than overshooting — a resumed arm can be there.
    assert seen[-1][1] == pytest.approx(0.02) and seen[-1][2] == 0.0
    # And the ramped values reach the row, so an annealed arm's history says what it ran under.
    row = algo.fields()['ppo']
    assert row['clip'] == pytest.approx(0.02) and row['learning_rate'] == 0.0


def test_the_ramp_is_one_function_for_all_three_schedules():
    for f in (ppo_schedules.entropy_coef_for, ppo_schedules.clip_for,
              ppo_schedules.learning_rate_for):
        assert f(0, 100, 1.0, 0.0) == 1.0
        assert f(50, 100, 1.0, 0.0) == 0.5
        assert f(250, 100, 1.0, 0.0) == 0.0     # clamped past the cap
        assert f(50, 100, 1.0) == 1.0           # no final, no ramp


@pytest.mark.parametrize('value', ['0', '1', '-0.1', '1.5'])
def test_a_clip_final_outside_the_open_interval_is_refused(value, monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    monkeypatch.setenv('SNEK_PPO_CLIP_FINAL', value)
    with pytest.raises(ValueError, match='SNEK_PPO_CLIP_FINAL'):
        train.build_config()


def test_a_negative_learning_rate_final_is_refused(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    monkeypatch.setenv('SNEK_PPO_LEARNING_RATE_FINAL', '-1e-5')
    with pytest.raises(ValueError, match='SNEK_PPO_LEARNING_RATE_FINAL'):
        train.build_config()
