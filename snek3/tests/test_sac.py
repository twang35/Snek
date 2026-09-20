"""Discrete SAC behind the seam: `algos/sac/agent.py`'s arithmetic, `algos/sac/algo.py`'s knobs and defaults,
the registry and the restore path (`plans/algoExploration/b-entropy.md` §2)."""

import math

import numpy as np
import pytest
import torch

import train
from algos.sac import agent as sac_agent
from algos.sac import algo as sac_algo
from env import constants
from tools import arch as arch_tools
from tools import checkpoints
from tools import restore

NAMES = ('sac', 'sac2')


def small_config(monkeypatch, name='sac', **env):
    monkeypatch.setenv('SNEK_ALGO', name)
    for key, value in env.items():
        monkeypatch.setenv('SNEK_' + key, str(value))
    config = train.build_config()
    config.update({'seed': 3, 'sac_replay_buffer_max_length': 500, 'sac_prefill': 40,
                   'sac_batch_size': 8, 'collect_envs': 2})
    return config


def build(monkeypatch, name='sac', **env):
    config = small_config(monkeypatch, name, **env)
    arch = arch_tools.build_arch(config['fc_layers'], constants.NUM_ACTIONS, constants.OBS_LEN,
                                 constants.OBS_ERA, algo=name)
    return train.ALGOS[name].build(config, arch), config, arch


# ---------------------------------------------------------------- the registry and the defaults

def test_both_names_are_registered_and_restorable():
    for name in NAMES:
        assert train.ALGOS[name].NAME == name
        assert name in restore.ALGORITHMS


def test_sac_defaults_are_the_2019_papers(monkeypatch):
    config = small_config(monkeypatch, 'sac')
    assert config['sac_alpha'] == 'auto' and config['sac_critic_combine'] == 'min'
    assert config['sac_q_clip'] == 0.0 and config['sac_entropy_penalty'] == 0.0
    assert config['sac_learning_rate'] == 3e-4 and config['sac_replay_ratio'] == 0.25
    assert config['sac_target_update_period'] == 2000 and config['sac_tau'] == 1.0
    assert config['sac_n_step'] == 1 and config['sac_priority_exponent'] == 0.0


def test_sac2_defaults_are_the_2022_papers_fixes(monkeypatch):
    config = small_config(monkeypatch, 'sac2')
    assert config['sac_alpha'] == '0.05' and config['sac_critic_combine'] == 'avg'
    assert config['sac_q_clip'] == 0.5 and config['sac_entropy_penalty'] == 0.5
    assert config['sac_learning_rate'] == 1e-5 and config['sac_replay_ratio'] == 0.1
    assert config['sac_tau'] == 0.005 and config['sac_target_update_period'] == 1 and config['sac_n_step'] == 3


def test_a_sac2_spec_can_put_alpha_back_on_auto(monkeypatch):
    algo, config, _ = build(monkeypatch, 'sac2', SAC_ALPHA='auto')
    assert algo.agent.auto_alpha and algo.agent.alpha_optimizer is not None


def test_a_fixed_alpha_builds_no_alpha_optimiser(monkeypatch):
    algo, _, _ = build(monkeypatch, 'sac', SAC_ALPHA='0.2')
    assert not algo.agent.auto_alpha and algo.agent.alpha_optimizer is None
    assert algo.agent.alpha == pytest.approx(0.2)


@pytest.mark.parametrize('knob', ['INITIAL_EPSILON', 'FORK_BRANCHES', 'LEARNING_RATE', 'PPO_CLIP',
                                  'DIST_QUANTILES', 'MUNCHAUSEN_ALPHA', 'RESET_INTERVAL'])
def test_a_foreign_knob_is_refused_by_name(monkeypatch, knob):
    monkeypatch.setenv('SNEK_ALGO', 'sac')
    monkeypatch.setenv('SNEK_' + knob, '1')
    with pytest.raises(ValueError, match=knob):
        train.build_config()


def test_ppo_refuses_the_sac_knobs(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    monkeypatch.setenv('SNEK_SAC_ALPHA', '0.05')
    with pytest.raises(ValueError, match='SAC_ALPHA'):
        train.build_config()


@pytest.mark.parametrize('env', [{'SAC_CRITIC_COMBINE': 'sum'}, {'SAC_ALPHA': 'warm'}, {'SAC_ALPHA': '-1'},
                                 {'SAC_TAU': '0'}, {'SAC_Q_CLIP': '-0.5'}])
def test_a_bad_value_is_refused(monkeypatch, env):
    monkeypatch.setenv('SNEK_ALGO', 'sac')
    for key, value in env.items():
        monkeypatch.setenv('SNEK_' + key, value)
    with pytest.raises(ValueError):
        train.build_config()


def test_every_sac_config_key_is_its_knob_lowercased(monkeypatch):
    import re
    source = open(sac_algo.__file__).read()
    knobs = {name.lower() for name in re.findall(r"tuned\('([A-Z0-9_]+)'", source)}
    keys = set(train.ALGOS['sac'].build_config(lambda name, default, cast=float: default))
    assert keys <= knobs, sorted(keys - knobs)


# ---------------------------------------------------------------- the arithmetic

def test_the_soft_value_of_a_uniform_policy_with_equal_critics_is_the_hand_constant():
    log_pi = torch.full((1, 3), math.log(1.0 / 3.0))
    q = torch.full((1, 3), 2.0)
    # sum_a (1/3) (2 - alpha log(1/3)) = 2 + alpha log 3
    assert float(sac_agent.soft_state_value(log_pi, q, 0.5)) == pytest.approx(2.0 + 0.5 * math.log(3.0))


def test_the_alpha_gradient_flips_sign_as_entropy_crosses_the_target():
    log_alpha = torch.zeros(1, requires_grad=True)
    target = 0.9
    below = sac_agent.alpha_loss(log_alpha, torch.tensor(0.5), target)
    grad_below, = torch.autograd.grad(below, log_alpha)
    above = sac_agent.alpha_loss(log_alpha, torch.tensor(1.05), target)
    grad_above, = torch.autograd.grad(above, log_alpha)
    assert grad_below < 0 < grad_above, 'alpha must rise when entropy is below target and fall above it'


def test_the_actor_loss_is_minimised_by_the_softmax_of_q_over_alpha():
    q = torch.tensor([[1.0, 0.0, -1.0]])
    alpha = 0.5
    best = torch.log_softmax(q / alpha, dim=1)
    best_loss = float(sac_agent.actor_loss(best, q, alpha))
    for other in (torch.log_softmax(torch.tensor([[0.0, 0.0, 0.0]]), dim=1),
                  torch.log_softmax(torch.tensor([[5.0, 0.0, -5.0]]), dim=1),
                  torch.log_softmax(q / (2 * alpha), dim=1)):
        assert float(sac_agent.actor_loss(other, q, alpha)) > best_loss


def test_avg_with_two_equal_critics_equals_min_and_differs_otherwise():
    a, b = torch.tensor([1.0, 3.0]), torch.tensor([1.0, 5.0])
    assert torch.equal(sac_agent.combine(a, a, 'avg'), sac_agent.combine(a, a, 'min'))
    assert torch.equal(sac_agent.combine(a, b, 'avg'), torch.tensor([1.0, 4.0]))
    assert torch.equal(sac_agent.combine(a, b, 'min'), torch.tensor([1.0, 3.0]))
    with pytest.raises(ValueError):
        sac_agent.combine(a, b, 'sum')


def test_the_q_clip_is_the_plain_error_inside_the_clip_and_the_larger_branch_outside():
    y = torch.tensor([0.0, 0.0])
    q_old = torch.tensor([0.0, 0.0])
    q = torch.tensor([0.3, 2.0])          # |q - q_old| = 0.3 < 0.5, and 2.0 > 0.5
    loss, won = sac_agent.clipped_critic_loss(q, q_old, y, 0.5)
    assert float(loss[0]) == pytest.approx(0.09) and not bool(won[0])
    # outside: clipped = q_old + 0.5 = 0.5 -> (0.5)^2 = 0.25 < plain 4.0, so the plain branch wins
    assert float(loss[1]) == pytest.approx(4.0) and not bool(won[1])
    # the clipped branch wins when q has moved *toward* y past the target: q_old far, q close
    loss2, won2 = sac_agent.clipped_critic_loss(torch.tensor([0.1]), torch.tensor([3.0]), torch.tensor([0.0]), 0.5)
    assert float(loss2[0]) == pytest.approx(2.5 ** 2) and bool(won2[0])


def test_the_entropy_penalty_is_zero_for_a_still_policy_and_grows_as_the_square():
    h = torch.tensor(1.0)
    assert float(sac_agent.entropy_penalty(1.0, h, 0.5)) == 0.0
    assert float(sac_agent.entropy_penalty(None, h, 0.5)) == 0.0
    assert float(sac_agent.entropy_penalty(0.8, h, 0.5)) == pytest.approx(0.5 * 0.5 * 0.04)
    assert float(sac_agent.entropy_penalty(0.6, h, 0.5)) == pytest.approx(0.5 * 0.5 * 0.16)


def test_polyak_at_tau_one_is_a_hard_copy_and_below_it_is_not(monkeypatch):
    algo, _, _ = build(monkeypatch, 'sac', SAC_TARGET_UPDATE_PERIOD='1', SAC_TAU='1.0')
    agent = algo.agent
    with torch.no_grad():
        for p in agent.q1.parameters():
            p.add_(1.0)
    agent.train_step = 1
    assert agent.maybe_update_target()
    for a, b in zip(agent.q1.parameters(), agent.q1_target.parameters()):
        assert torch.equal(a, b)
    algo2, _, _ = build(monkeypatch, 'sac', SAC_TARGET_UPDATE_PERIOD='1', SAC_TAU='0.5')
    agent2 = algo2.agent
    before = [p.detach().clone() for p in agent2.q1_target.parameters()]
    with torch.no_grad():
        for p in agent2.q1.parameters():
            p.add_(1.0)
    agent2.train_step = 1
    agent2.maybe_update_target()
    for old, new, live in zip(before, agent2.q1_target.parameters(), agent2.q1.parameters()):
        assert torch.allclose(new, 0.5 * old + 0.5 * live)


# ---------------------------------------------------------------- the seam

@pytest.mark.parametrize('name', NAMES)
def test_an_arm_prefills_advances_updates_and_reports(name, monkeypatch):
    algo, config, arch = build(monkeypatch, name)
    assert algo.prefill() >= 40
    step_before = algo.agent.train_step
    for _ in range(12):
        steps, transitions = algo.advance()
        assert steps == 1 and transitions >= 1
    assert algo.agent.train_step > step_before, 'the replay ratio bought no gradient step'
    fields = algo.fields()
    assert set(fields) == {'alpha', 'sac'} and fields['sac']['entropy'] is not None
    assert set(algo.on_eval([], {})) == {'alpha'}
    assert name in algo.describe() or 'alpha' in algo.describe()
    assert algo.log_note(fields).startswith('alpha')


def test_the_actor_samples_and_the_measured_policy_is_the_argmax(monkeypatch):
    algo, _, _ = build(monkeypatch, 'sac')
    obs = np.zeros((64, constants.OBS_LEN), dtype=np.float32)
    drawn = algo.agent.act(obs, 0.0, False)
    assert drawn.shape == (64,) and drawn.dtype == np.int64
    assert len(set(drawn.tolist())) > 1, 'a near-uniform opening policy must not act deterministically'
    greedy = algo.policy_fn(obs)
    assert len(set(greedy.tolist())) == 1


def test_the_checkpoint_is_the_actor_and_restores_through_the_registry(tmp_path, monkeypatch):
    algo, config, arch = build(monkeypatch, 'sac')
    policy_dir = str(tmp_path / 'arm')
    arch_tools.write_arch(policy_dir, arch)
    checkpoints.save(policy_dir, 7, algo.net)
    policy_fn, read_arch, step = restore.restore(policy_dir, 7)
    assert step == 7 and read_arch['algo'] == 'sac'
    obs = np.zeros((5, constants.OBS_LEN), dtype=np.float32)
    assert np.array_equal(policy_fn(obs), algo.policy_fn(obs))


def test_the_resume_state_round_trips_the_temperature_and_the_critics(monkeypatch):
    algo, _, _ = build(monkeypatch, 'sac')
    algo.prefill()
    for _ in range(8):
        algo.advance()
    state = algo.state_dict()
    other, _, _ = build(monkeypatch, 'sac')
    other.load_state_dict(state)
    assert other.agent.alpha == pytest.approx(algo.agent.alpha)
    assert other.agent.train_step == algo.agent.train_step
    for a, b in zip(other.agent.q2_target.parameters(), algo.agent.q2_target.parameters()):
        assert torch.equal(a, b)
