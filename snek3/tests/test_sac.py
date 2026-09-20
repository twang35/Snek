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


def test_the_entropy_penalty_is_per_state_zero_when_still_and_not_cancelled_by_opposite_moves():
    """Zhou et al.'s term is E_s[(H_old(s) - H(s))^2]: two states whose entropies moved by the same amount in
    opposite directions must be penalised, where a mean-to-mean difference would read zero."""
    old = torch.tensor([1.0, 0.6])
    assert torch.all(sac_agent.entropy_penalty(old, old.clone(), 0.5) == 0.0)
    assert torch.all(sac_agent.entropy_penalty(old, torch.tensor([0.8, 0.8]), 0.0) == 0.0), 'beta 0 is off'
    per_row = sac_agent.entropy_penalty(old, torch.tensor([0.8, 0.8]), 0.5)
    assert per_row.tolist() == pytest.approx([0.5 * 0.5 * 0.04, 0.5 * 0.5 * 0.04])
    assert float(per_row.mean()) > 0.0 and float((old.mean() - 0.8) ** 2) == pytest.approx(0.0)
    # the gradient flows to the current entropy, not the stored one
    h = torch.tensor([0.9, 0.7], requires_grad=True)
    sac_agent.entropy_penalty(old, h, 0.5).sum().backward()
    assert h.grad.tolist() == pytest.approx([0.5 * (0.9 - 1.0), 0.5 * (0.7 - 0.6)])


def _entropy_of(actor, obs):
    with torch.no_grad():
        log_pi = torch.log_softmax(actor(torch.as_tensor(np.asarray(obs, dtype=np.float32))), dim=1)
        return (-(log_pi.exp() * log_pi).sum(dim=1)).numpy()


def test_collection_stores_each_states_entropy_and_the_replay_returns_it_with_the_transition(monkeypatch):
    algo, config, _ = build(monkeypatch, 'sac2', SAC_N_STEP='1')
    banked = algo.collector.step(0.0)
    assert banked == config['collect_envs']
    stored = algo.buffer.aux[:banked]
    expected = _entropy_of(algo.agent.actor, algo.buffer.obs[:banked])
    assert stored == pytest.approx(expected, abs=1e-5), 'the collecting policy\'s entropy at that state'
    assert stored.std() >= 0.0 and np.all(stored > 0.0)
    algo.prefill()
    batch, indexes, _ = algo.buffer.sample(8, 0)
    assert batch['aux'] == pytest.approx(algo.buffer.aux[indexes])
    # and the penalty in an update reads it: with the actor unchanged since collection it is ~0
    td, metrics = algo.agent.update(batch, None)
    assert metrics['entropy_penalty'] == pytest.approx(0.0, abs=1e-6)
    # after the actor moves, the same rows carry a positive penalty
    with torch.no_grad():
        for p in algo.agent.actor.parameters():
            p.add_(0.5)
    _, metrics2 = algo.agent.update(batch, None)
    assert metrics2['entropy_penalty'] > 0.0


def test_the_critic_loss_is_mse_by_default_huber_by_knob_and_refuses_anything_else(monkeypatch):
    assert small_config(monkeypatch, 'sac')['sac_critic_loss'] == 'mse'
    assert small_config(monkeypatch, 'sac2')['sac_critic_loss'] == 'mse'
    assert small_config(monkeypatch, 'sac', SAC_CRITIC_LOSS='Huber')['sac_critic_loss'] == 'huber'
    with pytest.raises(ValueError, match='SAC_CRITIC_LOSS'):
        small_config(monkeypatch, 'sac', SAC_CRITIC_LOSS='l1')
    monkeypatch.delenv('SNEK_SAC_CRITIC_LOSS')
    # a large error is quadratic under mse and linear under huber: two agents, one batch, the critic loss differs.
    # The mse agent is built first: monkeypatch leaves the knob set for the rest of the test.
    algo_mse, _, _ = build(monkeypatch, 'sac')
    assert 'critic mse' in algo_mse.describe()
    algo, _, _ = build(monkeypatch, 'sac', SAC_CRITIC_LOSS='huber')
    assert 'critic huber' in algo.describe()
    algo.prefill(); algo_mse.prefill()
    batch, _, _ = algo.buffer.sample(8, 0)
    batch['reward'] = batch['reward'] + 100.0
    _, m_h = algo.agent.update(dict(batch), None)
    _, m_m = algo_mse.agent.update(dict(batch), None)
    assert m_m['critic_loss'] > 10 * m_h['critic_loss']


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
