"""`DistAgent` and the Munchausen branch of `DdqnAgent`. `algos/dist/agent.py`, `algos/dqn/agent.py`."""

import numpy as np
import pytest
import torch

from algos.dist import net as network
from algos.dist.agent import DistAgent
from algos.dqn.agent import DdqnAgent
from env import constants
from tools import arch as arch_tools


def an_arch(head=None, algo='dqn'):
    return arch_tools.build_arch([32], constants.NUM_ACTIONS, constants.OBS_LEN, constants.OBS_ERA,
                                 algo=algo, head=head)


HEADS = {
    'c51': {'type': 'c51', 'atoms': 11, 'v_min': -10.0, 'v_max': 110.0},
    'qrdqn': {'type': 'quantile', 'n': 8},
    'iqn': {'type': 'iqn', 'embedding': 16, 'n_tau': 8, 'k': 4},
    'fqf': {'type': 'fqf', 'embedding': 16, 'n': 8},
}


def a_batch(m=16, seed=0):
    rng = np.random.default_rng(seed)
    done = rng.random(m) < 0.25
    return {'obs': rng.random((m, constants.OBS_LEN)).astype(np.float32),
            'next_obs': rng.random((m, constants.OBS_LEN)).astype(np.float32),
            'action': rng.integers(0, constants.NUM_ACTIONS, m).astype(np.int64),
            'reward': np.where(done, rng.choice([-5.0, 100.0], m), rng.choice([0.0, 1.0], m)).astype(np.float32),
            'discount': np.where(done, 0.0, 0.99).astype(np.float32)}


def tensors(batch):
    return tuple(torch.as_tensor(batch[k]) for k in ('obs', 'action', 'reward', 'discount', 'next_obs'))


# ---------------------------------------------------------------- Munchausen on the scalar agent

def test_with_alpha_zero_the_target_is_the_double_dqn_target_to_the_bit():
    plain = DdqnAgent(an_arch(), seed=1)
    knobbed = DdqnAgent(an_arch(), seed=1, munchausen_alpha=0.0, munchausen_tau=0.03)
    obs, action, reward, discount, next_obs = tensors(a_batch())
    with torch.no_grad():
        best = plain.net(next_obs).argmax(dim=1, keepdim=True)
        expected = reward + discount * plain.target(next_obs).gather(1, best).squeeze(1)
        assert torch.equal(plain.scalar_target(obs, action, reward, discount, next_obs), expected)
        assert torch.equal(knobbed.scalar_target(obs, action, reward, discount, next_obs), expected)


def test_the_munchausen_target_carries_the_clipped_log_policy_and_the_soft_value():
    agent = DdqnAgent(an_arch(), seed=1, munchausen_alpha=0.9, munchausen_tau=0.03, munchausen_l0=-1.0)
    obs, action, reward, discount, next_obs = tensors(a_batch())
    tau = 0.03
    with torch.no_grad():
        q_s = agent.target(obs)
        log_pi_s = tau * torch.log_softmax(q_s / tau, dim=1)
        m = torch.clamp(log_pi_s.gather(1, action.unsqueeze(1)).squeeze(1), min=-1.0, max=0.0)
        q_next = agent.target(next_obs)
        pi = torch.softmax(q_next / tau, dim=1)
        log_pi_next = tau * torch.log_softmax(q_next / tau, dim=1)
        soft = (pi * (q_next - log_pi_next)).sum(dim=1)
        expected = reward + 0.9 * m + discount * soft
        got = agent.scalar_target(obs, action, reward, discount, next_obs)
    assert torch.allclose(got, expected, atol=1e-5)
    # The log-policy term is never positive and never below l0.
    assert float(m.max()) <= 0.0 and float(m.min()) >= -1.0


def test_the_soft_value_at_a_terminal_is_zero_because_discount_is():
    agent = DdqnAgent(an_arch(), seed=1, munchausen_alpha=0.9)
    obs, action, reward, discount, next_obs = tensors(a_batch())
    discount = torch.zeros_like(discount)
    with torch.no_grad():
        got = agent.scalar_target(obs, action, reward, discount, next_obs)
        q_s = agent.target(obs)
        m = torch.clamp(0.03 * torch.log_softmax(q_s / 0.03, dim=1).gather(1, action.unsqueeze(1)).squeeze(1),
                        min=-1.0, max=0.0)
    assert torch.allclose(got, reward + 0.9 * m, atol=1e-6)


def test_munchausen_knobs_are_validated():
    with pytest.raises(ValueError):
        DdqnAgent(an_arch(), munchausen_alpha=1.5)
    with pytest.raises(ValueError):
        DdqnAgent(an_arch(), munchausen_alpha=0.9, munchausen_tau=0.0)


def test_a_munchausen_update_runs_and_returns_td_errors_in_batch_shape():
    agent = DdqnAgent(an_arch(), seed=1, munchausen_alpha=0.9, learning_rate=1e-3)
    td, metrics = agent.update(a_batch())
    assert td.shape == (16,) and 'loss' in metrics


# ---------------------------------------------------------------- DistAgent

@pytest.mark.parametrize('rung', sorted(HEADS))
def test_every_rung_updates_and_its_priorities_are_finite_per_sample(rung):
    agent = DistAgent(an_arch(HEADS[rung], algo=rung), seed=1, learning_rate=1e-3)
    td, metrics = agent.update(a_batch())
    assert td.shape == (16,) and np.isfinite(td).all()
    assert metrics['train_step'] == 1 and np.isfinite(metrics['loss'])


@pytest.mark.parametrize('rung', sorted(HEADS))
def test_repeated_updates_on_one_batch_lower_its_loss(rung):
    torch.manual_seed(0)
    agent = DistAgent(an_arch(HEADS[rung], algo=rung), seed=1, learning_rate=3e-3,
                      target_update_period=0)
    batch = a_batch()
    # Windows rather than single updates: IQN draws fresh fractions every update, so one loss value is
    # a sample of the loss and two of them can go the wrong way by chance.
    trace = [agent.update(batch)[1]['loss'] for _ in range(80)]
    early, late = sum(trace[:10]) / 10, sum(trace[-10:]) / 10
    assert late < early, (rung, early, late)


@pytest.mark.parametrize('rung', sorted(HEADS))
def test_the_target_copy_and_the_resume_state_round_trip(rung):
    agent = DistAgent(an_arch(HEADS[rung], algo=rung), seed=1, target_update_period=1)
    agent.update(a_batch())
    assert all(torch.equal(a, b) for a, b in zip(agent.net.parameters(), agent.target.parameters()))
    state = agent.state_dict()
    fresh = DistAgent(an_arch(HEADS[rung], algo=rung), seed=9)
    fresh.load_state_dict(state)
    assert all(torch.equal(a, b) for a, b in zip(agent.net.parameters(), fresh.net.parameters()))
    if rung == 'fqf':
        assert 'fraction_optimizer' in state


def test_the_agent_builds_the_head_the_arch_names_and_its_greedy_policy_is_the_mean_read():
    agent = DistAgent(an_arch(HEADS['c51'], algo='c51'), seed=1)
    assert isinstance(agent.net, network.CategoricalNet)
    obs = a_batch()['obs']
    with torch.no_grad():
        expected = agent.net.q_values(torch.as_tensor(obs)).argmax(dim=1).numpy()
    assert (agent.policy_fn(obs) == expected).all()
    assert (agent.greedy_actions(obs) == expected).all()


def test_the_value_loss_does_not_move_fqf_fraction_net_and_its_own_loss_does():
    torch.manual_seed(0)
    agent = DistAgent(an_arch(HEADS['fqf'], algo='fqf'), seed=1, learning_rate=1e-3, fraction_lr=1e-2)
    before_fraction = agent.net.fraction.weight.detach().clone()
    before_head = agent.net.qnet.head.weight.detach().clone()
    # One update with the fraction optimiser silenced: the fraction net must not move.
    agent.fraction_optimizer = None
    agent.update(a_batch())
    assert torch.equal(agent.net.fraction.weight, before_fraction)
    assert not torch.equal(agent.net.qnet.head.weight, before_head)
    # Restore it: now the fraction net moves on its own loss.
    agent.fraction_optimizer = torch.optim.RMSprop(agent.net.fraction.parameters(), lr=1e-2)
    agent.update(a_batch(seed=1))
    assert not torch.equal(agent.net.fraction.weight, before_fraction)


def test_fraction_parameters_are_not_in_the_value_optimizer():
    agent = DistAgent(an_arch(HEADS['fqf'], algo='fqf'), seed=1)
    value_params = {id(p) for group in agent.optimizer.param_groups for p in group['params']}
    assert not any(id(p) in value_params for p in agent.net.fraction.parameters())
    assert id(agent.net.qnet.head.weight) in value_params


def test_risk_train_takes_the_target_action_under_the_cvar_read():
    torch.manual_seed(0)
    neutral = DistAgent(an_arch(HEADS['qrdqn'], algo='qrdqn'), seed=1, risk_alpha=0.25, risk_train=False)
    risky = DistAgent(an_arch(HEADS['qrdqn'], algo='qrdqn'), seed=1, risk_alpha=0.25, risk_train=True)
    next_obs = torch.as_tensor(a_batch(m=64)['next_obs'])
    with torch.no_grad():
        a_neutral = neutral._double_q_target_action(next_obs)
        a_risky = risky._double_q_target_action(next_obs)
        assert torch.equal(a_neutral, neutral.net.q_values(next_obs).argmax(dim=1))
        assert torch.equal(a_risky, risky.net.cvar_values(next_obs, 0.25).argmax(dim=1))
    # And the acting policy during training follows the same read.
    obs = a_batch(m=64)['obs']
    with torch.no_grad():
        assert (risky.greedy_actions(obs) == risky.net.cvar_values(torch.as_tensor(obs), 0.25).argmax(dim=1).numpy()).all()
    # The measured (stage-A) policy stays the mean read whatever `risk_train` says.
    with torch.no_grad():
        assert (risky.policy_fn(obs) == risky.net.q_values(torch.as_tensor(obs)).argmax(dim=1).numpy()).all()


def test_the_munchausen_mixture_weights_are_the_target_soft_policy_and_shifts_its_log_policy():
    agent = DistAgent(an_arch(HEADS['qrdqn'], algo='qrdqn'), seed=1, munchausen_alpha=0.9,
                      munchausen_tau=0.03, munchausen_l0=-1.0)
    obs, action, reward, discount, next_obs = tensors(a_batch())
    with torch.no_grad():
        weights, shifts, munchausen = agent._next_action_mixture(obs, action, next_obs)
        q_next = agent.target.q_values(next_obs)
        assert torch.allclose(weights, torch.softmax(q_next / 0.03, dim=1), atol=1e-6)
        assert torch.allclose(shifts, -0.03 * torch.log_softmax(q_next / 0.03, dim=1), atol=1e-6)
        assert torch.allclose(weights.sum(dim=1), torch.ones(16), atol=1e-6)
        assert float(munchausen.max()) <= 0.0 and float(munchausen.min()) >= -0.9


def test_without_munchausen_the_mixture_is_one_hot_on_the_online_argmax_with_no_shift():
    agent = DistAgent(an_arch(HEADS['c51'], algo='c51'), seed=1)
    obs, action, reward, discount, next_obs = tensors(a_batch())
    with torch.no_grad():
        weights, shifts, munchausen = agent._next_action_mixture(obs, action, next_obs)
        best = agent.net.q_values(next_obs).argmax(dim=1)
    assert torch.equal(weights.argmax(dim=1), best) and torch.all(weights.sum(dim=1) == 1.0)
    assert torch.all(shifts == 0.0) and torch.all(munchausen == 0.0)


def test_the_categorical_update_at_a_terminal_fits_the_reward_alone():
    """The loss the agent computes on a terminal batch equals the cross-entropy against the reward
    projected on its own -- the discount must zero the bootstrap, whatever s' says."""
    from algos.dist import losses
    agent = DistAgent(an_arch(HEADS['c51'], algo='c51'), seed=1)
    batch = a_batch(m=4)
    batch['reward'][:] = -5.0
    batch['discount'][:] = 0.0
    obs, action, reward, discount, next_obs = tensors(batch)
    got = agent._categorical_update(obs, action, reward, discount, next_obs)
    with torch.no_grad():
        support = agent.net.support
        mass = torch.zeros(4, agent.net.atoms)
        mass[:, 0] = 1.0                       # any distribution: it is scaled to a point by discount 0
        target = losses.project(mass, reward.view(4, 1).expand(4, agent.net.atoms), support)
        log_probs = agent.net.log_probs(obs).gather(
            1, action.view(-1, 1, 1).expand(-1, 1, agent.net.atoms)).squeeze(1)
        expected = losses.categorical_cross_entropy(log_probs, target)
    assert torch.allclose(got, expected, atol=1e-5)


def test_the_categorical_target_at_a_terminal_is_the_reward_projected():
    """A death (-5, discount 0) must put all target mass at the atom nearest -5, whatever s' says."""
    agent = DistAgent(an_arch(HEADS['c51'], algo='c51'), seed=1)
    batch = a_batch(m=4)
    batch['reward'][:] = -5.0
    batch['discount'][:] = 0.0
    obs, action, reward, discount, next_obs = tensors(batch)
    with torch.no_grad():
        # Reproduce the target the update builds.
        weights, shifts, munchausen = agent._next_action_mixture(obs, action, next_obs)
        target_probs = agent.target.probs(next_obs)
        support = agent.net.support
        b, a, n = target_probs.shape
        values = (reward + munchausen).view(b, 1, 1) + discount.view(b, 1, 1) * (support.view(1, 1, n) + shifts.view(b, a, 1))
        from algos.dist import losses
        projected = losses.project(target_probs.reshape(b * a, n), values.reshape(b * a, n), support)
        target = (projected.view(b, a, n) * weights.unsqueeze(2)).sum(dim=1)
    # Support -10..110 in 11 atoms is a 12-wide bin: -5 sits 5/12 of the way from -10 to 2.
    assert torch.allclose(target[:, 0], torch.full((4,), 7 / 12), atol=1e-5)
    assert torch.allclose(target[:, 1], torch.full((4,), 5 / 12), atol=1e-5)
    assert torch.allclose(target[:, 2:].sum(dim=1), torch.zeros(4), atol=1e-6)
