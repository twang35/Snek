"""The actor, the critic, and the categorical policy's three operations.

The load-bearing claim is the first one: **the PPO actor is the network DQN trains.** If it stops being
that, a PPO-vs-DQN result stops being a statement about the learning rule, and the champion warm-start
in the plan's backlog stops working — both silently, because a differently shaped actor still trains
and still reports numbers.
"""

import math

import numpy as np
import pytest
import torch

from dqn import net as qnet
from env import constants
from ppo import net as network
from tools import arch as arch_tools

LN3 = math.log(3.0)


def arch(algo='ppo', widths=(320,)):
    return arch_tools.build_arch(widths, constants.NUM_ACTIONS, constants.OBS_LEN,
                                 constants.OBS_ERA, algo=algo)


# --- the actor is QNet ---------------------------------------------------------------------------

def test_the_actor_is_the_same_network_dqn_trains_weight_for_weight():
    """**Identical, not merely the same shape**, and asserted at a fixed seed.

    Two consequences ride on this. A seed-matched PPO arm and DQN arm open from the same policy, so a
    difference between them is the learning rule. And the snek2 champion's converted weights load into
    a PPO actor unchanged, which is what lets "can PPO hold a policy DQN found" be asked separately
    from "can PPO find one".
    """
    actor = network.build(arch(), seed=11)
    q = qnet.build(arch(algo='dqn'), seed=11)
    left, right = actor.state_dict(), q.state_dict()
    assert sorted(left) == sorted(right)
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_a_dqn_checkpoint_loads_into_a_ppo_actor_with_strict_on():
    # The mechanical half of the claim above: `checkpoints.load` uses `strict=True`, so this is what a
    # champion warm-start would actually do.
    q = qnet.build(arch(algo='dqn'), seed=3)
    actor = network.build(arch(), seed=99)
    actor.load_state_dict(q.state_dict(), strict=True)
    sample = np.random.default_rng(0).random((5, constants.OBS_LEN)).astype(np.float32)
    with torch.no_grad():
        assert torch.equal(actor(torch.as_tensor(sample)), q(torch.as_tensor(sample)))


def test_the_actor_has_one_output_per_action_and_the_critic_has_one():
    actor = network.build(arch(), seed=1)
    critic = network.build_critic(arch(), seed=1)
    sample = torch.zeros(4, constants.OBS_LEN)
    assert actor(sample).shape == (4, constants.NUM_ACTIONS)
    assert critic(sample).shape == (4, 1)


def test_the_critic_is_not_a_copy_of_the_actor():
    """A shared seed would make them the same draws, which nobody intended.

    `dqn/net.py` draws from a *local* generator precisely so a seed pins the initialisation — which
    means two nets built with the same seed are the same net. The critic's seed is derived for that
    reason, and the derivation is what this asserts.
    """
    actor = network.build(arch(), seed=7)
    critic = network.build_critic(arch(), seed=7)
    assert not torch.equal(actor.hidden[0].weight, critic.hidden[0].weight)


def test_the_derived_critic_seed_is_repeatable_and_moves_with_the_arms_seed():
    assert network.critic_seed(5) == network.critic_seed(5)
    assert network.critic_seed(5) != network.critic_seed(6)
    assert network.critic_seed(None) is None


def test_the_critic_arch_is_never_the_one_written_to_disk():
    """`arch.json` describes the actor, because a checkpoint holds the actor.

    A critic built by mutating the arch in place would corrupt the sidecar the trainer then writes,
    and the corruption is a `num_actions` of 1 — which would make every restore refuse with a message
    about the environment having 3 actions.
    """
    original = arch()
    before = dict(original)
    network.build_critic(original, seed=1)
    assert original == before


# --- the greedy policy ---------------------------------------------------------------------------

def test_the_measured_policy_is_the_argmax_over_the_logits():
    """The analogue of DQN's greedy eval, and the same function object.

    Not a sample: stage A and stage B must measure a deterministic policy, or a 100-episode row and a
    500-episode row would differ by the sampling noise of the policy on top of the noise of the board.
    """
    actor = network.build(arch(), seed=2)
    policy_fn = network.greedy_policy_fn(actor)
    sample = np.random.default_rng(1).random((16, constants.OBS_LEN)).astype(np.float32)
    actions = policy_fn(sample)
    assert actions.shape == (16,)
    assert actions.dtype == np.int64
    with torch.no_grad():
        assert np.array_equal(actions, actor(torch.as_tensor(sample)).argmax(dim=1).numpy())


def test_the_greedy_policy_is_deterministic_across_calls():
    actor = network.build(arch(), seed=2)
    policy_fn = network.greedy_policy_fn(actor)
    sample = np.random.default_rng(1).random((32, constants.OBS_LEN)).astype(np.float32)
    assert np.array_equal(policy_fn(sample), policy_fn(sample))


# --- the categorical policy ----------------------------------------------------------------------

def test_a_uniform_three_way_policy_has_entropy_ln_three():
    """The number a collapsing policy is read against, so it is pinned rather than remembered."""
    logits = torch.zeros(4, 3)
    _, entropy = network.evaluate(logits, torch.zeros(4, dtype=torch.long))
    assert entropy.tolist() == pytest.approx([LN3] * 4, abs=1e-6)


def test_a_deterministic_policy_has_entropy_zero():
    logits = torch.tensor([[100.0, 0.0, 0.0]])
    _, entropy = network.evaluate(logits, torch.zeros(1, dtype=torch.long))
    assert float(entropy[0]) == pytest.approx(0.0, abs=1e-6)


def test_log_probabilities_sum_to_one_in_probability():
    logits = torch.tensor([[1.0, -2.0, 0.5], [0.0, 0.0, 0.0]])
    probs = network.log_softmax(logits).exp()
    assert probs.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0], abs=1e-6)


def test_the_log_probability_of_a_collapsed_action_stays_finite():
    """**Why every read goes through `log_softmax` and never through `softmax` then `log`.**

    `softmax` rounds a probability of ~1e-40 to 0 and the log is then `-inf`; the ratio
    `exp(logp - old_logp)` becomes NaN and poisons every parameter in the minibatch. `log_softmax`
    returns a large negative number, which is arithmetic the update survives.
    """
    # **-200 rather than -100, and the difference is the whole fixture.** At -100 float32 keeps a
    # subnormal 3.7e-44, so `log(softmax(...))` returns -99.98 and the naive spelling looks fine —
    # the first draft of this asserted at -100 and failed for that reason. float32 underflows to 0
    # somewhere past -104, and a trained policy that has committed to one of three actions gets there.
    logits = torch.tensor([[0.0, -200.0, -200.0]])
    logp, _ = network.evaluate(logits, torch.tensor([1]))
    assert torch.isfinite(logp).all()
    assert float(logp[0]) == pytest.approx(-200.0, abs=1e-3)
    # And the naive spelling is genuinely broken here, so the fixture's subject can violate it.
    assert torch.isinf(torch.log(torch.softmax(logits, dim=-1))[0, 1])


def test_sampling_returns_the_log_probability_of_the_action_it_drew():
    """The pair PPO's ratio needs. Recomputing it later is a different guarantee — see the agent."""
    logits = torch.tensor([[1.0, -2.0, 0.5]] * 64)
    generator = torch.Generator().manual_seed(4)
    actions, log_probs = network.sample(logits, generator=generator)
    expected = network.log_softmax(logits).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(log_probs, expected)


def test_sampling_covers_every_action_and_follows_the_logits():
    """A sampler that always returned the argmax would pass a shape check and nothing else."""
    logits = torch.tensor([[1.0, 0.0, -1.0]] * 3000)
    actions, _ = network.sample(logits, generator=torch.Generator().manual_seed(5))
    counts = np.bincount(actions.numpy(), minlength=3)
    assert (counts > 0).all(), counts
    # Ordered as the logits are, with wide margins so this cannot fail on sampling noise.
    assert counts[0] > counts[1] > counts[2]


def test_sampling_is_reproducible_from_its_own_generator():
    """And therefore independent of torch's global RNG, which the env and the tests both touch."""
    logits = torch.tensor([[0.4, 0.1, -0.3]] * 50)
    first, _ = network.sample(logits, generator=torch.Generator().manual_seed(8))
    torch.manual_seed(12345)
    second, _ = network.sample(logits, generator=torch.Generator().manual_seed(8))
    assert torch.equal(first, second)
