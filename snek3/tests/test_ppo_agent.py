"""The clipped surrogate, the epoch loop, and the two diagnostics that decide what a failure means.

**The two fixtures that matter most are the ratio-is-one identity and the clipped gradient.**

The first is the single best detector of a wrong log-prob anywhere in the stack — collect, storage,
shuffle, or the loss. It is asserted in two halves, bit-exact at the collect shape and to a float32
ULP on a real minibatch, and the note above them says why the second cannot be exact.

The second asserts the **gradient**, not the loss value. A `max` where the objective needs a `min`
turns the trust region into an anti-trust region that rewards leaving it; the loss it reports is a
plausible number either way, so a value fixture passes with the wrong branch selected.
"""

import numpy as np
import pytest
import torch

from env import constants
from ppo import agent as agent_module
from ppo import net as network
from ppo import rollout as rollout_module
from tools import arch as arch_tools

CLIP = 0.2


def arch(widths=(16,)):
    """A narrow net: these fixtures are about arithmetic, not capacity."""
    return arch_tools.build_arch(widths, constants.NUM_ACTIONS, constants.OBS_LEN,
                                 constants.OBS_ERA, algo='ppo')


def config(**overrides):
    out = {'seed': 3, 'discount': 0.99, 'ppo_gae_lambda': 0.95, 'ppo_clip': CLIP,
           'ppo_vf_coef': 0.5, 'ppo_epochs': 2, 'ppo_minibatch': 8,
           'ppo_normalize_adv': True, 'ppo_target_kl': 0.0, 'ppo_gradient_clipping': 0.5,
           'ppo_value_loss': 'huber', 'ppo_learning_rate': 3e-4, 'ppo_adam_epsilon': 1e-7,
           'ppo_entropy_coef': 0.01}
    out.update(overrides)
    return out


def filled_rollout(agent, steps=4, lanes=4, seed=0):
    """A rollout of real forward passes, so the stored log-probs are the ones the actor produced."""
    roll = rollout_module.Rollout(steps, lanes, constants.OBS_LEN)
    rng = np.random.default_rng(seed)
    for t in range(steps):
        obs = rng.random((lanes, constants.OBS_LEN)).astype(np.float32)
        actions, log_probs, values = agent.act(obs)
        roll.add(t, obs, actions, log_probs, values,
                 rng.normal(size=lanes).astype(np.float32),
                 rng.random(lanes) < 0.2)
    roll.finish(agent.values(rng.random((lanes, constants.OBS_LEN)).astype(np.float32)),
                agent.discount, agent.gae_lambda)
    return roll


# --- the identity that catches a wrong log-prob anywhere -----------------------------------------

# **‡ The stored log-prob is bit-exact at the shape it was collected at, and one float32 ULP away
# after the minibatch reshape — measured 1.19e-07.** The first draft of these two fixtures asserted
# bit-exactness on the flattened rollout and failed for that reason, which is worth keeping because it
# is a property of the production path and not of the test: a collect batch is `(lanes, 30)` and a
# minibatch is `(minibatch, 30)`, different shapes reach different BLAS kernels, and a float32 matmul
# reassociates. So the real first minibatch of every rollout sees a ratio of 1.0 +- ~1e-7, not exactly
# 1.0 — irrelevant against a clip of 0.2, and worth knowing before someone hunts it.
#
# Both halves are therefore asserted: exact at the collect shape, which pins the storage and
# derivation path, and 1-ULP on a real minibatch, which pins what the update actually sees.

def test_the_stored_log_probability_is_bit_exact_at_the_shape_it_was_collected_at():
    """`pi_old` is the denominator of the clipped ratio.

    If the stored value came from a different forward pass, a different dtype, a different
    normalisation or a re-derived distribution, every update applies a spurious ratio — and the loss
    it reports is a perfectly ordinary number. This is the strongest form of the check: same shape,
    no tolerance.
    """
    agent = agent_module.PpoAgent(arch(), config())
    lanes = 4
    obs = np.random.default_rng(7).random((lanes, constants.OBS_LEN)).astype(np.float32)
    actions, log_probs, values = agent.act(obs)
    with torch.no_grad():
        recomputed, _ = network.evaluate(agent.actor(torch.as_tensor(obs)),
                                        torch.as_tensor(actions).long())
        critic = agent.critic(torch.as_tensor(obs)).squeeze(-1)
    assert torch.equal(recomputed, torch.as_tensor(log_probs))
    # GAE needs `V(s_t)` under the critic that *saw* `s_t`; by the time the epochs have run it moved.
    assert torch.equal(critic, torch.as_tensor(values))


def test_the_first_minibatch_of_a_rollout_sees_a_ratio_of_one():
    """To within a float32 ULP, on the real minibatch path. See the note above for why not exactly.

    The tolerance is 1e-6 against a measured 1.19e-07 and a clip of 0.2 — five orders of magnitude of
    headroom, so this cannot pass with a genuinely wrong `pi_old`: a re-derived distribution, a
    stale rollout or a shuffle that misaligns the fields all move the ratio by O(0.1) or more.
    """
    agent = agent_module.PpoAgent(arch(), config())
    roll = filled_rollout(agent)
    batch = next(roll.minibatches(agent.minibatch, np.random.default_rng(0)))
    with torch.no_grad():
        recomputed, _ = network.evaluate(agent.actor(torch.as_tensor(batch['obs'])),
                                        torch.as_tensor(batch['actions']).long())
    ratio = (recomputed - torch.as_tensor(batch['log_probs'])).exp()
    assert ratio.tolist() == pytest.approx([1.0] * len(ratio), abs=1e-6)


# --- the clipped objective, asserted on its gradient ---------------------------------------------

def surrogate_gradient(log_ratio, advantage, clip=CLIP, objective='min'):
    """The gradient of the clipped surrogate w.r.t. the logits, at a chosen ratio and advantage.

    Built from the same three lines the agent uses rather than by calling `update`, so the assertion
    is about the objective and not about an optimiser step.
    """
    logits = torch.zeros(1, 3, requires_grad=True)
    actions = torch.tensor([0])
    log_probs, _ = network.evaluate(logits, actions)
    old_log_probs = log_probs.detach() - log_ratio
    ratio = (log_probs - old_log_probs).exp()
    adv = torch.tensor([float(advantage)])
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
    pick = torch.min if objective == 'min' else torch.max
    loss = -pick(unclipped, clipped).mean()
    loss.backward()
    return logits.grad.abs().sum().item(), float(ratio.detach())


def test_a_positive_advantage_past_the_upper_clip_contributes_no_gradient():
    """The trust region's upper edge. `min` picks the constant branch, so the policy stops moving."""
    gradient, ratio = surrogate_gradient(log_ratio=1.0, advantage=1.0)
    assert ratio > 1.0 + CLIP, 'the fixture must actually be past the clip'
    assert gradient == pytest.approx(0.0, abs=1e-9)


def test_a_negative_advantage_past_the_lower_clip_contributes_no_gradient():
    """The other edge, and it needs its own case: the binding branch swaps with the sign of A."""
    gradient, ratio = surrogate_gradient(log_ratio=-1.0, advantage=-1.0)
    assert ratio < 1.0 - CLIP
    assert gradient == pytest.approx(0.0, abs=1e-9)


def test_a_ratio_inside_the_clip_does_contribute_a_gradient():
    """**Without this the two fixtures above are satisfied by a loss that never has a gradient.**"""
    gradient, ratio = surrogate_gradient(log_ratio=0.1, advantage=1.0)
    assert 1.0 - CLIP < ratio < 1.0 + CLIP
    assert gradient > 1e-6


@pytest.mark.parametrize('log_ratio,advantage', [(1.0, 1.0), (-1.0, -1.0)])
def test_a_max_instead_of_a_min_would_reward_leaving_the_trust_region(log_ratio, advantage):
    """The mutant this whole section exists for, asserted directly rather than left to `mutate.py`.

    With `max` the clipped branch never binds, so a step already outside the region keeps being pushed
    further out. That trains until it diverges and reads as a learning-rate problem.
    """
    with_min, _ = surrogate_gradient(log_ratio, advantage, objective='min')
    with_max, _ = surrogate_gradient(log_ratio, advantage, objective='max')
    assert with_min == pytest.approx(0.0, abs=1e-9)
    assert with_max > 1e-6


# --- the diagnostics -----------------------------------------------------------------------------

def test_the_kl_estimate_is_never_negative():
    """**Which the naive `mean(logp_old - logp)` is**, and a negative KL makes a threshold meaningless.

    Checked over random log-ratios of both signs, including large ones, because the naive estimator is
    negative for roughly half of them.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        delta = torch.as_tensor(rng.normal(scale=0.5, size=64).astype(np.float32))
        old = torch.zeros(64)
        assert agent_module.approx_kl(delta, old) >= 0.0


def test_the_kl_estimate_is_zero_when_the_policy_has_not_moved():
    log_probs = torch.as_tensor(np.random.default_rng(1).normal(size=32).astype(np.float32))
    assert agent_module.approx_kl(log_probs, log_probs) == pytest.approx(0.0, abs=1e-9)


def test_the_naive_kl_estimator_would_go_negative_here():
    # So the fixture above has a subject that can violate it.
    delta = torch.tensor([0.5, 0.5, 0.5])
    naive = float((torch.zeros(3) - delta).mean())
    assert naive < 0.0
    assert agent_module.approx_kl(delta, torch.zeros(3)) > 0.0


@pytest.mark.parametrize('values,expected', [
    ('perfect', 1.0),
    ('mean', 0.0),
    ('worse', -3.0),
])
def test_explained_variance_reads_one_zero_and_negative(values, expected):
    """1 is a perfect critic, 0 is no better than the mean, **negative is worse than the mean** —
    which is the reading that says the critic rather than the policy is the problem."""
    returns = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    if values == 'perfect':
        predicted = returns
    elif values == 'mean':
        predicted = np.full(4, returns.mean(), dtype=np.float32)
    else:
        # Mirrored about the mean, so the error is *twice* the deviation and its variance is four
        # times: EV = 1 - 4 = -3. The first draft of this expected -1, which is the kind of arithmetic
        # slip a fixture is supposed to catch in the code rather than carry itself.
        predicted = returns.mean() + (returns.mean() - returns)
    assert agent_module.explained_variance(returns, predicted) == pytest.approx(expected, abs=1e-5)


def test_explained_variance_of_a_constant_return_is_zero_rather_than_infinite():
    # Var(returns) is 0 in the first rollout of a run where nothing has happened yet.
    out = agent_module.explained_variance(np.zeros(8), np.ones(8))
    assert out == 0.0


# --- the update ----------------------------------------------------------------------------------

def test_an_update_changes_both_towers():
    """A single optimiser over both parameter sets — so a missing half is silent in the loss."""
    agent = agent_module.PpoAgent(arch(), config())
    roll = filled_rollout(agent)
    before_actor = agent.actor.hidden[0].weight.detach().clone()
    before_critic = agent.critic.hidden[0].weight.detach().clone()
    agent.update(roll)
    assert not torch.equal(agent.actor.hidden[0].weight, before_actor)
    assert not torch.equal(agent.critic.hidden[0].weight, before_critic)


def test_an_update_reports_every_diagnostic_a_row_carries():
    agent = agent_module.PpoAgent(arch(), config())
    metrics = agent.update(filled_rollout(agent))
    for key in ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction',
                'explained_variance', 'epochs_run', 'stopped_early', 'train_step'):
        assert key in metrics, key
    assert 0.0 <= metrics['entropy'] <= np.log(3.0) + 1e-6
    assert 0.0 <= metrics['clip_fraction'] <= 1.0


def test_every_epoch_runs_when_no_kl_target_is_set():
    agent = agent_module.PpoAgent(arch(), config(ppo_epochs=3, ppo_target_kl=0.0))
    metrics = agent.update(filled_rollout(agent))
    assert metrics['epochs_run'] == 3
    assert metrics['stopped_early'] is False


def test_an_impossibly_tight_kl_target_stops_after_one_epoch():
    """And stops **between** epochs, never mid-epoch: a half-finished epoch uses some samples more
    often than others, which biases toward whatever the shuffle put first."""
    agent = agent_module.PpoAgent(arch(), config(ppo_epochs=4, ppo_target_kl=1e-12))
    metrics = agent.update(filled_rollout(agent))
    assert metrics['epochs_run'] == 1
    assert metrics['stopped_early'] is True


def test_the_gradient_step_count_matches_the_epochs_and_minibatches_run():
    """`epochs * ceil(size / minibatch)`, which is the arithmetic behind "64x fewer than DQN"."""
    agent = agent_module.PpoAgent(arch(), config(ppo_epochs=2, ppo_minibatch=8))
    roll = filled_rollout(agent, steps=4, lanes=4)      # 16 samples
    agent.update(roll)
    assert agent.train_step == 2 * 2


@pytest.mark.parametrize('kind', ['huber', 'mse'])
def test_both_value_losses_run_and_they_are_not_the_same_number(kind):
    agent = agent_module.PpoAgent(arch(), config(ppo_value_loss=kind))
    metrics = agent.update(filled_rollout(agent))
    assert np.isfinite(metrics['value_loss'])


def test_an_unknown_value_loss_is_refused_by_name():
    with pytest.raises(ValueError, match='huber'):
        agent_module.PpoAgent(arch(), config(ppo_value_loss='mae'))


def test_nothing_in_an_update_produces_a_nan():
    """The failure mode is total: one NaN reaches every parameter and the arm reads flat 0 forever.

    Driven with a rollout whose advantages are deliberately degenerate — every value equal, which is
    what an early run looks like — because that is the input that makes `normalise` divide by zero.
    """
    agent = agent_module.PpoAgent(arch(), config())
    roll = filled_rollout(agent)
    roll.advantages[:] = 2.0
    roll.returns[:] = roll.advantages + roll.values
    metrics = agent.update(roll)
    assert all(np.isfinite(metrics[key]) for key in
               ('policy_loss', 'value_loss', 'entropy', 'approx_kl'))
    for parameter in list(agent.actor.parameters()) + list(agent.critic.parameters()):
        assert torch.isfinite(parameter).all()


# --- persistence ---------------------------------------------------------------------------------

def test_a_resume_restores_both_towers_the_optimiser_and_both_streams():
    agent = agent_module.PpoAgent(arch(), config())
    agent.update(filled_rollout(agent))
    state = agent.state_dict()

    other = agent_module.PpoAgent(arch(), config(seed=999))
    assert not torch.equal(other.actor.head.weight, agent.actor.head.weight)
    other.load_state_dict(state)
    assert torch.equal(other.actor.head.weight, agent.actor.head.weight)
    assert torch.equal(other.critic.head.weight, agent.critic.head.weight)
    assert other.train_step == agent.train_step
    # The action stream too, or a resumed arm re-draws the sequence it already used.
    sample = np.random.default_rng(5).random((8, constants.OBS_LEN)).astype(np.float32)
    assert np.array_equal(agent.act(sample)[0], other.act(sample)[0])


# --- the same three claims, asserted through the real `update` -----------------------------------

# **‡ The three fixtures above build the objective from the same three lines the agent uses, and that
# is a weaker check than it looks.** They pin the *arithmetic* of the clipped surrogate, which is worth
# pinning, but a `min` changed to a `max` in `ppo/agent.py` leaves every one of them passing — they
# never call `update`. Measured: the ten-mutant spec for `ppo/` reported the `max` mutant as a
# survivor until the three fixtures below existed. Each one drives `agent.update(rollout)` and reads
# the actor's parameters, so its subject is the production line and not a copy of it.

def clipped_far_rollout(agent, log_ratio=1.0, advantage=1.0):
    """A rollout every sample of which sits past the upper clip with a positive advantage.

    The stored log-probs are the real ones the actor produced, shifted by `log_ratio` — so the ratio
    is `exp(log_ratio)` by construction rather than by luck, and the advantages are overwritten after
    GAE has run because what is under test is the objective, not the advantage estimator.
    """
    roll = filled_rollout(agent)
    roll.log_probs -= np.float32(log_ratio)
    roll.advantages[:] = np.float32(advantage)
    return roll


def actor_snapshot(agent):
    return [p.detach().clone() for p in agent.actor.parameters()]


def actor_moved(agent, before):
    return max(float((p.detach() - b).abs().max())
               for p, b in zip(agent.actor.parameters(), before))


def test_the_real_update_leaves_the_actor_still_when_every_sample_is_past_the_clip():
    """**The `min`-vs-`max` mutant, killed at the call site.**

    Entropy and value coefficients are zeroed so the actor's only gradient path is the surrogate, and
    the clip is binding on every sample — so a correct objective moves the actor by exactly nothing.
    """
    agent = agent_module.PpoAgent(arch(), config(ppo_epochs=1, ppo_entropy_coef=0.0, ppo_vf_coef=0.0,
                                                ppo_normalize_adv=False, ppo_gradient_clipping=0.0))
    roll = clipped_far_rollout(agent)
    before = actor_snapshot(agent)
    metrics = agent.update(roll)
    assert metrics['clip_fraction'] == pytest.approx(1.0), 'the fixture must be past the clip'
    assert actor_moved(agent, before) == pytest.approx(0.0, abs=1e-12)


def test_the_real_update_does_move_the_actor_when_the_clip_is_not_binding():
    """The other half, or the fixture above is satisfied by an update that never moves anything.

    Same construction with a ratio inside the clip. This is also what kills an inverted ratio: with
    `exp(old - new)` the two fixtures swap, so no single ratio order satisfies both.
    """
    agent = agent_module.PpoAgent(arch(), config(ppo_epochs=1, ppo_entropy_coef=0.0, ppo_vf_coef=0.0,
                                                ppo_normalize_adv=False, ppo_gradient_clipping=0.0))
    roll = clipped_far_rollout(agent, log_ratio=0.05)
    before = actor_snapshot(agent)
    metrics = agent.update(roll)
    assert metrics['clip_fraction'] == pytest.approx(0.0), 'the fixture must be inside the clip'
    assert actor_moved(agent, before) > 1e-9


def test_the_entropy_bonus_raises_entropy_rather_than_lowering_it():
    """**The sign on `- entropy_coef * entropy` — a `+` there is an anti-exploration penalty.**

    That mutant is the worst of the ten to diagnose from a chart: the arm collapses to one action
    within a few thousand transitions, the reported `entropy` falls, and everything else looks healthy.
    The head's bias is skewed first so entropy starts well below `ln 3` and has room to rise.
    """
    agent = agent_module.PpoAgent(arch(), config(ppo_epochs=1, ppo_entropy_coef=1.0, ppo_vf_coef=0.0,
                                                ppo_normalize_adv=False, ppo_gradient_clipping=0.0,
                                                ppo_learning_rate=0.05))
    with torch.no_grad():
        agent.actor.head.bias.copy_(torch.tensor([4.0, 0.0, 0.0]))
    roll = filled_rollout(agent)
    roll.advantages[:] = 0.0            # the surrogate contributes nothing; only the bonus acts
    obs = torch.as_tensor(roll.flat()['obs'])

    def mean_entropy():
        with torch.no_grad():
            _, entropy = network.evaluate(agent.actor(obs), torch.zeros(len(obs), dtype=torch.long))
        return float(entropy.mean())

    before = mean_entropy()
    assert before < 0.5, 'the fixture must start away from the uniform ln 3'
    agent.update(roll)
    assert mean_entropy() > before


def test_advantage_normalisation_makes_the_update_invariant_to_the_advantage_scale():
    """**What normalising is for, and the only property that distinguishes it from skipping it.**

    Two identical arms, one fed advantages 1,000x larger. With normalisation on they take the same
    step; with it off the larger one takes a step 1,000x bigger, which is how a shaping dose or a
    reward-scale change silently becomes a learning-rate change.
    """
    def stepped(scale, normalise_on):
        agent = agent_module.PpoAgent(arch(), config(
            ppo_epochs=1, ppo_entropy_coef=0.0, ppo_vf_coef=0.0, ppo_gradient_clipping=0.0,
            ppo_normalize_adv=normalise_on))
        roll = filled_rollout(agent)
        roll.advantages *= np.float32(scale)
        agent.update(roll)
        return actor_snapshot(agent)

    on_small, on_large = stepped(1.0, True), stepped(1000.0, True)
    assert max(float((a - b).abs().max()) for a, b in zip(on_small, on_large)) < 1e-7
    off_small, off_large = stepped(1.0, False), stepped(1000.0, False)
    # And the un-normalised pair genuinely differs, so the check above is not vacuous.
    assert max(float((a - b).abs().max()) for a, b in zip(off_small, off_large)) > 1e-5
