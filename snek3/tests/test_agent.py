"""The DDQN agent: the double-Q split, the target network, and the exploration shield.

Two things here change the training while leaving every log line plausible. **The double-Q split** —
argmax from the online net, value from the target net — degrades to plain DQN if the two are
confused, which is a slow overestimation bias and not a crash. **The shield asymmetry** — random
moves steered, the greedy action never — is the whole reason the shield is safe, and shielding the
greedy action too would look like an improvement for a while and then fail every eval, because evals
run unshielded.
"""

import numpy as np
import pytest
import torch

from dqn import agent as agent_module
from dqn.agent import DdqnAgent, safe_actions, shielded_choice
from env import constants
from tools import arch as arch_tools

SAFE_START, SAFE_STOP = constants.block_ranges()['body_and_wall']


def small_arch(fc=(4,), num_actions=3):
    return arch_tools.build_arch(list(fc), num_actions, constants.OBS_LEN, constants.OBS_ERA)


def make_agent(**kwargs):
    kwargs.setdefault('seed', 0)
    return DdqnAgent(small_arch(), **kwargs)


def observations(rows, safety):
    """`rows` observations whose safety block is `safety` and whose other values are zero."""
    obs = np.zeros((rows, constants.OBS_LEN), dtype=np.float32)
    obs[:, SAFE_START:SAFE_STOP] = np.asarray(safety, dtype=np.float32)
    return obs


def batch_of(rows, obs_value=0.7, next_value=0.3, action=0, reward=0.0, discount=0.99):
    """A batch with a **non-zero** observation, which is not incidental.

    With an all-zero observation the hidden pre-activations are zero (biases initialise to zero), so
    the relu output is zero, so every weight gradient below the head is zero and the head's weight
    gradient is zero too — only the head bias moves. Two target-network fixtures passed vacuously
    against that batch: they compared parameters that could not have changed.
    """
    return {'obs': np.full((rows, constants.OBS_LEN), obs_value, np.float32),
            'next_obs': np.full((rows, constants.OBS_LEN), next_value, np.float32),
            'action': np.full(rows, action, np.int64),
            'reward': np.full(rows, reward, np.float32),
            'discount': np.full(rows, discount, np.float32)}


# --- the safety block ------------------------------------------------------------------------

def test_the_safety_slice_comes_from_the_layout_table():
    # Hardcoding 6:9 would keep passing if a block were inserted before it, and the shield would
    # then be masking on food distances.
    assert (agent_module._SAFETY_START, agent_module._SAFETY_STOP) == (SAFE_START, SAFE_STOP)
    assert constants.OBS_BLOCKS[1][0] == 'body_and_wall'


def test_safe_actions_reads_one_as_safe():
    # Polarity: 1 means the move is survivable. Inverted, the shield would steer *into* walls.
    mask = safe_actions(observations(1, [[1, 0, 1]]))
    assert list(mask[0]) == [True, False, True]


# --- the shield -----------------------------------------------------------------------------

def test_a_guided_draw_only_ever_picks_a_safe_move():
    rng = np.random.default_rng(0)
    obs = observations(200, [[1, 0, 0]] * 200)
    drawn = shielded_choice(obs, np.ones(200, bool), rng, 3)
    assert set(drawn.tolist()) == {0}


def test_a_guided_draw_spreads_over_all_the_safe_moves():
    # It must be uniform over the allowed set, not merely legal — always picking the lowest safe
    # index would pass the test above and collect a badly skewed distribution.
    rng = np.random.default_rng(0)
    obs = observations(600, [[1, 0, 1]] * 600)
    counts = np.bincount(shielded_choice(obs, np.ones(600, bool), rng, 3), minlength=3)
    assert counts[1] == 0
    assert min(counts[0], counts[2]) > 200, counts


def test_an_unguided_draw_can_pick_a_fatal_move():
    # This is what makes a guided fraction of 0 reproduce unshielded behaviour exactly.
    rng = np.random.default_rng(0)
    obs = observations(300, [[1, 0, 0]] * 300)
    drawn = shielded_choice(obs, np.zeros(300, bool), rng, 3)
    assert set(drawn.tolist()) == {0, 1, 2}


def test_a_boxed_in_row_lets_everything_through_rather_than_having_no_choice():
    # No safe move means nothing to be steered to. Without this the row has an empty allowed set.
    rng = np.random.default_rng(0)
    obs = observations(300, [[0, 0, 0]] * 300)
    drawn = shielded_choice(obs, np.ones(300, bool), rng, 3)
    assert set(drawn.tolist()) == {0, 1, 2}


def test_guided_is_resolved_per_row_not_per_call():
    # Lanes are independent games, so one lane being guided must not shield another.
    rng = np.random.default_rng(0)
    obs = observations(400, [[1, 0, 0]] * 400)
    guided = np.zeros(400, bool)
    guided[:200] = True
    drawn = shielded_choice(obs, guided, rng, 3)
    assert set(drawn[:200].tolist()) == {0}
    assert set(drawn[200:].tolist()) == {0, 1, 2}


def test_a_scalar_guided_flag_applies_to_every_row():
    rng = np.random.default_rng(0)
    obs = observations(50, [[1, 0, 0]] * 50)
    assert set(shielded_choice(obs, True, rng, 3).tolist()) == {0}


def test_a_safety_block_that_does_not_match_the_action_count_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        shielded_choice(observations(2, [[1, 0, 0]] * 2), True, rng, 4)


# --- acting ---------------------------------------------------------------------------------

def test_epsilon_zero_is_purely_greedy():
    agent = make_agent()
    obs = np.random.default_rng(1).random((32, constants.OBS_LEN)).astype(np.float32)
    assert list(agent.act(obs, 0.0)) == list(agent.greedy_actions(obs))


def test_a_negative_epsilon_is_treated_as_zero_rather_than_inverted():
    agent = make_agent()
    obs = np.random.default_rng(1).random((16, constants.OBS_LEN)).astype(np.float32)
    assert list(agent.act(obs, -1.0)) == list(agent.greedy_actions(obs))


def test_epsilon_one_still_lets_the_greedy_action_be_drawn():
    """The exploration draw is uniform over *all* allowed actions, greedy included.

    That is standard epsilon-greedy and it is what the schedule is calibrated against: with three
    actions the effective non-greedy rate is `epsilon * 2/3`. Excluding the greedy action would
    silently rescale every epsilon in `dqn/schedules.py` by 1.5.
    """
    agent = make_agent()
    obs = observations(1, [[1, 1, 1]])
    greedy = int(agent.greedy_actions(obs)[0])
    drawn = [int(agent.act(obs, 1.0)[0]) for _ in range(200)]
    assert greedy in drawn
    assert len(set(drawn)) == 3


def test_roughly_epsilon_of_the_actions_are_explored():
    """The explore rate is epsilon, measured rather than assumed.

    Set up so an explored action is unambiguous: the head bias makes action 1 greedy everywhere, and
    only action 0 is safe with the shield on, so every explored row plays 0 and every greedy row
    plays 1. A skip here would have been no coverage at all, so the greedy action is forced rather
    than hoped for.
    """
    agent = make_agent(seed=3)
    with torch.no_grad():
        agent.net.head.weight.zero_()
        agent.net.head.bias.copy_(torch.tensor([0.0, 1.0, 0.0]))
    obs = observations(4000, [[1, 0, 0]] * 4000)
    assert set(agent.greedy_actions(obs).tolist()) == {1}
    explored = (agent.act(obs, 0.25, guided=True) == 0).mean()
    assert explored == pytest.approx(0.25, abs=0.03), explored


def test_the_policy_fn_has_the_shape_the_engine_expects():
    agent = make_agent()
    obs = np.zeros((5, constants.OBS_LEN), np.float32)
    actions = agent.policy_fn(obs)
    assert actions.shape == (5,) and actions.dtype == np.int64


# --- learning -------------------------------------------------------------------------------

def test_the_target_starts_as_an_exact_copy():
    # Otherwise the first updates bootstrap off a second, unrelated random initialisation.
    agent = make_agent()
    for online, lagged in zip(agent.net.parameters(), agent.target.parameters()):
        assert torch.equal(online, lagged)


def test_the_target_does_not_take_gradients():
    agent = make_agent()
    assert not any(parameter.requires_grad for parameter in agent.target.parameters())


def test_the_target_is_copied_on_the_period_and_not_between():
    agent = make_agent(target_update_period=4)
    before = agent.target.head.weight.detach().clone()
    for _ in range(3):
        agent.update(batch_of(8, reward=1.0))
    assert torch.equal(agent.target.head.weight, before), 'copied early'
    agent.update(batch_of(8, reward=1.0))
    assert not torch.equal(agent.target.head.weight, before), 'never copied'


def test_a_period_of_zero_freezes_the_target_instead_of_dividing_by_zero():
    agent = make_agent(target_update_period=0)
    before = agent.target.head.weight.detach().clone()
    for _ in range(5):
        agent.update(batch_of(8, reward=1.0))
    assert torch.equal(agent.target.head.weight, before)


def test_a_tau_below_one_moves_the_target_only_part_of_the_way():
    agent = make_agent(target_update_period=1, target_update_tau=0.5)
    agent.update(batch_of(8, reward=1.0))
    for online, lagged in zip(agent.net.parameters(), agent.target.parameters()):
        assert not torch.equal(online, lagged), 'a hard copy, not a Polyak average'


def test_the_target_uses_the_online_argmax_and_the_target_value():
    """The double-Q split, asserted as an exact value against both ways it can collapse.

    `reward + discount * Q_target(s', argmax_a Q_online(s', a))`. Two wrong versions are one
    character away and neither crashes:

    - **plain DQN**, `max_a Q_target(s', a)` — the overestimation bias double-Q exists to remove;
    - **no target network at all**, `Q_online(s', a*)`, which is `max_a Q_online(s', a)`.

    An earlier version of this fixture only asserted that the td error was *closer* to the double-Q
    value than to plain DQN's, and the second mutation survived it.
    """
    agent = make_agent()
    # Drive the two nets apart, or all three candidates coincide and the fixture proves nothing.
    with torch.no_grad():
        agent.target.head.weight.mul_(-1.0)
        agent.target.head.bias.add_(0.5)

    batch = batch_of(1, obs_value=0.7, next_value=0.3, action=0, reward=0.25, discount=0.9)
    obs = torch.as_tensor(batch['obs'])
    next_obs = torch.as_tensor(batch['next_obs'])
    with torch.no_grad():
        online_next = agent.net(next_obs)
        target_next = agent.target(next_obs)
        chosen = float(agent.net(obs)[0, 0])
        best = int(online_next.argmax())
        double_q = 0.25 + 0.9 * float(target_next[0, best])
        plain_dqn = 0.25 + 0.9 * float(target_next.max())
        no_target = 0.25 + 0.9 * float(online_next.max())

    assert len({round(double_q, 4), round(plain_dqn, 4), round(no_target, 4)}) == 3, \
        'the three candidates coincide here, so this fixture cannot distinguish them'

    td_error, _ = agent.update(batch)
    assert td_error[0] == pytest.approx(double_q - chosen, abs=1e-5)


def test_a_terminal_transition_bootstraps_off_nothing():
    # discount 0 means the target is the reward alone. This is the bug that folding `done` into
    # `discount` makes unrepresentable rather than merely fixed.
    agent = make_agent()
    batch = batch_of(1, next_value=5.0, reward=-5.0, discount=0.0)
    with torch.no_grad():
        chosen = float(agent.net(torch.as_tensor(batch['obs']))[0, 0])
    td_error, _ = agent.update(batch)
    assert td_error[0] == pytest.approx(-5.0 - chosen, abs=1e-5)


def test_importance_weights_scale_the_loss():
    heavy = make_agent(seed=5)
    light = make_agent(seed=5)
    batch = batch_of(8, reward=1.0)
    _, heavy_metrics = heavy.update(batch, np.full(8, 4.0, np.float32))
    _, light_metrics = light.update(batch, np.full(8, 1.0, np.float32))
    assert heavy_metrics['loss'] == pytest.approx(4.0 * light_metrics['loss'], rel=1e-4)


def test_importance_weights_can_be_switched_off():
    agent = make_agent(use_is_weights=False)
    plain = make_agent(use_is_weights=False)
    batch = batch_of(8, reward=1.0)
    _, weighted = agent.update(batch, np.full(8, 7.0, np.float32))
    _, unweighted = plain.update(batch, None)
    assert weighted['loss'] == pytest.approx(unweighted['loss'], rel=1e-6)


def test_the_loss_is_huber_so_one_perfect_game_cannot_dominate_a_batch():
    # A perfect game's reward is 100 against a typical step's ~0.001. Squared error would make one
    # such transition worth ~10,000 of the others.
    agent = make_agent()
    _, small = agent.update(batch_of(1, reward=1.0, discount=0.0))
    fresh = make_agent()
    _, large = fresh.update(batch_of(1, reward=101.0, discount=0.0))
    # Huber is linear past delta=1, so a 100x larger error costs ~100x, not ~10,000x.
    assert large['loss'] / small['loss'] < 300


def test_gradient_clipping_reports_the_norm_and_is_off_by_default():
    _, metrics = make_agent().update(batch_of(8, reward=50.0))
    assert 'grad_norm' not in metrics
    _, clipped = make_agent(gradient_clipping=0.5).update(batch_of(8, reward=50.0))
    assert clipped['grad_norm'] > 0.0


def test_update_returns_one_td_error_per_row():
    td_error, _ = make_agent().update(batch_of(16, reward=1.0))
    assert td_error.shape == (16,)


def test_the_train_step_counts_updates():
    agent = make_agent()
    for expected in (1, 2, 3):
        _, metrics = agent.update(batch_of(4))
        assert metrics['train_step'] == expected


# --- persistence ----------------------------------------------------------------------------

def test_a_state_dict_round_trip_restores_the_nets_the_optimiser_and_the_step():
    agent = make_agent(seed=11)
    for _ in range(5):
        agent.update(batch_of(8, reward=1.0), np.ones(8, np.float32))
    state = agent.state_dict()

    restored = make_agent(seed=99)
    restored.load_state_dict(state)
    assert restored.train_step == agent.train_step
    for online, other in zip(agent.net.parameters(), restored.net.parameters()):
        assert torch.equal(online, other)
    for lagged, other in zip(agent.target.parameters(), restored.target.parameters()):
        assert torch.equal(lagged, other)
    # The optimiser's Adam moments matter: restoring without them restarts the moment estimates and
    # the first steps after a resume are effectively at a different learning rate.
    assert restored.optimizer.state_dict()['state'].keys() == \
        agent.optimizer.state_dict()['state'].keys()


def test_the_seed_pins_the_network_initialisation_not_only_the_coins():
    """Two agents with the same seed must be the same network.

    Found by a fixture that expected two same-seeded agents to compute the same loss and got 0.4963
    against 0.4993: `nn.init` reads torch's *global* RNG, so the second agent built in a process
    continued the first one's stream. Every seed-matched arm comparison would have differed in its
    initialisation, which is the one thing a seed exists to pin.
    """
    first, second = make_agent(seed=42), make_agent(seed=42)
    for one, other in zip(first.net.parameters(), second.net.parameters()):
        assert torch.equal(one, other)
    different = make_agent(seed=43)
    assert not all(torch.equal(one, other)
                   for one, other in zip(first.net.parameters(), different.net.parameters()))


def test_an_unseeded_agent_is_still_usable():
    # `seed=None` means "do not care", not "crash". Every path in tools/ builds a net it is about
    # to overwrite from a checkpoint.
    agent = DdqnAgent(small_arch(), seed=None)
    assert agent.greedy_actions(np.zeros((2, constants.OBS_LEN), np.float32)).shape == (2,)


def test_a_restored_agent_explores_the_same_way():
    agent = make_agent(seed=11)
    obs = observations(64, [[1, 0, 0]] * 64)
    agent.act(obs, 0.5, guided=True)
    state = agent.state_dict()

    restored = make_agent(seed=99)
    restored.load_state_dict(state)
    assert list(restored.act(obs, 0.5, guided=True)) == list(agent.act(obs, 0.5, guided=True))
