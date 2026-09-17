"""The distributional heads and their reads. `algos/dist/net.py`."""

import math

import pytest
import torch

from algos.dist import net as network
from env import constants
from tools import arch as arch_tools


def an_arch(head, algo='c51', widths=(32,)):
    return arch_tools.build_arch(list(widths), constants.NUM_ACTIONS, constants.OBS_LEN,
                                 constants.OBS_ERA, algo=algo, head=head)


C51 = {'type': 'c51', 'atoms': 5, 'v_min': 0.0, 'v_max': 4.0}
QUANT = {'type': 'quantile', 'n': 8}
IQN = {'type': 'iqn', 'embedding': 16, 'n_tau': 8, 'k': 4}
FQF = {'type': 'fqf', 'embedding': 16, 'n': 8}


def observations(m=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(m, constants.OBS_LEN, generator=generator)


# ---------------------------------------------------------------- build

@pytest.mark.parametrize('head, kind', [(C51, network.CategoricalNet), (QUANT, network.QuantileNet),
                                        (IQN, network.ImplicitNet), (FQF, network.ImplicitNet)])
def test_build_picks_the_network_the_head_names(head, kind):
    assert isinstance(network.build(an_arch(head)), kind)


def test_a_missing_or_unknown_head_is_refused_by_name():
    with pytest.raises(ValueError):
        network.build(arch_tools.build_arch([32], 3, constants.OBS_LEN, constants.OBS_ERA, algo='c51'))
    with pytest.raises(ValueError):
        network.build(an_arch({'type': 'gaussian'}))


def test_the_seed_pins_the_initialisation_and_a_different_seed_moves_it():
    for head in (C51, QUANT, IQN, FQF):
        a = network.build(an_arch(head), seed=5).state_dict()
        b = network.build(an_arch(head), seed=5).state_dict()
        c = network.build(an_arch(head), seed=6).state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a), head['type']
        assert any(not torch.equal(a[k], c[k]) for k in a), head['type']


def test_every_head_answers_q_values_and_cvar_values_in_the_action_shape():
    obs = observations()
    for head in (C51, QUANT, IQN, FQF):
        net = network.build(an_arch(head), seed=1)
        assert net.q_values(obs).shape == (3, constants.NUM_ACTIONS)
        assert net.cvar_values(obs, 0.25).shape == (3, constants.NUM_ACTIONS)


# ---------------------------------------------------------------- c51

def test_the_categorical_mean_of_a_one_hot_distribution_is_its_atom():
    net = network.build(an_arch(C51), seed=1)
    with torch.no_grad():
        net.qnet.head.weight.zero_()
        # Put all the mass of action 0 on atom 3 (value 3.0), the others uniform.
        bias = net.qnet.head.bias.view(constants.NUM_ACTIONS, 5)
        bias.zero_()
        bias[0, 3] = 50.0
    q = net.q_values(observations(1))
    assert math.isclose(float(q[0, 0]), 3.0, abs_tol=1e-4)
    assert math.isclose(float(q[0, 1]), 2.0, abs_tol=1e-4), 'uniform over 0..4 has mean 2'


def test_the_categorical_cvar_is_the_mean_of_the_lower_tail():
    net = network.build(an_arch(C51), seed=1)
    with torch.no_grad():
        net.qnet.head.weight.zero_()
        net.qnet.head.bias.zero_()          # uniform over atoms 0, 1, 2, 3, 4 for every action
    # The lower 40% of a uniform over {0,1,2,3,4} is atoms 0 and 1: mean 0.5.
    cvar = net.cvar_values(observations(1), 0.4)
    assert torch.allclose(cvar, torch.full((1, constants.NUM_ACTIONS), 0.5), atol=1e-4)
    # alpha = 1 recovers the mean.
    assert torch.allclose(net.cvar_values(observations(1), 1.0), net.q_values(observations(1)), atol=1e-5)


def test_the_categorical_support_is_the_arch_and_the_probabilities_sum_to_one():
    net = network.build(an_arch(C51), seed=1)
    assert torch.allclose(net.support, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(net.probs(observations()).sum(dim=2), torch.ones(3, constants.NUM_ACTIONS))


# ---------------------------------------------------------------- quantile

def test_the_quantile_fractions_are_the_midpoints_and_the_mean_is_the_plain_average():
    net = network.build(an_arch(QUANT, algo='qrdqn'), seed=1)
    assert torch.allclose(net.fractions('cpu'), torch.tensor([(2 * i - 1) / 16 for i in range(1, 9)]))
    obs = observations()
    assert torch.allclose(net.q_values(obs), net.quantiles(obs).mean(dim=2))


def test_the_quantile_cvar_averages_the_lowest_sorted_quantiles():
    net = network.build(an_arch(QUANT, algo='qrdqn'), seed=1)
    obs = observations(1)
    sorted_q, _ = net.quantiles(obs).sort(dim=2)
    assert torch.allclose(net.cvar_values(obs, 0.25), sorted_q[:, :, :2].mean(dim=2))   # ceil(0.25 * 8) = 2
    assert torch.allclose(net.cvar_values(obs, 1.0), net.q_values(obs))


# ---------------------------------------------------------------- iqn / fqf

def test_the_cosine_embedding_at_tau_zero_is_relu_of_the_row_sums():
    net = network.build(an_arch(IQN, algo='iqn'), seed=1)
    taus = torch.zeros(2, 3)
    # cos(pi i 0) = 1 for every i, so the embedding is relu(W 1 + b) for every tau.
    expected = torch.relu(net.tau_embed(torch.ones(2, 3, IQN['embedding'])))
    assert torch.allclose(net.embed(taus), expected)


def test_the_cosine_embedding_uses_the_harmonics_pi_i():
    net = network.build(an_arch(IQN, algo='iqn'), seed=1)
    assert torch.allclose(net.harmonics, math.pi * torch.arange(1, IQN['embedding'] + 1).float())
    # At tau = 1, cos(pi i) alternates -1, +1, -1, ...; without the pi it would be cos(i), which is
    # neither, and the embedding would be a different function of tau.
    taus = torch.ones(1, 1)
    signs = torch.tensor([(-1.0) ** i for i in range(1, IQN['embedding'] + 1)]).view(1, 1, -1)
    assert torch.allclose(net.embed(taus), torch.relu(net.tau_embed(signs)), atol=1e-5)


def test_iqn_quantile_values_have_the_sampled_fraction_axis():
    net = network.build(an_arch(IQN, algo='iqn'), seed=1)
    obs = observations(2)
    features = net.features(obs)
    taus = torch.rand(2, 5)
    assert net.quantiles_at(features, taus).shape == (2, 5, constants.NUM_ACTIONS)


def test_iqn_quantiles_are_read_at_the_fraction_asked_for_not_a_fixed_grid():
    net = network.build(an_arch(IQN, algo='iqn'), seed=1)
    features = net.features(observations(1))
    low = net.quantiles_at(features, torch.tensor([[0.05]]))
    high = net.quantiles_at(features, torch.tensor([[0.95]]))
    assert not torch.allclose(low, high), 'a network that ignored tau would be QR-DQN with one quantile'


def test_the_cvar_read_scales_the_acting_fractions_by_alpha():
    net = network.build(an_arch(IQN, algo='iqn'), seed=1)
    features = net.features(observations(4))
    torch.manual_seed(3)
    neutral = net.acting_taus(features, 1.0)
    torch.manual_seed(3)
    tail = net.acting_taus(features, 0.25)
    assert torch.allclose(tail, neutral * 0.25)
    assert float(tail.max()) <= 0.25


def test_fqf_proposes_monotone_fractions_pinned_at_zero_and_one():
    net = network.build(an_arch(FQF, algo='fqf'), seed=2)
    taus, hats, entropy = net.propose(net.features(observations(5)))
    assert taus.shape == (5, FQF['n'] + 1) and hats.shape == (5, FQF['n'])
    assert torch.all(taus[:, 0] == 0.0) and torch.all(taus[:, -1] == 1.0)
    assert torch.all(taus[:, 1:] >= taus[:, :-1])
    assert torch.allclose(hats, (taus[:, :-1] + taus[:, 1:]) / 2)
    assert entropy.shape == (5,) and float(entropy.min()) > 0.0


def test_fqf_proposals_start_near_the_quantile_midpoints():
    # The fraction net is initialised small, so the first proposals are QR-DQN's grid to within noise.
    net = network.build(an_arch(FQF, algo='fqf'), seed=2)
    _, hats, _ = net.propose(net.features(observations(2)))
    grid = torch.tensor([(2 * i - 1) / 16 for i in range(1, 9)]).expand(2, 8)
    assert torch.allclose(hats, grid, atol=0.02)


def test_fqf_mean_is_the_width_weighted_sum_over_its_bins():
    net = network.build(an_arch(FQF, algo='fqf'), seed=2)
    obs = observations(2)
    features = net.features(obs)
    taus, hats, _ = net.propose(features)
    values = net.quantiles_at(features, hats)
    widths = (taus[:, 1:] - taus[:, :-1]).unsqueeze(2)
    assert torch.allclose(net.q_values(obs), (widths * values).sum(dim=1), atol=1e-6)


# ---------------------------------------------------------------- the policy reads

def test_the_greedy_policy_is_the_argmax_of_the_mean_and_the_variant_of_the_tail():
    net = network.build(an_arch(QUANT, algo='qrdqn'), seed=4)
    obs = observations(6).numpy()
    mean_policy = network.greedy_policy_fn(net)
    tail_policy = network.greedy_policy_fn(net, variant='cvar:0.25')
    with torch.no_grad():
        t = torch.as_tensor(obs)
        assert (mean_policy(obs) == net.q_values(t).argmax(dim=1).numpy()).all()
        assert (tail_policy(obs) == net.cvar_values(t, 0.25).argmax(dim=1).numpy()).all()
    assert mean_policy(obs).dtype.kind == 'i'


def test_a_variant_that_is_not_cvar_or_out_of_range_names_itself():
    assert network.parse_variant(None) is None
    assert network.parse_variant('mean') is None
    assert network.parse_variant('cvar:0.1') == ('cvar', 0.1)
    for bad in ('var:0.5', 'cvar:0', 'cvar:1.5', 'cvar:x'):
        with pytest.raises(ValueError):
            network.parse_variant(bad)


def test_the_tail_read_can_disagree_with_the_mean_read():
    """The whole point of the variant: two actions with equal means and different tails."""
    net = network.build(an_arch(QUANT, algo='qrdqn'), seed=1)
    with torch.no_grad():
        net.qnet.head.weight.zero_()
        bias = net.qnet.head.bias.view(constants.NUM_ACTIONS, QUANT['n'])
        bias.zero_()
        bias[0] = torch.tensor([-4.0, -4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0])     # mean 0, fat tails
        bias[1] = torch.tensor([-0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5])     # mean 0, thin tails
        bias[2] = torch.full((QUANT['n'],), -1.0)                              # mean -1
    obs = observations(1)
    q = net.q_values(obs)
    assert math.isclose(float(q[0, 0]), 0.0, abs_tol=1e-6) and math.isclose(float(q[0, 1]), 0.0, abs_tol=1e-6)
    assert int(net.cvar_values(obs, 0.25).argmax()) == 1, 'the thin-tailed action is the safer one'
