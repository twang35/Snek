"""The reads over a return distribution -- `algos/dist/reads.py` and the heads' `distribution()`.

Every read is pinned on a hand-built `(values, masses)` table, because the reads are arithmetic that
looks interchangeable (a partial expectation and a conditional mean differ by one division) and the
plan's predictions (`plans/quantile-reads.md` §2) rest on which one each read uses.
"""

import math

import pytest
import torch

from algos.dist import net as network
from algos.dist import reads
from env import constants
from tools import arch as arch_tools


def table(*rows):
    """`(values, masses)` for one state: each row is one action's list of (value, mass) pairs, padded
    to the same length with zero-mass entries so a ragged table still stacks."""
    width = max(len(row) for row in rows)
    values = torch.tensor([[v for v, _ in row] + [0.0] * (width - len(row)) for row in rows])
    masses = torch.tensor([[w for _, w in row] + [0.0] * (width - len(row)) for row in rows])
    return values.unsqueeze(0), masses.unsqueeze(0)


def equal(n, *values):
    return [(float(v), 1.0 / n) for v in values]


def score(variant, values, masses):
    return reads.scores(values, masses, reads.parse_variant(variant))[0][0].tolist()


# ---------------------------------------------------------------- parsing

def test_the_variants_parse_and_the_mean_is_none():
    assert reads.parse_variant(None) is None
    assert reads.parse_variant('') is None
    assert reads.parse_variant('mean') is None
    assert reads.parse_variant('mean:fixed') == ('mean', 'fixed')
    assert reads.parse_variant('cvar:0.25') == ('cvar', 0.25)
    assert reads.parse_variant('leastneg') == ('leastneg', 0.0)
    assert reads.parse_variant('leastneg:-2') == ('leastneg', -2.0)
    assert reads.parse_variant('leastnegmean') == ('leastnegmean', 0.0)
    assert reads.parse_variant('leastnegmean:-2') == ('leastnegmean', -2.0)
    assert reads.parse_variant('mix:0.7') == ('mix', 0.7)
    assert reads.parse_variant('mixmass:0.7') == ('mixmass', 0.7)
    assert reads.parse_variant('above:30') == ('above', 30.0)
    assert reads.parse_variant('abovemean:10') == ('abovemean', 10.0)
    assert network.parse_variant is reads.parse_variant


@pytest.mark.parametrize('bad', ['gaussian', 'cvar:0', 'cvar:1.5', 'mean:sampled', 'mix:1.2',
                                 'mixmass:-0.1', 'above', 'abovemean:', 'leastneg:x'])
def test_a_malformed_variant_names_itself(bad):
    with pytest.raises(ValueError):
        reads.parse_variant(bad)


# ---------------------------------------------------------------- the worked examples

def test_the_users_mix_example_averages_within_each_side_then_mixes():
    # -5, -1, -1, -1, -1 | 10 x5: the negative side averages to -1.8, the positive to 10.
    values, masses = table(equal(10, -5, -1, -1, -1, -1, 10, 10, 10, 10, 10))
    assert math.isclose(score('mix:0.3', values, masses)[0], 0.3 * -1.8 + 0.7 * 10, abs_tol=1e-5)
    assert math.isclose(score('mix:0.5', values, masses)[0], 0.5 * -1.8 + 0.5 * 10, abs_tol=1e-5)
    assert math.isclose(score('mean:fixed', values, masses)[0], 4.1, abs_tol=1e-5)


def test_the_plans_variant_three_example_gives_four_and_a_half_not_seven_point_eight():
    values, masses = table(equal(10, -1, -1, 10, 10, 10, 10, 10, 10, 10, 10))
    assert math.isclose(score('mix:0.5', values, masses)[0], 4.5, abs_tol=1e-5)
    assert math.isclose(score('mean:fixed', values, masses)[0], 7.8, abs_tol=1e-5)


def test_the_conditional_mean_reads_are_mass_blind_and_mixmass_is_not():
    # Plan §2: A = one at -5 and thirty-one at 20; B = sixteen at -0.1 and sixteen at 20.
    a = equal(32, *([-5] + [20] * 31))
    b = equal(32, *([-0.1] * 16 + [20] * 16))
    values, masses = table(a, b)
    mix = score('mix:0.5', values, masses)
    assert math.isclose(mix[0], 7.5, abs_tol=1e-4) and math.isclose(mix[1], 9.95, abs_tol=1e-4)
    assert mix[1] > mix[0]                                   # mix prefers B
    mean = score('mean:fixed', values, masses)
    assert mean[0] > mean[1]                                 # the mean prefers A
    mixmass = score('mixmass:0.7', values, masses)
    # 0.7 * (-5/32) + 0.3 * (31*20/32) = 5.703 ; 0.7 * (-1.6/32) + 0.3 * (16*20/32) = 2.965
    assert math.isclose(mixmass[0], 0.7 * (-5 / 32) + 0.3 * (620 / 32), abs_tol=1e-4)
    assert mixmass[0] > mixmass[1]                           # mass kept: A again
    least = score('leastnegmean', values, masses)
    assert least[1] > least[0]                               # -0.1 is closer to zero than -5
    least_mass = score('leastneg', values, masses)
    assert math.isclose(least_mass[0], -5 / 32, abs_tol=1e-6) and math.isclose(least_mass[1], -1.6 / 32, abs_tol=1e-6)
    assert least_mass[1] > least_mass[0]                     # B's losses are smaller in total mass x size too


def test_above_keeps_mass_and_abovemean_discards_it():
    # Plan §1 read 9: nine at 20 and one at 35 against five at 20 and five at 32.
    values, masses = table(equal(10, *([20] * 9 + [35])), equal(10, *([20] * 5 + [32] * 5)))
    above = score('above:30', values, masses)
    assert math.isclose(above[0], 3.5, abs_tol=1e-5) and math.isclose(above[1], 16.0, abs_tol=1e-5)
    abovemean = score('abovemean:30', values, masses)
    assert math.isclose(abovemean[0], 35.0, abs_tol=1e-5) and math.isclose(abovemean[1], 32.0, abs_tol=1e-5)


# ---------------------------------------------------------------- empty sets, fallbacks, ties

def test_an_empty_negative_side_scores_zero_which_is_the_best_leastneg_can_do():
    values, masses = table(equal(4, 1, 2, 3, 4), equal(4, -1, 2, 3, 4))
    least, tie = reads.scores(values, masses, reads.parse_variant('leastnegmean'))
    assert least[0].tolist() == [0.0, -1.0]
    assert math.isclose(float(tie[0, 0]), 2.5) and math.isclose(float(tie[0, 1]), 3.0)
    assert reads.actions(values, masses, reads.parse_variant('leastnegmean')).tolist() == [0]
    least, tie = reads.scores(values, masses, reads.parse_variant('leastneg'))
    assert least[0].tolist() == [0.0, -0.25]                # the loss's share of the mean: -1 / 4
    assert math.isclose(float(tie[0, 0]), 2.5) and math.isclose(float(tie[0, 1]), 2.25)
    assert reads.actions(values, masses, reads.parse_variant('leastneg')).tolist() == [0]
    # mix on an all-positive action is (1 - a) times its mean, since the negative term is 0.
    assert math.isclose(score('mix:0.3', values, masses)[0], 0.7 * 2.5, abs_tol=1e-6)


def test_leastneg_with_a_threshold_ignores_small_losses():
    # Action 0 risks a starve-sized -0.5 twice, action 1 a death-sized -5 once; under -2 only the death counts.
    values, masses = table(equal(4, -0.5, -0.5, 10, 10), equal(4, -5, 12, 12, 12))
    assert reads.actions(values, masses, reads.parse_variant('leastnegmean')).tolist() == [0]  # -0.5 > -5
    assert reads.actions(values, masses, reads.parse_variant('leastneg')).tolist() == [0]      # -0.25 > -1.25
    for variant, expected in (('leastnegmean:-2', [0.0, -5.0]), ('leastneg:-2', [0.0, -1.25])):
        least, tie = reads.scores(values, masses, reads.parse_variant(variant))
        assert least[0].tolist() == expected, variant
        assert reads.actions(values, masses, reads.parse_variant(variant)).tolist() == [0]
    # And when neither action has a death-sized loss, the tie-break (the positive side) decides.
    values, masses = table(equal(4, -0.5, -0.5, 10, 10), equal(4, -1, 12, 12, 12))
    for variant in ('leastneg:-2', 'leastnegmean:-2'):
        assert reads.actions(values, masses, reads.parse_variant(variant)).tolist() == [1], variant


def test_leastneg_is_mass_aware_and_leastnegmean_is_not():
    # One quantile at -8 against four at -1: the same total loss share (-8/32 = -0.25 each side of
    # zero... no: -8/32 = -0.25 and -4/32 = -0.125), so the mass-aware read prefers the four small
    # losses and the conditional mean prefers the single big one over the average -1.
    values, masses = table(equal(32, *([-8] + [10] * 31)), equal(32, *([-1] * 4 + [10] * 28)))
    mass_aware = score('leastneg', values, masses)
    assert math.isclose(mass_aware[0], -8 / 32, abs_tol=1e-6) and math.isclose(mass_aware[1], -4 / 32, abs_tol=1e-6)
    assert reads.actions(values, masses, reads.parse_variant('leastneg')).tolist() == [1]
    blind = score('leastnegmean', values, masses)
    assert blind == pytest.approx([-8.0, -1.0])
    assert reads.actions(values, masses, reads.parse_variant('leastnegmean')).tolist() == [1]
    # Flip it: one at -3 against four at -1. Total share -3/32 < -4/32 is false (-0.094 > -0.125), so
    # mass-aware takes the single loss; the conditional mean still takes the four at -1.
    values, masses = table(equal(32, *([-3] + [10] * 31)), equal(32, *([-1] * 4 + [10] * 28)))
    assert reads.actions(values, masses, reads.parse_variant('leastneg')).tolist() == [0]
    assert reads.actions(values, masses, reads.parse_variant('leastnegmean')).tolist() == [1]


def test_leastneg_ties_break_on_the_positive_side_and_a_lone_best_ignores_it():
    values, masses = table(equal(4, -1, 1, 1, 1), equal(4, -1, 9, 9, 9), equal(4, -3, 50, 50, 50))
    for variant in ('leastneg', 'leastnegmean'):
        assert reads.actions(values, masses, reads.parse_variant(variant)).tolist() == [1], variant


def test_a_threshold_read_falls_back_to_the_mean_when_no_action_reaches_it():
    values, masses = table(equal(4, 1, 2, 3, 4), equal(4, 5, 5, 5, 5))
    for variant in ('above:30', 'abovemean:30'):
        assert score(variant, values, masses) == pytest.approx([2.5, 5.0])
    # ... but not when one action does: the others read 0 and lose.
    values, masses = table(equal(4, 1, 2, 3, 40), equal(4, 5, 5, 5, 5))
    assert score('above:30', values, masses) == pytest.approx([10.0, 0.0])
    assert score('abovemean:30', values, masses) == pytest.approx([40.0, 0.0])


def test_the_fallback_is_per_state():
    values, masses = table(equal(4, 1, 2, 3, 40), equal(4, 5, 5, 5, 5))
    quiet_values, quiet_masses = table(equal(4, 1, 2, 3, 4), equal(4, 5, 5, 5, 5))
    both_v = torch.cat([values, quiet_values]); both_m = torch.cat([masses, quiet_masses])
    scored, _ = reads.scores(both_v, both_m, reads.parse_variant('above:30'))
    assert scored[0].tolist() == pytest.approx([10.0, 0.0])
    assert scored[1].tolist() == pytest.approx([2.5, 5.0])


def test_unequal_masses_weight_the_conditional_mean_as_fqf_needs():
    # Two negative quantiles, one carrying three times the other's mass: -1 (0.3), -5 (0.1), 10 (0.6).
    values, masses = table([(-1.0, 0.3), (-5.0, 0.1), (10.0, 0.6)])
    least = score('leastnegmean', values, masses)[0]
    assert math.isclose(least, (-0.3 - 0.5) / 0.4, abs_tol=1e-6)       # -2.0, not the plain -3.0
    assert math.isclose(score('leastneg', values, masses)[0], -0.8, abs_tol=1e-6)
    assert math.isclose(score('mixmass:0.5', values, masses)[0], 0.5 * -0.8 + 0.5 * 6.0, abs_tol=1e-6)


def test_a_set_under_the_mass_floor_counts_as_empty():
    # A softmax leaves 1e-6 on a -10 atom; leastneg must not read that as a confident -10.
    values, masses = table([(-10.0, 1e-6), (5.0, 1.0 - 1e-6)], [(-0.5, 0.5), (5.0, 0.5)])
    least = score('leastnegmean', values, masses)
    assert least[0] == 0.0 and least[1] == -0.5
    assert reads.MIN_MASS < 1.0 / 32                          # inert on QR-DQN's n = 32


# ---------------------------------------------------------------- the heads' distribution()

def an_arch(head, algo='c51'):
    return arch_tools.build_arch([32], constants.NUM_ACTIONS, constants.OBS_LEN, constants.OBS_ERA,
                                 algo=algo, head=head)


HEADS = [{'type': 'c51', 'atoms': 5, 'v_min': 0.0, 'v_max': 4.0},
         {'type': 'quantile', 'n': 8},
         {'type': 'iqn', 'embedding': 16, 'n_tau': 8, 'k': 4},
         {'type': 'fqf', 'embedding': 16, 'n': 8}]


@pytest.mark.parametrize('head', HEADS, ids=[h['type'] for h in HEADS])
def test_every_head_states_a_distribution_whose_mean_is_its_mean(head):
    net = network.build(an_arch(head), seed=1)
    obs = torch.rand(3, constants.OBS_LEN, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        values, masses = net.distribution(obs)
        assert values.shape == masses.shape and values.shape[:2] == (3, constants.NUM_ACTIONS)
        assert torch.allclose(masses.sum(dim=2), torch.ones(3, constants.NUM_ACTIONS), atol=1e-5)
        mean = reads.mean(values, masses)
        if head['type'] == 'iqn':
            # IQN's q_values samples its fractions; the fixed read is its own reference.
            again, _ = net.distribution(obs)
            assert torch.equal(values, again)
        else:
            assert torch.allclose(mean, net.q_values(obs), atol=1e-4), head['type']


def test_the_implicit_distribution_is_at_fixed_midpoint_fractions():
    net = network.build(an_arch({'type': 'iqn', 'embedding': 16, 'n_tau': 8, 'k': 4}, algo='iqn'), seed=3)
    obs = torch.rand(2, constants.OBS_LEN)
    with torch.no_grad():
        features = net.features(obs)
        taus = torch.tensor([[0.125, 0.375, 0.625, 0.875]] * 2)
        expected = net.quantiles_at(features, taus).transpose(1, 2)
        values, masses = net.distribution(obs)
    assert torch.allclose(values, expected)
    assert torch.allclose(masses, torch.full_like(masses, 0.25))


def test_fqf_masses_are_the_proposed_bin_widths():
    net = network.build(an_arch({'type': 'fqf', 'embedding': 16, 'n': 8}, algo='fqf'), seed=4)
    with torch.no_grad():
        net.fraction.weight.uniform_(-2.0, 2.0)                # make the proposals uneven
        obs = torch.rand(2, constants.OBS_LEN)
        taus_full, _, _ = net.propose(net.features(obs))
        _, masses = net.distribution(obs)
    widths = taus_full[:, 1:] - taus_full[:, :-1]
    assert torch.allclose(masses[:, 0, :], widths) and torch.allclose(masses[:, 1, :], widths)
    assert float(widths.std()) > 1e-3


# ---------------------------------------------------------------- the policy seam

@pytest.mark.parametrize('head', HEADS, ids=[h['type'] for h in HEADS])
def test_greedy_policy_fn_dispatches_each_variant_to_its_read(head):
    net = network.build(an_arch(head), seed=5)
    obs = torch.rand(6, constants.OBS_LEN, generator=torch.Generator().manual_seed(6)).numpy()
    t = torch.as_tensor(obs)
    with torch.no_grad():
        values, masses = net.distribution(t)
        for variant in ('mean:fixed', 'leastneg', 'leastneg:-2', 'leastnegmean', 'leastnegmean:-2', 'mix:0.7',
                        'mixmass:0.7', 'above:0.5', 'abovemean:0.5'):
            expected = reads.actions(values, masses, reads.parse_variant(variant)).numpy()
            assert (network.greedy_policy_fn(net, variant=variant)(obs) == expected).all(), variant
        if head['type'] != 'iqn':   # IQN samples its fractions on both of these reads
            assert (network.greedy_policy_fn(net, variant='cvar:0.25')(obs)
                    == net.cvar_values(t, 0.25).argmax(dim=1).numpy()).all()
            assert (network.greedy_policy_fn(net)(obs) == net.q_values(t).argmax(dim=1).numpy()).all()


def test_the_reads_disagree_with_the_mean_somewhere_on_a_random_quantile_net():
    # If no read ever changed an action the pass would measure the control seven times over.
    net = network.build(an_arch({'type': 'quantile', 'n': 8}, algo='qrdqn'), seed=7)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(9)
        net.qnet.head.weight.normal_(0.0, 3.0, generator=generator)   # wide, sign-mixed quantiles
        net.qnet.head.bias.normal_(0.0, 1.0, generator=generator)
    obs = torch.rand(256, constants.OBS_LEN, generator=torch.Generator().manual_seed(8)).numpy()
    mean = network.greedy_policy_fn(net)(obs)
    for variant in ('leastneg', 'leastnegmean', 'mix:0.7', 'mix:0.5', 'mixmass:0.7'):
        assert (network.greedy_policy_fn(net, variant=variant)(obs) != mean).any(), variant
