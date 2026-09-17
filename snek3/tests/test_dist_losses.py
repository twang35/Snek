"""The distributional losses on hand-worked examples. `algos/dist/losses.py`."""

import math

import pytest
import torch

from algos.dist import losses


def support(v_min, v_max, atoms):
    return torch.linspace(v_min, v_max, atoms)


# ---------------------------------------------------------------- the categorical projection

def test_a_value_between_two_atoms_splits_its_mass_in_proportion_to_distance():
    sup = support(0.0, 2.0, 3)                      # atoms at 0, 1, 2
    probs = torch.tensor([[1.0, 0.0, 0.0]])
    values = torch.tensor([[0.25, 5.0, 5.0]])       # the one unit of mass sits at 0.25
    out = losses.project(probs, values, sup)
    # 0.25 is a quarter of the way from atom 0 to atom 1: 3/4 stays at 0, 1/4 moves to 1.
    assert torch.allclose(out, torch.tensor([[0.75, 0.25, 0.0]]))


def test_a_value_beyond_the_support_clips_to_the_end_atom_with_all_its_mass():
    sup = support(0.0, 2.0, 3)
    probs = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.0, 0.0, 7.5]])
    assert torch.allclose(losses.project(probs, values, sup), torch.tensor([[0.0, 0.0, 1.0]]))
    values = torch.tensor([[0.0, 0.0, -3.0]])
    assert torch.allclose(losses.project(probs, values, sup), torch.tensor([[1.0, 0.0, 0.0]]))


def test_a_value_exactly_on_an_atom_gives_that_atom_everything_once():
    sup = support(0.0, 2.0, 3)
    probs = torch.tensor([[0.4, 0.6, 0.0]])
    values = torch.tensor([[1.0, 1.0, 9.0]])
    out = losses.project(probs, values, sup)
    assert torch.allclose(out, torch.tensor([[0.0, 1.0, 0.0]]))
    assert math.isclose(float(out.sum()), 1.0)


def test_the_projection_conserves_mass_on_a_random_batch():
    torch.manual_seed(0)
    sup = support(-10.0, 110.0, 51)
    probs = torch.softmax(torch.randn(64, 51), dim=1)
    values = -20.0 + 150.0 * torch.rand(64, 51)
    out = losses.project(probs, values, sup)
    assert torch.allclose(out.sum(dim=1), torch.ones(64), atol=1e-5)
    assert float(out.min()) >= 0.0


def test_the_identity_projection_is_the_identity():
    sup = support(-1.0, 1.0, 5)
    probs = torch.softmax(torch.randn(3, 5), dim=1)
    assert torch.allclose(losses.project(probs, sup.expand(3, 5), sup), probs, atol=1e-6)


def test_cross_entropy_is_zero_only_against_itself_and_positive_otherwise():
    target = torch.tensor([[0.2, 0.5, 0.3]])
    own = losses.categorical_cross_entropy(torch.log(target), target)
    other = losses.categorical_cross_entropy(torch.log(torch.tensor([[0.6, 0.2, 0.2]])), target)
    # CE against itself is the entropy; the gap to any other distribution is the KL, which is > 0.
    entropy = -(target * torch.log(target)).sum()
    assert torch.allclose(own, entropy.view(1))
    assert float(other) > float(own)


# ---------------------------------------------------------------- the quantile Huber

def test_the_quantile_huber_is_zero_when_a_single_quantile_meets_its_target():
    online = torch.tensor([[2.0]])
    taus = torch.tensor([[0.5]])
    assert float(losses.quantile_huber(online, taus, online.clone())) == 0.0


def test_every_online_quantile_is_compared_with_every_target_sample():
    # Identical sets are *not* a zero loss: the (i, j) cross terms remain, which is the paper's loss
    # (Eq. 10) and not a per-index match. Permuting the target samples changes nothing.
    online = torch.tensor([[1.0, 2.0, 3.0]])
    taus = torch.tensor([[1 / 6, 3 / 6, 5 / 6]])
    same = losses.quantile_huber(online, taus, online.clone())
    permuted = losses.quantile_huber(online, taus, torch.tensor([[3.0, 1.0, 2.0]]))
    assert float(same) > 0.0 and torch.allclose(same, permuted)


def test_at_kappa_zero_the_quantile_huber_is_the_pinball_loss():
    online = torch.tensor([[0.0]])
    taus = torch.tensor([[0.25]])
    # u = target - online = +1: rho = tau * |u| = 0.25.  u = -1: rho = (1 - tau) * |u| = 0.75.
    assert math.isclose(float(losses.quantile_huber(online, taus, torch.tensor([[1.0]]), kappa=0.0)), 0.25)
    assert math.isclose(float(losses.quantile_huber(online, taus, torch.tensor([[-1.0]]), kappa=0.0)), 0.75)


def test_the_asymmetry_weight_is_tau_for_undershoot_and_one_minus_tau_for_overshoot():
    # Inside the Huber's quadratic zone with kappa 1: huber(u)/kappa = u^2 / 2.
    online = torch.tensor([[0.0]])
    taus = torch.tensor([[0.1]])
    up = float(losses.quantile_huber(online, taus, torch.tensor([[0.5]])))
    down = float(losses.quantile_huber(online, taus, torch.tensor([[-0.5]])))
    assert math.isclose(up, 0.1 * 0.125, rel_tol=1e-6)
    assert math.isclose(down, 0.9 * 0.125, rel_tol=1e-6)


def test_target_weights_replace_the_uniform_average_over_target_samples():
    online = torch.tensor([[0.0]])
    taus = torch.tensor([[0.5]])
    target = torch.tensor([[1.0, -3.0]])
    # Pinball at tau 0.5 is |u| / 2: the two samples cost 0.5 and 1.5.
    uniform = losses.quantile_huber(online, taus, target, kappa=0.0)
    assert math.isclose(float(uniform), (0.5 + 1.5) / 2)
    all_first = losses.quantile_huber(online, taus, target, kappa=0.0,
                                      target_weights=torch.tensor([[1.0, 0.0]]))
    assert math.isclose(float(all_first), 0.5)
    skewed = losses.quantile_huber(online, taus, target, kappa=0.0,
                                   target_weights=torch.tensor([[0.25, 0.75]]))
    assert math.isclose(float(skewed), 0.25 * 0.5 + 0.75 * 1.5)


def test_the_loss_has_a_gradient_toward_the_target_only_through_the_online_quantiles():
    online = torch.tensor([[0.0, 0.0]], requires_grad=True)
    taus = torch.tensor([[0.25, 0.75]])
    target = torch.tensor([[2.0, 2.0]], requires_grad=True)
    losses.quantile_huber(online, taus, target).sum().backward()
    assert (online.grad < 0).all(), 'the online quantiles must move up toward a higher target'
    # The target is a constant in the fit: `torch.no_grad()` at the call site, and the loss itself
    # differentiates it only through the Huber, which the agent never asks for.


# ---------------------------------------------------------------- FQF's fraction gradient

def test_the_fraction_gradient_is_zero_when_the_quantile_function_is_linear():
    # F^-1(tau) = 10 tau: equal spacing is optimal, so every interior fraction's gradient is 0.
    n = 4
    taus = torch.tensor([[0.25, 0.5, 0.75]])
    hats = torch.tensor([[0.125, 0.375, 0.625, 0.875]])
    grad = losses.fraction_gradient(10.0 * taus, 10.0 * hats)
    assert torch.allclose(grad, torch.zeros(1, n - 1), atol=1e-6)


def test_the_fraction_gradient_has_the_curvature_sign_of_the_quantile_function():
    # For a convex F^-1 (tau^2) the value at tau_i sits below the mean of the two midpoints' values,
    # so 2F(tau_i) - F(hat_i) - F(hat_{i-1}) < 0; for a concave one (sqrt) it is above, > 0. A linear
    # F^-1 gives 0 (the test above). This is the sign the fraction net descends.
    taus = torch.tensor([[0.5]])
    hats = torch.tensor([[0.25, 0.75]])
    convex = losses.fraction_gradient(taus ** 2, hats ** 2)
    concave = losses.fraction_gradient(taus.sqrt(), hats.sqrt())
    assert math.isclose(float(convex), 2 * 0.25 - 0.5625 - 0.0625)
    assert float(convex) < 0.0 < float(concave)
    assert not convex.requires_grad, 'the gradient is a constant to feed the fraction net'
