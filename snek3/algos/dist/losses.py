"""The distributional losses, as pure tensor functions with the arithmetic that the fixtures pin.

| function | paper | fits |
|---|---|---|
| `project` | Bellemare, Dabney & Munos 2017, Alg. 1 | a categorical distribution shifted and scaled onto a fixed support |
| `categorical_cross_entropy` | the same | the online log-probabilities against a projected target |
| `quantile_huber` | Dabney, Rowland, Bellemare & Munos 2018, Eq. 9-10 | online quantiles at fractions tau against target samples, optionally weighted |
| `fraction_gradient` | Yang, Zhao, Du, Wei & Liu 2019, Eq. 10 | the fraction proposal's Wasserstein gradient, closed form |

Every function is per-sample (`(m,)` losses), so prioritised replay's importance weights apply outside.
"""

import torch
import torch.nn.functional as F


def project(probs, values, support):
    """Project mass `probs` sitting at `values` onto the fixed `support`. `(B, n) x (B, n) -> (B, atoms)`.

    Each value is clipped into `[v_min, v_max]`, its position on the support computed, and its mass
    split between the two neighbouring atoms in proportion to its distance from each (a value exactly
    on an atom gives that atom everything). The rows of the result sum to the rows of `probs`.
    """
    atoms = support.shape[0]
    v_min, v_max = float(support[0]), float(support[-1])
    delta = (v_max - v_min) / (atoms - 1)
    clipped = torch.clamp(values, v_min, v_max)
    position = (clipped - v_min) / delta                                   # (B, n), in [0, atoms-1]
    lower = position.floor()
    upper = position.ceil()
    # A value exactly on an atom has lower == upper; give it the whole mass once, not twice.
    on_atom = (upper == lower)
    weight_upper = position - lower
    weight_lower = 1.0 - weight_upper
    weight_lower = torch.where(on_atom, torch.ones_like(weight_lower), weight_lower)
    weight_upper = torch.where(on_atom, torch.zeros_like(weight_upper), weight_upper)
    out = torch.zeros(probs.shape[0], atoms, device=probs.device, dtype=probs.dtype)
    out.scatter_add_(1, lower.long(), probs * weight_lower)
    out.scatter_add_(1, upper.long(), probs * weight_upper)
    return out


def categorical_cross_entropy(log_probs, target_probs):
    """`-(sum_atoms target * log online)`, per sample: `(B, atoms) x (B, atoms) -> (B,)`."""
    return -(target_probs * log_probs).sum(dim=1)


def quantile_huber(online, taus, target, kappa=1.0, target_weights=None):
    """The quantile Huber loss. `online (B, N)` at fractions `taus (B, N)`, `target (B, M)` samples.

    For every pair (i, j): u = target_j - online_i; rho = |tau_i - 1[u < 0]| * huber_kappa(u) / kappa
    (kappa -> 0 recovers the pinball loss, which is why the division is there). The result sums over
    the target samples (weighted by `target_weights (B, M)` when given, else 1/M each) and averages
    over the online quantiles, as the papers write it.
    """
    u = target.unsqueeze(1) - online.unsqueeze(2)                          # (B, N, M)
    if kappa > 0.0:
        huber = F.huber_loss(online.unsqueeze(2).expand_as(u), target.unsqueeze(1).expand_as(u),
                             reduction='none', delta=kappa) / kappa
    else:
        huber = u.abs()
    indicator = (u.detach() < 0.0).float()
    rho = (taus.unsqueeze(2) - indicator).abs() * huber                    # (B, N, M)
    if target_weights is None:
        per_online = rho.mean(dim=2)
    else:
        per_online = (rho * target_weights.unsqueeze(1)).sum(dim=2)
    return per_online.mean(dim=1)


def fraction_gradient(values_at_taus, values_at_hats):
    """FQF's closed-form gradient of the 1-Wasserstein loss with respect to the interior fractions.

    `values_at_taus (B, n-1)` are F^-1 at tau_1..tau_{n-1}; `values_at_hats (B, n)` at the midpoints
    tau_hat_0..tau_hat_{n-1}. dW/dtau_i = 2 F^-1(tau_i) - F^-1(tau_hat_i) - F^-1(tau_hat_{i-1}).
    Detached: it is a gradient to feed the fraction net, not a quantity to differentiate.
    """
    return (2.0 * values_at_taus - values_at_hats[:, 1:] - values_at_hats[:, :-1]).detach()
