"""`RainbowAgent`: `DistAgent`'s plumbing over a `RainbowNet`, acting with the noise on, and **its own update**.

What is this module's and not the parent's:

| | |
|---|---|
| the network | `algos/rainbow/net.py`'s, rebuilt over the parent's (which builds A's head for the same sidecar) |
| acting | **noise on** (`c-value-stack.md`, decided 2026-09-20): the collector's greedy action is a noisy forward, one fresh draw per act, and epsilon-greedy over that when the schedule says so. Rainbow's paper cell runs epsilon 0, so this is its whole exploration |
| measuring | `policy_fn` is `net.greedy_policy_fn`: noise off. Stage A, restore, `watch.py` and every pass read the `mu` weights |
| learning | `update` below, with the noise on in both the online and the target net so each forward is its own draw (Fortunato et al. §3.2: independent noise for the online and target networks) |
| double-Q | `double=False` (BTR) takes the target action from the **target** net's own argmax; with Munchausen on there is no argmax and the flag has no effect, as the plan says |

**Why the update is this module's** (2026-09-23, the review of `c-value-stack.md`'s implementation):
Group A's `DistAgent.update` departs from both papers in three places, and Group A's closed results
were measured on it, so it is left as it is and the papers' forms live here:

| | Group A (`algos/dist/`) | here, as the papers and the BTR code |
|---|---|---|
| Munchausen target, quantile heads | the soft policy's **mixture**: all `A x M` shifted samples, weighted `pi(a') / M` | **averaged over actions inside each target sample**: `sum_a' pi(a') (Z_j(s', a') - tau log pi(a'))`, `M` targets (Vieillard et al. 2020 App. B; BTR `Agent.py` `.sum(2)`). `pi` is read off the same target samples |
| quantile loss reduction | mean over the online quantiles, mean over the targets | **sum** over the online quantiles, mean over the targets (Dabney et al. 2018 Eq. 3, IQN Eq. 10, BTR): `N` times Group A's, which matters through the gradient clip and Adam's epsilon |
| priority | the per-sample loss | c51: the **KL** of the target from the online distribution, `CE - H(target)` (Rainbow §3); quantile heads: the **pairwise absolute TD error**, summed over online and averaged over target (BTR `loss_v`) |

The c51 Munchausen target stays the mixture: a categorical target has no sample index to average
within, and the projected mixture is the distribution the soft policy implies.

`munchausen_logpi` picks the net that reads the taken action's `tau log pi(a|s)`: `target` as
Vieillard et al., or `online` as BTR's released code (`self.net.qvals(states)`).
"""

import numpy as np
import torch
import torch.nn.functional as F

from algos.dist import losses
from algos.dist.agent import DistAgent
from algos.dqn.agent import build_adam
from algos.rainbow import net as network

MUNCHAUSEN_LOGPI = ('target', 'online')


def quantile_loss(online, taus, target, kappa):
    """`(loss (B,), priority (B,))` for `online (B, N)` at `taus (B, N)` against `target (B, M)`.

    The loss sums the pairwise quantile Huber over the `N` online quantiles and averages it over the `M`
    targets -- Group A's `losses.quantile_huber` (which averages both) times `N`. The priority is the
    absolute pairwise TD error under the same reduction.
    """
    loss = losses.quantile_huber(online, taus, target, kappa=kappa) * online.shape[1]
    priority = (target.unsqueeze(1) - online.unsqueeze(2)).detach().abs().sum(dim=1).mean(dim=1)
    return loss, priority


def categorical_kl(log_probs, target):
    """`KL(target || online)` per sample: the cross-entropy less the target's entropy, `0 log 0 = 0`."""
    return (torch.xlogy(target, target) - target * log_probs).sum(dim=1).clamp_min(0.0)


class RainbowAgent(DistAgent):

    def __init__(self, arch, double=True, munchausen_logpi='target', **kwargs):
        seed = kwargs.get('seed')
        device = kwargs.get('device', 'cpu')
        super().__init__(arch, **kwargs)
        self.net = network.build(arch, device, seed=seed)
        self.target = network.build(arch, device)
        self.target.load_state_dict(self.net.state_dict())
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        # Eval mode for good: a spectral-normed linear runs a power iteration on every training-mode
        # forward, and a target whose weights drift between copies is not a target. Its `_u`/`_v` arrive
        # with each hard copy, so its normalised weight is the online net's at the copy.
        self.target.eval()
        self.optimizer = build_adam(self.net.parameters(), float(kwargs.get('learning_rate', 1e-5)),
                                    float(kwargs.get('adam_epsilon', 1e-7)))
        self.double = bool(double)
        self.noisy = bool(network.trunk_of(arch)['noisy'])
        if munchausen_logpi not in MUNCHAUSEN_LOGPI:
            raise ValueError('munchausen_logpi must be one of {0}, got {1!r}'.format(MUNCHAUSEN_LOGPI,
                                                                                    munchausen_logpi))
        self.munchausen_logpi = munchausen_logpi

    # ---------------------------------------------------------------- acting

    @property
    def policy_fn(self):
        return network.greedy_policy_fn(self.net, self.device)

    def greedy_actions(self, observations):
        """The collector's greedy action: the noisy network's argmax, a fresh draw per call."""
        self.net.eval()
        network.set_noise(self.net, self.noisy)
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=self.device)
            return self.net.q_values(tensor).argmax(dim=1).cpu().numpy().astype(np.int64)

    def fresh_net(self, seed):
        return network.build(self.arch, self.device, seed=seed)

    # ---------------------------------------------------------------- the target action

    def _double_q_target_action(self, next_obs):
        if self.double:
            return super()._double_q_target_action(next_obs)
        return self.target.q_values(next_obs).argmax(dim=1)

    def _next_action_mixture(self, obs, action, next_obs):
        if self.double or self.munchausen_alpha > 0.0:
            return super()._next_action_mixture(obs, action, next_obs)
        best = self.target.q_values(next_obs).argmax(dim=1)
        weights = F.one_hot(best, self.num_actions).float()
        return weights, torch.zeros_like(weights), torch.zeros(obs.shape[0], device=obs.device)

    def _munchausen_reward(self, obs, action):
        """`alpha * clip(tau log pi(a|s), l0, 0)` of the taken action, read off `munchausen_logpi`'s net."""
        source = self.net if self.munchausen_logpi == 'online' else self.target
        _, log_pi = self._soft_policy(source.q_values(obs))
        return torch.clamp(log_pi.gather(1, action.unsqueeze(1)).squeeze(1),
                           min=self.munchausen_l0, max=0.0) * self.munchausen_alpha

    # ---------------------------------------------------------------- the update

    def update(self, batch, weights=None):
        network.set_noise(self.net, self.noisy)
        network.set_noise(self.target, self.noisy)
        self.net.train()
        obs = torch.as_tensor(batch['obs'], device=self.device)
        next_obs = torch.as_tensor(batch['next_obs'], device=self.device)
        action = torch.as_tensor(batch['action'], device=self.device).long()
        reward = torch.as_tensor(batch['reward'], device=self.device).float()
        discount = torch.as_tensor(batch['discount'], device=self.device).float()

        if self.head_type == 'c51':
            per_sample, priority = self._categorical(obs, action, reward, discount, next_obs)
        else:
            per_sample, priority = self._quantile(obs, action, reward, discount, next_obs)

        losses_ = per_sample
        if weights is not None and self.use_is_weights:
            losses_ = losses_ * torch.as_tensor(np.asarray(weights, dtype=np.float32), device=self.device)
        loss = losses_.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = None
        if self.gradient_clipping > 0.0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.gradient_clipping))
        self.optimizer.step()

        self.train_step += 1
        self.maybe_update_target()
        self.maybe_reset()
        self.maybe_anneal()
        priority = priority.detach()
        metrics = {'loss': float(loss.detach()), 'train_step': self.train_step,
                   'mean_abs_td': float(priority.mean()), 'resets': self.resets}
        if grad_norm is not None:
            metrics['grad_norm'] = grad_norm
        return priority.cpu().numpy(), metrics

    def _categorical(self, obs, action, reward, discount, next_obs):
        """Cross-entropy to learn from, KL to prioritise by. The target is Group A's: the projected
        distribution at the target action, or the projected Munchausen mixture."""
        log_probs = self.net.log_probs(obs).gather(
            1, action.view(-1, 1, 1).expand(-1, 1, self.net.atoms)).squeeze(1)          # (B, atoms)
        with torch.no_grad():
            weights, shifts, munchausen = self._next_action_mixture(obs, action, next_obs)
            if self.munchausen_alpha > 0.0:
                munchausen = self._munchausen_reward(obs, action)
            target_probs = self.target.probs(next_obs)                                   # (B, A, atoms)
            support = self.net.support
            batch, actions, atoms = target_probs.shape
            values = (reward + munchausen).view(batch, 1, 1) + discount.view(batch, 1, 1) * (
                support.view(1, 1, atoms) + shifts.view(batch, actions, 1))
            projected = losses.project(target_probs.reshape(batch * actions, atoms),
                                       values.reshape(batch * actions, atoms), support)
            target = (projected.view(batch, actions, atoms) * weights.unsqueeze(2)).sum(dim=1)
        return losses.categorical_cross_entropy(log_probs, target), categorical_kl(log_probs.detach(), target)

    def _quantile(self, obs, action, reward, discount, next_obs):
        """QR-DQN's fixed fractions or IQN's sampled ones; the target per sample index."""
        if self.head_type == 'quantile':
            online = self.net.quantiles(obs).gather(
                1, action.view(-1, 1, 1).expand(-1, 1, self.net.n)).squeeze(1)          # (B, N)
            taus = self.net.fractions(obs.device).unsqueeze(0).expand_as(online)
            with torch.no_grad():
                target_quantiles = self.target.quantiles(next_obs)                       # (B, A, M)
        else:
            features = self.net.features(obs)
            taus = self.net.sample_taus(obs.shape[0], self.n_tau)
            online = self.net.quantiles_at(features, taus).gather(
                2, action.view(-1, 1, 1).expand(-1, taus.shape[1], 1)).squeeze(2)      # (B, N)
            with torch.no_grad():
                target_taus = self.net.sample_taus(obs.shape[0], self.n_tau_prime)
                target_quantiles = self.target.quantiles_at(self.target.features(next_obs),
                                                            target_taus).transpose(1, 2)  # (B, A, M)
        with torch.no_grad():
            target = self._quantile_target(obs, action, reward, discount, next_obs, target_quantiles)
        return quantile_loss(online, taus, target, self.kappa)

    def _quantile_target(self, obs, action, reward, discount, next_obs, target_quantiles):
        """`(B, M)` target samples from the target net's `(B, A, M)`.

        Double-Q or the target argmax: the `M` samples at that action, shifted and scaled. Munchausen:
        for each sample index `j`, `sum_a' pi(a') (Z_j(a') - tau log pi(a'))`, with `pi` the soft policy
        of the target's mean over these same samples -- M-IQN's form and BTR's code.
        """
        batch, actions, m = target_quantiles.shape
        if self.munchausen_alpha <= 0.0:
            best = self._double_q_target_action(next_obs)
            chosen = target_quantiles.gather(1, best.view(-1, 1, 1).expand(-1, 1, m)).squeeze(1)
            return reward.unsqueeze(1) + discount.unsqueeze(1) * chosen
        pi_next, log_pi_next = self._soft_policy(target_quantiles.mean(dim=2))          # (B, A)
        soft = (pi_next.unsqueeze(2) * (target_quantiles - log_pi_next.unsqueeze(2))).sum(dim=1)  # (B, M)
        munchausen = self._munchausen_reward(obs, action)
        return (reward + munchausen).unsqueeze(1) + discount.unsqueeze(1) * soft
