"""`DistAgent`: `DdqnAgent` with a distributional head and the loss that fits it.

Acting, the exploration shield, the target copy, the optimiser and the persistence are the parent's,
untouched; only `update` differs, and it differs by head:

| head | target | loss | priority |
|---|---|---|---|
| `c51` | the target net's distribution at the double-Q action (or the Munchausen mixture), shifted by the reward, scaled by the discount, projected onto the support | cross-entropy | the loss |
| `quantile` | the target net's quantiles at that action, shifted and scaled | quantile Huber at the fixed fractions | the loss |
| `iqn` | `n_tau_prime` target samples at fresh fractions | quantile Huber at `n_tau` fresh online fractions | the loss |
| `fqf` | the target's quantiles at the proposed midpoints | quantile Huber at the proposed midpoints; plus the fraction net's own step on its Wasserstein gradient | the loss |

The double-Q action is the argmax of the **online** mean at the next state, as `DdqnAgent` takes it,
so every rung differs from A1 only in the head. With Munchausen on (`alpha > 0`) the next-state target
is the target net's soft policy's **mixture** over actions, each action's distribution shifted by
`-tau log pi(a'|s')`, and the reward carries the clipped log-policy of the taken action -- the M-IQN
form of Vieillard et al. 2020, App. B, applied to every head through the mixture.

Priorities are the per-sample loss, as Rainbow feeds its KL back; `PrioritizedReplay` takes the
absolute value.
"""

import numpy as np
import torch
import torch.nn.functional as F

from algos.dist import losses
from algos.dist import net as network
from algos.dqn.agent import DdqnAgent, build_adam


class DistAgent(DdqnAgent):

    def __init__(self, arch, kappa=1.0, n_tau=64, n_tau_prime=64, fraction_lr=2.5e-9,
                 fraction_entropy=0.001, risk_alpha=1.0, risk_train=False, **kwargs):
        # The parent builds `self.net` and `self.target` through `algos.dqn.net.build`; rebuild them
        # through this package's `build`, keeping everything else the parent set up.
        seed = kwargs.get('seed')
        device = kwargs.get('device', 'cpu')
        super().__init__(arch, **kwargs)
        self.net = network.build(arch, device, seed=seed)
        self.target = network.build(arch, device)
        self.target.load_state_dict(self.net.state_dict())
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.head_type = network.head_of(arch)['type']
        self.kappa = float(kappa)
        self.n_tau, self.n_tau_prime = int(n_tau), int(n_tau_prime)
        self.risk_alpha = float(risk_alpha)
        self.risk_train = bool(risk_train)
        # The value parameters (trunk, embedding, head) on Adam; FQF's fraction net on its own
        # RMSProp, as the authors' code (centred, no momentum, eps 1e-5), so the two rates are
        # independent -- the proposal's is nine orders of magnitude smaller.
        value_parameters = [p for name, p in self.net.named_parameters() if not name.startswith('fraction.')]
        self.optimizer = build_adam(value_parameters, float(kwargs.get('learning_rate', 1e-5)),
                                    float(kwargs.get('adam_epsilon', 1e-7)))
        self.fraction_optimizer = None
        self.fraction_entropy = float(fraction_entropy)
        if self.head_type == 'fqf':
            self.fraction_optimizer = torch.optim.RMSprop(self.net.fraction.parameters(),
                                                          lr=float(fraction_lr), alpha=0.95,
                                                          eps=1e-5, centered=True, momentum=0.0)

    # ---------------------------------------------------------------- resets

    def fresh_net(self, seed):
        return network.build(self.arch, self.device, seed=seed)

    def optimizers(self):
        return [self.optimizer] + ([self.fraction_optimizer] if self.fraction_optimizer is not None else [])

    # ---------------------------------------------------------------- acting

    @property
    def policy_fn(self):
        """The greedy policy the trainer's stage A measures: always the mean read. The risk-sensitive
        *training* policy (`risk_train`) is a property of collection, and the eval's own risk read is a
        pass's `--policy-variant` (`a-return-tail.md` §5)."""
        return network.greedy_policy_fn(self.net, self.device)

    def greedy_actions(self, observations):
        self.net.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=self.device)
            if self.risk_train and self.risk_alpha < 1.0:
                values = self.net.cvar_values(tensor, self.risk_alpha)
            else:
                values = self.net.q_values(tensor)
            return values.argmax(dim=1).cpu().numpy().astype(np.int64)

    # ---------------------------------------------------------------- the soft policy (Munchausen)

    def _soft_policy(self, q_values):
        """`(pi, tau * log pi)` from the target net's mean Q at temperature tau."""
        tau = self.munchausen_tau
        log_pi = tau * F.log_softmax(q_values / tau, dim=1)
        return torch.softmax(q_values / tau, dim=1), log_pi

    def _next_action_mixture(self, obs, action, next_obs):
        """What the target is built from at s': `(weights (B, A), shifts (B, A), munchausen (B,))`.

        Double-Q: weight 1 on the online argmax, no shift, no reward term. Munchausen: the target net's
        soft policy as the weights, `-tau log pi(a'|s')` as each action's shift, and the clipped
        `alpha * tau log pi(a|s)` of the taken action as the reward term.
        """
        batch = obs.shape[0]
        if self.munchausen_alpha <= 0.0:
            best = self.net.q_values(next_obs).argmax(dim=1)
            weights = F.one_hot(best, self.num_actions).float()
            return weights, torch.zeros_like(weights), torch.zeros(batch, device=obs.device)
        _, log_pi_s = self._soft_policy(self.target.q_values(obs))
        munchausen = torch.clamp(log_pi_s.gather(1, action.unsqueeze(1)).squeeze(1),
                                 min=self.munchausen_l0, max=0.0) * self.munchausen_alpha
        pi_next, log_pi_next = self._soft_policy(self.target.q_values(next_obs))
        # If the risk-sensitive *training* policy is on, the paper's agent takes the target's argmax
        # under the distorted read too; the Munchausen soft policy has no argmax to distort, so the
        # two knobs together keep the soft form.
        return pi_next, -log_pi_next, munchausen

    def _double_q_target_action(self, next_obs):
        """The target action for the quantile heads without Munchausen: argmax of the online mean, or
        of its CVaR when the paper's risk-sensitive training form is on (`risk_train`)."""
        if self.risk_train and self.risk_alpha < 1.0:
            return self.net.cvar_values(next_obs, self.risk_alpha).argmax(dim=1)
        return self.net.q_values(next_obs).argmax(dim=1)

    # ---------------------------------------------------------------- the update

    def update(self, batch, weights=None):
        self.net.train()
        obs = torch.as_tensor(batch['obs'], device=self.device)
        next_obs = torch.as_tensor(batch['next_obs'], device=self.device)
        action = torch.as_tensor(batch['action'], device=self.device).long()
        reward = torch.as_tensor(batch['reward'], device=self.device).float()
        discount = torch.as_tensor(batch['discount'], device=self.device).float()

        if self.head_type == 'c51':
            per_sample, fraction_loss = self._categorical_update(obs, action, reward, discount, next_obs), None
        elif self.head_type == 'quantile':
            per_sample, fraction_loss = self._quantile_update(obs, action, reward, discount, next_obs), None
        else:
            per_sample, fraction_loss = self._implicit_update(obs, action, reward, discount, next_obs)

        losses_ = per_sample
        if weights is not None and self.use_is_weights:
            losses_ = losses_ * torch.as_tensor(np.asarray(weights, dtype=np.float32), device=self.device)
        loss = losses_.mean()

        self.optimizer.zero_grad(set_to_none=True)
        if self.fraction_optimizer is not None:
            self.fraction_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if fraction_loss is not None:
            fraction_loss.backward()
        grad_norm = None
        if self.gradient_clipping > 0.0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.gradient_clipping))
        self.optimizer.step()
        if self.fraction_optimizer is not None:
            self.fraction_optimizer.step()

        self.train_step += 1
        self.maybe_update_target()
        self.maybe_reset()
        self.maybe_anneal()
        metrics = {'loss': float(loss.detach()), 'train_step': self.train_step,
                   'mean_abs_td': float(per_sample.detach().abs().mean()), 'resets': self.resets}
        if grad_norm is not None:
            metrics['grad_norm'] = grad_norm
        if fraction_loss is not None:
            metrics['fraction_loss'] = float(fraction_loss.detach())
        return per_sample.detach().cpu().numpy(), metrics

    # ------------------------------------------------------------ c51

    def _categorical_update(self, obs, action, reward, discount, next_obs):
        log_probs = self.net.log_probs(obs).gather(
            1, action.view(-1, 1, 1).expand(-1, 1, self.net.atoms)).squeeze(1)          # (B, atoms)
        with torch.no_grad():
            weights, shifts, munchausen = self._next_action_mixture(obs, action, next_obs)
            target_probs = self.target.probs(next_obs)                                   # (B, A, atoms)
            support = self.net.support
            batch, actions, atoms = target_probs.shape
            # Every action's distribution, shifted by r + alpha*m + gamma*(-tau log pi(a')) and scaled by
            # gamma, projected, then mixed by the weights. Double-Q is the one-hot case of the same code.
            values = (reward + munchausen).view(batch, 1, 1) + discount.view(batch, 1, 1) * (
                support.view(1, 1, atoms) + shifts.view(batch, actions, 1))
            projected = losses.project(target_probs.reshape(batch * actions, atoms),
                                       values.reshape(batch * actions, atoms), support)
            target = (projected.view(batch, actions, atoms) * weights.unsqueeze(2)).sum(dim=1)
        return losses.categorical_cross_entropy(log_probs, target)

    # ------------------------------------------------------------ qr-dqn

    def _quantile_update(self, obs, action, reward, discount, next_obs):
        online = self.net.quantiles(obs).gather(
            1, action.view(-1, 1, 1).expand(-1, 1, self.net.n)).squeeze(1)              # (B, n)
        taus = self.net.fractions(obs.device).unsqueeze(0).expand_as(online)
        with torch.no_grad():
            target, target_weights = self._quantile_target(obs, action, reward, discount, next_obs,
                                                           self.target.quantiles(next_obs))
        return losses.quantile_huber(online, taus, target, kappa=self.kappa, target_weights=target_weights)

    def _quantile_target(self, obs, action, reward, discount, next_obs, target_quantiles,
                         sample_weights=None):
        """`(target samples (B, M), weights (B, M) or None)` from the target net's `(B, A, n)` quantiles.

        Double-Q: the `n` quantiles at the target action, shifted and scaled, weighted by
        `sample_weights (B, n)` if given (FQF's bin widths) else 1/n. Munchausen: every action's `n`
        quantiles shifted by its own `-tau log pi`, all `A * n` of them, weighted `pi(a') * w_j`.
        """
        batch, actions, n = target_quantiles.shape
        if sample_weights is None:
            sample_weights = torch.full((batch, n), 1.0 / n, device=target_quantiles.device)
        if self.munchausen_alpha <= 0.0:
            best = self._double_q_target_action(next_obs)
            chosen = target_quantiles.gather(1, best.view(-1, 1, 1).expand(-1, 1, n)).squeeze(1)
            return reward.unsqueeze(1) + discount.unsqueeze(1) * chosen, sample_weights
        weights, shifts, munchausen = self._next_action_mixture(obs, action, next_obs)
        shifted = (reward + munchausen).view(batch, 1, 1) + discount.view(batch, 1, 1) * (
            target_quantiles + shifts.unsqueeze(2))                                      # (B, A, n)
        mixture_weights = weights.unsqueeze(2) * sample_weights.unsqueeze(1)             # (B, A, n)
        return shifted.reshape(batch, actions * n), mixture_weights.reshape(batch, actions * n)

    # ------------------------------------------------------------ iqn / fqf

    def _implicit_update(self, obs, action, reward, discount, next_obs):
        features = self.net.features(obs)
        batch = obs.shape[0]
        fraction_loss = None
        if self.head_type == 'fqf':
            taus_full, tau_hats, entropy = self.net.propose(features)
            # Detached: the value loss must not move the fraction net; its own loss below does.
            online_taus = tau_hats.detach()                                               # (B, n)
            with torch.no_grad():
                target_features = self.target.features(next_obs)
                target_taus_full, target_hats, _ = self.target.propose(target_features)
            # The fraction net's own loss: its Wasserstein gradient against the *online* quantile
            # function at the taus and the hats, both read without gradient to the value parameters,
            # plus the entropy bonus of the released code.
            with torch.no_grad():
                at_hats = self.net.quantiles_at(features, tau_hats).gather(
                    2, action.view(-1, 1, 1).expand(-1, tau_hats.shape[1], 1)).squeeze(2)
                at_taus = self.net.quantiles_at(features, taus_full[:, 1:-1]).gather(
                    2, action.view(-1, 1, 1).expand(-1, taus_full.shape[1] - 2, 1)).squeeze(2)
                gradient = losses.fraction_gradient(at_taus, at_hats)
            fraction_loss = (gradient * taus_full[:, 1:-1]).sum(dim=1).mean() \
                - self.fraction_entropy * entropy.mean()
            target_sample_taus = target_hats
        else:
            online_taus = self.net.sample_taus(batch, self.n_tau)
            target_sample_taus = self.net.sample_taus(batch, self.n_tau_prime)
            with torch.no_grad():
                target_features = self.target.features(next_obs)

        online = self.net.quantiles_at(features, online_taus).gather(
            2, action.view(-1, 1, 1).expand(-1, online_taus.shape[1], 1)).squeeze(2)   # (B, N)
        with torch.no_grad():
            # The paper's risk-sensitive agent (`risk_train`) distorts the fractions the *argmax* is
            # taken under -- `_double_q_target_action` -- and reads the target's values at
            # undistorted fractions, which is what these are.
            target_quantiles = self.target.quantiles_at(target_features, target_sample_taus)
            target_quantiles = target_quantiles.transpose(1, 2)                          # (B, A, M)
            # FQF weights the target's samples by their bins' widths rather than 1/M.
            sample_weights = None
            if self.head_type == 'fqf':
                sample_weights = target_taus_full[:, 1:] - target_taus_full[:, :-1]
            target, target_weights = self._quantile_target(obs, action, reward, discount, next_obs,
                                                           target_quantiles, sample_weights)
        per_sample = losses.quantile_huber(online, online_taus, target, kappa=self.kappa,
                                           target_weights=target_weights)
        return per_sample, fraction_loss

    # ---------------------------------------------------------------- persistence

    def state_dict(self):
        state = super().state_dict()
        if self.fraction_optimizer is not None:
            state['fraction_optimizer'] = self.fraction_optimizer.state_dict()
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if self.fraction_optimizer is not None and state.get('fraction_optimizer') is not None:
            self.fraction_optimizer.load_state_dict(state['fraction_optimizer'])
