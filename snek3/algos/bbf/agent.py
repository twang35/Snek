"""`BbfAgent`: Bigger, Better, Faster (Schwarzer et al. 2023) as one agent -- the paper's recipe on this
game's MLP, and nothing of this codebase's plumbing.

| piece | the paper | here |
|---|---|---|
| network | IMPALA CNN x4 width, dueling C51 (51 atoms), no noisy nets | `algos/rainbow/net.py`'s `RainbowNet` with `dueling` on and `noisy` off, the trunk `SNEK_FC_LAYERS` (the plan's analogue of x4 is `1280,2048`); the checkpoint is that net alone, so `tools/restore.py` reads a `bbf` checkpoint as a Rainbow one |
| target | EMA, tau 0.005, every update | `ema_update` on the Q net and on the SPR heads, every update |
| optimiser | AdamW lr 1e-4, eps 1.5e-4, weight decay 0.1, grad clip 10 | one AdamW over the Q net and the SPR heads |
| TD loss | C51 cross-entropy on the n-step target, double-Q (target net at the online argmax), no reward clipping | `algos/dist/losses.project` and `categorical_cross_entropy`; the n-step return and `gamma ** n` arrive **in the batch**, computed by `SequentialReplay.sample` at the cycle's current n and gamma |
| SPR | latents rolled K = 5 steps through a transition model on (latent, action), projected and predicted, against the target encoder's projection of the true next observations; normalised-L2, weight 5 | `SprHeads`: transition `Linear(width + actions, width) -> ReLU -> Linear(width, width)`, projection `Linear(width, dim)`, predictor `Linear(dim, dim)`; the target latent is the **EMA target net's** trunk and the EMA copy of the projection. Loss `2 - 2 cos` per step, masked past a terminal, summed over K, mean over the batch |
| resets | every 40k gradient steps: encoder and transition model shrink-and-perturbed (alpha 0.5), projection, predictor and Q head re-initialised, optimiser cleared, target copied | `resets.shrink_and_perturb` on the Q net (trunk shrunk, streams replaced), the same rule by hand on the SPR heads; `resets.clear_optimizer`; both targets copied. The cycle restarts (`last_reset_step`) and the algorithm reads the anneal off `cycle_values()` |
| exploration | epsilon-greedy, 1 -> 0 over 2001 moves after the 2000-move prefill | `act` is uniform epsilon-greedy; the schedule is the algorithm's |
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from algos.dist import losses
from algos.dqn import resets
from algos.rainbow import net as network


def ema_update(target, online, tau):
    """`target <- (1 - tau) target + tau online`, parameters and buffers."""
    tau = float(tau)
    with torch.no_grad():
        for t, o in zip(target.parameters(), online.parameters()):
            t.mul_(1.0 - tau).add_(o, alpha=tau)
        for t, o in zip(target.buffers(), online.buffers()):
            t.copy_(o)


def spr_loss(predictions, targets, mask):
    """`(B, K, dim) x (B, K, dim) x (B, K) -> (B,)`: the normalised-L2 distance `2 - 2 cos` at every step
    still inside the episode, summed over the K steps."""
    predictions = F.normalize(predictions, dim=-1)
    targets = F.normalize(targets, dim=-1)
    per_step = 2.0 - 2.0 * (predictions * targets).sum(dim=-1)
    return (per_step * mask).sum(dim=1)


class SprHeads(nn.Module):
    """The transition model, the projection and the predictor. Names matter: `transition.*` is shrunk at a
    reset like the encoder, everything else is replaced, as BBF's `reset_weights` does."""

    def __init__(self, width, num_actions, projection_dim, generator=None, transition_width=0):
        super().__init__()
        self.num_actions = int(num_actions)
        # BBF's transition model is two 64-channel convolutions beside an encoder many times their size; the
        # analogue is a hidden width well under the latent's. 0 means the latent's own width.
        hidden = int(transition_width) if int(transition_width) > 0 else int(width)
        self.transition = nn.Sequential(nn.Linear(width + self.num_actions, hidden), nn.ReLU(), nn.Linear(hidden, width))
        self.projection = nn.Linear(width, projection_dim)
        self.predictor = nn.Linear(projection_dim, projection_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                bound = 1.0 / float(module.weight.shape[1]) ** 0.5
                nn.init.uniform_(module.weight, -bound, bound, generator=generator)
                nn.init.zeros_(module.bias)

    def step(self, latent, action):
        one_hot = F.one_hot(action.long(), self.num_actions).float()
        return torch.relu(self.transition(torch.cat([latent, one_hot], dim=-1)))

    def rollout(self, latent, actions):
        """`(B, width) x (B, K) -> (B, K, width)`: the latents K steps on."""
        out = []
        for k in range(actions.shape[1]):
            latent = self.step(latent, actions[:, k])
            out.append(latent)
        return torch.stack(out, dim=1)

    def predict(self, latents):
        return self.predictor(self.projection(latents))


class BbfAgent(object):

    def __init__(self, arch, learning_rate=1e-4, adam_epsilon=1.5e-4, weight_decay=0.1, gradient_clipping=10.0,
                 target_tau=0.005, seed=None, device='cpu', double=True, spr_weight=5.0, spr_steps=5,
                 projection_dim=512, transition_width=0, reset_interval=40000, reset_alpha=0.5, reset_stop_after=0,
                 cycle=None):
        self.arch = arch
        self.device = device
        self.seed = seed
        self.num_actions = int(arch['num_actions'])
        self.gradient_clipping = float(gradient_clipping)
        self.target_tau = float(target_tau)
        self.double = bool(double)
        self.spr_weight = float(spr_weight)
        self.spr_steps = int(spr_steps)
        self.projection_dim = int(projection_dim)
        self.transition_width = int(transition_width)
        self.learning_rate, self.adam_epsilon, self.weight_decay = float(learning_rate), float(adam_epsilon), float(weight_decay)
        if network.head_of(arch)['type'] != 'c51':
            raise ValueError('BBF is dueling C51; the arch head is {0!r}'.format(arch['head']))
        self.net = network.build(arch, device, seed=seed)
        self.target = network.build(arch, device)
        self.target.load_state_dict(self.net.state_dict())
        self.spr = self._fresh_spr(seed)
        self.target_spr = self._fresh_spr(None)
        self.target_spr.load_state_dict(self.spr.state_dict())
        for module in (self.target, self.target_spr):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.optimizer = self._build_optimizer()
        self.rng = np.random.default_rng(seed)
        self.train_step = 0
        self.reset_schedule = resets.ResetSchedule(reset_interval, reset_alpha, reset_stop_after)
        self.cycle = cycle if cycle is not None else resets.CycleSchedule()
        self.resets = 0
        self.last_reset_step = 0
        self.last = {}

    def _fresh_spr(self, seed):
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed) + 7)
        return SprHeads(self.net.trunk.width, self.num_actions, self.projection_dim, generator,
                        transition_width=self.transition_width).to(self.device)

    def _build_optimizer(self):
        return torch.optim.AdamW(list(self.net.parameters()) + list(self.spr.parameters()),
                                 lr=self.learning_rate, eps=self.adam_epsilon, weight_decay=self.weight_decay)

    # ---------------------------------------------------------------- acting

    @property
    def policy_fn(self):
        return network.greedy_policy_fn(self.net, self.device)

    def greedy_actions(self, observations):
        self.net.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=self.device)
            return self.net.q_values(tensor).argmax(dim=1).cpu().numpy().astype(np.int64)

    def act(self, observations, epsilon, guided=False):
        """Uniform epsilon-greedy; `guided` (the shield) is ignored, BBF has none."""
        observations = np.asarray(observations, dtype=np.float32)
        actions = self.greedy_actions(observations)
        explore = self.rng.random(observations.shape[0]) < float(epsilon)
        if not explore.any():
            return actions
        drawn = self.rng.integers(0, self.num_actions, size=observations.shape[0])
        return np.where(explore, drawn, actions).astype(np.int64)

    # ---------------------------------------------------------------- the cycle

    @property
    def steps_since_reset(self):
        return self.train_step - self.last_reset_step

    def cycle_values(self):
        since = self.steps_since_reset
        return self.cycle.n_step_at(since), self.cycle.gamma_at(since)

    # ---------------------------------------------------------------- learning

    def _categorical_loss(self, obs, action, reward, discount, next_obs):
        log_probs = self.net.log_probs(obs).gather(1, action.view(-1, 1, 1).expand(-1, 1, self.net.atoms)).squeeze(1)
        with torch.no_grad():
            if self.double:
                best = self.net.q_values(next_obs).argmax(dim=1)
            else:
                best = self.target.q_values(next_obs).argmax(dim=1)
            target_probs = self.target.probs(next_obs).gather(
                1, best.view(-1, 1, 1).expand(-1, 1, self.net.atoms)).squeeze(1)              # (B, atoms)
            support = self.net.support
            values = reward.view(-1, 1) + discount.view(-1, 1) * support.view(1, -1)
            target = losses.project(target_probs, values, support)
        return losses.categorical_cross_entropy(log_probs, target)

    def _spr_loss(self, obs, batch):
        if self.spr_weight <= 0.0 or self.spr_steps <= 0:
            return torch.zeros(obs.shape[0], device=self.device)
        spr_actions = torch.as_tensor(batch['spr_action'], device=self.device).long()[:, :self.spr_steps]
        spr_next = torch.as_tensor(batch['spr_next_obs'], device=self.device)[:, :self.spr_steps]
        mask = torch.as_tensor(batch['spr_mask'], device=self.device).float()[:, :self.spr_steps]
        latents = self.spr.rollout(self.net.features(obs), spr_actions)                          # (B, K, width)
        predictions = self.spr.predict(latents)
        with torch.no_grad():
            flat = spr_next.reshape(-1, spr_next.shape[-1])
            targets = self.target_spr.projection(self.target.features(flat)).view(
                spr_next.shape[0], spr_next.shape[1], -1)
        return spr_loss(predictions, targets, mask)

    def update(self, batch, weights=None):
        """One AdamW step on the C51 loss plus `spr_weight` times the SPR loss. Returns `(per-row TD loss for
        the priorities, metrics)`."""
        self.net.train()
        self.spr.train()
        obs = torch.as_tensor(batch['obs'], device=self.device)
        next_obs = torch.as_tensor(batch['next_obs'], device=self.device)
        action = torch.as_tensor(batch['action'], device=self.device).long()
        reward = torch.as_tensor(batch['reward'], device=self.device).float()
        discount = torch.as_tensor(batch['discount'], device=self.device).float()

        td = self._categorical_loss(obs, action, reward, discount, next_obs)
        weighted = td
        if weights is not None:
            weighted = td * torch.as_tensor(np.asarray(weights, dtype=np.float32), device=self.device)
        spr = self._spr_loss(obs, batch)
        loss = weighted.mean() + self.spr_weight * spr.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = None
        if self.gradient_clipping > 0.0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                list(self.net.parameters()) + list(self.spr.parameters()), self.gradient_clipping))
        self.optimizer.step()

        self.train_step += 1
        ema_update(self.target, self.net, self.target_tau)
        ema_update(self.target_spr, self.spr, self.target_tau)
        self.maybe_reset()
        self.last = {'loss': float(loss.detach()), 'td_loss': float(td.detach().mean()),
                     'spr_loss': float(spr.detach().mean()), 'train_step': self.train_step, 'resets': self.resets}
        if grad_norm is not None:
            self.last['grad_norm'] = grad_norm
        return td.detach().cpu().numpy(), dict(self.last)

    # ---------------------------------------------------------------- resets

    def maybe_reset(self):
        if not self.reset_schedule.due(self.train_step):
            return False
        fresh_seed = resets.reset_seed(self.seed if self.seed is not None else 0, self.resets + 1)
        fresh = network.build(self.arch, self.device, seed=fresh_seed)
        resets.shrink_and_perturb(self.net, fresh, self.reset_schedule.alpha)
        fresh_spr = self._fresh_spr(fresh_seed)
        alpha = float(self.reset_schedule.alpha)
        with torch.no_grad():
            for (name, parameter), new in zip(self.spr.named_parameters(), fresh_spr.parameters()):
                if name.startswith('transition.'):
                    parameter.mul_(alpha).add_(new, alpha=1.0 - alpha)
                else:
                    parameter.copy_(new)
        self.target.load_state_dict(self.net.state_dict())
        self.target_spr.load_state_dict(self.spr.state_dict())
        resets.clear_optimizer(self.optimizer)
        self.resets += 1
        self.last_reset_step = self.train_step
        return True

    # ---------------------------------------------------------------- persistence

    def state_dict(self):
        return {'net': self.net.state_dict(), 'target': self.target.state_dict(),
                'spr': self.spr.state_dict(), 'target_spr': self.target_spr.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'train_step': self.train_step,
                'rng': self.rng.bit_generator.state, 'resets': self.resets, 'last_reset_step': self.last_reset_step}

    def load_state_dict(self, state):
        self.net.load_state_dict(state['net'])
        self.target.load_state_dict(state['target'])
        self.spr.load_state_dict(state['spr'])
        self.target_spr.load_state_dict(state['target_spr'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.train_step = int(state['train_step'])
        self.resets = int(state.get('resets', 0))
        self.last_reset_step = int(state.get('last_reset_step', 0))
        if state.get('rng') is not None:
            self.rng.bit_generator.state = state['rng']
