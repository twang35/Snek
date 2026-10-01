"""`R2d2Agent`: the learner of Kapturowski et al. 2019 (`plans/algoExploration/e-memory.md` §2), on one
fixed-grid batch from `SequenceReplay`.

| piece | the paper | here |
|---|---|---|
| burn-in | the first 40 of a stored 80 replayed from the stored state, no gradient, online and target nets each | `unroll` over slots `[0, burn_in)` under `no_grad`, both nets, from the window's stored state; a window with fewer burn-in rows is `fresh` at its first real row and starts from zero there |
| loss | squared TD on the n-step double-Q target at every loss step, summed over the steps and averaged over the batch; IS-weighted per sequence | the same, over slots `[burn_in, burn_in + block)` masked by `valid`; the bootstrap at slot `t + n` is the target net at the online argmax; a step past the episode's end contributes nothing |
| rescaling | `h(x) = sign(x)(sqrt(|x| + 1) - 1) + eps x`; target `h(R + gamma^n h^-1(Q_target))` | `rescale` / `unscale` below, `eps` 1e-3; **off** (`rescale=False`) for the C51 recipe, whose support is the return itself |
| priority | `eta max|delta| + (1 - eta) mean|delta|` over the sequence, eta 0.9 | computed here per window, handed back for `update_priorities` |
| target | a copy every 2500 updates | `target_update_period` |
| optimiser | Adam 1e-4, eps 1e-3, grad clip 40 | the same |
| acting | epsilon-greedy per actor on its own epsilon | `act` takes one epsilon per lane |

**The C51 recipe** (`head: c51`, the plan's §2b): the same burn-in and geometry, the loss the categorical
cross-entropy against the projected n-step target distribution at the double-Q action (`algos/dist/losses`),
the priority the same eta-mix over the per-step cross-entropy, rescaling refused at construction.
"""

import numpy as np
import torch

from algos.dist import losses
from algos.r2d2 import net as network


def rescale(x, eps=1e-3):
    """`h(x) = sign(x) (sqrt(|x| + 1) - 1) + eps x` (Pohlen et al. 2018, R2D2 §2.3)."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def unscale(x, eps=1e-3):
    """`h^-1(x) = sign(x) (((sqrt(1 + 4 eps (|x| + 1 + eps)) - 1) / (2 eps))^2 - 1)`."""
    inner = (torch.sqrt(1.0 + 4.0 * eps * (torch.abs(x) + 1.0 + eps)) - 1.0) / (2.0 * eps)
    return torch.sign(x) * (inner * inner - 1.0)


def n_step_targets(reward, done, valid, bootstrap, gamma, n, first, length):
    """`(T, B)` rewards, done and valid flags, `(T, B)` bootstrap values (the target's value of slot t's
    state) -> `(L, B)` returns for the `L = length` loss slots starting at `first`, and the `(L, B)` mask of
    slots that carry a loss.

    For loss slot t: `sum_k gamma^k r_{t+k}` over k < n while the step before t+k was not a terminal and the
    row exists, plus `gamma^n V(s_{t+n})` when the n rows after t all exist and none ended the episode. A row
    that ended the episode contributes its reward and nothing after it.
    """
    T = reward.shape[0]
    returns, masks = [], []
    for t in range(first, first + length):
        total = torch.zeros_like(reward[0])
        alive = valid[t].to(reward.dtype)                       # the slot itself must hold a row
        discount = torch.ones_like(reward[0])
        for k in range(n):
            slot = t + k
            if slot >= T:
                alive = torch.zeros_like(alive)
                break
            total = total + alive * discount * reward[slot]
            # Past a terminal row nothing follows; past a missing row the window has no more to say.
            alive = alive * (1.0 - done[slot].to(reward.dtype))
            discount = discount * gamma
            if slot + 1 < T:
                alive = alive * valid[slot + 1].to(reward.dtype)
            else:
                alive = torch.zeros_like(alive)
        if t + n < T:
            total = total + alive * discount * bootstrap[t + n]
        returns.append(total)
        masks.append(valid[t])
    return torch.stack(returns, dim=0), torch.stack(masks, dim=0)


def mix_priority(abs_td, mask, eta):
    """`(L, B) |delta| x (L, B) mask -> (B,)`: `eta max + (1 - eta) mean` over the steps that carried a loss, the
    paper's sequence priority (eta 0.9)."""
    magnitude = abs_td * mask
    count = mask.sum(dim=0).clamp(min=1.0)
    return eta * magnitude.max(dim=0).values + (1.0 - eta) * magnitude.sum(dim=0) / count


class R2d2Agent(object):

    def __init__(self, arch, learning_rate=1e-4, adam_epsilon=1e-3, gradient_clipping=40.0, target_update_period=2500,
                 discount=0.997, n_step=5, burn_in=40, block=40, rescale_targets=True, rescale_eps=1e-3,
                 priority_eta=0.9, seed=None, device='cpu'):
        self.arch = arch
        self.device = device
        self.num_actions = int(arch['num_actions'])
        self.gradient_clipping = float(gradient_clipping)
        self.target_update_period = int(target_update_period)
        self.gamma = float(discount)
        self.n_step = int(n_step)
        self.burn_in, self.block = int(burn_in), int(block)
        self.rescale_targets = bool(rescale_targets)
        self.rescale_eps = float(rescale_eps)
        self.priority_eta = float(priority_eta)
        self.net = network.build(arch, device, seed=seed)
        self.target = network.build(arch, device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        if self.net.head_type == 'c51' and self.rescale_targets:
            raise ValueError('the C51 recipe runs with value rescaling off (SNEK_R2D2_RESCALE=0): its support '
                             'is the return itself and h() would move it off the atoms')
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(learning_rate), eps=float(adam_epsilon))
        self.rng = np.random.default_rng(seed)
        self.train_step = 0
        self.last = {}

    # ---------------------------------------------------------------- acting

    @property
    def policy_fn(self):
        return network.greedy_policy_fn(self.net, self.device)

    @property
    def state_width(self):
        return self.net.state_width

    def act(self, observations, prev_action, prev_reward, state, fresh, epsilons):
        """One step of every lane: epsilon-greedy on the lane's own epsilon, carrying the LSTM state. Returns
        `(actions (n,) int64, next state (n, state_width))`, the state advanced by the greedy pass whether the
        lane explored or not -- the net saw the observation either way, which is what the stored sequence will
        replay."""
        self.net.eval()
        with torch.no_grad():
            obs = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=self.device)
            out, next_state = self.net.step(obs, torch.as_tensor(np.asarray(prev_action, dtype=np.int64), device=self.device),
                                            torch.as_tensor(np.asarray(prev_reward, dtype=np.float32), device=self.device),
                                            torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device),
                                            torch.as_tensor(np.asarray(fresh, dtype=bool), device=self.device))
            greedy = self.net.q_of(out).argmax(dim=1).cpu().numpy().astype(np.int64)
        n = greedy.shape[0]
        explore = self.rng.random(n) < np.asarray(epsilons, dtype=np.float64)
        drawn = self.rng.integers(0, self.num_actions, size=n)
        actions = np.where(explore, drawn, greedy).astype(np.int64)
        return actions, next_state.cpu().numpy()

    # ---------------------------------------------------------------- learning

    def _tensors(self, batch):
        """The batch on the device, time-major `(T, B, ...)`."""
        def tm(key, dtype):
            return torch.as_tensor(np.asarray(batch[key]), device=self.device).to(dtype).transpose(0, 1).contiguous()
        return {'obs': tm('obs', torch.float32), 'action': tm('action', torch.int64), 'reward': tm('reward', torch.float32),
                'done': tm('done', torch.bool), 'prev_action': tm('prev_action', torch.int64),
                'prev_reward': tm('prev_reward', torch.float32), 'valid': tm('valid', torch.bool), 'fresh': tm('fresh', torch.bool),
                'state': torch.as_tensor(np.asarray(batch['state'], dtype=np.float32), device=self.device)}

    def _unroll_both(self, b):
        """Burn-in under `no_grad` from the stored state, then the loss block and the lookahead with the gradient on
        for the online net. Returns `(online outputs, target outputs)` over the slots from `burn_in` on, `(T', B, A, out)`."""
        T = b['obs'].shape[0]
        head, tail = slice(0, self.burn_in), slice(self.burn_in, T)
        online_state, target_state = b['state'], b['state'].clone()
        if self.burn_in > 0:
            with torch.no_grad():
                _, online_state = self.net.unroll(b['obs'][head], b['prev_action'][head], b['prev_reward'][head],
                                                  online_state, b['fresh'][head])
                _, target_state = self.target.unroll(b['obs'][head], b['prev_action'][head], b['prev_reward'][head],
                                                     target_state, b['fresh'][head])
        online, _ = self.net.unroll(b['obs'][tail], b['prev_action'][tail], b['prev_reward'][tail], online_state, b['fresh'][tail])
        with torch.no_grad():
            target, _ = self.target.unroll(b['obs'][tail], b['prev_action'][tail], b['prev_reward'][tail], target_state,
                                           b['fresh'][tail])
        return online, target

    def update(self, batch, weights=None):
        """One Adam step. Returns `(per-window priorities, metrics)`."""
        self.net.train()
        b = self._tensors(batch)
        online, target = self._unroll_both(b)                        # (T', B, A, out) from slot burn_in
        tail = slice(self.burn_in, b['obs'].shape[0])
        reward, done, valid = b['reward'][tail], b['done'][tail], b['valid'][tail]
        action = b['action'][tail]
        L = self.block
        if self.net.head_type == 'c51':
            per_step, mask = self._categorical(online, target, action, reward, done, valid, L)
        else:
            per_step, mask = self._scalar(online, target, action, reward, done, valid, L)
        maskf = mask.to(per_step.dtype)
        per_window = (per_step * maskf).sum(dim=0)                                        # (B,)
        if weights is not None:
            per_window = per_window * torch.as_tensor(np.asarray(weights, dtype=np.float32), device=self.device)
        loss = per_window.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = None
        if self.gradient_clipping > 0.0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.gradient_clipping))
        self.optimizer.step()
        self.train_step += 1
        if self.target_update_period > 0 and self.train_step % self.target_update_period == 0:
            self.target.load_state_dict(self.net.state_dict())
        with torch.no_grad():
            priority = mix_priority(self.last_abs_td, maskf, self.priority_eta)
        self.last = {'loss': float(loss.detach()), 'td_loss': float((per_step.detach() * maskf).sum() / maskf.sum().clamp(min=1.0)),
                     'train_step': self.train_step}
        if grad_norm is not None:
            self.last['grad_norm'] = grad_norm
        return priority.cpu().numpy(), dict(self.last)

    def _scalar(self, online, target, action, reward, done, valid, L):
        q_all = online.squeeze(-1)                                                          # (T', B, A)
        q = q_all[:L].gather(2, action[:L].unsqueeze(-1)).squeeze(-1)                       # (L, B)
        with torch.no_grad():
            target_q = target.squeeze(-1)                                                   # (T', B, A)
            best = q_all.detach().argmax(dim=2, keepdim=True)
            boot = target_q.gather(2, best).squeeze(-1)                                     # (T', B)
            if self.rescale_targets:
                boot = unscale(boot, self.rescale_eps)
            returns, mask = n_step_targets(reward, done, valid, boot, self.gamma, self.n_step, 0, L)
            if self.rescale_targets:
                returns = rescale(returns, self.rescale_eps)
        delta = returns - q
        self.last_abs_td = delta.detach().abs()
        return 0.5 * delta * delta, mask

    def _categorical(self, online, target, action, reward, done, valid, L):
        atoms = self.net.atoms
        support = self.net.support
        log_probs = torch.log_softmax(online[:L], dim=-1)                                   # (L, B, A, atoms)
        taken = log_probs.gather(2, action[:L].view(L, -1, 1, 1).expand(-1, -1, 1, atoms)).squeeze(2)  # (L, B, atoms)
        with torch.no_grad():
            q_online = self.net.q_of(online.detach())                                       # (T', B, A)
            best = q_online.argmax(dim=2)                                                   # (T', B)
            target_probs = torch.softmax(target, dim=-1).gather(
                2, best.view(*best.shape, 1, 1).expand(-1, -1, 1, atoms)).squeeze(2)        # (T', B, atoms)
            # Rewards and the discount reaching the bootstrap slot, computed with a zero bootstrap value and a
            # unit one: the difference is the discount factor that multiplies the bootstrap distribution.
            zeros = torch.zeros(reward.shape, device=self.device)
            rewards_only, mask = n_step_targets(reward, done, valid, zeros, self.gamma, self.n_step, 0, L)
            with_unit, _ = n_step_targets(reward, done, valid, torch.ones_like(zeros), self.gamma, self.n_step, 0, L)
            factor = with_unit - rewards_only                                               # (L, B): gamma^n or 0
            B = reward.shape[1]
            boot_probs = torch.zeros(L, B, atoms, device=self.device)
            for i in range(L):
                slot = i + self.n_step
                if slot < target_probs.shape[0]:
                    boot_probs[i] = target_probs[slot]
            values = rewards_only.unsqueeze(-1) + factor.unsqueeze(-1) * support.view(1, 1, -1)
            # Where nothing is bootstrapped the whole mass sits on the return itself.
            no_boot = factor.unsqueeze(-1) == 0.0
            point = torch.zeros_like(boot_probs)
            point[..., 0] = 1.0
            probs = torch.where(no_boot, point, boot_probs)
            projected = losses.project(probs.view(L * B, atoms), values.view(L * B, atoms), support).view(L, B, atoms)
        per_step = losses.categorical_cross_entropy(taken.view(L * B, atoms), projected.view(L * B, atoms)).view(L, B)
        self.last_abs_td = per_step.detach()
        return per_step, mask

    # ---------------------------------------------------------------- persistence

    def state_dict(self):
        return {'net': self.net.state_dict(), 'target': self.target.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'train_step': self.train_step, 'rng': self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.net.load_state_dict(state['net'])
        self.target.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.train_step = int(state['train_step'])
        if state.get('rng') is not None:
            self.rng.bit_generator.state = state['rng']
