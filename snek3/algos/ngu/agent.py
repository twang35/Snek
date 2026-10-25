"""`NguAgent`: the learner of Badia et al. 2020 (`plans/algoExploration/f-ngu-agent57.md` §1-2) on one fixed-grid
batch from `NguReplay`.

| piece | the paper | here |
|---|---|---|
| burn-in | R2D2's | `algos/r2d2/agent.py`'s: both nets from the stored state under `no_grad` |
| target | **transformed Retrace**, λ 0.95: `h(Q(x_s,a_s;θ⁻) + Σ_k γ^{k−s} (Π c) δ_k)` in `h⁻¹` space; π ε-soft (0.01) greedy on the **online** net, every value on the **target** net (NGU App. E, Agent57 App. C.2) | `retrace_targets`: `c_k = λ min(1, π(a_k\\|x_k)/μ_k)`, `δ_k = r_k + γ_j V_π(x_{k+1}) − Q(x_k,a_k)`, `Δ_k = δ_k + γ_j c_{k+1} Δ_{k+1}` backwards over the window's rows; the trace stops at the window's last row, bootstrapping on `V_π` of that row's state (nothing after it is stored), and at a terminal row `δ = r − Q`. γ_j is the **window's mixture's** |
| loss | squared TD at every loss step, summed over the block, averaged over the batch; IS exponent 0 | the same; `weights` all 1 at β_IS = 0 |
| priority | `η max + (1 − η) mean` of the TD error (η 1.0, App. A) | `mix_priority` |
| the auxiliary losses | the inverse-dynamics classifier and the RND predictor on the **last 5** steps of each sampled sequence (App. A) | `_aux`: the last `embed_steps` valid non-terminal pairs per window; Adam 5e-4 each, L2 1e-5 on the embedding |
| optimiser | Adam 1e-4, ε 1e-4, clip 40, target copy every 1,500 updates | `learning_rate`, `adam_epsilon`, `gradient_clipping`, `target_update_period` |
| acting | ε-greedy per lane on its own ε, reporting μ | `act` returns `(actions, state, mu)` with `mu = 1 − ε + ε/A` on the greedy action and `ε/A` otherwise |
"""

import numpy as np
import torch

from algos.ngu import net as network
from algos.r2d2.agent import rescale, unscale, mix_priority


def retrace_targets(q_online, q_target, action, reward, done, valid, mu, gamma, lam, eps, L, rescale_targets, rescale_eps):
    """`(T', B, A)` online and target outputs from the burn-in slot on, `(T', B)` rows, `gamma (B,)` -> `(L, B)` targets in
    the net's (h) space for the first L slots, and the `(L, B)` mask of slots carrying a loss."""
    T, B, A = q_online.shape
    greedy = q_online.argmax(dim=-1)                                                       # (T', B)
    pi = torch.full_like(q_online, eps / A)
    pi.scatter_add_(2, greedy.unsqueeze(-1), torch.full((T, B, 1), 1.0 - eps, device=q_online.device, dtype=q_online.dtype))
    qt = unscale(q_target, rescale_eps) if rescale_targets else q_target
    v = (pi * qt).sum(dim=-1)                                                              # V_π(x_s) on the target net
    q_taken = qt.gather(2, action.unsqueeze(-1)).squeeze(-1)
    pi_taken = pi.gather(2, action.unsqueeze(-1)).squeeze(-1)
    c = lam * torch.clamp(pi_taken / mu.clamp(min=1e-8), max=1.0)
    validf, donef = valid.to(qt.dtype), done.to(qt.dtype)
    has_next = torch.cat([validf[1:], torch.zeros_like(validf[:1])], dim=0)
    v_next = torch.cat([v[1:], torch.zeros_like(v[:1])], dim=0)
    g = gamma.to(qt.dtype).unsqueeze(0)                                                    # (1, B)
    cont = g * (1.0 - donef) * has_next                                                    # the discount reaching the next row
    delta = reward + cont * v_next - q_taken
    # A row with no stored successor and no terminal has no TD error the window can state: the trace stops there.
    delta = torch.where((donef > 0.0) | (has_next > 0.0), delta, torch.zeros_like(delta)) * validf
    deltas = torch.zeros_like(delta)
    following = torch.zeros(B, device=qt.device, dtype=qt.dtype)
    for s in range(T - 1, -1, -1):
        c_next = c[s + 1] if s + 1 < T else torch.zeros_like(following)
        following = delta[s] + cont[s] * c_next * following
        deltas[s] = following
    target = q_taken + deltas
    if rescale_targets:
        target = rescale(target, rescale_eps)
    return target[:L], valid[:L]


def _last_pairs(valid, done, steps):
    """`(T, B)` masks -> `(T−1, B)` bool: the last `steps` slots s per window where rows s and s+1 both exist and s is
    not terminal -- the pairs the auxiliary losses train on."""
    pair = valid[:-1] & valid[1:] & ~done[:-1]
    rank_from_end = torch.flip(torch.cumsum(torch.flip(pair.to(torch.int64), dims=[0]), dim=0), dims=[0])
    return pair & (rank_from_end <= int(steps))


class NguAgent(object):

    def __init__(self, arch, learning_rate=1e-4, adam_epsilon=1e-4, gradient_clipping=40.0, target_update_period=1500,
                 burn_in=40, block=80, rescale_targets=True, rescale_eps=1e-3, priority_eta=1.0, retrace_lambda=0.95,
                 target_epsilon=0.01, embed_lr=5e-4, embed_l2=1e-5, embed_steps=5, seed=None, device='cpu'):
        self.arch = arch
        self.device = device
        self.num_actions = int(arch['num_actions'])
        self.gradient_clipping = float(gradient_clipping)
        self.target_update_period = int(target_update_period)
        self.burn_in, self.block = int(burn_in), int(block)
        self.rescale_targets = bool(rescale_targets)
        self.rescale_eps = float(rescale_eps)
        self.priority_eta = float(priority_eta)
        self.retrace_lambda = float(retrace_lambda)
        self.target_epsilon = float(target_epsilon)
        self.embed_steps = int(embed_steps)
        self.net = network.build(arch, device, seed=seed)
        self.target = network.build(arch, device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        core = [p for name, p in self.net.named_parameters()
                if not name.startswith(('embed_', 'classifier', 'rnd_'))]
        self.optimizer = torch.optim.Adam(core, lr=float(learning_rate), eps=float(adam_epsilon))
        embed = list(self.net.embed_first.parameters()) + list(self.net.embed_second.parameters()) + \
            list(self.net.classifier.parameters())
        self.embed_optimizer = torch.optim.Adam(embed, lr=float(embed_lr), weight_decay=float(embed_l2))
        self.rnd_optimizer = torch.optim.Adam(self.net.rnd_predictor.parameters(), lr=float(embed_lr))
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

    def act(self, observations, prev_action, prev_reward, prev_reward_int, mixture, state, fresh, epsilons):
        """One step of every lane -> `(actions, next state, mu)`: ε-greedy on the lane's own ε, the state advanced by
        the greedy pass either way, and μ the probability the lane's policy gave the action it took."""
        self.net.eval()
        t = lambda a, dtype: torch.as_tensor(np.asarray(a), device=self.device).to(dtype)
        with torch.no_grad():
            out, next_state = self.net.step(t(observations, torch.float32), t(prev_action, torch.int64),
                                            t(prev_reward, torch.float32), t(prev_reward_int, torch.float32),
                                            t(mixture, torch.int64), t(state, torch.float32), t(fresh, torch.bool))
            greedy = self.net.q_of(out).argmax(dim=1).cpu().numpy().astype(np.int64)
        n = greedy.shape[0]
        eps = np.asarray(epsilons, dtype=np.float64)
        explore = self.rng.random(n) < eps
        drawn = self.rng.integers(0, self.num_actions, size=n)
        actions = np.where(explore, drawn, greedy).astype(np.int64)
        mu = np.where(actions == greedy, 1.0 - eps + eps / self.num_actions, eps / self.num_actions).astype(np.float32)
        return actions, next_state.cpu().numpy(), mu

    # ---------------------------------------------------------------- learning

    def _tensors(self, batch):
        def tm(key, dtype):
            return torch.as_tensor(np.asarray(batch[key]), device=self.device).to(dtype).transpose(0, 1).contiguous()
        out = {'obs': tm('obs', torch.float32), 'action': tm('action', torch.int64), 'reward': tm('reward', torch.float32),
               'done': tm('done', torch.bool), 'prev_action': tm('prev_action', torch.int64),
               'prev_reward': tm('prev_reward', torch.float32), 'prev_reward_int': tm('prev_reward_int', torch.float32),
               'mu': tm('mu', torch.float32), 'valid': tm('valid', torch.bool), 'fresh': tm('fresh', torch.bool),
               'state': torch.as_tensor(np.asarray(batch['state'], dtype=np.float32), device=self.device),
               'mixture': torch.as_tensor(np.asarray(batch['mixture'], dtype=np.int64), device=self.device)}
        if self.net.intrinsic['input'] == 'board':
            out['board'] = torch.as_tensor(np.asarray(batch['board']), device=self.device).transpose(0, 1).contiguous()
        return out

    def _unroll_both(self, b):
        T = b['obs'].shape[0]
        head, tail = slice(0, self.burn_in), slice(self.burn_in, T)
        online_state, target_state = b['state'], b['state'].clone()
        args = lambda sl: (b['obs'][sl], b['prev_action'][sl], b['prev_reward'][sl], b['prev_reward_int'][sl], b['mixture'])
        if self.burn_in > 0:
            with torch.no_grad():
                _, online_state = self.net.unroll(*args(head), online_state, b['fresh'][head])
                _, target_state = self.target.unroll(*args(head), target_state, b['fresh'][head])
        online, _ = self.net.unroll(*args(tail), online_state, b['fresh'][tail])
        with torch.no_grad():
            target, _ = self.target.unroll(*args(tail), target_state, b['fresh'][tail])
        return online, target

    def update(self, batch, weights=None):
        """One Adam step on the Q-network, one on the embedding and one on the RND predictor. Returns
        `(per-window priorities, metrics)`."""
        self.net.train()
        b = self._tensors(batch)
        online, target = self._unroll_both(b)
        T = b['obs'].shape[0]
        tail = slice(self.burn_in, T)
        L = self.block
        q_all = online.squeeze(-1)
        q = q_all[:L].gather(2, b['action'][tail][:L].unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            gamma = self.net.gammas[b['mixture']]
            returns, mask = retrace_targets(q_all.detach(), target.squeeze(-1), b['action'][tail], b['reward'][tail],
                                            b['done'][tail], b['valid'][tail], b['mu'][tail], gamma, self.retrace_lambda,
                                            self.target_epsilon, L, self.rescale_targets, self.rescale_eps)
        delta = returns - q
        maskf = mask.to(delta.dtype)
        per_step = 0.5 * delta * delta
        per_window = (per_step * maskf).sum(dim=0)
        if weights is not None:
            per_window = per_window * torch.as_tensor(np.asarray(weights, dtype=np.float32), device=self.device)
        loss = per_window.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = None
        if self.gradient_clipping > 0.0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.gradient_clipping))
        self.optimizer.step()
        aux = self._aux(b)
        self.train_step += 1
        if self.target_update_period > 0 and self.train_step % self.target_update_period == 0:
            self.target.load_state_dict(self.net.state_dict())
        with torch.no_grad():
            priority = mix_priority(delta.detach().abs(), maskf, self.priority_eta)
        self.last = {'loss': float(loss.detach()), 'td_loss': float((per_step.detach() * maskf).sum() / maskf.sum().clamp(min=1.0)),
                     'train_step': self.train_step}
        self.last.update(aux)
        if grad_norm is not None:
            self.last['grad_norm'] = grad_norm
        return priority.cpu().numpy(), dict(self.last)

    def _aux(self, b):
        """The inverse-dynamics and RND losses on the last `embed_steps` pairs of each window."""
        pairs = _last_pairs(b['valid'], b['done'], self.embed_steps)                           # (T-1, B)
        picked = pairs.nonzero(as_tuple=True)
        if picked[0].numel() == 0:
            return {}
        s, lanes = picked
        if self.net.intrinsic['input'] == 'board':
            now = self.net.intrinsic_input(b['board'][s, lanes], None)
            nxt = self.net.intrinsic_input(b['board'][s + 1, lanes], None)
        else:
            now = b['obs'][s, lanes]
            nxt = b['obs'][s + 1, lanes]
        logits = self.net.classify(now, nxt)
        taken = b['action'][s, lanes]
        embed_loss = torch.nn.functional.cross_entropy(logits, taken)
        self.embed_optimizer.zero_grad(set_to_none=True)
        embed_loss.backward()
        self.embed_optimizer.step()
        rnd_loss = self.net.rnd_error(now.detach()).mean()
        self.rnd_optimizer.zero_grad(set_to_none=True)
        rnd_loss.backward()
        self.rnd_optimizer.step()
        with torch.no_grad():
            accuracy = float((logits.argmax(dim=-1) == taken).to(torch.float32).mean())
        return {'embed_loss': float(embed_loss.detach()), 'embed_acc': accuracy, 'rnd_loss': float(rnd_loss.detach()),
                'aux_pairs': int(taken.numel())}

    # ---------------------------------------------------------------- persistence

    def state_dict(self):
        return {'net': self.net.state_dict(), 'target': self.target.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'embed_optimizer': self.embed_optimizer.state_dict(), 'rnd_optimizer': self.rnd_optimizer.state_dict(),
                'train_step': self.train_step, 'rng': self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.net.load_state_dict(state['net'])
        self.target.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.embed_optimizer.load_state_dict(state['embed_optimizer'])
        self.rnd_optimizer.load_state_dict(state['rnd_optimizer'])
        self.train_step = int(state['train_step'])
        if state.get('rng') is not None:
            self.rng.bit_generator.state = state['rng']
