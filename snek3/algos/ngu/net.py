"""NGU's network (Badia et al. 2020, Fig. 1 and App. A; `plans/algoExploration/f-ngu-agent57.md` §1-2): R2D2's
trunk -> LSTM -> dueling streams, with the cell fed **five** things beside the trunk's features -- the one-hot of
the lane's mixture index, the one-hot of the previous action, the previous extrinsic reward and the previous
intrinsic reward -- and the intrinsic modules on the same `nn.Module`, so one checkpoint carries everything a
restored policy needs to compute `r^i` as the trainer did (§0, the checkpoint row):

| piece | the paper | here |
|---|---|---|
| core | R2D2's conv -> LSTM 512 -> dueling | `R2d2Net`'s layout over the 42-value observation (`algos/r2d2/net.py`); `recurrent` from the sidecar as E2 |
| conditioning | one-hot β_j ⊕ a_{t−1} ⊕ r^e_{t−1} ⊕ r^i_{t−1} into the LSTM | `_cell_input`; `prev_input` off drops the three previous-step inputs and keeps the index |
| embedding f | conv -> 32, trained by inverse dynamics | `embed_first`, `embed_second` over the **board** (400 values, `board.py`) -> 128 -> 32; `classifier` 64 -> 128 -> A |
| RND | two conv nets -> 128, the target frozen | `rnd_target` (frozen), `rnd_predictor`, 400 -> 128 -> 128; the error's running mean / std as buffers (`rnd_mean`, `rnd_m2`, `rnd_count`) |
| the ladders | β_i = β_max σ(10 (2i − (N−2)) / (N−2)), β_0 = 0, β_{N−1} = β_max; γ_i the log-interpolation | `beta_ladder`, `gamma_ladder`, non-persistent buffers from the sidecar's `conditioning` |

**The sidecar.** `recurrent` (E2's), `head` (`scalar`; the C51 recipe is refused here), `conditioning`
`{mixtures, beta_max, gamma_top, gamma_min, split}` and `intrinsic` `{input, embed_dim, embed_hidden, rnd_hidden,
rnd_out, knn_k, kernel_epsilon, cluster_distance, pseudo_count, max_similarity, memory, lifelong_max}` -- the shapes
and every constant that decides the value of `r^i`, so a checkpoint reproduces the trainer's inputs bit for bit.

The state is R2D2's flat `(n, 2 x hidden)`.
"""

import math

import numpy as np
import torch
from torch import nn

from algos.dqn import net as qnet
from algos.ngu import board as board_codes
from algos.r2d2 import net as r2d2_net
from algos.stateful import StatefulPolicy, one_hot_previous

SPLITS = ('none',)
INPUTS = ('board', 'obs')
EXPLOIT_INDEX = 0


def beta_ladder(n, beta_max):
    """NGU App. A: β_0 = 0, β_{N−1} = β_max, β_i = β_max σ(10 (2i − (N − 2)) / (N − 2)) between."""
    n = int(n)
    if n < 2:
        return np.array([0.0], dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    betas = float(beta_max) / (1.0 + np.exp(-10.0 * (2.0 * i - (n - 2)) / (n - 2)))
    betas[0], betas[-1] = 0.0, float(beta_max)
    return betas


def gamma_ladder(n, gamma_top, gamma_min):
    """NGU App. A: γ_i = 1 − exp(((N−1−i) log(1−γ_max) + i log(1−γ_min)) / (N−1)), γ_0 the top (exploitative)."""
    n = int(n)
    if n < 2:
        return np.array([float(gamma_top)], dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    inner = ((n - 1 - i) * math.log(1.0 - float(gamma_top)) + i * math.log(1.0 - float(gamma_min))) / (n - 1)
    return 1.0 - np.exp(inner)


def conditioning_of(arch):
    spec = arch.get('conditioning')
    if not spec:
        raise ValueError('an ngu arch needs a conditioning block {mixtures, beta_max, gamma_top, gamma_min, split}')
    out = {'mixtures': int(spec['mixtures']), 'beta_max': float(spec['beta_max']), 'gamma_top': float(spec['gamma_top']),
           'gamma_min': float(spec['gamma_min']), 'split': str(spec.get('split', 'none'))}
    if out['mixtures'] < 1:
        raise ValueError('conditioning.mixtures must be at least 1')
    if out['split'] not in SPLITS:
        raise ValueError('conditioning.split must be one of {0}, got {1!r}'.format(SPLITS, out['split']))
    return out


INTRINSIC_FIELDS = ('input', 'embed_dim', 'embed_hidden', 'rnd_hidden', 'rnd_out', 'knn_k', 'kernel_epsilon',
                    'cluster_distance', 'pseudo_count', 'max_similarity', 'memory', 'lifelong_max')
_INTRINSIC_INTS = ('embed_dim', 'embed_hidden', 'rnd_hidden', 'rnd_out', 'knn_k', 'memory')


def intrinsic_of(arch):
    spec = arch.get('intrinsic')
    if not spec:
        raise ValueError('an ngu arch needs an intrinsic block {0}'.format(INTRINSIC_FIELDS))
    missing = [field for field in INTRINSIC_FIELDS if field not in spec]
    if missing:
        raise ValueError('arch intrinsic block is missing {0}'.format(missing))
    out = {}
    for field in INTRINSIC_FIELDS:
        if field == 'input':
            out[field] = str(spec[field])
        elif field in _INTRINSIC_INTS:
            out[field] = int(spec[field])
        else:
            out[field] = float(spec[field])
    if out['input'] not in INPUTS:
        raise ValueError('intrinsic.input must be one of {0}, got {1!r}'.format(INPUTS, out['input']))
    return out


def _mlp(widths, generator, head=True):
    layers = []
    for i in range(len(widths) - 1):
        linear = nn.Linear(widths[i], widths[i + 1])
        if i == len(widths) - 2 and head:
            qnet.head_init(linear, generator)
        else:
            qnet.he_init(linear, generator)
        layers.append(linear)
        if i < len(widths) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class NguNet(nn.Module):

    def __init__(self, arch, seed=None):
        super().__init__()
        spec, head = r2d2_net.recurrent_of(arch), r2d2_net.head_of(arch)
        if head['type'] != 'scalar':
            raise ValueError('ngu runs the scalar head; the C51 recipe is not built for it')
        cond, intr = conditioning_of(arch), intrinsic_of(arch)
        self.conditioning, self.intrinsic = cond, intr
        generator = qnet.make_generator(seed)
        self.obs_len = int(arch['obs_len'])
        widths = [self.obs_len] + [int(width) for width in arch['fc_layer_params']]
        self.num_actions = int(arch['num_actions'])
        self.kind = spec['type']
        self.hidden_size = spec['hidden']
        self.prev_input = spec['prev_input']
        self.head_type = 'scalar'
        self.mixtures = cond['mixtures']
        self.hidden = nn.ModuleList([nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        for layer in self.hidden:
            qnet.he_init(layer, generator)
        cell_in = widths[-1] + self.mixtures + (self.num_actions + 2 if self.prev_input else 0)
        if self.kind == 'lstm':
            self.cell = nn.LSTMCell(cell_in, self.hidden_size)
            bound = 1.0 / math.sqrt(self.hidden_size)
            for parameter in self.cell.parameters():
                nn.init.uniform_(parameter, -bound, bound, generator=generator)
        else:
            self.cell = nn.Linear(cell_in, self.hidden_size)
            qnet.he_init(self.cell, generator)
        self.outputs = 1
        self.value = r2d2_net._stream(self.hidden_size, spec['stream_width'], 1, generator)
        self.advantage = r2d2_net._stream(self.hidden_size, spec['stream_width'], self.num_actions, generator)

        # --- the intrinsic modules
        self.intrinsic_len = board_codes.BOARD_LEN if intr['input'] == 'board' else self.obs_len
        self.embed_first = nn.Linear(self.intrinsic_len, intr['embed_hidden'])
        self.embed_second = nn.Linear(intr['embed_hidden'], intr['embed_dim'])
        qnet.he_init(self.embed_first, generator)
        qnet.he_init(self.embed_second, generator)
        self.classifier = _mlp([2 * intr['embed_dim'], intr['embed_hidden'], self.num_actions], generator)
        self.rnd_target = _mlp([self.intrinsic_len, intr['rnd_hidden'], intr['rnd_out']], generator, head=False)
        self.rnd_predictor = _mlp([self.intrinsic_len, intr['rnd_hidden'], intr['rnd_out']], generator, head=False)
        for parameter in self.rnd_target.parameters():
            parameter.requires_grad_(False)
        self.register_buffer('rnd_mean', torch.zeros(()))
        self.register_buffer('rnd_m2', torch.zeros(()))
        self.register_buffer('rnd_count', torch.zeros(()))
        self.register_buffer('betas', torch.as_tensor(beta_ladder(self.mixtures, cond['beta_max']), dtype=torch.float32),
                             persistent=False)
        self.register_buffer('gammas', torch.as_tensor(gamma_ladder(self.mixtures, cond['gamma_top'], cond['gamma_min']),
                                                       dtype=torch.float32), persistent=False)

    # ------------------------------------------------------------ the state

    @property
    def recurrent(self):
        return self.kind == 'lstm'

    @property
    def state_width(self):
        return 2 * self.hidden_size if self.recurrent else 0

    def initial_state(self, n, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros((int(n), self.state_width), dtype=torch.float32, device=device)

    # ------------------------------------------------------------ the core

    def features(self, observations):
        values = observations
        for layer in self.hidden:
            values = torch.relu(layer(values))
        return values

    def _cell_input(self, features, prev_action, prev_reward, prev_reward_int, mixture):
        index = torch.nn.functional.one_hot(mixture.to(torch.int64), self.mixtures).to(features.dtype)
        parts = [features, index]
        if self.prev_input:
            parts += [one_hot_previous(prev_action, self.num_actions), prev_reward.to(features.dtype).unsqueeze(-1),
                      prev_reward_int.to(features.dtype).unsqueeze(-1)]
        return torch.cat(parts, dim=-1)

    def _streams(self, hidden):
        lead = hidden.shape[:-1]
        advantage = self.advantage(hidden).view(*lead, self.num_actions, 1)
        value = self.value(hidden).view(*lead, 1, 1)
        return value + advantage - advantage.mean(dim=-2, keepdim=True)

    def step(self, observations, prev_action, prev_reward, prev_reward_int, mixture, state, fresh=None):
        """One step of every row -> `(outputs (n, actions, 1), new state)`. A `fresh` row's state is zeroed first."""
        features = self._cell_input(self.features(observations), prev_action, prev_reward, prev_reward_int, mixture)
        if not self.recurrent:
            return self._streams(torch.relu(self.cell(features))), state
        if fresh is not None:
            state = state * (~fresh).to(state.dtype).unsqueeze(-1)
        h, c = self.cell(features, (state[:, :self.hidden_size], state[:, self.hidden_size:]))
        return self._streams(h), torch.cat([h, c], dim=-1)

    def unroll(self, observations, prev_action, prev_reward, prev_reward_int, mixture, state, fresh):
        """`(T, n, ...)` sequences from a `(n, state_width)` state; `mixture (n,)` is per lane, fixed over the window."""
        outputs = []
        for t in range(observations.shape[0]):
            out, state = self.step(observations[t], prev_action[t], prev_reward[t], prev_reward_int[t], mixture, state, fresh[t])
            outputs.append(out)
        return torch.stack(outputs, dim=0), state

    def q_of(self, outputs):
        return outputs.squeeze(-1)

    def forward(self, observations):
        """One step at the exploitative index from a zero state with no previous inputs: Q `(n, actions)`."""
        n = observations.shape[0]
        device = observations.device
        out, _ = self.step(observations, torch.full((n,), -1, dtype=torch.int64, device=device),
                           torch.zeros(n, device=device), torch.zeros(n, device=device),
                           torch.full((n,), EXPLOIT_INDEX, dtype=torch.int64, device=device), self.initial_state(n, device))
        return self.q_of(out)

    # ------------------------------------------------------------ the intrinsic modules

    def intrinsic_input(self, codes, observations):
        """What the intrinsic modules read: the decoded board, or the observation under `input: obs`."""
        if self.intrinsic['input'] == 'board':
            return board_codes.decode(codes)
        return torch.as_tensor(observations, dtype=torch.float32)

    def embed(self, inputs):
        return self.embed_second(torch.relu(self.embed_first(inputs)))

    def classify(self, inputs_now, inputs_next):
        """Logits of the action between two consecutive inputs, from the live embedding."""
        return self.classifier(torch.cat([self.embed(inputs_now), self.embed(inputs_next)], dim=-1))

    def rnd_error(self, inputs):
        """`(m,)`: the predictor's mean squared error against the frozen target."""
        with torch.no_grad():
            target = self.rnd_target(inputs)
        return ((self.rnd_predictor(inputs) - target) ** 2).mean(dim=-1)

    def rnd_std(self):
        count = self.rnd_count
        return torch.where(count > 1.0, torch.sqrt(self.rnd_m2 / (count - 1.0).clamp(min=1.0)), torch.zeros_like(count))

    @torch.no_grad()
    def update_rnd_stats(self, error):
        """Welford over a batch of errors, into the running mean / variance buffers."""
        error = error.detach().reshape(-1).to(torch.float32)
        n = float(error.numel())
        if n == 0:
            return
        batch_mean = error.mean()
        batch_m2 = ((error - batch_mean) ** 2).sum()
        total = self.rnd_count + n
        delta = batch_mean - self.rnd_mean
        self.rnd_mean += delta * n / total
        self.rnd_m2 += batch_m2 + delta * delta * self.rnd_count * n / total
        self.rnd_count.copy_(total)


def build(arch, device='cpu', seed=None):
    """**The signature `tools/restore.py` calls.**"""
    return NguNet(arch, seed=seed).to(device)


class NguPolicy(StatefulPolicy):
    """The greedy read of an `NguNet` at one mixture index, behind the engine's seam. Beside the LSTM state, the
    previous action and the previous extrinsic reward every `StatefulPolicy` carries, it carries the previous
    intrinsic reward per lane and computes it as the trainer did: from the board the engine hands `begin`
    (`needs_board`), through the checkpoint's embedding and a per-lane episodic memory emptied with the lane,
    the RND statistics frozen at the checkpoint's. `a_{-1}` is drawn from a generator seeded once, so the same
    episodes play the same way on every measurement."""

    def __init__(self, net, device='cpu', mixture=EXPLOIT_INDEX):
        from algos.ngu.intrinsic import LaneNovelty
        super().__init__(None, net.state_width, device=device)
        self.net = net
        self.mixture = int(mixture)
        if not 0 <= self.mixture < net.mixtures:
            raise ValueError('mixture {0} is outside the net\'s {1}'.format(mixture, net.mixtures))
        self.needs_board = net.intrinsic['input'] == 'board'
        self.novelty = LaneNovelty(net, 0, device=device, training=False)
        self.prev_reward_int = torch.zeros((0,), dtype=torch.float32, device=device)
        self.generator = torch.Generator(device='cpu')
        self.generator.manual_seed(20260101)
        self._fresh = None
        self._board = None

    def _grow(self, needed):
        have = self.state.shape[0]
        super()._grow(needed)
        if needed > have:
            self.prev_reward_int = torch.cat([self.prev_reward_int, torch.zeros((needed - have,), dtype=torch.float32,
                                                                                device=self.device)])
            self.novelty.grow(needed)

    def begin(self, rows, fresh, prev_reward, board=None):
        super().begin(rows, fresh, prev_reward)
        self._fresh = torch.as_tensor(np.asarray(fresh, dtype=bool), device=self.device)
        self._board = None if board is None else torch.as_tensor(np.asarray(board, dtype=np.uint8), device=self.device)
        if self.needs_board and self._board is None:
            raise ValueError('this policy reads the board: begin(..., board=<(m, 100) uint8>)')

    def __call__(self, observations):
        observations = np.asarray(observations, dtype=np.float32)
        m = observations.shape[0]
        if self._rows is None:
            rows = np.arange(m)
            # The fallback for a caller without `begin`; a board-reading policy cannot take it and says so.
            self.begin(rows, np.full(m, not self._began, dtype=bool), np.zeros(m, dtype=np.float32))
        index, fresh = self._rows, self._fresh
        self._rows = None
        if index.shape[0] != m:
            raise ValueError('begin() named {0} rows but the call carries {1} observations'.format(index.shape[0], m))
        with torch.no_grad():
            obs = torch.as_tensor(observations, device=self.device)
            inputs = self.net.intrinsic_input(self._board, obs)
            rows = index.cpu().numpy()
            is_fresh = fresh.cpu().numpy()
            if is_fresh.any():
                self.novelty.begin_episode(rows[is_fresh], inputs[fresh])
                self.prev_reward_int[index[fresh]] = 0.0
                drawn = torch.randint(0, self.net.num_actions, (int(is_fresh.sum()),), generator=self.generator)
                self.prev_action[index[fresh]] = drawn.to(self.device)
            if (~is_fresh).any():
                reward = self.novelty.reward(rows[~is_fresh], inputs[~fresh])
                self.prev_reward_int[index[~fresh]] = torch.as_tensor(reward, device=self.device)
            mixture = torch.full((m,), self.mixture, dtype=torch.int64, device=self.device)
            out, state = self.net.step(obs, self.prev_action[index], self.prev_reward[index], self.prev_reward_int[index],
                                       mixture, self.state[index])
            actions = self.net.q_of(out).argmax(dim=1).to(torch.int64)
            self.prev_action[index] = actions
            if self.state_width:
                self.state[index] = state
        return actions.cpu().numpy()


def greedy_policy_fn(net, device='cpu'):
    """The exploitative index: β = 0, the top γ -- what the protocol measures and what NGU's paper evaluates."""
    net.eval()
    return NguPolicy(net, device=device, mixture=EXPLOIT_INDEX)


def variant_policy_fn(net, device='cpu', variant=None):
    """`mixture:<i>`: the greedy read at mixture i (`plans/algoExploration/f-ngu-agent57.md` §1, restore)."""
    name, _, value = str(variant).partition(':')
    if name != 'mixture' or not value.isdigit():
        raise ValueError('ngu knows the variant mixture:<i>, got {0!r}'.format(variant))
    net.eval()
    return NguPolicy(net, device=device, mixture=int(value))
