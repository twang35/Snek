"""R2D2's network (Kapturowski et al. 2019, §2.3 and Figure 8): trunk -> [previous action one-hot, previous
reward] -> LSTM (512) -> dueling streams (512 wide) -> a scalar Q per action, or C51's atoms per action for the
local variant. `plans/algoExploration/e-memory.md` §2 and §2b.

| piece | the paper | here |
|---|---|---|
| encoder | the Nature DQN convolutions | `QNet`'s hidden stack over `fc_layer_params` (the series' analogue), `algos/dqn/net.py`'s initialiser |
| recurrent inputs | the encoder's features concatenated with the one-hot previous action and the previous reward | the same; `prev_input` off feeds the features alone |
| the cell | LSTM, 512 | `nn.LSTMCell(hidden)`; **`recurrent: dense`** is the paper's feed-forward ablation: a `Linear(hidden) + ReLU` in the cell's place, no state (`ffr2d2`, the plan's control) |
| head | dueling, 512-wide streams | `V + A - mean_a(A)` over a value stream and an advantage stream, each `Linear(stream_width) -> ReLU -> Linear(out)`; `out` is 1 (`scalar`) or the atoms (`c51`) |

**The state is one flat `(n, 2 x hidden)` tensor**, `(h, c)` side by side, as `algos/ppo/net.py`'s tower
lays it out, so `StatefulPolicy`, the collector and the replay carry one array per lane whatever the cell;
a dense net's state is `(n, 0)`.

`step` zeroes the state of a `fresh` row before using it, and `unroll` runs a stored sequence from a stored
state with the gradient on -- the learner's burn-in runs the same `unroll` under `no_grad`.
"""

import math

import torch
from torch import nn

from algos.dqn import net as qnet
from algos.stateful import StatefulPolicy, one_hot_previous

HEAD_TYPES = ('scalar', 'c51')
RECURRENT_KINDS = ('lstm', 'dense')


def recurrent_of(arch):
    spec = arch.get('recurrent')
    if not spec:
        raise ValueError('an r2d2 arch needs a recurrent block {type, hidden, stream_width, prev_input}')
    kind = spec.get('type')
    if kind not in RECURRENT_KINDS:
        raise ValueError('arch recurrent.type must be one of {0}, got {1!r}'.format(RECURRENT_KINDS, kind))
    hidden = int(spec.get('hidden', 0))
    if hidden < 1:
        raise ValueError('arch recurrent.hidden must be at least 1, got {0}'.format(hidden))
    return {'type': kind, 'hidden': hidden, 'stream_width': int(spec.get('stream_width', 0)),
            'prev_input': bool(spec.get('prev_input', True))}


def head_of(arch):
    head = arch.get('head') or {'type': 'scalar'}
    if head.get('type') not in HEAD_TYPES:
        raise ValueError('an r2d2 arch head.type must be one of {0}, got {1!r}'.format(HEAD_TYPES, head))
    return head


def _stream(width, hidden, outputs, generator):
    if not hidden:
        linear = nn.Linear(width, outputs)
        qnet.head_init(linear, generator)
        return linear
    first, second = nn.Linear(width, hidden), nn.Linear(hidden, outputs)
    qnet.he_init(first, generator)
    qnet.head_init(second, generator)
    return nn.Sequential(first, nn.ReLU(), second)


class R2d2Net(nn.Module):

    def __init__(self, arch, seed=None):
        super().__init__()
        spec, head = recurrent_of(arch), head_of(arch)
        generator = qnet.make_generator(seed)
        widths = [int(arch['obs_len'])] + [int(width) for width in arch['fc_layer_params']]
        self.num_actions = int(arch['num_actions'])
        self.kind = spec['type']
        self.hidden_size = spec['hidden']
        self.prev_input = spec['prev_input']
        self.head_type = head['type']
        self.hidden = nn.ModuleList([nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        for layer in self.hidden:
            qnet.he_init(layer, generator)
        cell_in = widths[-1] + (self.num_actions + 1 if self.prev_input else 0)
        if self.kind == 'lstm':
            self.cell = nn.LSTMCell(cell_in, self.hidden_size)
            bound = 1.0 / math.sqrt(self.hidden_size)
            for parameter in self.cell.parameters():
                nn.init.uniform_(parameter, -bound, bound, generator=generator)
        else:
            self.cell = nn.Linear(cell_in, self.hidden_size)
            qnet.he_init(self.cell, generator)
        if self.head_type == 'c51':
            self.atoms = int(head['atoms'])
            self.v_min, self.v_max = float(head['v_min']), float(head['v_max'])
            if self.atoms < 2 or self.v_max <= self.v_min:
                raise ValueError('c51 needs atoms >= 2 and v_max > v_min, got {0}'.format(head))
            self.register_buffer('support', torch.linspace(self.v_min, self.v_max, self.atoms))
            self.outputs = self.atoms
        else:
            self.outputs = 1
        self.value = _stream(self.hidden_size, spec['stream_width'], self.outputs, generator)
        self.advantage = _stream(self.hidden_size, spec['stream_width'], self.num_actions * self.outputs, generator)

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

    # ------------------------------------------------------------ one step

    def features(self, observations):
        values = observations
        for layer in self.hidden:
            values = torch.relu(layer(values))
        return values

    def _cell_input(self, features, prev_action, prev_reward):
        if not self.prev_input:
            return features
        one_hot = one_hot_previous(prev_action, self.num_actions)
        return torch.cat([features, one_hot, prev_reward.to(features.dtype).unsqueeze(-1)], dim=-1)

    def _streams(self, hidden):
        """`(n, hidden) -> (n, actions, outputs)`: `V + A - mean_a(A)` per output."""
        lead = hidden.shape[:-1]
        advantage = self.advantage(hidden).view(*lead, self.num_actions, self.outputs)
        value = self.value(hidden).view(*lead, 1, self.outputs)
        return value + advantage - advantage.mean(dim=-2, keepdim=True)

    def step(self, observations, prev_action, prev_reward, state, fresh=None):
        """One step of every row -> `(outputs (n, actions, out), new state)`. A `fresh` row's state is zeroed
        first; a dense net ignores the state and returns it unchanged."""
        features = self._cell_input(self.features(observations), prev_action, prev_reward)
        if not self.recurrent:
            return self._streams(torch.relu(self.cell(features))), state
        if fresh is not None:
            state = state * (~fresh).to(state.dtype).unsqueeze(-1)
        h, c = self.cell(features, (state[:, :self.hidden_size], state[:, self.hidden_size:]))
        return self._streams(h), torch.cat([h, c], dim=-1)

    def unroll(self, observations, prev_action, prev_reward, state, fresh):
        """`(T, n, ...)` sequences from a `(n, state_width)` state -> `(outputs (T, n, actions, out), final
        state)`. The cell runs forward in time; nothing flows into `state`'s past."""
        outputs = []
        for t in range(observations.shape[0]):
            out, state = self.step(observations[t], prev_action[t], prev_reward[t], state, fresh[t])
            outputs.append(out)
        return torch.stack(outputs, dim=0), state

    # ------------------------------------------------------------ reads

    def q_of(self, outputs):
        """`(..., actions, out) -> (..., actions)`: the scalar Q, or the categorical mean."""
        if self.head_type == 'c51':
            return (torch.softmax(outputs, dim=-1) * self.support).sum(dim=-1)
        return outputs.squeeze(-1)

    def forward(self, observations):
        """One step from a zero state with no previous action or reward: Q-values `(n, actions)`. For a caller
        that treats the net as feed-forward; the stateful callers use `step` and `unroll`."""
        n = observations.shape[0]
        device = observations.device
        out, _ = self.step(observations, torch.full((n,), -1, dtype=torch.int64, device=device),
                           torch.zeros(n, device=device), self.initial_state(n, device))
        return self.q_of(out)


def build(arch, device='cpu', seed=None):
    """**The signature `tools/restore.py` calls.**"""
    return R2d2Net(arch, seed=seed).to(device)


def greedy_policy_fn(net, device='cpu'):
    """A `StatefulPolicy` over the net: argmax of Q, the LSTM state, the previous action and the previous
    reward carried per lane (`algos/stateful.py`). Eval mode, no autograd."""
    net.eval()

    def step_fn(observations, prev_action, prev_reward, state):
        out, state = net.step(observations, prev_action, prev_reward, state)
        return net.q_of(out).argmax(dim=1), state

    return StatefulPolicy(step_fn, net.state_width, device=device)
