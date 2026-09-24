"""The Rainbow / Beyond-the-Rainbow network, and the module `tools/restore.py` loads its checkpoints through.

One `RainbowNet` assembled from two sidecar fields (`tools/arch.py`, `OPTIONAL_FIELDS`):

| field | what it decides |
|---|---|
| `arch['head']` | the distribution read off the streams, in Group A's formats: `c51` (`atoms`, `v_min`, `v_max`), `quantile` (`n`), `iqn` (`embedding`, `n_tau`, `k`) |
| `arch['trunk']` | `dueling`, `noisy`, `noisy_sigma`, `residual`, `blocks`, `spectral_norm`, `layer_norm`, and optionally `stream_hidden` |

**Separate from `algos/dist/net.py` on purpose** (decided 2026-09-20): Group A's heads build their own
`QNet` and read features by walking it, so dueling streams and a residual trunk cannot be composed onto
them from outside, and Group A's code is not edited. The agent (`agent.py`) is `DistAgent`'s plumbing
with its own update (the papers' Munchausen target, loss reduction and priorities), and it and the
inherited helpers call this module through the same reads as the dist agent: `log_probs`/`probs`/`support`/`atoms` for c51, `quantiles`/`fractions`/`n` for the quantile
head, `features`/`sample_taus`/`quantiles_at` for IQN, and `q_values` for every acting read.

The trunk with `residual` off is `QNet`'s hidden stack to the initialiser, so a `rainbow` arm with
dueling and noisy off differs from A2's c51 arm by nothing but the module name. With `residual` on it is
the plan's MLP analogue of BTR's IMPALA stack (`c-value-stack.md` §2, C2): a stem to `fc_layer_params[-1]`,
then `blocks` residual blocks of two linears with ReLUs, spectral normalisation on those two linears
only (the paper's placement -- the residual path, never the stem or the head) and an optional layer norm
at each block's input (the paper's post-submission variant, off in the paper cell).

The streams are combined as `V + A - mean_a(A)` per atom, quantile or tau. With `stream_hidden` off they
are one linear each (`value: width -> outputs`, `advantage: width -> actions * outputs`) -- BBF's layout
and every checkpoint written before 2026-09-23. With it on (`rainbow` and `btr` since then) each stream
is a hidden linear, a ReLU and the output linear, the dueling paper's split of Nature DQN's 512-unit layer
into one per stream, which Rainbow (Table 4) and BTR (Table D6; `networks.py`'s `fc1V`/`fc1A`) both keep.
The MLP analogue: the last width of `fc_layer_params` moves out of the shared stack into each stream, so
a plain single-stream net is still `QNet` weight for weight; the residual trunk keeps its width and each
stream adds a hidden layer of that width, as BTR's streams follow its IMPALA stack. `NoisyLinear` for
**both** layers of each stream when `noisy`, as both papers have it; otherwise the hidden layer takes
`QNet`'s hidden initialiser and the output its head initialiser. For IQN the streams read the
feature-embedding product one tau at a time, as A4's head does, so the dueling combine is per tau.
"""

import math

import torch
from torch import nn
from torch.nn.utils import parametrizations

from algos.rainbow.noisy import NoisyLinear, noisy_layers, set_noise

HEAD_TYPES = ('c51', 'quantile', 'iqn')
TRUNK_FIELDS = ('dueling', 'noisy', 'noisy_sigma', 'residual', 'blocks', 'spectral_norm', 'layer_norm')
# Optional in the sidecar, off when absent: a trunk written before the field existed is the one-linear layout.
OPTIONAL_TRUNK_FIELDS = ('stream_hidden',)

# snek2's He-normal with Keras' truncation correction, as `algos/dqn/net.QNet.reset_parameters`.
_TRUNC_CORRECTION = 0.87962566103423978


def head_of(arch):
    head = arch.get('head')
    if not head or head.get('type') not in HEAD_TYPES:
        raise ValueError('a rainbow arch needs head.type in {0}, got {1!r}'.format(HEAD_TYPES, head))
    return head


def trunk_of(arch):
    trunk = arch.get('trunk')
    if not trunk:
        raise ValueError('a rainbow arch needs a trunk field with {0}'.format(TRUNK_FIELDS))
    missing = [field for field in TRUNK_FIELDS if field not in trunk]
    if missing:
        raise ValueError('rainbow trunk is missing {0}: {1!r}'.format(missing, trunk))
    return trunk


def _he_init(linear, generator):
    fan_in = linear.weight.shape[1]
    stddev = math.sqrt(2.0 / fan_in) / _TRUNC_CORRECTION
    nn.init.trunc_normal_(linear.weight, std=stddev, a=-2 * stddev, b=2 * stddev, generator=generator)
    nn.init.zeros_(linear.bias)


def _head_init(linear, generator):
    nn.init.uniform_(linear.weight, -0.03, 0.03, generator=generator)
    nn.init.zeros_(linear.bias)


def _stream_layer(in_features, out_features, noisy, sigma, generator, init):
    if noisy:
        return NoisyLinear(in_features, out_features, sigma_zero=sigma, generator=generator)
    linear = nn.Linear(in_features, out_features)
    init(linear, generator)
    return linear


def _stream(in_features, hidden, out_features, noisy, sigma, generator):
    """One stream: the output linear alone (`hidden` 0), or `hidden linear -> ReLU -> output`. A
    `Sequential` so its parameters are `advantage.0.*`, never `hidden.*`: `resets.partition` reads
    `hidden.` as the trunk, and a stream is head-side."""
    if not hidden:
        return _stream_layer(in_features, out_features, noisy, sigma, generator, _head_init)
    return nn.Sequential(_stream_layer(in_features, hidden, noisy, sigma, generator, _he_init), nn.ReLU(),
                         _stream_layer(hidden, out_features, noisy, sigma, generator, _head_init))


def _spectral_norm(linear, generator):
    """`parametrizations.spectral_norm`, whose constructor draws `_u`/`_v` from the global RNG and runs
    15 power iterations on them. Seeded from the arm's generator by forking the global RNG, so a seed
    pins the state dict **and** the estimates are converged at build: overwriting `_u`/`_v` after
    construction (this module until 2026-09-23) left norms of 5 to 60 in an eval-mode target until its
    first hard copy, and the early TD errors raised `max_priority` for the whole run."""
    if generator is None:
        return parametrizations.spectral_norm(linear)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2 ** 62, (1,), generator=generator)))
        return parametrizations.spectral_norm(linear)


# ---------------------------------------------------------------- the trunk

class ResidualBlock(nn.Module):
    """`x + L2(relu(L1(relu(x))))`, IMPALA's block with linears for convolutions. Spectral norm on L1
    and L2 when asked; layer norm on the block's input when asked."""

    def __init__(self, width, spectral_norm, layer_norm, generator):
        super().__init__()
        self.norm = nn.LayerNorm(width) if layer_norm else None
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)
        _he_init(self.first, generator)
        _he_init(self.second, generator)
        if spectral_norm:
            self.first = _spectral_norm(self.first, generator)
            self.second = _spectral_norm(self.second, generator)

    def forward(self, inputs):
        hidden = inputs if self.norm is None else self.norm(inputs)
        hidden = self.second(torch.relu(self.first(torch.relu(hidden))))
        return inputs + hidden


class Trunk(nn.Module):
    """`obs_len -> features (width)`. `QNet`'s hidden stack, or the stem-and-blocks residual MLP."""

    def __init__(self, obs_len, fc_layer_params, trunk, generator):
        super().__init__()
        widths = [int(obs_len)] + [int(width) for width in fc_layer_params]
        self.width = widths[-1]
        self.residual = bool(trunk['residual'])
        if self.residual:
            self.stem = nn.Linear(widths[0], self.width)
            _he_init(self.stem, generator)
            self.blocks = nn.ModuleList([
                ResidualBlock(self.width, bool(trunk['spectral_norm']), bool(trunk['layer_norm']), generator)
                for _ in range(int(trunk['blocks']))])
        else:
            self.hidden = nn.ModuleList([nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
            for layer in self.hidden:
                _he_init(layer, generator)

    def forward(self, observations):
        if self.residual:
            values = torch.relu(self.stem(observations))
            for block in self.blocks:
                values = block(values)
            return torch.relu(values)
        values = observations
        for layer in self.hidden:
            values = torch.relu(layer(values))
        return values


# ---------------------------------------------------------------- the network

class RainbowNet(nn.Module):

    def __init__(self, arch, seed=None):
        super().__init__()
        head, trunk = head_of(arch), trunk_of(arch)
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
        self.head_type = head['type']
        self.num_actions = int(arch['num_actions'])
        self.dueling = bool(trunk['dueling'])
        self.noisy = bool(trunk['noisy'])
        self.stream_hidden = bool(trunk.get('stream_hidden', False))
        if self.head_type == 'c51':
            self.atoms = int(head['atoms'])
            self.v_min, self.v_max = float(head['v_min']), float(head['v_max'])
            if self.atoms < 2 or self.v_max <= self.v_min:
                raise ValueError('c51 needs atoms >= 2 and v_max > v_min, got {0}'.format(head))
            self.register_buffer('support', torch.linspace(self.v_min, self.v_max, self.atoms))
            self.outputs = self.atoms
        elif self.head_type == 'quantile':
            self.n = int(head['n'])
            if self.n < 1:
                raise ValueError('quantile head needs n >= 1, got {0}'.format(head))
            self.outputs = self.n
        else:
            self.embedding = int(head['embedding'])
            self.n_tau = int(head.get('n_tau', 8))
            self.k = int(head.get('k', 8))
            if self.embedding < 1:
                raise ValueError('iqn needs embedding >= 1, got {0}'.format(head))
            self.outputs = 1
        widths = [int(width) for width in arch['fc_layer_params']]
        # With stream hidden layers the plain trunk hands its last width to the streams; the residual
        # trunk keeps it and the streams repeat it.
        shared = widths[:-1] if self.stream_hidden and not trunk['residual'] else widths
        stream_width = widths[-1] if self.stream_hidden else 0
        self.trunk = Trunk(arch['obs_len'], shared, trunk, generator)
        width = self.trunk.width
        if self.head_type == 'iqn':
            self.tau_embed = nn.Linear(self.embedding, width)
            bound = 1.0 / math.sqrt(self.embedding)
            nn.init.uniform_(self.tau_embed.weight, -bound, bound, generator=generator)
            nn.init.zeros_(self.tau_embed.bias)
            self.register_buffer('harmonics', math.pi * torch.arange(1, self.embedding + 1, dtype=torch.float32))
        sigma = float(trunk['noisy_sigma'])
        self.advantage = _stream(width, stream_width, self.num_actions * self.outputs, self.noisy, sigma, generator)
        self.value = _stream(width, stream_width, self.outputs, self.noisy, sigma, generator) if self.dueling else None

    @property
    def noisy_layers(self):
        return noisy_layers(self)

    @property
    def delta(self):
        return (self.v_max - self.v_min) / (self.atoms - 1)

    # ------------------------------------------------------------ the streams

    def features(self, observations):
        return self.trunk(observations)

    def streams(self, features):
        """`(..., width) -> (..., actions, outputs)`: the advantage stream alone, or the dueling
        combine `V + A - mean_a(A)`, the mean over the action axis for every output separately."""
        lead = features.shape[:-1]
        advantage = self.advantage(features).view(*lead, self.num_actions, self.outputs)
        if not self.dueling:
            return advantage
        value = self.value(features).view(*lead, 1, self.outputs)
        return value + advantage - advantage.mean(dim=-2, keepdim=True)

    # ------------------------------------------------------------ c51

    def logits(self, observations):
        return self.streams(self.features(observations))

    def log_probs(self, observations):
        return torch.log_softmax(self.logits(observations), dim=2)

    def probs(self, observations):
        return torch.softmax(self.logits(observations), dim=2)

    # ------------------------------------------------------------ quantile

    def quantiles(self, observations):
        return self.streams(self.features(observations))

    def fractions(self, device):
        return (torch.arange(self.n, device=device, dtype=torch.float32) + 0.5) / float(self.n)

    # ------------------------------------------------------------ iqn

    def embed(self, taus):
        cosines = torch.cos(taus.unsqueeze(2) * self.harmonics)
        return torch.relu(self.tau_embed(cosines))

    def quantiles_at(self, features, taus):
        """`(m, n_tau, actions)`: the streams over the feature-embedding product, one tau at a time."""
        product = features.unsqueeze(1) * self.embed(taus)
        return self.streams(product).squeeze(-1)

    def sample_taus(self, batch, n, generator=None):
        return torch.rand(batch, n, device=self.tau_embed.weight.device, generator=generator)

    # ------------------------------------------------------------ the acting read

    def q_values(self, observations):
        if self.head_type == 'c51':
            return (self.probs(observations) * self.support).sum(dim=2)
        if self.head_type == 'quantile':
            return self.quantiles(observations).mean(dim=2)
        features = self.features(observations)
        return self.quantiles_at(features, self.sample_taus(features.shape[0], self.k)).mean(dim=1)

    def forward(self, observations):
        return self.q_values(observations)


# ---------------------------------------------------------------- the restore module's contract

def build(arch, device='cpu', seed=None):
    """The network an `arch.json` with `head` and `trunk` describes. **The signature `tools/restore.py` calls.**"""
    return RainbowNet(arch, seed=seed).to(device)


def greedy_policy_fn(net, device='cpu'):
    """`(m, obs_len) float32 -> (m,) int64`: argmax over the mean, **noise off**, eval mode, no autograd.

    The noise is switched off inside every call rather than once at construction, because the trainer
    interleaves this read with `update()` and `act()`, both of which switch it back on; a policy function
    that measured a noisy network would report exploration as skill.
    """
    net.eval()
    set_noise(net, False)

    def policy_fn(observations):
        set_noise(net, False)
        with torch.no_grad():
            batch = torch.as_tensor(observations, dtype=torch.float32, device=device)
            return net.q_values(batch).argmax(dim=1).to(torch.int64).cpu().numpy()

    return policy_fn
