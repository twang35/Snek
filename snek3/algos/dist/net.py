"""The distributional networks, and the module `tools/restore.py` loads a checkpoint through.

Every head sits on `algos/dqn/net.py`'s trunk -- the same hidden stack, the same initialisers, built by
`QNet` itself so the seed pins the weights the way it does for every other arm -- and replaces only the
output. `arch['head']` (see `tools/arch.py`'s `OPTIONAL_FIELDS`) says which head and how wide:

| `head['type']` | network | output |
|---|---|---|
| `c51` | `QNet` with `num_actions * atoms` outputs, read as logits over a fixed support | `(m, actions, atoms)` log-probabilities |
| `quantile` | `QNet` with `num_actions * n` outputs, read as quantile values at fixed fractions | `(m, actions, n)` values |
| `iqn` | `QNet`'s trunk, a cosine embedding of sampled fractions multiplied into the trunk's features, `QNet`'s head over the product | `(m, n_tau, actions)` values at the sampled fractions |
| `fqf` | `iqn` plus a fraction-proposal linear over the trunk's features | as `iqn`, at the proposed fractions |

**One interface for the agent and the policy**: `q_values(obs)` is the mean of the distribution per
action, `(m, actions)`, and `cvar_values(obs, alpha)` is the mean of its lower `alpha` tail, which is
what the risk-sensitive read (`a-return-tail.md` §5) acts on. The greedy `policy_fn` is the argmax of
either, in the `(m, obs_len) float32 -> (m,) int64` shape the engine takes; a `variant` string
(`cvar:0.25`) selects the tail read and is the only way anything but the mean is measured.
"""

import math

import numpy as np
import torch
from torch import nn

from algos.dqn import net as qnet

HEAD_TYPES = ('c51', 'quantile', 'iqn', 'fqf')


def head_of(arch):
    head = arch.get('head')
    if not head or head.get('type') not in HEAD_TYPES:
        raise ValueError('a distributional arch needs head.type in {0}, got {1!r}'.format(
            HEAD_TYPES, head))
    return head


# ---------------------------------------------------------------- shared

def _features(trunk, observations):
    """`QNet`'s hidden stack applied to a batch: the features its own head would read."""
    values = observations
    for layer in trunk.hidden:
        values = torch.relu(layer(values))
    return values


def _midpoint_fractions(n, device):
    """QR-DQN's fixed fractions `(2i - 1) / 2n`, i = 1..n."""
    return (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / float(n)


class CategoricalNet(nn.Module):
    """C51. `num_actions * atoms` logits from `QNet`, a fixed support from `v_min` to `v_max`."""

    head_type = 'c51'

    def __init__(self, arch, seed=None):
        super().__init__()
        head = head_of(arch)
        self.num_actions = int(arch['num_actions'])
        self.atoms = int(head['atoms'])
        self.v_min, self.v_max = float(head['v_min']), float(head['v_max'])
        if self.atoms < 2 or self.v_max <= self.v_min:
            raise ValueError('c51 needs atoms >= 2 and v_max > v_min, got {0}'.format(head))
        self.qnet = qnet.QNet(arch['obs_len'], arch['fc_layer_params'],
                              self.num_actions * self.atoms, seed=seed)
        self.register_buffer('support', torch.linspace(self.v_min, self.v_max, self.atoms))

    @property
    def delta(self):
        return (self.v_max - self.v_min) / (self.atoms - 1)

    def logits(self, observations):
        return self.qnet(observations).view(-1, self.num_actions, self.atoms)

    def log_probs(self, observations):
        return torch.log_softmax(self.logits(observations), dim=2)

    def probs(self, observations):
        return torch.softmax(self.logits(observations), dim=2)

    def q_values(self, observations):
        return (self.probs(observations) * self.support).sum(dim=2)

    def cvar_values(self, observations, alpha):
        """Mean of the lower `alpha` of the mass, per action: the CVaR_alpha of the categorical."""
        probs = self.probs(observations)
        cumulative = probs.cumsum(dim=2)
        # Mass of each atom that lies below the alpha quantile: the whole atom while the cumulative
        # is under alpha, the remainder for the atom that crosses it, nothing above.
        below = torch.clamp(alpha - (cumulative - probs), min=0.0)
        mass = torch.minimum(below, probs)
        return (mass * self.support).sum(dim=2) / max(alpha, 1e-8)

    def forward(self, observations):
        return self.q_values(observations)


class QuantileNet(nn.Module):
    """QR-DQN. `num_actions * n` quantile values from `QNet`, at the fixed midpoint fractions."""

    head_type = 'quantile'

    def __init__(self, arch, seed=None):
        super().__init__()
        head = head_of(arch)
        self.num_actions = int(arch['num_actions'])
        self.n = int(head['n'])
        if self.n < 1:
            raise ValueError('quantile head needs n >= 1, got {0}'.format(head))
        self.qnet = qnet.QNet(arch['obs_len'], arch['fc_layer_params'],
                              self.num_actions * self.n, seed=seed)

    def quantiles(self, observations):
        """`(m, actions, n)`, the value at fraction `(2i - 1) / 2n` for i = 1..n."""
        return self.qnet(observations).view(-1, self.num_actions, self.n)

    def fractions(self, device):
        return _midpoint_fractions(self.n, device)

    def q_values(self, observations):
        return self.quantiles(observations).mean(dim=2)

    def cvar_values(self, observations, alpha):
        """Mean of the lowest `ceil(alpha * n)` quantiles, sorted, per action."""
        sorted_q, _ = self.quantiles(observations).sort(dim=2)
        k = max(1, int(math.ceil(alpha * self.n)))
        return sorted_q[:, :, :k].mean(dim=2)

    def forward(self, observations):
        return self.q_values(observations)


class ImplicitNet(nn.Module):
    """IQN, and FQF when `head['type'] == 'fqf'`.

    The trunk is `QNet`'s hidden stack and the value head is `QNet`'s `head`, so a checkpoint's trunk is
    the same tensor set every other arm has. The cosine embedding `relu(Linear(cos(pi i tau)))`,
    i = 1..embedding, is multiplied into the features (Dabney et al. 2018, Eq. 4), and the head reads
    the product one fraction at a time: `(m, n_tau, actions)`.

    FQF adds `fraction`: a linear over the (detached) features producing `n` logits, whose softmax's
    cumulative sum is the proposed fractions tau_1..tau_{n-1} (with tau_0 = 0 and tau_n = 1) and whose
    midpoints tau_hat are where the quantile values are read for acting and for the loss.
    """

    def __init__(self, arch, seed=None):
        super().__init__()
        head = head_of(arch)
        self.head_type = head['type']
        self.num_actions = int(arch['num_actions'])
        self.embedding = int(head['embedding'])
        self.n_tau = int(head.get('n_tau', 64))
        self.k = int(head.get('k', 32))
        self.n = int(head.get('n', 32))
        if self.embedding < 1:
            raise ValueError('iqn needs embedding >= 1, got {0}'.format(head))
        self.qnet = qnet.QNet(arch['obs_len'], arch['fc_layer_params'], self.num_actions, seed=seed)
        width = int(arch['fc_layer_params'][-1])
        self.tau_embed = nn.Linear(self.embedding, width)
        self.fraction = nn.Linear(width, self.n) if self.head_type == 'fqf' else None
        self.reset_extra_parameters(seed)
        self.register_buffer('harmonics', math.pi * torch.arange(1, self.embedding + 1,
                                                                 dtype=torch.float32))

    def reset_extra_parameters(self, seed=None):
        """The embedding and the fraction proposal, from a generator derived from the seed so the trunk
        (seeded by `QNet` itself) and these do not share a stream."""
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.tau_embed.weight.device)
            generator.manual_seed(int(seed) * 7919 + 17)
        bound = 1.0 / math.sqrt(self.tau_embed.weight.shape[1])
        nn.init.uniform_(self.tau_embed.weight, -bound, bound, generator=generator)
        nn.init.zeros_(self.tau_embed.bias)
        if self.fraction is not None:
            # Near-uniform proposals at the start (FQF's released code initialises the fraction net
            # small), so the first fractions are QR-DQN's midpoints to within noise.
            nn.init.uniform_(self.fraction.weight, -0.01, 0.01, generator=generator)
            nn.init.zeros_(self.fraction.bias)

    # ------------------------------------------------------------ the quantile function

    def features(self, observations):
        return _features(self.qnet, observations)

    def embed(self, taus):
        """`(m, n_tau) -> (m, n_tau, width)`: relu(W cos(pi i tau) + b)."""
        cosines = torch.cos(taus.unsqueeze(2) * self.harmonics)
        return torch.relu(self.tau_embed(cosines))

    def quantiles_at(self, features, taus):
        """`(m, n_tau, actions)`: the quantile values at each `taus[:, j]`."""
        product = features.unsqueeze(1) * self.embed(taus)
        return self.qnet.head(product)

    def sample_taus(self, batch, n, generator=None):
        return torch.rand(batch, n, device=self.tau_embed.weight.device, generator=generator)

    # ------------------------------------------------------------ FQF's fractions

    def propose(self, features):
        """`(taus (m, n+1), tau_hats (m, n), entropy (m,))` from the fraction net, features detached.

        tau_0 = 0 and tau_n = 1 by construction; the interior fractions are the cumulative softmax, so
        they are monotone and in (0, 1) whatever the logits.
        """
        logits = self.fraction(features.detach())
        probs = torch.softmax(logits, dim=1)
        cumulative = probs.cumsum(dim=1)
        zeros = torch.zeros(probs.shape[0], 1, device=probs.device)
        taus = torch.cat([zeros, cumulative], dim=1)
        taus = torch.cat([taus[:, :-1], torch.ones_like(zeros)], dim=1)   # pin tau_n to exactly 1
        tau_hats = (taus[:, :-1] + taus[:, 1:]) / 2.0
        entropy = -(probs * torch.log_softmax(logits, dim=1)).sum(dim=1)
        return taus, tau_hats, entropy

    # ------------------------------------------------------------ reads

    def acting_taus(self, features, alpha=1.0):
        """The fractions the greedy read averages over: FQF's proposed midpoints, IQN's `k` uniform
        draws; both scaled by `alpha` for the CVaR read (tau <- alpha * tau, Dabney et al. §3.1)."""
        if self.head_type == 'fqf':
            _, tau_hats, _ = self.propose(features)
            taus = tau_hats
        else:
            taus = self.sample_taus(features.shape[0], self.k)
        return taus * float(alpha)

    def q_values(self, observations, alpha=1.0):
        features = self.features(observations)
        taus = self.acting_taus(features, alpha)
        values = self.quantiles_at(features, taus)                  # (m, k, actions)
        if self.head_type == 'fqf' and alpha >= 1.0:
            # FQF's mean is the fraction-weighted sum over the proposed bins, not a plain average.
            taus_full, _, _ = self.propose(features)
            widths = (taus_full[:, 1:] - taus_full[:, :-1]).unsqueeze(2)
            return (widths * values).sum(dim=1)
        return values.mean(dim=1)

    def cvar_values(self, observations, alpha):
        return self.q_values(observations, alpha=alpha)

    def forward(self, observations):
        return self.q_values(observations)


# ---------------------------------------------------------------- the restore module's contract

def build(arch, device='cpu', seed=None):
    """The network an `arch.json` with a `head` describes. **The signature `tools/restore.py` calls.**"""
    kind = head_of(arch)['type']
    if kind == 'c51':
        net = CategoricalNet(arch, seed=seed)
    elif kind == 'quantile':
        net = QuantileNet(arch, seed=seed)
    else:
        net = ImplicitNet(arch, seed=seed)
    return net.to(device)


def parse_variant(variant):
    """`None` -> the mean; `'cvar:0.25'` -> `('cvar', 0.25)`. Anything else names itself in the error."""
    if variant is None or variant == '' or variant == 'mean':
        return None
    kind, _, value = str(variant).partition(':')
    if kind != 'cvar':
        raise ValueError('unknown policy variant {0!r}; this head knows cvar:<alpha>'.format(variant))
    alpha = float(value)
    if not 0.0 < alpha <= 1.0:
        raise ValueError('cvar alpha must be in (0, 1], got {0}'.format(value))
    return kind, alpha


def greedy_policy_fn(net, device='cpu', variant=None):
    """`(m, obs_len) float32 -> (m,) int64`: argmax over the mean, or over the CVaR tail for a variant.

    Eval mode and no autograd, as `algos/dqn/net.greedy_policy_fn`. IQN's mean read draws `k` fractions
    per call from torch's global generator; the measurement engine's episodes are seeded through the
    game, not the policy, so a run is repeatable to the game and not to the draw -- as the paper's
    agent is. A seed for the draw is a later knob if a byte-exact replay is wanted.
    """
    net.eval()
    parsed = parse_variant(variant)

    def policy_fn(observations):
        with torch.no_grad():
            batch = torch.as_tensor(observations, dtype=torch.float32, device=device)
            values = net.q_values(batch) if parsed is None else net.cvar_values(batch, parsed[1])
            return values.argmax(dim=1).to(torch.int64).cpu().numpy()

    return policy_fn


def variant_policy_fn(net, device='cpu', variant=None):
    """The name `tools/restore.policy_fn_for` looks for when a variant is asked of a module."""
    return greedy_policy_fn(net, device=device, variant=variant)
