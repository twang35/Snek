"""The Q-network, and the greedy `policy_fn` the measurement engine takes.

One architecture built in one place, because a second copy is how the trainer and the evaluator
quietly drift apart. The shape always comes from an `arch.json` (see `tools/arch.py`) rather than
from an environment variable, which is what removes snek2's silent-default failure — a checkpoint
trained at `100,200,100` restored into a `50,100,50` net, loading without a word and playing like a
beginner.

**`policy_fn` is the only seam between torch and the rest of snek3.** It is a plain callable of
shape `(m, obs_len) float32 -> (m,) int64`, so `vectorized/` imports no torch and the whole
measurement stack can be exercised against a hand-written heuristic. Do not widen it.
"""

import math

import torch
from torch import nn


def he_init(linear, generator=None):
    """snek2's hidden-layer initialiser: He-normal, truncated at 2 sigma, with Keras' truncation correction
    (`QNet.reset_parameters` on why the 0.8796). Module-level so the recurrent towers (`algos/ppo/net.py`,
    `algos/r2d2/net.py`) build their trunks with the same draw rather than a copy of these lines."""
    fan_in = linear.weight.shape[1]
    stddev = math.sqrt(2.0 / fan_in) / 0.87962566103423978
    nn.init.trunc_normal_(linear.weight, std=stddev, a=-2 * stddev, b=2 * stddev, generator=generator)
    nn.init.zeros_(linear.bias)


def head_init(linear, generator=None):
    """snek2's output-layer initialiser: uniform in [-0.03, 0.03], zero bias, so the opening policy is near
    uniform."""
    nn.init.uniform_(linear.weight, -0.03, 0.03, generator=generator)
    nn.init.zeros_(linear.bias)


def make_generator(seed, device='cpu'):
    """A local `torch.Generator` at `seed`, or None for None. The reason it is local is in
    `QNet.reset_parameters`."""
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


class QNet(nn.Module):
    """`obs_len -> fc_layer_params (relu) -> num_actions`, with no activation on the head.

    Deliberately the same network snek2 trained, down to the initialisers, so a snek2 checkpoint
    converts by transposing its kernels and a snek3 arm starts from the same distribution. See
    `tools/import_tf_checkpoint.py`.
    """

    def __init__(self, obs_len, fc_layer_params, num_actions, seed=None):
        super().__init__()
        widths = [int(obs_len)] + [int(width) for width in fc_layer_params]
        self.hidden = nn.ModuleList(
            [nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        self.head = nn.Linear(widths[-1], int(num_actions))
        self.reset_parameters(seed)

    def reset_parameters(self, seed=None):
        """snek2's initialisers, spelled out in torch.

        The hidden layers used Keras `VarianceScaling(scale=2.0, mode='fan_in',
        distribution='truncated_normal')`. That is He-normal with one wrinkle: Keras divides the
        standard deviation by 0.8796 to correct for the truncation at ±2σ pulling the realised
        variance *down*, so a plain `trunc_normal_` at He's σ would start ~12% narrower than every
        snek2 arm did. The head used `RandomUniform(-0.03, 0.03)` with zero bias, which keeps the
        opening Q-values near zero and the opening policy near uniform.

        **`seed` draws from a local `torch.Generator`, not the global one, and it is not optional for
        a comparison.** Two nets built in one process with the same configured seed were *different
        networks* before this existed, because `nn.init` reads torch's global RNG and the second call
        continues where the first stopped. Every "seed-matched" arm would have differed in its
        initialisation — the one thing a seed is supposed to pin — and nothing in a run report would
        have shown it. This is the same class of defect that disqualified cpprb, whose buffer
        silently ignored `seed=`.
        """
        generator = make_generator(seed, self.head.weight.device)
        for layer in self.hidden:
            he_init(layer, generator)
        head_init(self.head, generator)

    def forward(self, observations):
        values = observations
        for layer in self.hidden:
            values = torch.relu(layer(values))
        return self.head(values)


def build(arch, device='cpu', seed=None):
    """A `QNet` sized by an `arch.json` dict. The sidecar is authoritative.

    `seed` pins the initialisation. Leave it None when the weights are about to be overwritten by a
    checkpoint restore, which is every path in `tools/`.
    """
    net = QNet(arch['obs_len'], arch['fc_layer_params'], arch['num_actions'], seed=seed)
    return net.to(device)


def greedy_policy_fn(net, device='cpu'):
    """`(m, obs_len) float32 -> (m,) int64`, the argmax over Q. No epsilon.

    Puts the net in eval mode and keeps it out of autograd, both of which are correctness rather
    than speed here: a stray `requires_grad` graph over a 500-episode measurement holds every
    intermediate activation of every step alive.
    """
    net.eval()

    def policy_fn(observations):
        with torch.no_grad():
            batch = torch.as_tensor(observations, dtype=torch.float32, device=device)
            return net(batch).argmax(dim=1).to(torch.int64).cpu().numpy()

    return policy_fn
