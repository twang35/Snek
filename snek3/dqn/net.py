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


class QNet(nn.Module):
    """`obs_len -> fc_layer_params (relu) -> num_actions`, with no activation on the head.

    Deliberately the same network snek2 trained, down to the initialisers, so a snek2 checkpoint
    converts by transposing its kernels and a snek3 arm starts from the same distribution. See
    `tools/import_tf_checkpoint.py`.
    """

    def __init__(self, obs_len, fc_layer_params, num_actions):
        super().__init__()
        widths = [int(obs_len)] + [int(width) for width in fc_layer_params]
        self.hidden = nn.ModuleList(
            [nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        self.head = nn.Linear(widths[-1], int(num_actions))
        self.reset_parameters()

    def reset_parameters(self):
        """snek2's initialisers, spelled out in torch.

        The hidden layers used Keras `VarianceScaling(scale=2.0, mode='fan_in',
        distribution='truncated_normal')`. That is He-normal with one wrinkle: Keras divides the
        standard deviation by 0.8796 to correct for the truncation at ±2σ pulling the realised
        variance *down*, so a plain `trunc_normal_` at He's σ would start ~12% narrower than every
        snek2 arm did. The head used `RandomUniform(-0.03, 0.03)` with zero bias, which keeps the
        opening Q-values near zero and the opening policy near uniform.
        """
        for layer in self.hidden:
            fan_in = layer.weight.shape[1]
            stddev = math.sqrt(2.0 / fan_in) / 0.87962566103423978
            nn.init.trunc_normal_(layer.weight, std=stddev, a=-2 * stddev, b=2 * stddev)
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.head.weight, -0.03, 0.03)
        nn.init.zeros_(self.head.bias)

    def forward(self, observations):
        values = observations
        for layer in self.hidden:
            values = torch.relu(layer(values))
        return self.head(values)


def build(arch, device='cpu'):
    """A `QNet` sized by an `arch.json` dict. The sidecar is authoritative."""
    net = QNet(arch['obs_len'], arch['fc_layer_params'], arch['num_actions'])
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
