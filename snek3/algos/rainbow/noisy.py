"""`NoisyLinear`: a linear layer with factorised Gaussian noise on its weights (Fortunato et al. 2018).

Rainbow's exploration. Each weight is `mu + sigma * eps` with `eps` the outer product of two scaled
Gaussian vectors, `f(x) = sign(x) sqrt(|x|)`, resampled **on every forward while `active`**; the
network learns `mu` and `sigma`, so the noise shrinks where the value is certain. With `active` off the
layer is `nn.Linear` with the `mu` weights, which is what every greedy read of a checkpoint uses
(`net.greedy_policy_fn`). The initialisation is the paper's: `mu ~ U(-1/sqrt(p), 1/sqrt(p))`,
`sigma = sigma_0 / sqrt(p)`, `p` the fan-in.

`active` is a plain attribute rather than `self.training`, on purpose: the residual trunk's spectral
norm runs a power iteration on every training-mode forward, so the acting path stays in eval mode and
turns the noise on by name (`set_noise`). The two states are then independent and each is set where it
is wanted -- the agent turns noise on to act and to learn, the policy function turns it off to measure.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, sigma_zero=0.5, generator=None):
        super().__init__()
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.sigma_zero = float(sigma_zero)
        self.weight_mu = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.weight_sigma = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias_mu = nn.Parameter(torch.empty(self.out_features))
        self.bias_sigma = nn.Parameter(torch.empty(self.out_features))
        self.active = True
        self.reset_parameters(generator)

    def reset_parameters(self, generator=None):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound, generator=generator)
        nn.init.uniform_(self.bias_mu, -bound, bound, generator=generator)
        nn.init.constant_(self.weight_sigma, self.sigma_zero / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.sigma_zero / math.sqrt(self.in_features))

    @staticmethod
    def _scaled_noise(size, device):
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()

    @torch.no_grad()
    def resample(self):
        """A fresh factorised draw `(weight_epsilon (out, in), bias_epsilon (out,))`:
        `eps_ij = f(eps_i) f(eps_j)`, `eps_bias = f(eps_i)`. Fresh tensors, not buffers written in place:
        an update runs two online forwards before its backward, and the first's saved noise must survive
        the second."""
        eps_in = self._scaled_noise(self.in_features, self.weight_mu.device)
        eps_out = self._scaled_noise(self.out_features, self.weight_mu.device)
        return torch.outer(eps_out, eps_in), eps_out

    def forward(self, inputs):
        if not self.active:
            return F.linear(inputs, self.weight_mu, self.bias_mu)
        weight_epsilon, bias_epsilon = self.resample()
        return F.linear(inputs, self.weight_mu + self.weight_sigma * weight_epsilon,
                        self.bias_mu + self.bias_sigma * bias_epsilon)

    def extra_repr(self):
        return 'in_features={0}, out_features={1}, sigma_zero={2}'.format(
            self.in_features, self.out_features, self.sigma_zero)


def noisy_layers(module):
    return [layer for layer in module.modules() if isinstance(layer, NoisyLinear)]


def set_noise(module, active):
    """Turns every `NoisyLinear` under `module` on or off. Returns how many it found."""
    layers = noisy_layers(module)
    for layer in layers:
        layer.active = bool(active)
    return len(layers)
