"""`RainbowAgent`: Group A's `DistAgent` losses over a `RainbowNet`, acting with the noise on.

What is this module's and not the parent's:

| | |
|---|---|
| the network | `algos/rainbow/net.py`'s, rebuilt over the parent's (which builds A's head for the same sidecar) |
| acting | **noise on** (`c-value-stack.md`, decided 2026-09-20): the collector's greedy action is a noisy forward, one fresh draw per act, and epsilon-greedy over that when the schedule says so. Rainbow's paper cell runs epsilon 0, so this is its whole exploration |
| measuring | `policy_fn` is `net.greedy_policy_fn`: noise off. Stage A, restore, `watch.py` and every pass read the `mu` weights |
| learning | the parent's `update`, with the noise on in both the online and the target net so each forward is its own draw (Fortunato et al. §3.2: independent noise for the online and target networks) |
| double-Q | `double=False` (BTR) takes the target action from the **target** net's own argmax; with Munchausen on there is no argmax and the flag has no effect, as the plan says |
"""

import numpy as np
import torch

from algos.dist.agent import DistAgent
from algos.dqn.agent import build_adam
from algos.rainbow import net as network
import torch.nn.functional as F


class RainbowAgent(DistAgent):

    def __init__(self, arch, double=True, **kwargs):
        seed = kwargs.get('seed')
        device = kwargs.get('device', 'cpu')
        super().__init__(arch, **kwargs)
        self.net = network.build(arch, device, seed=seed)
        self.target = network.build(arch, device)
        self.target.load_state_dict(self.net.state_dict())
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        # Eval mode for good: a spectral-normed linear runs a power iteration on every training-mode
        # forward, and a target whose weights drift between copies is not a target. Its `_u`/`_v` arrive
        # with each hard copy, so its normalised weight is the online net's at the copy.
        self.target.eval()
        self.optimizer = build_adam(self.net.parameters(), float(kwargs.get('learning_rate', 1e-5)),
                                    float(kwargs.get('adam_epsilon', 1e-7)))
        self.double = bool(double)
        self.noisy = bool(network.trunk_of(arch)['noisy'])

    # ---------------------------------------------------------------- acting

    @property
    def policy_fn(self):
        return network.greedy_policy_fn(self.net, self.device)

    def greedy_actions(self, observations):
        """The collector's greedy action: the noisy network's argmax, a fresh draw per call."""
        self.net.eval()
        network.set_noise(self.net, self.noisy)
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=self.device)
            return self.net.q_values(tensor).argmax(dim=1).cpu().numpy().astype(np.int64)

    # ---------------------------------------------------------------- learning

    def update(self, batch, weights=None):
        network.set_noise(self.net, self.noisy)
        network.set_noise(self.target, self.noisy)
        return super().update(batch, weights)

    def fresh_net(self, seed):
        return network.build(self.arch, self.device, seed=seed)

    def _double_q_target_action(self, next_obs):
        if self.double:
            return super()._double_q_target_action(next_obs)
        return self.target.q_values(next_obs).argmax(dim=1)

    def _next_action_mixture(self, obs, action, next_obs):
        if self.double or self.munchausen_alpha > 0.0:
            return super()._next_action_mixture(obs, action, next_obs)
        best = self.target.q_values(next_obs).argmax(dim=1)
        weights = F.one_hot(best, self.num_actions).float()
        return weights, torch.zeros_like(weights), torch.zeros(obs.shape[0], device=obs.device)
