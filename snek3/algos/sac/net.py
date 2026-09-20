"""The actor and the two critics. The actor is PPO's: DQN's `QNet` read as logits, so a checkpoint is
watchable, recordable and measurable by every eval shard with no new sidecar field -- an argmax over
logits and an argmax over Q are the same operation. The critics are `QNet`s too, one Q value per action,
built on derived seeds so neither opens as a copy of the actor (`algos/ppo/net.py` on why derived)."""

import numpy as np

from algos.dqn import net as qnet
from algos.ppo import net as ppo_net

# Which sub-streams of the arm's seed the two critics take. The actor takes the seed itself, so a
# seed-matched `sac` arm and `dqn` arm open from the same network.
CRITIC_SEED_STREAMS = (2, 3)


def critic_seed(seed, stream):
    if seed is None:
        return None
    return int(np.random.SeedSequence([int(seed), int(stream)]).generate_state(1, dtype=np.uint32)[0])


def build(arch, device='cpu', seed=None):
    """The actor, sized by an `arch.json`. **The signature `tools/restore.py` calls.**"""
    return qnet.build(arch, device=device, seed=seed)


def build_critic(arch, device='cpu', seed=None, stream=CRITIC_SEED_STREAMS[0]):
    """One critic: the same trunk, `num_actions` outputs read as Q(s, .). Never written to disk."""
    return qnet.build(arch, device=device, seed=critic_seed(seed, stream))


def greedy_policy_fn(net, device='cpu'):
    """The argmax over the logits: the measured policy, as for PPO (`algos/ppo/net.py` on why not a sample)."""
    return ppo_net.greedy_policy_fn(net, device=device)


log_softmax = ppo_net.log_softmax
sample = ppo_net.sample
