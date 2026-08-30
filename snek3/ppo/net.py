"""The actor, the critic, and the three things you can ask of a categorical policy.

**The actor is the same network DQN trains, read as logits instead of as Q-values.** Not "the same
shape" — the same class, the same initialisers, built by the same factory. That is the point of the
comparison: if PPO and DQN differ, it has to be the learning rule and not the function class. It also
makes two things free that would otherwise be work — `greedy_policy_fn` is DQN's, because an argmax
over logits and an argmax over Q are the same operation; and the snek2 champion's converted weights
load straight into a `PolicyNet`, which is how "can PPO hold a policy DQN found" gets asked
separately from "can PPO find one".

**‡ `ppo/` imports two things from `dqn/` and that is deliberate.** This module takes the network and
its initialisers from [`../dqn/net.py`](../dqn/net.py); `agent.py` takes `build_adam` from
[`../dqn/agent.py`](../dqn/agent.py). Both carry measured facts — Keras' truncation correction, which
a plain `trunc_normal_` gets 12% wrong, and the exhausted-generator trap that silently trains an
optimiser over no parameters — so a second copy is a second thing to get wrong rather than a
decoupling. If a third algorithm arrives, they move to a module both can reach; two is not enough to
pay for that.

## The critic

`30 -> 320 -> 1`, its own tower, no shared trunk. Three reasons and the third decides it:

- the actor is then *exactly* `QNet`, per above;
- `vf_coef` becomes nearly inert, because a value loss of ~1,600 at initialisation (V starts near 0
  and the true value is ~40 at gamma=0.9975) cannot contaminate a policy gradient it shares no
  parameters with;
- **the actor is its own `nn.Module`, so `ckpt-<step>.pt` can hold `actor.state_dict()` and nothing
  else.** `arch.json`'s fields are "every field is required", so a shared trunk — whose checkpoint
  would carry a value head — would have needed a new field and invalidated every committed sidecar.
  Stage B measures the policy, and the policy is the actor.

The critic's seed is **derived** from the arm's rather than equal to it. `dqn/net.py` draws from a
local `torch.Generator`, so two nets built with the same seed are the same network; an actor and a
critic that opened as transposes of one another would be a coincidence nobody intended.
"""

import numpy as np
import torch

from dqn import net as qnet

# Which sub-stream of the arm's seed the critic takes. The actor takes the seed itself, so an actor's
# initialisation is identical to the DQN arm's at the same `SNEK_SEED` — which is what makes a
# seed-matched PPO-vs-DQN pair start from the same policy.
CRITIC_SEED_STREAM = 1


def critic_seed(seed):
    """A seed for the critic, derived from the arm's. `None` stays `None`."""
    if seed is None:
        return None
    return int(np.random.SeedSequence([int(seed), CRITIC_SEED_STREAM])
               .generate_state(1, dtype=np.uint32)[0])


def build(arch, device='cpu', seed=None):
    """The actor, sized by an `arch.json`. **The signature `tools/restore.py` calls.**

    Returns a `dqn.net.QNet` — see the module docstring. Its `num_actions` outputs are read as
    logits, which is a difference in interpretation and not in the tensor.
    """
    return qnet.build(arch, device=device, seed=seed)


def build_critic(arch, device='cpu', seed=None):
    """The critic: the same trunk with a single output.

    The arch is copied with `num_actions` set to 1 rather than being built through
    `arch_tools.build_arch`, because this shape is never written to disk and must never be mistaken
    for the policy's. `arch.json` describes the actor, which is what a checkpoint holds.
    """
    return qnet.build(dict(arch, num_actions=1), device=device, seed=critic_seed(seed))


def greedy_policy_fn(net, device='cpu'):
    """`(m, obs_len) float32 -> (m,) int64`, the argmax over the logits. No sampling.

    Literally DQN's, because the operation is the same one. **The measured policy is the argmax and
    not a sample from pi**, which is the deliberate choice: it is the analogue of DQN's greedy eval,
    it is what `watch.py` and `record_gif.py` show, and it makes a PPO stage-B row and a DQN stage-B
    row the same kind of number. Measuring the stochastic policy is a different question and a later
    knob.
    """
    return qnet.greedy_policy_fn(net, device=device)


# ---------------------------------------------------------------- the categorical policy

def log_softmax(logits):
    """Log-probabilities over the action axis.

    **Every read of the policy goes through this, never through `softmax` and a `log`.** The pair
    loses precision exactly where PPO is most sensitive: an action whose probability has collapsed
    has a large negative log-prob, `softmax` rounds it to 0, and the log is then `-inf`, which makes
    the ratio `exp(logp - old_logp)` a NaN that propagates into every parameter in the minibatch.
    """
    return torch.log_softmax(logits, dim=-1)


def sample(logits, generator=None):
    """One action per row, drawn from the policy. Returns `(actions, log_probs)`.

    The log-prob comes back with the action because that is the pair PPO's ratio needs, and computing
    it later from re-derived logits is how a collect-time and a loss-time log-prob drift apart. The
    first epoch's first minibatch must see a ratio of exactly 1.0; `tests/test_ppo_agent.py` pins it.

    `generator` is the policy's own seeded `torch.Generator`. It must not be torch's global one: the
    env draws its food from numpy and the two streams must stay independent, or an arm's decisions
    would depend on how many food cells were rejected.
    """
    logp = log_softmax(logits)
    actions = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
    return actions, logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def evaluate(logits, actions):
    """`(log_prob_of_actions, entropy)` under `logits`. The loss's half of `sample`.

    Entropy is over the whole distribution, not of the sampled action — it is the quantity the bonus
    is a bonus on. On three actions it maxes at ln 3 = 1.0986, which is the number to read a
    collapsing policy against.
    """
    logp = log_softmax(logits)
    chosen = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    entropy = -(logp.exp() * logp).sum(dim=-1)
    return chosen, entropy
