"""The actor, the critic, and the three things you can ask of a categorical policy.

**The actor is the same network DQN trains, read as logits instead of as Q-values.** Not "the same
shape" — the same class, the same initialisers, built by the same factory. That is the point of the
comparison: if PPO and DQN differ, it has to be the learning rule and not the function class. It also
makes two things free that would otherwise be work — `greedy_policy_fn` is DQN's, because an argmax
over logits and an argmax over Q are the same operation; and the snek2 champion's converted weights
load straight into a `PolicyNet`, which is how "can PPO hold a policy DQN found" gets asked
separately from "can PPO find one".

**‡ `algos/ppo/` imports two things from `algos/dqn/` and that is deliberate.** This module takes the network and
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

The critic's seed is **derived** from the arm's rather than equal to it. `algos/dqn/net.py` draws from a
local `torch.Generator`, so two nets built with the same seed are the same network; an actor and a
critic that opened as transposes of one another would be a coincidence nobody intended.
"""

import math

import numpy as np
import torch
from torch import nn

from algos.dqn import net as qnet
from algos.stateful import StatefulPolicy

# `arch['recurrent']['type']` for a PPO tower: a GRU cell or an LSTM cell between the trunk and the head.
RECURRENT_KINDS = ('gru', 'lstm')

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


def recurrent_of(arch):
    """The sidecar's `recurrent` block as `{'type', 'hidden'}`, or None for a feed-forward arch. Refuses an
    unknown cell or a width below 1 by name rather than building a tower that is silently feed-forward."""
    spec = arch.get('recurrent')
    if not spec:
        return None
    kind = spec.get('type')
    if kind not in RECURRENT_KINDS:
        raise ValueError('arch recurrent.type must be one of {0}, got {1!r}'.format(RECURRENT_KINDS, kind))
    hidden = int(spec.get('hidden', 0))
    if hidden < 1:
        raise ValueError('arch recurrent.hidden must be at least 1, got {0}'.format(hidden))
    return {'type': kind, 'hidden': hidden}


class RecurrentTower(nn.Module):
    """`obs_len -> fc_layer_params (relu) -> GRU | LSTM (hidden) -> outputs`: E1's actor and E1's critic
    (`plans/algoExploration/e-memory.md` §2), each its own tower as the feed-forward pair is.

    The trunk is `QNet`'s hidden stack to the initialiser, so a recurrent arm differs from b27's in the cell
    and nothing else. The cell's parameters take torch's own uniform `(-1/sqrt(hidden), 1/sqrt(hidden))`,
    drawn from the arm's generator so a seed pins them. The head is `QNet`'s head initialiser.

    **The state is one flat `(n, state_width)` tensor**: a GRU's hidden, or an LSTM's `(h, c)` side by side,
    so the rollout stores one array per tower and `StatefulPolicy` carries one per lane whatever the cell.
    `step` zeroes the state of a `fresh` row *before* using it, which is what `done` resets inside a sequence
    and what an episode's first step starts from.
    """

    def __init__(self, obs_len, fc_layer_params, outputs, kind, hidden, seed=None):
        super().__init__()
        if kind not in RECURRENT_KINDS:
            raise ValueError('kind must be one of {0}, got {1!r}'.format(RECURRENT_KINDS, kind))
        if int(hidden) < 1:
            raise ValueError('hidden must be at least 1, got {0}'.format(hidden))
        widths = [int(obs_len)] + [int(width) for width in fc_layer_params]
        self.kind = kind
        self.hidden_size = int(hidden)
        self.hidden = nn.ModuleList([nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        # `nn.LSTM` / `nn.GRU` rather than the `*Cell` modules, for `unroll`: the fused kernel runs a whole
        # segment between episode starts in one call, where a cell call per step on a two-lane minibatch is
        # Python overhead 256 times over (measured 2026-09-30: 1,021 transitions/s against 54,340 feed-forward).
        # `step` is the same module on a length-1 sequence.
        if kind == 'lstm':
            self.cell = nn.LSTM(widths[-1], self.hidden_size)
        else:
            self.cell = nn.GRU(widths[-1], self.hidden_size)
        self.head = nn.Linear(self.hidden_size, int(outputs))
        self.reset_parameters(seed)

    @property
    def state_width(self):
        return self.hidden_size * (2 if self.kind == 'lstm' else 1)

    def reset_parameters(self, seed=None):
        generator = qnet.make_generator(seed, self.head.weight.device)
        for layer in self.hidden:
            qnet.he_init(layer, generator)
        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.cell.parameters():
            nn.init.uniform_(parameter, -bound, bound, generator=generator)
        qnet.head_init(self.head, generator)

    def initial_state(self, n, device=None):
        return torch.zeros((int(n), self.state_width), dtype=torch.float32,
                           device=device or self.head.weight.device)

    def features(self, observations):
        values = observations
        for layer in self.hidden:
            values = torch.relu(layer(values))
        return values

    def _run(self, features, state):
        """`(L, n, in) x (n, state_width) -> (outputs (L, n, hidden), state)`: the cell over a segment with no
        reset inside it."""
        if self.kind == 'lstm':
            h = state[:, :self.hidden_size].unsqueeze(0).contiguous()
            c = state[:, self.hidden_size:].unsqueeze(0).contiguous()
            out, (h, c) = self.cell(features, (h, c))
            return out, torch.cat([h[0], c[0]], dim=-1)
        out, h = self.cell(features, state.unsqueeze(0).contiguous())
        return out, h[0]

    def step(self, observations, state, fresh=None):
        """One step of every row: `(n, obs_len) x (n, state_width) [x (n,) bool] -> (outputs (n, k), state)`."""
        if fresh is not None:
            state = state * (~fresh).to(state.dtype).unsqueeze(-1)
        out, state = self._run(self.features(observations).unsqueeze(0), state)
        return self.head(out[0]), state

    def unroll(self, observations, state, fresh):
        """`(T, n, obs_len) x (n, state_width) x (T, n) bool -> (outputs (T, n, k), final state)`: the cell
        run forward through a stored sequence from its stored initial state, the state zeroed at every
        `fresh` step, with gradient. Truncated backpropagation: nothing flows into `state`'s past.

        The sequence is cut at every step where some row is fresh and each piece runs through the fused
        kernel in one call, which is `step` applied T times (a fixture pins the equality) at a fraction of
        the cost."""
        T, n = observations.shape[0], observations.shape[1]
        features = self.features(observations.reshape(T * n, -1)).reshape(T, n, -1)
        starts = fresh.any(dim=1).nonzero().flatten().tolist()
        cuts = sorted(set([0] + starts + [T]))
        outputs = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            state = state * (~fresh[a]).to(state.dtype).unsqueeze(-1)
            out, state = self._run(features[a:b], state)
            outputs.append(out)
        return self.head(torch.cat(outputs, dim=0)), state

    def forward(self, observations, state=None):
        """One step from `state`, or from a zero state: the outputs alone. For a caller that treats the tower
        as a feed-forward net; the stateful callers use `step` and `unroll`."""
        if state is None:
            state = self.initial_state(observations.shape[0], observations.device)
        return self.step(observations, state)[0]


def build(arch, device='cpu', seed=None):
    """The actor, sized by an `arch.json`. **The signature `tools/restore.py` calls.**

    Returns a `dqn.net.QNet` — see the module docstring — or, when the sidecar carries a `recurrent`
    block, a `RecurrentTower` with `num_actions` outputs. Either way the outputs are read as logits,
    which is a difference in interpretation and not in the tensor.
    """
    spec = recurrent_of(arch)
    if spec is None:
        return qnet.build(arch, device=device, seed=seed)
    return RecurrentTower(arch['obs_len'], arch['fc_layer_params'], arch['num_actions'],
                          spec['type'], spec['hidden'], seed=seed).to(device)


def build_critic(arch, device='cpu', seed=None):
    """The critic: the same trunk with a single output, and the same cell when the actor has one
    (**its own**, not the actor's -- the two towers share nothing, as the feed-forward pair share nothing).

    The arch is copied with `num_actions` set to 1 rather than being built through
    `arch_tools.build_arch`, because this shape is never written to disk and must never be mistaken
    for the policy's. `arch.json` describes the actor, which is what a checkpoint holds.
    """
    spec = recurrent_of(arch)
    if spec is None:
        return qnet.build(dict(arch, num_actions=1), device=device, seed=critic_seed(seed))
    return RecurrentTower(arch['obs_len'], arch['fc_layer_params'], 1, spec['type'], spec['hidden'],
                          seed=critic_seed(seed)).to(device)


def greedy_policy_fn(net, device='cpu'):
    """`(m, obs_len) float32 -> (m,) int64`, the argmax over the logits. No sampling.

    Literally DQN's, because the operation is the same one. **The measured policy is the argmax and
    not a sample from pi**, which is the deliberate choice: it is the analogue of DQN's greedy eval,
    it is what `watch.py` and `record_gif.py` show, and it makes a PPO stage-B row and a DQN stage-B
    row the same kind of number. Measuring the stochastic policy is a different question and a later
    knob.

    A `RecurrentTower` gets a `StatefulPolicy` (`algos/stateful.py`): the same callable, carrying the
    cell's state per lane, zeroed where the engine says a lane is fresh. E1's net takes no previous
    action or reward, so those two inputs are ignored here.
    """
    if not isinstance(net, RecurrentTower):
        return qnet.greedy_policy_fn(net, device=device)
    net.eval()

    def step_fn(observations, prev_action, prev_reward, state):
        logits, state = net.step(observations, state)
        return logits.argmax(dim=1), state

    return StatefulPolicy(step_fn, net.state_width, device=device)


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
