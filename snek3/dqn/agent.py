"""Double DQN: the network, its target copy, the optimiser, and the exploration shield.

The seam every algorithm in this project sits on, so `train.py` needs to know nothing about DQN:

    agent.act(obs, epsilon, guided)  -> actions        # (n, 30) -> (n,)
    agent.update(batch, indexes, weights) -> metrics   # one optimisation step
    agent.policy_fn                  -> callable       # greedy, for engine.measure
    agent.state_dict() / load_state_dict()

**Double DQN, not DQN**: the *online* network chooses the next action and the *target* network values
it, which removes the max-operator bias that plain DQN's `max_a Q_target(s', a)` carries. The reward
and the terminal flag arrive already folded into the batch's `discount` field — see
[`replay.py`](replay.py) — so the target here is literally `reward + discount * Q_target(s', a*)`
with no terminal branch to get wrong.

## The exploration shield

**An epsilon-greedy action drawn at random is sampled from the moves that do not kill the snake this
step. The network's own argmax is never overridden.** That asymmetry is the entire design:

| branch | shielded? | why |
|---|---|---|
| the epsilon coin picked a random move | **yes** | a blunder the agent did not choose teaches it nothing |
| the network's argmax is fatal | **no** | it must eat the death reward and learn, exactly as before |

Overriding a fatal *greedy* action would leave `Q(s, a_fatal)` never updated toward the death reward
for the states where the network is wrong, so those values would drift on generalisation alone — and
evals run unshielded, so the arm would walk into walls it was never allowed to learn about.

The shield exists because a mastery-gated epsilon schedule deadlocks without it: with epsilon pinned
at 0.05, 3.3% of collected actions are random, a random move with a long snake is usually fatal, so
the buffer fills with trajectories that die before the endgame, the greedy policy never learns to
finish, the perfect rate stays 0, and the schedule never descends. Four snek2 arms sat there for up
to 942k steps.

**One step deep, deliberately.** Snake's hard problem is sealing itself into a region it cannot
escape, and that is untouched — an arm still has to learn it. All this removes is "the coin flipped
and the snake drove into its own body".
"""

import numpy as np
import torch
import torch.nn.functional as F

from dqn import net as network
from env import constants

# Where "is this move survivable" lives in the observation, read from the layout table rather than
# written as 6:9. The block is the only place in the vector that states legality, and a future block
# inserted before it would silently move it — a hardcoded slice would then mask on food distances.
SAFETY_RANGE = constants.block_ranges()['body_and_wall']
_SAFETY_START, _SAFETY_STOP = SAFETY_RANGE

# An observation value above this counts as "safe". The block is 0/1, so the midpoint is arbitrary
# and only has to be strictly between them.
SAFE_THRESHOLD = 0.5


def safe_actions(observations):
    """The `(n, num_actions)` boolean mask of moves that do not kill the snake this step.

    Read from the observation rather than recomputed. `env.observations.body_and_wall_collisions`
    already does this exactly, including the one case a naive check gets wrong: the cell the tail is
    vacating this step *is* safe to move into.
    """
    return np.asarray(observations)[:, _SAFETY_START:_SAFETY_STOP] > SAFE_THRESHOLD


def shielded_choice(observations, guided, rng, num_actions):
    """One uniformly random action per row, restricted to safe moves where the shield applies.

    Everything is allowed in two cases, and both matter:

    - **`guided` is False for that row.** This is what makes a guided fraction of 0 reproduce
      unshielded behaviour exactly, rather than approximately.
    - **No move is safe.** A boxed-in snake has nothing to be steered to, so it takes a move and
      dies. Without this the row would have no allowed action at all and the draw would be
      undefined.
    """
    allowed = safe_actions(observations)
    if allowed.shape[1] != num_actions:
        raise ValueError('the safety block is {0} wide but there are {1} actions'.format(
            allowed.shape[1], num_actions))
    guided = np.asarray(guided, dtype=bool).reshape(-1, 1)
    allowed = allowed | ~allowed.any(axis=1, keepdims=True) | ~guided

    # Uniform over the allowed columns of each row: one random key per cell, masked cells pushed
    # below every allowed one, then the argmax. Vectorised, and it needs no per-row Python loop.
    keys = rng.random(allowed.shape)
    keys = np.where(allowed, keys, -1.0)
    return np.argmax(keys, axis=1).astype(np.int64)


class DdqnAgent(object):
    """Double DQN with a hard-copied target network.

    `discount` is not held here: it reaches the loss through the batch's `discount` field, which the
    collector computes once as `gamma**n` or 0 for a terminal transition. One place rather than two
    that can disagree about whether a transition ended an episode.
    """

    def __init__(self, arch, learning_rate=1e-5, adam_epsilon=1e-7, target_update_period=8,
                 target_update_tau=1.0, gradient_clipping=0.0, use_is_weights=True,
                 seed=None, device='cpu'):
        self.arch = arch
        self.device = device
        self.num_actions = int(arch['num_actions'])
        # The seed reaches the *initialisation*, not only the exploration coins. Without that two
        # arms launched with the same `SNEK_SEED` start from different weights.
        self.net = network.build(arch, device, seed=seed)
        # The target starts as an exact copy, so the first updates are ordinary DQN updates rather
        # than updates against a randomly initialised critic.
        self.target = network.build(arch, device)
        self.target.load_state_dict(self.net.state_dict())
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(learning_rate),
                                          eps=float(adam_epsilon))
        self.target_update_period = int(target_update_period)
        self.target_update_tau = float(target_update_tau)
        self.gradient_clipping = float(gradient_clipping)
        self.use_is_weights = bool(use_is_weights)
        # Its own Generator, independent of the replay buffer's — see the note in replay.py on why
        # two consumers must not share one.
        self.rng = np.random.default_rng(seed)
        self.train_step = 0

    # ---------------------------------------------------------------- acting

    @property
    def policy_fn(self):
        """The greedy policy, in the shape `vectorized/engine.py` expects."""
        return network.greedy_policy_fn(self.net, self.device)

    def greedy_actions(self, observations):
        self.net.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=self.device)
            return self.net(tensor).argmax(dim=1).cpu().numpy().astype(np.int64)

    def act(self, observations, epsilon, guided=False):
        """Epsilon-greedy actions, with the random branch shielded where `guided` says so.

        **The exploration draw is uniform over all allowed actions and may re-pick the greedy one**,
        which is the standard epsilon-greedy semantics: at epsilon e with three actions the effective
        non-greedy rate is `e * 2/3`. Changing that silently rescales every epsilon in the schedule.
        """
        observations = np.asarray(observations, dtype=np.float32)
        actions = self.greedy_actions(observations)
        explore = self.rng.random(observations.shape[0]) < float(epsilon)
        # One fast path, not two. An `epsilon <= 0` early-out sat here as well and was redundant —
        # `random() < 0.0` is already all-False — but it read like a semantic guard, and a mutation
        # removing it survived the suite, which is what redundant guards do.
        if not explore.any():
            return actions
        drawn = shielded_choice(observations, guided, self.rng, self.num_actions)
        return np.where(explore, drawn, actions).astype(np.int64)

    # ---------------------------------------------------------------- learning

    def update(self, batch, weights=None):
        """One optimisation step. Returns `(td_errors, metrics)`.

        `td_errors` go straight back to `PrioritizedReplay.update_priorities`.
        """
        self.net.train()
        obs = torch.as_tensor(batch['obs'], device=self.device)
        next_obs = torch.as_tensor(batch['next_obs'], device=self.device)
        action = torch.as_tensor(batch['action'], device=self.device).long()
        reward = torch.as_tensor(batch['reward'], device=self.device).float()
        discount = torch.as_tensor(batch['discount'], device=self.device).float()

        chosen = self.net(obs).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # The double-Q split: argmax from the online net, value from the target net.
            best = self.net(next_obs).argmax(dim=1, keepdim=True)
            bootstrap = self.target(next_obs).gather(1, best).squeeze(1)
            target = reward + discount * bootstrap

        td_error = target - chosen
        # Huber, element-wise, then a weighted mean — matching what snek2's agent did. Huber rather
        # than squared error because a perfect game's reward is 100 against a typical step's ~0.001,
        # so one terminal transition in a batch of 128 would otherwise dominate the gradient.
        losses = F.huber_loss(chosen, target, reduction='none', delta=1.0)
        if weights is not None and self.use_is_weights:
            losses = losses * torch.as_tensor(np.asarray(weights, dtype=np.float32),
                                              device=self.device)
        loss = losses.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = None
        if self.gradient_clipping > 0.0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                                             self.gradient_clipping))
        self.optimizer.step()

        self.train_step += 1
        self.maybe_update_target()
        metrics = {'loss': float(loss.detach()), 'train_step': self.train_step,
                   'mean_abs_td': float(td_error.detach().abs().mean())}
        if grad_norm is not None:
            metrics['grad_norm'] = grad_norm
        return td_error.detach().cpu().numpy(), metrics

    def maybe_update_target(self):
        """Copies the online weights into the target every `target_update_period` updates.

        `tau` of 1.0 is a hard copy, which is what snek2 ran. A smaller tau makes it a Polyak
        average, and combining a short period with a small tau is how a target network stops being a
        target — so both are exposed rather than one being hidden inside the other.
        """
        if self.target_update_period <= 0 or self.train_step % self.target_update_period:
            return False
        if self.target_update_tau >= 1.0:
            self.target.load_state_dict(self.net.state_dict())
            return True
        with torch.no_grad():
            for online, lagged in zip(self.net.parameters(), self.target.parameters()):
                lagged.mul_(1.0 - self.target_update_tau).add_(online, alpha=self.target_update_tau)
        return True

    # ---------------------------------------------------------------- persistence

    def state_dict(self):
        """Everything a resume needs beyond the policy weights themselves."""
        return {'model': self.net.state_dict(), 'target': self.target.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'train_step': self.train_step,
                'rng': self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.net.load_state_dict(state['model'])
        self.target.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.train_step = int(state.get('train_step', 0))
        if state.get('rng') is not None:
            self.rng.bit_generator.state = state['rng']
