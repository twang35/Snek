"""Double DQN: the network, its target copy, the optimiser, and the exploration shield.

The seam every algorithm in this project sits on, so `train.py` needs to know nothing about DQN:

    agent.act(obs, epsilon, guided)  -> actions        # (n, OBS_LEN) -> (n,)
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

from algos.dqn import net as network
from algos.dqn import resets
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


def build_adam(parameters, learning_rate, epsilon):
    """Adam, fused where the device supports it. Same algorithm, one kernel instead of dozens.

    **Public, and `algos/ppo/agent.py` imports it rather than copying it.** Both halves of this docstring
    are measured facts about *this* net shape, and the materialisation below is a trap that a second
    copy would eventually walk into again — see the paragraph on the exhausted generator.

    **Bit-identical, and that is the reason it can be turned on without re-running anything.** The
    fused path is the same update arithmetic in a single kernel rather than ~40 ops over four tiny
    tensors; on this net (26 -> 320 -> 3, 9,283 parameters; 30 -> 320 -> 3 was 10,560) the per-op dispatch *is* the cost, so a
    whole `forward + backward + step` fell from **240.7 us to 162.1 us** — 19% off the learn step,
    which is 12M gradient steps of an arm. Verified over 2,000 steps from a fixed seed: **max
    absolute parameter difference 0.0**, so a seeded arm reproduces exactly as before.

    Falls back rather than raising, because `fused` is device-gated: it is supported on CPU and CUDA
    but not on every backend, and losing a training run to an optimiser flag would be a poor trade
    for 19%. A backend that refuses it simply runs the unfused path at the old speed.
    """
    # Materialised before the attempt, and this is the whole reason the helper exists rather than a
    # bare try/except at the call site. `net.parameters()` is a **generator**, and
    # `Optimizer.__init__` consumes it before torch validates `fused` — so a backend that rejects
    # fused would leave the fallback iterating an exhausted generator and building an optimiser over
    # **no parameters at all**. That does not raise: it trains forever with every weight frozen at
    # its initialisation, which reads as an arm that simply never learns.
    parameters = list(parameters)
    try:
        return torch.optim.Adam(parameters, lr=learning_rate, eps=epsilon, fused=True)
    except (RuntimeError, ValueError):
        return torch.optim.Adam(parameters, lr=learning_rate, eps=epsilon)


class DdqnAgent(object):
    """Double DQN with a hard-copied target network.

    `discount` is not held here: it reaches the loss through the batch's `discount` field, which the
    collector computes once as `gamma**n` or 0 for a terminal transition. One place rather than two
    that can disagree about whether a transition ended an episode.
    """

    def __init__(self, arch, learning_rate=1e-5, adam_epsilon=1e-7, target_update_period=8,
                 target_update_tau=1.0, gradient_clipping=0.0, use_is_weights=True,
                 seed=None, device='cpu', munchausen_alpha=0.0, munchausen_tau=0.03,
                 munchausen_l0=-1.0, reset_interval=0, reset_alpha=0.5, reset_stop_after=0,
                 cycle=None, on_cycle=None):
        self.arch = arch
        self.device = device
        self.seed = seed
        # Shrink-and-perturb resets on a gradient-step schedule, off at interval 0 (Group D, row D1).
        # See `algos/dqn/resets.py`; `maybe_reset` runs after every update, like the target copy.
        self.reset_schedule = resets.ResetSchedule(reset_interval, reset_alpha, reset_stop_after)
        self.resets = 0
        # BBF's within-cycle anneal of n-step and gamma (`resets.CycleSchedule`), read after every update
        # from the gradient steps since the last reset; `on_cycle(n_step, gamma)` is the algorithm's hook
        # that moves the collector and the env. Off (`cycle` None or disabled) it is never called.
        self.cycle = cycle if cycle is not None else resets.CycleSchedule()
        self.on_cycle = on_cycle
        self.last_reset_step = 0
        self.num_actions = int(arch['num_actions'])
        # Munchausen (Vieillard, Pietquin & Geist 2020): `alpha > 0` adds the clipped, scaled
        # log-policy of the taken action to the reward and replaces the double-Q bootstrap with the
        # soft (entropy-regularised) value of the next state, both computed from the **target** net's
        # Q-values at temperature `tau`, as the paper does. **`alpha == 0` is plain double DQN, to the
        # bit** -- the soft target is not used at all then, not used with a zero weight -- and a
        # fixture pins that, because A1's control ran before this existed. Group A, row A6.
        self.munchausen_alpha = float(munchausen_alpha)
        self.munchausen_tau = float(munchausen_tau)
        self.munchausen_l0 = float(munchausen_l0)
        if self.munchausen_alpha < 0.0 or self.munchausen_alpha > 1.0:
            raise ValueError('munchausen_alpha must be in [0, 1], got {0}'.format(munchausen_alpha))
        if self.munchausen_alpha > 0.0 and self.munchausen_tau <= 0.0:
            raise ValueError('munchausen_tau must be positive when alpha is on, got {0}'.format(
                munchausen_tau))
        # The seed reaches the *initialisation*, not only the exploration coins. Without that two
        # arms launched with the same `SNEK_SEED` start from different weights.
        self.net = network.build(arch, device, seed=seed)
        # The target starts as an exact copy, so the first updates are ordinary DQN updates rather
        # than updates against a randomly initialised critic.
        self.target = network.build(arch, device)
        self.target.load_state_dict(self.net.state_dict())
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

        self.optimizer = build_adam(self.net.parameters(), float(learning_rate),
                                    float(adam_epsilon))
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
            target = self.scalar_target(obs, action, reward, discount, next_obs)

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
        self.maybe_reset()
        self.maybe_anneal()
        metrics = {'loss': float(loss.detach()), 'train_step': self.train_step,
                   'mean_abs_td': float(td_error.detach().abs().mean()), 'resets': self.resets}
        if grad_norm is not None:
            metrics['grad_norm'] = grad_norm
        return td_error.detach().cpu().numpy(), metrics

    def scalar_target(self, obs, action, reward, discount, next_obs):
        """The regression target for the chosen action, under `torch.no_grad()` by the caller.

        Double DQN by default: argmax from the online net, value from the target net. With Munchausen
        on, `r + alpha * clip(tau * log pi(a|s), l0, 0) + discount * sum_a' pi(a'|s') (q'(s',a') - tau
        * log pi(a'|s'))`, every pi from the target net's Q at temperature tau (the paper's Eq. 7-8);
        `discount` is already `gamma**n` or 0 at a terminal, so the soft term vanishes there as the
        bootstrap does.
        """
        if self.munchausen_alpha <= 0.0:
            best = self.net(next_obs).argmax(dim=1, keepdim=True)
            bootstrap = self.target(next_obs).gather(1, best).squeeze(1)
            return reward + discount * bootstrap
        tau = self.munchausen_tau
        # tau * log_softmax(q / tau) == q - tau * logsumexp(q / tau): the paper's scaled log-policy,
        # computed in the stable form.
        log_pi_s = tau * F.log_softmax(self.target(obs) / tau, dim=1)
        munchausen = torch.clamp(log_pi_s.gather(1, action.unsqueeze(1)).squeeze(1),
                                 min=self.munchausen_l0, max=0.0)
        q_next = self.target(next_obs)
        log_pi_next = tau * F.log_softmax(q_next / tau, dim=1)
        soft_value = (torch.softmax(q_next / tau, dim=1) * (q_next - log_pi_next)).sum(dim=1)
        return reward + self.munchausen_alpha * munchausen + discount * soft_value

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

    # ---------------------------------------------------------------- resets

    def fresh_net(self, seed):
        """A network of this agent's shape at a fresh initialisation. The dist agent overrides the builder."""
        return network.build(self.arch, self.device, seed=seed)

    def optimizers(self):
        """Every optimiser whose state a reset clears."""
        return [self.optimizer]

    def maybe_reset(self):
        """Shrink-and-perturb the online net when the schedule says so; the target becomes its copy.

        Runs on the same `train_step` the target copy gates on, after it, so a reset step's target is
        the reset net rather than the pre-reset one. Returns whether a reset happened.
        """
        if not self.reset_schedule.due(self.train_step):
            return False
        fresh = self.fresh_net(resets.reset_seed(self.seed, self.resets + 1))
        resets.shrink_and_perturb(self.net, fresh, self.reset_schedule.alpha)
        self.target.load_state_dict(self.net.state_dict())
        for optimizer in self.optimizers():
            resets.clear_optimizer(optimizer)
        self.resets += 1
        self.last_reset_step = self.train_step
        return True

    @property
    def steps_since_reset(self):
        return self.train_step - self.last_reset_step

    def cycle_values(self):
        """`(n_step, gamma)` the anneal asks for now."""
        since = self.steps_since_reset
        return self.cycle.n_step_at(since), self.cycle.gamma_at(since)

    def maybe_anneal(self):
        """Hands the cycle's current `(n_step, gamma)` to `on_cycle` after every update while the anneal is
        on. The hook is cheap when nothing changed (the collector compares), so it is called every step
        rather than on a change test here, which keeps the reset step itself -- where both values jump
        back -- on the same path as every other step."""
        if not self.cycle.enabled or self.on_cycle is None:
            return False
        n_step, gamma = self.cycle_values()
        self.on_cycle(n_step, gamma)
        return True

    # ---------------------------------------------------------------- persistence

    def state_dict(self):
        """Everything a resume needs beyond the policy weights themselves."""
        return {'model': self.net.state_dict(), 'target': self.target.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'train_step': self.train_step,
                'rng': self.rng.bit_generator.state, 'resets': self.resets,
                'last_reset_step': self.last_reset_step}

    def load_state_dict(self, state):
        self.net.load_state_dict(state['model'])
        self.target.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.train_step = int(state.get('train_step', 0))
        self.resets = int(state.get('resets', 0))
        self.last_reset_step = int(state.get('last_reset_step', 0))
        if state.get('rng') is not None:
            self.rng.bit_generator.state = state['rng']
