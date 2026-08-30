"""The on-policy rollout buffer and GAE. Pure numpy — no torch, so this is testable on its own.

`(T, N)` preallocated arrays, filled once, read `epochs` times, then overwritten. That is the whole
difference from `dqn/replay.py`'s 271 lines of sum tree: there is no sampling, no priority, no
eviction and no importance weight, because every sample in a PPO update was produced by the policy
being updated.

## GAE, and the one line that matters

    delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

**`done_t` means "the action at t ended the episode", and it gates both terms.** `VecSnake` auto-
resets inside `step()`, so `obs[t+1]` on a lane that just died is a *fresh episode's* first state and
`V(s_{t+1})` is that new episode's value. Multiplying it out is the same trick `dqn/collect.py` uses
when it stores `discount=0` on a terminal transition, and the same bug snek2 shipped in its n-step
window when it had no episode check at all. Advantage must not flow across an episode boundary in
either term: not through the bootstrap, and not through the recursion.

**Truncation is not a termination.** A rollout ending mid-episode is bootstrapped off `last_values`,
which is `V` of the state the next rollout will start from — correct, and the reason `done` here only
ever means a real death, starve or win. There is no time limit in this env whose expiry would need
the other treatment: the starve budget's expiry *is* a terminal state of the MDP, and the observation
carries the budget, so bootstrapping 0 there is right.

## Horizon, stated because it is the design's main risk

`1 / (1 - gamma * lambda)` is the number of steps an advantage actually sees, and at this project's
gamma it is short:

| gamma | lambda | horizon |
|---|---|---:|
| 0.9975 | 0.95 | **19** |
| 0.9975 | 0.98 | **44.5** |
| 0.9975 | 1.00 | 400 — the discount's own horizon, and pure Monte-Carlo variance |
| 0.99 | 0.98 | 33.6 |

**A perfect game is ~950 moves from the opening, so the +100 is not visible to GAE at any usable
lambda.** It reaches the policy through the critic, exactly as it reaches DQN's policy through a
1-step bootstrap — DQN got to 98.7% that way, so this is a risk to watch and not a reason it cannot
work. It is why lambda's default is 0.98 rather than the conventional 0.95, why the shaping terms
matter more here than for DQN, and why `explained_variance` is in every eval row.
"""

import numpy as np


def horizon(discount, gae_lambda):
    """`1 / (1 - gamma * lambda)`, the steps an advantage sees. For the report, and for arguing."""
    decay = float(discount) * float(gae_lambda)
    if decay >= 1.0:
        return float('inf')
    return 1.0 / (1.0 - decay)


class Rollout(object):
    """`steps` x `lanes` of on-policy experience, plus the GAE pass over it.

    Allocated once for the life of the arm and overwritten in place, so a 12M-transition run does no
    per-rollout allocation. `add` is called with `t` explicitly rather than tracking a cursor: the
    collector already has the loop index, and a cursor is one more piece of state that can disagree
    with the caller about where it is.
    """

    def __init__(self, steps, lanes, obs_len):
        self.steps = int(steps)
        self.lanes = int(lanes)
        self.obs_len = int(obs_len)
        if self.steps < 1 or self.lanes < 1:
            raise ValueError('a rollout needs at least 1 step and 1 lane, got {0}x{1}'.format(
                self.steps, self.lanes))
        shape = (self.steps, self.lanes)
        self.obs = np.zeros(shape + (self.obs_len,), dtype=np.float32)
        self.actions = np.zeros(shape, dtype=np.int64)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros(shape, dtype=bool)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self._finished = False

    @property
    def size(self):
        """Transitions in a full rollout. Also the arm's step increment — see `ppo/algo.py`."""
        return self.steps * self.lanes

    def add(self, t, obs, actions, log_probs, values, rewards, dones):
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.dones[t] = dones
        self._finished = False

    def finish(self, last_values, discount, gae_lambda):
        """The backward GAE pass. Fills `advantages` and `returns`; returns the advantages.

        `last_values` is `V` of the state each lane is *now* in — after the last stored step — which
        is what a truncated rollout bootstraps off. It is ignored on a lane whose last stored step was
        terminal, by the same `(1 - done)` that gates every other bootstrap.

        `returns = advantages + values` rather than a separately accumulated discounted sum. The two
        are the same quantity for the value target, and computing it twice is how they come to differ.
        """
        last_values = np.asarray(last_values, dtype=np.float32).reshape(self.lanes)
        discount = float(discount)
        gae_lambda = float(gae_lambda)
        advantage = np.zeros(self.lanes, dtype=np.float32)
        for t in range(self.steps - 1, -1, -1):
            # 1.0 where the episode continued past t, 0.0 where the action at t ended it.
            alive = (~self.dones[t]).astype(np.float32)
            next_value = last_values if t == self.steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + discount * alive * next_value - self.values[t]
            advantage = delta + discount * gae_lambda * alive * advantage
            self.advantages[t] = advantage
        self.returns[:] = self.advantages + self.values
        self._finished = True
        return self.advantages

    # ---------------------------------------------------------------- reading it back

    def flat(self):
        """The rollout as flat `(T*N, ...)` arrays. One dict, so a caller cannot mismatch them."""
        if not self._finished:
            raise ValueError('finish() has not run: advantages and returns are stale')
        total = self.size
        return {'obs': self.obs.reshape(total, self.obs_len),
                'actions': self.actions.reshape(total),
                'log_probs': self.log_probs.reshape(total),
                'values': self.values.reshape(total),
                'advantages': self.advantages.reshape(total),
                'returns': self.returns.reshape(total)}

    def minibatches(self, minibatch, rng):
        """Yields shuffled minibatches covering the rollout exactly once.

        **Exactly once, with no sample dropped and none repeated**, which is what makes "4 epochs"
        mean four passes rather than four samples-with-replacement of unknown coverage. A trailing
        partial batch is yielded rather than discarded: dropping it would silently shorten every epoch
        by up to `minibatch - 1` samples, and at the planned 16,384 / 256 it would be invisible
        because the sizes happen to divide.

        `rng` is the shuffle's own `numpy.Generator` — not the policy's and not the env's.
        """
        flat = self.flat()
        order = rng.permutation(self.size)
        step = max(1, int(minibatch))
        for start in range(0, self.size, step):
            index = order[start:start + step]
            yield {key: value[index] for key, value in flat.items()}


def normalise(advantages):
    """Zero mean, unit standard deviation. The standard PPO advantage treatment.

    **The guard is on the standard deviation, not on the length.** A minibatch of one has sd 0, and so
    does a minibatch in which every advantage happens to be equal — which is not exotic early in a run
    where nothing has happened yet. Dividing by it gives NaN and poisons every parameter; falling back
    to the un-scaled centred advantage is the behaviour that degrades rather than explodes.
    """
    advantages = np.asarray(advantages, dtype=np.float32)
    centred = advantages - advantages.mean()
    spread = float(centred.std())
    if spread < 1e-8:
        return centred
    return centred / spread
