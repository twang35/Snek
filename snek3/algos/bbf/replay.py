"""`SequentialReplay`: BBF's replay -- transitions kept in the order they were played, so the n-step
return is computed **when a batch is drawn**, at the n and gamma the anneal asks for then, and the K
observations after a state are there for SPR.

Why not `algos/dqn/replay.py`: the shared replay stores each transition with its return already summed
over the collector's window, so a change of n or gamma reaches new transitions only. BBF's schedule moves
both every gradient step and its code (Dopamine's `PrioritizedReplayBuffer` with the horizon set per
sample) recomputes the return from the raw rewards each time. That needs the raw rows in order, which
also gives SPR its sequences for nothing.

## Layout

One circular buffer, rows in insertion order. The collector (`algos/dqn/collect.py`, at `n_step` 1 with
the fork off) banks **one row per lane per step, lanes in order**, so the row after row `i` in the same
lane is `i + lanes`. `add` checks that: the new row's `obs` must equal the lane's previous `next_obs`
unless that row was terminal, so a collector that ever broke the ordering fails loudly here instead of
feeding SPR sequences spliced from two games.

A row is **drawable** once its `horizon` successors exist -- `horizon = max(n, K)` in-lane steps, so the
newest `horizon * lanes` rows are skipped when sampling (resampled from the tree, then uniform among the
drawable rows if the tree keeps handing back new ones). The terminal row of an episode ends every
sequence through it: the n-step sum stops at the first `done`, the bootstrap discount is 0 past it, and
SPR's mask is 0 for every step after it.

## Priorities

Prioritised, as BBF's is (exponent 0.5), on the sum tree of `algos/dqn/replay.py`. New rows enter at
the running maximum; `update_priorities` takes the per-row loss. The importance weights are Dopamine's
form, `p ** -0.5` **normalised by the batch maximum** -- not the shared replay's mean-1 normalisation --
because the paper's learning rate was tuned under that form.
"""

import os

import numpy as np

from algos.dqn.replay import SumTree, PRIORITY_EPSILON

BUFFER_FILENAME = 'replay.npz'


class SequentialReplay(object):

    def __init__(self, capacity, obs_len, lanes=1, horizon=10, alpha=0.5, seed=None):
        self.capacity = int(capacity)
        self.obs_len = int(obs_len)
        self.lanes = int(lanes)
        self.horizon = int(horizon)
        self.alpha = float(alpha)
        if self.lanes < 1 or self.horizon < 1:
            raise ValueError('lanes and horizon must be at least 1, got {0}, {1}'.format(lanes, horizon))
        if self.capacity < 2 * (self.horizon + 1) * self.lanes:
            raise ValueError('capacity {0} is too small for {1} lane(s) at horizon {2}'.format(
                self.capacity, self.lanes, self.horizon))
        self.obs = np.zeros((self.capacity, self.obs_len), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_len), dtype=np.float32)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=bool)
        self.size = 0
        self.write = 0
        self.max_priority = 1.0
        tree_size = 1
        while tree_size < self.capacity:
            tree_size *= 2
        self.tree = SumTree(tree_size)
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- writing

    def add(self, obs, action, reward, next_obs, discount, aux=0.0):
        """One transition; `discount == 0` is the terminal flag (the collector's convention). Returns the
        row's index."""
        slot = self.write
        lane_previous = slot - self.lanes
        if self.size >= self.lanes:
            previous = lane_previous % self.capacity
            if not self.done[previous] and not np.array_equal(self.next_obs[previous], np.asarray(obs, dtype=np.float32)):
                raise ValueError('row {0} does not continue lane row {1}: the collector must bank one row per '
                                 'lane per step, lanes in order (n_step 1, fork off)'.format(slot, previous))
        self.obs[slot] = obs
        self.next_obs[slot] = next_obs
        self.action[slot] = int(action)
        self.reward[slot] = float(reward)
        self.done[slot] = float(discount) == 0.0
        self.tree.set_one(slot, self.max_priority ** self.alpha)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return slot

    # ---------------------------------------------------------------- reading

    def age(self, indexes):
        """Rows written since each index, as a count of rows (0 for the newest)."""
        return (self.write - 1 - np.asarray(indexes, dtype=np.int64)) % self.capacity

    def drawable(self, indexes):
        indexes = np.asarray(indexes, dtype=np.int64)
        return (indexes < self.size) & (self.age(indexes) >= self.horizon * self.lanes)

    @property
    def n_drawable(self):
        return max(0, self.size - self.horizon * self.lanes)

    def _drawable_uniform(self, count):
        # The drawable rows are the `n_drawable` oldest; walk back from the newest.
        offsets = self.rng.integers(self.horizon * self.lanes, self.size, size=count)
        return (self.write - 1 - offsets) % self.capacity

    def successors(self, indexes, steps):
        """`(B, steps)` row indexes of the in-lane successors 1..steps of each index."""
        indexes = np.asarray(indexes, dtype=np.int64)
        offsets = (np.arange(1, steps + 1, dtype=np.int64) * self.lanes).reshape(1, -1)
        return (indexes.reshape(-1, 1) + offsets) % self.capacity

    def sample(self, batch_size, n_step, gamma, spr_steps):
        """A batch with the n-step return at `(n_step, gamma)` and `spr_steps` future observations.

        Returns `(batch, indexes, weights)` or None while nothing is drawable. `batch` holds `obs`,
        `action`, `reward` (the n-step sum, truncated at the first terminal), `discount` (`gamma ** n`, 0 if
        a terminal fell inside the window), `next_obs` (the observation the bootstrap reads, n steps on),
        `spr_next_obs` `(B, K, obs_len)`, `spr_action` `(B, K)` (the actions taken from `obs` on) and
        `spr_mask` `(B, K)` (1 while the sequence is still in the same episode).
        """
        n_step, spr_steps = int(n_step), int(spr_steps)
        if n_step < 1 or n_step > self.horizon or spr_steps < 0 or spr_steps > self.horizon:
            raise ValueError('n_step {0} and spr_steps {1} must be in [1, {2}] and [0, {2}]'.format(
                n_step, spr_steps, self.horizon))
        if self.n_drawable <= 0 or self.tree.total <= 0.0:
            return None
        batch_size = int(batch_size)
        indexes = self.tree.find(self.rng.random(batch_size) * self.tree.total)
        for _ in range(3):
            bad = ~self.drawable(indexes)
            if not bad.any():
                break
            indexes[bad] = self.tree.find(self.rng.random(int(bad.sum())) * self.tree.total)
        bad = ~self.drawable(indexes)
        if bad.any():
            indexes[bad] = self._drawable_uniform(int(bad.sum()))

        steps = max(n_step, spr_steps)
        rows = np.concatenate([indexes.reshape(-1, 1), self.successors(indexes, max(steps - 1, 0))], axis=1) \
            if steps > 1 else indexes.reshape(-1, 1)                                    # (B, steps): t .. t+steps-1
        done = self.done[rows]                                                           # (B, steps)
        # alive[:, k] is 1 while no terminal row sits at t .. t+k-1: row t+k is still this episode's.
        alive = np.concatenate([np.ones((batch_size, 1), dtype=np.float32),
                                np.cumprod(1.0 - done[:, :-1].astype(np.float32), axis=1)], axis=1)
        gammas = float(gamma) ** np.arange(n_step, dtype=np.float32)
        reward = (self.reward[rows[:, :n_step]] * alive[:, :n_step] * gammas).sum(axis=1)
        # The bootstrap: gamma^n if every one of the n rows was non-terminal.
        survived = alive[:, n_step - 1] * (1.0 - done[:, n_step - 1].astype(np.float32))
        discount = (float(gamma) ** n_step) * survived
        next_obs = self.next_obs[rows[:, n_step - 1]]

        batch = {'obs': self.obs[indexes], 'action': self.action[indexes],
                 'reward': reward.astype(np.float32), 'discount': discount.astype(np.float32),
                 'next_obs': next_obs}
        if spr_steps > 0:
            batch['spr_next_obs'] = self.next_obs[rows[:, :spr_steps]]                   # obs at t+1 .. t+K
            batch['spr_action'] = self.action[rows[:, :spr_steps]]                        # a_t .. a_{t+K-1}
            # Step k (obs at t+k) is in-episode iff no terminal among rows t .. t+k-1.
            batch['spr_mask'] = (alive[:, :spr_steps] * (1.0 - done[:, :spr_steps].astype(np.float32))).astype(np.float32)
        priorities = self.tree.nodes[indexes + self.tree.size]
        probabilities = priorities / self.tree.total
        weights = probabilities ** -0.5
        weights = (weights / weights.max()).astype(np.float32)
        return batch, indexes, weights

    def update_priorities(self, indexes, losses):
        priorities = np.abs(np.asarray(losses, dtype=np.float64)) + PRIORITY_EPSILON
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self.tree.set(indexes, priorities ** self.alpha)

    # ---------------------------------------------------------------- persistence

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, BUFFER_FILENAME)
        staging = path + '.partial.npz'
        np.savez(staging, obs=self.obs[:self.size], next_obs=self.next_obs[:self.size],
                 action=self.action[:self.size], reward=self.reward[:self.size], done=self.done[:self.size],
                 priorities=self.tree.nodes[self.tree.size:self.tree.size + self.size],
                 size=self.size, write=self.write, max_priority=self.max_priority, lanes=self.lanes)
        os.replace(staging, path)
        return path

    def load(self, directory):
        path = os.path.join(directory, BUFFER_FILENAME)
        if not os.path.exists(path):
            return False
        with np.load(path) as data:
            size = int(data['size'])
            if size > self.capacity:
                raise ValueError('saved buffer holds {0} rows, capacity is {1}'.format(size, self.capacity))
            if int(data['lanes']) != self.lanes:
                raise ValueError('saved buffer has {0} lane(s), this run {1}'.format(int(data['lanes']), self.lanes))
            self.obs[:size] = data['obs']
            self.next_obs[:size] = data['next_obs']
            self.action[:size] = data['action']
            self.reward[:size] = data['reward']
            self.done[:size] = data['done']
            self.size = size
            self.write = int(data['write']) % self.capacity
            self.max_priority = float(data['max_priority'])
            self.tree.set(np.arange(size), data['priorities'])
            # The game is not saved with the buffer: the resumed collector starts a fresh episode, so the row
            # that follows each lane's newest saved row is another game's. Ending the saved sequence there
            # costs those rows their bootstrap (one row per lane per resume) and keeps every n-step sum and
            # SPR sequence inside one game, which is the whole point of this replay.
            for lane in range(min(self.lanes, size)):
                self.done[(self.write - 1 - lane) % self.capacity] = True
        return True
