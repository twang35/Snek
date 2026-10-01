"""R2D2's replay: rows in play order, stored once, and **windows** over them -- a loss block of 40 steps
with up to 40 steps of burn-in before it and `n` steps of lookahead after -- each with the recurrent state
the collector carried into its first row, prioritised per window (`plans/algoExploration/e-memory.md` §2,
the replay row; the geometry is §0's, decided 2026-09-30).

## Rows

One ring over `(obs, action, reward, done, prev_action, prev_reward)`, written **one row per lane per step,
lanes in order**, so the row after row `i` in the same lane is `i + lanes` (the layout `algos/bbf/replay.py`
uses, for the same reason: a window is then a start row and a count, never a copy). The ring is sized so a
live window's rows are never overwritten: every row belongs to exactly one loss block and a block holds at
most `block` rows, so `windows` live windows span at most `windows x block` rows, plus the rows of the
windows still being cut (`2 x (seq_length + lookahead) x lanes`). `sample` checks the age anyway.

## Windows

Each is `(start, burn, length, ahead, state)`: the first stored row, how many burn-in rows precede the loss
block (0 to `burn_in`), how many loss rows the block holds (1 to `block`), how many lookahead rows follow
(0 to `lookahead`; fewer only when the episode ended), and the `(h, c)` the collector carried into `start`.
The collector cuts them (`collect.py`); this module stores and samples them. **A sampled batch is laid out
on a fixed grid of `burn_in + block + lookahead` slots**: the loss block always starts at slot `burn_in`, a
window with fewer burn-in rows than `burn_in` has its first `burn_in - burn` slots padded and is marked
`fresh` at its first real row (the episode's first step, whose state is the zero state), and a short block
or lookahead is padded at the end. `valid` says which slots hold rows. The fixed grid is what lets the
learner burn in every window of a batch in one vectorised unroll and take the loss at the same slots.

## Priorities

Per window: `eta * max |delta| + (1 - eta) * mean |delta|` over the valid loss steps (the paper's mix, eta
0.9), raised to `alpha` 0.9; a new window enters at the running maximum (Ape-X's actors compute an initial
TD priority on their local copy; here that is a 120-step unroll per new window on the live net and is not
done -- the departure the plan's §0 names). Importance weights `(N p)^-beta`, beta 0.6 held, **max-normalised
over the batch**, the papers' form.
"""

import os

import numpy as np

from algos.dqn.replay import SumTree, PRIORITY_EPSILON

BUFFER_FILENAME = 'replay.npz'


class SequenceReplay(object):

    def __init__(self, windows, obs_len, lanes, state_width, seq_length=80, burn_in=40, lookahead=5,
                 alpha=0.9, beta=0.6, eta=0.9, seed=None):
        self.windows_capacity = int(windows)
        self.obs_len = int(obs_len)
        self.lanes = int(lanes)
        self.state_width = int(state_width)
        self.seq_length, self.burn_in, self.lookahead = int(seq_length), int(burn_in), int(lookahead)
        self.alpha, self.beta, self.eta = float(alpha), float(beta), float(eta)
        if self.windows_capacity < 1 or self.lanes < 1:
            raise ValueError('windows and lanes must be at least 1')
        if not 0 <= self.burn_in < self.seq_length:
            raise ValueError('burn_in {0} must be in [0, seq_length {1})'.format(self.burn_in, self.seq_length))
        if self.lookahead < 1:
            raise ValueError('lookahead (the n-step) must be at least 1')
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError('eta {0} must be in [0, 1]'.format(self.eta))
        self.block = self.seq_length - self.burn_in
        self.slots = self.burn_in + self.block + self.lookahead
        self.row_capacity = self.windows_capacity * self.block + 2 * self.slots * self.lanes

        cap = self.row_capacity
        self.obs = np.zeros((cap, self.obs_len), dtype=np.float32)
        self.action = np.zeros(cap, dtype=np.int64)
        self.reward = np.zeros(cap, dtype=np.float32)
        self.done = np.zeros(cap, dtype=bool)
        self.prev_action = np.full(cap, -1, dtype=np.int64)
        self.prev_reward = np.zeros(cap, dtype=np.float32)
        self.rows = 0
        self.row_write = 0

        w = self.windows_capacity
        self.start = np.zeros(w, dtype=np.int64)
        self.burn = np.zeros(w, dtype=np.int64)
        self.length = np.zeros(w, dtype=np.int64)
        self.ahead = np.zeros(w, dtype=np.int64)
        self.state = np.zeros((w, self.state_width), dtype=np.float32)
        self.n_windows = 0
        self.win_write = 0
        tree_size = 1
        while tree_size < w:
            tree_size *= 2
        self.tree = SumTree(tree_size)
        self.max_priority = 1.0
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- writing

    def add_row(self, obs, action, reward, done, prev_action, prev_reward):
        """One row; returns its index. **Lanes in order, every lane every step** is the caller's contract."""
        slot = self.row_write
        self.obs[slot] = obs
        self.action[slot] = int(action)
        self.reward[slot] = float(reward)
        self.done[slot] = bool(done)
        self.prev_action[slot] = int(prev_action)
        self.prev_reward[slot] = float(prev_reward)
        self.row_write = (slot + 1) % self.row_capacity
        self.rows = min(self.rows + 1, self.row_capacity)
        return slot

    def add_window(self, start, burn, length, ahead, state):
        """A finished window over stored rows. Enters at the running maximum priority. Returns its slot."""
        burn, length, ahead = int(burn), int(length), int(ahead)
        if not 0 <= burn <= self.burn_in or not 1 <= length <= self.block or not 0 <= ahead <= self.lookahead:
            raise ValueError('window burn {0} length {1} ahead {2} outside [0,{3}] x [1,{4}] x [0,{5}]'.format(
                burn, length, ahead, self.burn_in, self.block, self.lookahead))
        slot = self.win_write
        self.start[slot] = int(start) % self.row_capacity
        self.burn[slot], self.length[slot], self.ahead[slot] = burn, length, ahead
        self.state[slot] = state
        self.tree.set_one(slot, self.max_priority ** self.alpha)
        self.win_write = (slot + 1) % self.windows_capacity
        self.n_windows = min(self.n_windows + 1, self.windows_capacity)
        return slot

    # ---------------------------------------------------------------- reading

    def row_age(self, indexes):
        return (self.row_write - 1 - np.asarray(indexes, dtype=np.int64)) % self.row_capacity

    def window_rows(self, slot):
        """The row indexes a window covers, oldest first: burn-in, loss block, lookahead."""
        count = int(self.burn[slot] + self.length[slot] + self.ahead[slot])
        return (self.start[slot] + np.arange(count, dtype=np.int64) * self.lanes) % self.row_capacity

    def sample(self, batch_size):
        """`(batch, slots, weights)` or None while fewer than `batch_size` windows exist.

        `batch` is a dict of `(B, slots, ...)` arrays on the fixed grid (module docstring): `obs`, `action`,
        `reward`, `done`, `prev_action` (-1 where none), `prev_reward`, `valid`, `fresh`; and per window
        `burn`, `length`, `ahead`, `state (B, state_width)`.
        """
        batch_size = int(batch_size)
        if self.n_windows < batch_size or self.tree.total <= 0.0:
            return None
        slots = self.tree.find(self.rng.random(batch_size) * self.tree.total)
        T = self.slots
        obs = np.zeros((batch_size, T, self.obs_len), dtype=np.float32)
        action = np.zeros((batch_size, T), dtype=np.int64)
        reward = np.zeros((batch_size, T), dtype=np.float32)
        done = np.zeros((batch_size, T), dtype=bool)
        prev_action = np.full((batch_size, T), -1, dtype=np.int64)
        prev_reward = np.zeros((batch_size, T), dtype=np.float32)
        valid = np.zeros((batch_size, T), dtype=bool)
        fresh = np.zeros((batch_size, T), dtype=bool)
        for b, slot in enumerate(slots):
            rows = self.window_rows(slot)
            if rows.size and self.row_age(rows[0]) >= self.rows:
                raise RuntimeError('window {0} reaches a row the ring has overwritten; the row capacity bound '
                                   'does not hold'.format(slot))
            first = self.burn_in - int(self.burn[slot])
            span = slice(first, first + rows.size)
            obs[b, span] = self.obs[rows]
            action[b, span] = self.action[rows]
            reward[b, span] = self.reward[rows]
            done[b, span] = self.done[rows]
            prev_action[b, span] = self.prev_action[rows]
            prev_reward[b, span] = self.prev_reward[rows]
            valid[b, span] = True
            # A window with fewer burn-in rows than the grid begins at its episode's first step.
            fresh[b, first] = self.burn[slot] < self.burn_in
        priorities = self.tree.nodes[slots + self.tree.size]
        probabilities = priorities / self.tree.total
        weights = (self.n_windows * probabilities) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)
        batch = {'obs': obs, 'action': action, 'reward': reward, 'done': done, 'prev_action': prev_action,
                 'prev_reward': prev_reward, 'valid': valid, 'fresh': fresh,
                 'burn': self.burn[slots].copy(), 'length': self.length[slots].copy(),
                 'ahead': self.ahead[slots].copy(), 'state': self.state[slots].copy()}
        return batch, slots, weights

    def update_priorities(self, slots, priorities):
        priorities = np.abs(np.asarray(priorities, dtype=np.float64)) + PRIORITY_EPSILON
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self.tree.set(slots, priorities ** self.alpha)

    # ---------------------------------------------------------------- persistence

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, BUFFER_FILENAME)
        staging = path + '.partial.npz'
        rows, w = self.rows, self.n_windows
        np.savez(staging, obs=self.obs[:rows], action=self.action[:rows], reward=self.reward[:rows],
                 done=self.done[:rows], prev_action=self.prev_action[:rows], prev_reward=self.prev_reward[:rows],
                 rows=rows, row_write=self.row_write,
                 start=self.start[:w], burn=self.burn[:w], length=self.length[:w], ahead=self.ahead[:w],
                 state=self.state[:w], n_windows=w, win_write=self.win_write,
                 priorities=self.tree.nodes[self.tree.size:self.tree.size + w], max_priority=self.max_priority,
                 lanes=self.lanes, row_capacity=self.row_capacity)
        os.replace(staging, path)
        return path

    def load(self, directory):
        """Repopulates from `save()`. The game is not saved with the buffer, so the collector starts fresh
        episodes; the windows still being cut at the save were never stored and are simply lost."""
        path = os.path.join(directory, BUFFER_FILENAME)
        if not os.path.exists(path):
            return False
        with np.load(path) as data:
            if int(data['lanes']) != self.lanes or int(data['row_capacity']) != self.row_capacity:
                raise ValueError('saved buffer has {0} lane(s) and {1} rows of capacity; this run {2} and {3}'.format(
                    int(data['lanes']), int(data['row_capacity']), self.lanes, self.row_capacity))
            rows, w = int(data['rows']), int(data['n_windows'])
            if w > self.windows_capacity:
                raise ValueError('saved buffer holds {0} windows, capacity is {1}'.format(w, self.windows_capacity))
            self.obs[:rows] = data['obs']
            self.action[:rows] = data['action']
            self.reward[:rows] = data['reward']
            self.done[:rows] = data['done']
            self.prev_action[:rows] = data['prev_action']
            self.prev_reward[:rows] = data['prev_reward']
            self.rows, self.row_write = rows, int(data['row_write']) % self.row_capacity
            self.start[:w], self.burn[:w], self.length[:w], self.ahead[:w] = (
                data['start'], data['burn'], data['length'], data['ahead'])
            self.state[:w] = data['state']
            self.n_windows, self.win_write = w, int(data['win_write']) % self.windows_capacity
            self.max_priority = float(data['max_priority'])
            self.tree.set(np.arange(w), data['priorities'])
        return True
