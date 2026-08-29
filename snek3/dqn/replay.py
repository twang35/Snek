"""Prioritised experience replay: numpy ring buffers plus a numpy sum tree.

**A sum tree of our own rather than cpprb**, for one disqualifying reason: cpprb silently ignores
`seed=`, and a run that cannot be reproduced from its seed is not something to build a comparison on.
The tree is ~40 lines and samples a batch of 128 from 100,000 in well under a millisecond, so nothing
is given up.

**This buffer stores flat transitions and knows nothing about streams, forks or episodes.** That is a
deliberate change from snek2, where the buffer held a per-stream deque of the last `n+1` driver
frames and stitched items out of them. The stitching belongs next to the thing that knows what a
stream *is* — [`collect.py`](collect.py), which owns the forking — and keeping it there means the
buffer cannot fabricate a transition by interleaving two games, which is the failure that made the
per-stream window necessary in the first place: with one shared window, frame k of one game was
stored next to frame k+1 of another, silently, on every interleaved add.

So a transition is five arrays and no context:

| field | shape | meaning |
|---|---|---|
| `obs` | `(cap, obs_len)` | the state acted in |
| `action` | `(cap,)` | what was played |
| `reward` | `(cap,)` | the (possibly n-step accumulated) reward |
| `next_obs` | `(cap, obs_len)` | the state `n` steps later |
| `discount` | `(cap,)` | `gamma**n`, or **0 for a terminal transition** |

**`discount` folds the terminal flag into the number the Bellman target already needs**, so there is
no `done` field to disagree with it. A truncated episode — one that stopped for a step cap rather
than by dying — keeps a non-zero discount, which is correct and is awkward to express with a boolean.

`next_obs` is stored rather than reconstructed from the following slot. That reconstruction is the
usual memory optimisation and it is exactly what is unsafe here: with several interleaved streams the
following slot belongs to a different game. The cost is 12 MB at the default capacity.
"""

import os

import numpy as np

# Added to every priority, so a transition whose TD error came back exactly zero is still reachable.
# Without it a slot can fall out of the sampling distribution permanently.
PRIORITY_EPSILON = 1e-6

BUFFER_FILENAME = 'replay.npz'


def normalize_is_weights(weights):
    """Rescales importance-sampling weights to mean 1.0.

    **Mean, not max, and this is not cosmetic.** Dividing by the largest weight in the buffer — which
    is what the textbook formulation and cpprb both do — makes a batch's weights average about 0.09
    at beta=0.4 and 0.003 at beta=1.0. Passed to the optimiser as-is those are a blanket 11x to 370x
    cut to the learning rate, worsening as beta anneals; measured in snek2 as avg_score 34.8 against
    50.0 at step 24k versus a uniform buffer.

    Only the weights *relative* to each other carry the bias correction, so normalising to mean 1.0
    keeps the correction exactly while leaving the average gradient magnitude — and therefore the
    tuned learning rate — where it was.
    """
    mean = weights.mean()
    if mean <= 0:
        return np.ones_like(weights)
    return weights / mean


class SumTree(object):
    """Fixed-size array sum tree over `size` leaves, where `size` is a power of two.

    A power of two so **every leaf sits at the same depth**, which is what lets a whole batch descend
    the tree in lockstep: the sampling loop is `log2(size)` vectorised numpy steps rather than one
    Python descent per sample. Unwritten leaves hold 0 and are therefore unreachable, which is how
    the tree can be full-size from the start while the buffer is still filling.
    """

    def __init__(self, size):
        self.size = int(size)
        if self.size & (self.size - 1):
            raise ValueError('size must be a power of two, got {0}'.format(size))
        # Leaves live at [size, 2*size); node i has children 2i and 2i+1; index 0 is unused.
        self.nodes = np.zeros(2 * self.size, dtype=np.float64)

    @property
    def total(self):
        return float(self.nodes[1])

    def set(self, leaves, values):
        """Sets several leaf priorities and repairs the tree above them.

        Repaired level by level, so setting a batch of 128 costs `log2(size)` small numpy operations
        rather than 128 independent walks to the root.

        **Duplicate parents are left in rather than deduplicated, and that is a measured choice.** A
        batch of 128 shares ancestors near the root, so `np.unique` per level looks like the obvious
        saving — but every duplicate entry reads the same two children and therefore computes the
        same sum, so the repeated scatter writes are idempotent and uniqueness buys only a shorter
        array. It does not pay for itself: 17 `np.unique` sorts cost 0.167 ms against 0.067 ms
        without, which was 18% of a whole gradient step.

        The termination check reads `parents[0]` for the whole batch because every leaf sits at the
        same depth, so all indexes reach the root on the same iteration — the same property `find`
        relies on to descend in lockstep.
        """
        indexes = np.asarray(leaves, dtype=np.int64) + self.size
        self.nodes[indexes] = np.asarray(values, dtype=np.float64)
        if not indexes.size:
            return
        parents = indexes >> 1
        while True:
            self.nodes[parents] = self.nodes[2 * parents] + self.nodes[2 * parents + 1]
            if parents[0] == 1:
                break
            parents >>= 1

    def find(self, targets):
        """Leaf indexes whose cumulative ranges contain `targets`. Vectorised over the batch."""
        targets = np.asarray(targets, dtype=np.float64).copy()
        indexes = np.ones(targets.shape, dtype=np.int64)
        while indexes[0] < self.size:
            left = 2 * indexes
            left_sum = self.nodes[left]
            go_right = targets > left_sum
            targets = np.where(go_right, targets - left_sum, targets)
            indexes = left + go_right
        return indexes - self.size


class PrioritizedReplay(object):
    """A capacity-bounded prioritised buffer. FIFO eviction, proportional sampling.

    `alpha` is the priority exponent and `beta` the importance-sampling exponent, annealed by
    `beta_for(step)`. New transitions enter at the highest priority seen, so everything is trained on
    at least once before its priority reflects a real TD error.
    """

    def __init__(self, capacity, obs_len, alpha=0.6, initial_beta=0.4, final_beta=1.0,
                 beta_anneal_steps=300000, seed=None):
        self.capacity = int(capacity)
        self.obs_len = int(obs_len)
        self.alpha = float(alpha)
        self.initial_beta = float(initial_beta)
        self.final_beta = float(final_beta)
        self.beta_anneal_steps = int(beta_anneal_steps)
        # Its own Generator, so the buffer's sampling stream is independent of the collector's
        # exploration coins. Two consumers sharing one Generator makes either one's call count part
        # of the other's results, which is a reproducibility trap rather than a saving.
        self.rng = np.random.default_rng(seed)

        self.obs = np.zeros((self.capacity, self.obs_len), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_len), dtype=np.float32)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.discount = np.zeros(self.capacity, dtype=np.float32)

        tree_size = 1
        while tree_size < self.capacity:
            tree_size *= 2
        self.tree = SumTree(tree_size)
        self.size = 0
        self.write = 0
        self.max_priority = 1.0

    def add(self, obs, action, reward, next_obs, discount):
        """Stores one transition at the write cursor, evicting the oldest when full."""
        slot = self.write
        self.obs[slot] = obs
        self.action[slot] = action
        self.reward[slot] = reward
        self.next_obs[slot] = next_obs
        self.discount[slot] = discount
        self.tree.set([slot], [self.max_priority ** self.alpha])
        self.write = (slot + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return slot

    def beta_for(self, step):
        """The importance-sampling exponent: linear to `final_beta`, then held."""
        if step >= self.beta_anneal_steps:
            return self.final_beta
        progress = step / self.beta_anneal_steps
        return self.initial_beta + progress * (self.final_beta - self.initial_beta)

    def sample(self, batch_size, step):
        """Returns `(batch, indexes, is_weights)`, or None when there is nothing to sample yet.

        `batch` is a dict of numpy arrays with the field names above. `indexes` go straight back to
        `update_priorities`.
        """
        if self.size == 0 or self.tree.total <= 0.0:
            return None
        targets = self.rng.random(int(batch_size)) * self.tree.total
        indexes = self.tree.find(targets)
        # No unwritten-slot guard here, and that is deliberate rather than an omission. `find` cannot
        # return a leaf above `size`: a right turn needs `target > left_sum` and therefore positive
        # weight beneath it, a left turn into an empty subtree needs `target <= 0` with leaf 0
        # unwritten, and leaf 0 is written whenever the buffer holds anything at all. The padding
        # leaves are the only zero-priority ones, because every written slot carries
        # `PRIORITY_EPSILON`. `tests/test_replay.py` sweeps the property rather than trusting the
        # argument; a guard here would have been unreachable code that no fixture could exercise.
        priorities = self.tree.nodes[indexes + self.tree.size]
        probabilities = priorities / self.tree.total
        weights = (self.size * probabilities) ** (-self.beta_for(step))
        batch = {'obs': self.obs[indexes], 'action': self.action[indexes],
                 'reward': self.reward[indexes], 'next_obs': self.next_obs[indexes],
                 'discount': self.discount[indexes]}
        return batch, indexes, normalize_is_weights(weights)

    def update_priorities(self, indexes, td_errors):
        """Feeds absolute TD errors back as priorities."""
        priorities = np.abs(np.asarray(td_errors, dtype=np.float64)) + PRIORITY_EPSILON
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self.tree.set(indexes, priorities ** self.alpha)

    def save(self, directory):
        """Writes the buffer, priorities included, atomically."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, BUFFER_FILENAME)
        staging = path + '.partial.npz'
        np.savez(staging,
                 obs=self.obs[:self.size], next_obs=self.next_obs[:self.size],
                 action=self.action[:self.size], reward=self.reward[:self.size],
                 discount=self.discount[:self.size],
                 # Priorities are saved too, unlike snek2's, where cpprb's exporter dropped them and
                 # a restart re-learned them over a few thousand steps. Saving them costs one array.
                 priorities=self.tree.nodes[self.tree.size:self.tree.size + self.size],
                 size=self.size, write=self.write, max_priority=self.max_priority)
        os.replace(staging, path)
        return path

    def load(self, directory):
        """Repopulates from a previous `save()`. Returns True if anything loaded."""
        path = os.path.join(directory, BUFFER_FILENAME)
        if not os.path.exists(path):
            return False
        with np.load(path) as data:
            size = int(data['size'])
            if size > self.capacity:
                raise ValueError('saved buffer holds {0} transitions, capacity is {1}'.format(
                    size, self.capacity))
            self.obs[:size] = data['obs']
            self.next_obs[:size] = data['next_obs']
            self.action[:size] = data['action']
            self.reward[:size] = data['reward']
            self.discount[:size] = data['discount']
            self.size = size
            self.write = int(data['write']) % self.capacity
            self.max_priority = float(data['max_priority'])
            self.tree.set(np.arange(size), data['priorities'])
        return True
