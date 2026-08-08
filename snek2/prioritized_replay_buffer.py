"""Prioritized experience replay backed by cpprb's C++ sum tree.

This replaces tf_agents' PyUniformReplayBuffer, whose pure-Python sampling was
the single largest cost in the training loop: 6.0 ms of an 11.5 ms step, because
each batch of 128 did 128 separate get_single() calls, every one of them taking a
lock and nest-mapping over 6 numpy buffers. cpprb samples in 0.062 ms while also
doing the priority lookup and returning importance-sampling weights.

Items are stored as consecutive *pairs* of driver trajectories — consecutive
*within one stream*, see add() — which is exactly what as_dataset(num_steps=2)
used to hand the agent, so the batches the agent trains on are unchanged apart
from the sampling distribution.

Priorities are not persisted: cpprb's save_transitions() keeps the transitions
but resets priorities to the max. They re-learn within a few thousand steps as
TD errors come back, so a restart costs a little sample efficiency, not
correctness.
"""
import collections
import os

import numpy as np
import tensorflow as tf
from cpprb import PrioritizedReplayBuffer

BUFFER_FILENAME = 'buffer.npz'

# The stream every caller gets unless it asks for another one: the main collect line. Forked
# branches take ids of their own — see add().
DEFAULT_STREAM = 0


def normalize_is_weights(weights):
    """Rescales importance-sampling weights to average 1.0.

    cpprb divides by the largest weight in the whole buffer, not the batch, so a
    batch's weights average about 0.09 at beta=0.4 and 0.003 at beta=1.0. Passed
    to agent.train() as-is those act as a blanket ~11x to ~370x cut to the
    learning rate, which slowed learning badly and got worse as beta annealed
    (measured: avg_score 34.8 vs 50.0 at step 24k against a uniform buffer).

    Only the weights *relative* to each other carry the bias correction, so
    rescaling to mean 1.0 keeps the correction while leaving the average gradient
    magnitude -- and therefore the tuned learning_rate -- unchanged.
    """
    mean = weights.mean()
    if mean <= 0:
        return np.ones_like(weights)
    return weights / mean


class TrajectoryPrioritizedReplayBuffer:

    def __init__(self, data_spec, capacity, alpha=0.6, initial_beta=0.4, final_beta=1.0,
                 beta_anneal_steps=1000000, sequence_length=2):
        self._data_spec = data_spec
        self._capacity = capacity
        self._initial_beta = initial_beta
        self._final_beta = final_beta
        self._beta_anneal_steps = beta_anneal_steps
        self._sequence_length = sequence_length

        # cpprb is keyed by flat field name, so mirror tf.nest's flatten order and
        # rebuild the Trajectory with pack_sequence_as on the way out. Each field
        # carries a leading time dimension holding the trajectory window: 2 for
        # single-step DQN, n+1 when the agent uses n_step_update=n.
        flat_specs = tf.nest.flatten(data_spec)
        self._field_names = ['field{0}'.format(i) for i in range(len(flat_specs))]
        env_dict = {}
        for name, spec in zip(self._field_names, flat_specs):
            env_dict[name] = {'shape': (sequence_length,) + tuple(spec.shape), 'dtype': spec.dtype}

        self._buffer = PrioritizedReplayBuffer(capacity, env_dict, alpha=alpha)
        # Trailing window of trajectories per stream, held back so each stored item is a run
        # of consecutive steps. maxlen makes a window slide one step at a time.
        self._windows = {DEFAULT_STREAM: collections.deque(maxlen=sequence_length)}

    def add(self, traj, stream=DEFAULT_STREAM):
        """Observer for PyDriver. Stores one window per step once that stream's window is full.

        **The window is per stream, and that is not an optimization.** A stored item is literally
        "the last `sequence_length` calls for this stream, in call order", with no episode id and
        no discontinuity check. One shared window was fine while a single environment was the only
        thing adding, but the forking collector round-robins several independent games through
        here — and with one window, frame k of one game would be stored next to frame k+1 of
        another, fabricating a transition that never happened, silently, on every interleaved add.

        One code path rather than a fast path for stream 0 plus a slow path for the rest. Two
        paths is precisely how the two copies of _rebuild_grid drifted apart in Snake.py.
        """
        window = self._windows.get(stream)
        if window is None:
            window = self._windows[stream] = collections.deque(maxlen=self._sequence_length)
        window.append(tf.nest.flatten(traj))
        if len(window) < self._sequence_length:
            return
        item = {}
        for index, name in enumerate(self._field_names):
            item[name] = np.stack([frame[index] for frame in window])
        self._buffer.add(**item)

    def fork_stream(self, parent, child):
        """Starts `child` holding a copy of `parent`'s window.

        Not a fabrication: a fork's history up to the fork state *is* its parent's history, frame
        for frame, so those frames legitimately belong to both streams. What differs from the
        parent's item is the action taken at the fork and everything after it.

        This is what makes a short branch representable at `n_step_update > 1`. With an empty
        window a child needs `n + 1` adds before it stores anything, so at n=3 a branch that forks
        and dies two steps later would contribute **nothing** — and short fatal branches are
        exactly what the feature exists to collect.

        The cost, at n=1: the child's first item repeats a transition the parent also stores, since
        both pair the same frame 0 with a next-observation from the same state. That is ~1 duplicate
        per fork. Accepted rather than special-cased on `sequence_length`, because a rule that
        changes shape with n is how this quietly breaks when someone tries n=3.
        """
        self._windows[child] = collections.deque(self._windows.get(parent, ()),
                                                 maxlen=self._sequence_length)

    def close_stream(self, stream):
        """Drops a retired branch's window, so the dict does not grow for the whole run.

        Only the *partial* window goes — at most `sequence_length - 1` frames that never completed
        an item. Everything the stream already stored stays in the buffer and keeps being sampled
        until FIFO eviction, exactly like a main-line transition.

        Correctness does not depend on this being called. The collector allocates stream ids from
        a monotonic counter and never reuses one, so a missed close leaks a few floats rather than
        splicing a new branch onto a dead one's tail. Do not "simplify" the id allocation to
        anything that can repeat, or that stops being true.
        """
        if stream == DEFAULT_STREAM:
            raise ValueError('the default stream is the main collect line and never closes')
        self._windows.pop(stream, None)

    def sample(self, batch_size, train_step):
        """Returns (experience, indexes, is_weights) for agent.train()."""
        batch = self._buffer.sample(batch_size, beta=self.beta(train_step))
        flat = [batch[name] for name in self._field_names]
        experience = tf.nest.pack_sequence_as(self._data_spec, flat)
        return experience, batch['indexes'], normalize_is_weights(batch['weights'])

    def update_priorities(self, indexes, td_errors):
        # cpprb adds its own epsilon, so a zero TD error still gets sampled again.
        self._buffer.update_priorities(indexes, np.abs(np.asarray(td_errors, dtype=np.float64)))

    def beta(self, train_step):
        """Importance-sampling exponent, annealed to final_beta then held."""
        if train_step >= self._beta_anneal_steps:
            return self._final_beta
        progress = train_step / self._beta_anneal_steps
        return self._initial_beta + progress * (self._final_beta - self._initial_beta)

    @property
    def size(self):
        return self._buffer.get_stored_size()

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, BUFFER_FILENAME)
        # Write beside the target and rename, so a crash mid-write can't leave a
        # half-written buffer where the next run would try to load one. The name
        # has to end in .npz or save_transitions() appends it and the rename misses.
        partial = path + '.partial.npz'
        self._buffer.save_transitions(partial)
        os.replace(partial, path)

    def restore(self, directory):
        """Repopulates from a previous save(). Returns True if anything loaded."""
        path = os.path.join(directory, BUFFER_FILENAME)
        if not os.path.exists(path):
            return False
        self._buffer.load_transitions(path)
        return True
