"""R2D2's collector: N lanes in lockstep, each carrying an LSTM state, its previous action and reward and its
own epsilon, banking one row per lane per step and cutting the rows into windows for `SequenceReplay`.

## The loss-block rule (`plans/algoExploration/e-memory.md` §0, decided 2026-09-30)

A loss block of `block` (80) steps starts every `stride` (40) steps of an episode from its first step, so
consecutive windows overlap by `block - stride` (the paper's 40) and every step is a loss position `block / stride`
times. A block's window is the block with **as many burn-in rows as the episode has
before it, up to `burn_in`** -- 40 for every block but the first, 0 for the first, whose stored state is the
exact zero state the lane had at the episode's start -- and the `lookahead` (n) rows after it, which the
n-step target of the block's last steps reads. Mid-episode this is the paper's geometry exactly: 120 stored,
40 burn-in, 80 loss, consecutive windows' loss blocks 40 apart.

A window is handed to the replay once it is complete: its block is full (or the episode ended) **and**
its `lookahead` rows exist (or the episode ended, in which case the bootstrap past the end is 0 and the
rows that would have followed are not needed). So every stored window is drawable at once.

## What a lane carries

| per lane | |
|---|---|
| `state` | the net's `(h, c)` carried into the next step; zero after a death |
| `prev_action`, `prev_reward` | the inputs the next step feeds the cell; -1 and 0 after a death |
| `fresh` | True on the episode's first step |
| `epsilon` | the lane's own, for the whole run (the Ape-X ladder, `schedules.apex_epsilons`) or the shared linear one |
| `recent` | the last `burn_in + 1` rows' `(row, state carried into it)`, so a block's window can start up to `burn_in` rows back with the right state |
| `pending` | the blocks of the current episode not yet handed over |

**No fork and no shield.** A forked lane would need a copied state and a copied sequence prefix, and the
paper has neither; refused by construction rather than by name, since the knobs are not read here.
"""

from collections import deque

import numpy as np


class Collector(object):

    def __init__(self, vec, agent, buffer, epsilons, stride=None, seed=None):
        self.vec = vec
        self.agent = agent
        self.buffer = buffer
        n = vec.n
        if n != buffer.lanes:
            raise ValueError('the env has {0} lanes and the replay was built for {1}'.format(n, buffer.lanes))
        self.epsilons = np.asarray(epsilons, dtype=np.float64).reshape(-1)
        if self.epsilons.shape[0] != n:
            raise ValueError('{0} epsilon(s) for {1} lane(s)'.format(self.epsilons.shape[0], n))
        self.block, self.burn_in, self.lookahead = buffer.block, buffer.burn_in, buffer.lookahead
        self.stride = int(stride) if stride else self.block
        if not 1 <= self.stride <= self.block:
            raise ValueError('stride {0} must be in [1, block {1}]'.format(self.stride, self.block))
        self.state = np.zeros((n, buffer.state_width), dtype=np.float32)
        self.prev_action = np.full(n, -1, dtype=np.int64)
        self.prev_reward = np.zeros(n, dtype=np.float32)
        self.fresh = np.ones(n, dtype=bool)
        self.episode_len = np.zeros(n, dtype=np.int64)
        self.recent = [deque(maxlen=self.burn_in + 1) for _ in range(n)]
        self.pending = [[] for _ in range(n)]
        self.rng = np.random.default_rng(seed)
        self.counters = {name: 0 for name in ('episodes', 'transitions', 'perfect_games', 'windows')}
        self.obs = vec.reset_all()
        if self.obs is None:
            self.obs = vec.observe()

    def set_epsilons(self, epsilons):
        self.epsilons = np.asarray(epsilons, dtype=np.float64).reshape(-1)

    # ---------------------------------------------------------------- the loop

    def step(self):
        """One lockstep advance of every lane. Returns the rows banked (one per lane)."""
        actions, next_state = self.agent.act(self.obs, self.prev_action, self.prev_reward, self.state,
                                             self.fresh, self.epsilons)
        # The state each row was stepped *from*, after the fresh zeroing, is what a window starting at that row
        # restores. `act` zeroed fresh lanes before using the state, so zero it here the same way.
        carried = self.state * (~self.fresh)[:, None]
        previous = self.obs
        self.obs, rewards, done, info = self.vec.step(actions)
        n = self.vec.n
        for lane in range(n):
            row = self.buffer.add_row(previous[lane], actions[lane], rewards[lane], done[lane],
                                      self.prev_action[lane], self.prev_reward[lane])
            self._bank(lane, row, carried[lane], bool(done[lane]))
        self.counters['transitions'] += n
        finished = np.flatnonzero(done)
        if finished.size:
            self.counters['episodes'] += int(finished.size)
            self.counters['perfect_games'] += int(np.count_nonzero(info['perfect']))
        self.state = np.asarray(next_state, dtype=np.float32)
        self.prev_action = np.asarray(actions, dtype=np.int64).copy()
        self.prev_reward = np.asarray(rewards, dtype=np.float32).copy()
        self.fresh = np.asarray(done, dtype=bool).copy()
        if finished.size:
            self.state[finished] = 0.0
            self.prev_action[finished] = -1
            self.prev_reward[finished] = 0.0
        return n

    # ---------------------------------------------------------------- the windows

    def _bank(self, lane, row, state, done):
        recent = self.recent[lane]
        recent.append((row, state))
        pending = self.pending[lane]
        position = int(self.episode_len[lane])              # this row's index within its episode
        if position % self.stride == 0:
            burn = min(self.burn_in, position)
            first_row, first_state = recent[-1 - burn]        # `burn` rows back: the window's first row
            pending.append({'start': first_row, 'burn': burn, 'state': first_state.copy(), 'length': 0,
                            'ahead': 0, 'closed': False})
        self.episode_len[lane] = position + 1
        for window in pending:
            if not window['closed']:
                window['length'] += 1
                if window['length'] == self.block:
                    window['closed'] = True
            else:
                window['ahead'] += 1
        if done:
            for window in pending:
                self._emit(window)
            del pending[:]
            recent.clear()
            self.episode_len[lane] = 0
            return
        complete = [w for w in pending if w['closed'] and w['ahead'] >= self.lookahead]
        for window in complete:
            self._emit(window)
            pending.remove(window)

    def _emit(self, window):
        self.buffer.add_window(window['start'], window['burn'], window['length'],
                               min(window['ahead'], self.lookahead), window['state'])
        self.counters['windows'] += 1

    # ---------------------------------------------------------------- reporting

    def snapshot(self):
        out = dict(self.counters)
        out['pending_windows'] = int(sum(len(p) for p in self.pending))
        return out
