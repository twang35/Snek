"""NGU's collector: R2D2's lockstep lanes and loss-block rule (`algos/r2d2/collect.py`) with the intrinsic
reward computed per lane per step (`plans/algoExploration/f-ngu-agent57.md` §2, F1's row):

| per lane, beyond R2D2's | |
|---|---|
| `mixture` | the index j: fixed (lane j holds j, NGU App. A read as j mod N) or redrawn uniformly per episode (`assign: episode`, F2's control) |
| `prev_reward_int` | `r^i_{t−1}`, a cell input beside the previous action and extrinsic reward |
| `prev_action` at an episode start | **random** (Agent57 App. E), not "none" |
| the episodic memory and the embedding snapshot | `intrinsic.LaneNovelty` in training mode: emptied and re-snapshotted at the lane's first board |

**The order of a step.** At `x_t` (board `b_t`): a fresh lane begins its episode (memory emptied, snapshot taken,
`b_t`'s key stored); the agent acts on `(x_t, a_{t−1}, r^e_{t−1}, r^i_{t−1}, j)` and reports μ(a_t\\|x_t); the env
steps to `x_{t+1}`; `r^i_{t+1}` is computed from `b_{t+1}` against the memory (which holds `b_0 .. b_t`) and
`b_{t+1}` stored; the row `(x_t, a_t, r^e_{t+1} + β_j r^i_{t+1}, done, a_{t−1}, r^e_{t−1}, r^i_{t−1}, μ, b_t, r^i_{t+1})`
is banked. **A terminal row's `r^i` is 0**: the env has already reset the lane, the terminal board is gone, and
the transition bootstraps nothing.
"""

from collections import deque

import numpy as np
import torch

ASSIGNMENTS = ('fixed', 'episode')


class NguCollector(object):

    def __init__(self, vec, agent, buffer, epsilons, novelty, mixtures, assign='fixed', stride=None, seed=None):
        self.vec, self.agent, self.buffer, self.novelty = vec, agent, buffer, novelty
        n = vec.n
        if n != buffer.lanes:
            raise ValueError('the env has {0} lanes and the replay was built for {1}'.format(n, buffer.lanes))
        self.epsilons = np.asarray(epsilons, dtype=np.float64).reshape(-1)
        if self.epsilons.shape[0] != n:
            raise ValueError('{0} epsilon(s) for {1} lane(s)'.format(self.epsilons.shape[0], n))
        if assign not in ASSIGNMENTS:
            raise ValueError('assign must be one of {0}, got {1!r}'.format(ASSIGNMENTS, assign))
        self.assign = assign
        self.num_mixtures = int(agent.net.mixtures)
        self.block, self.burn_in, self.lookahead = buffer.block, buffer.burn_in, buffer.lookahead
        self.stride = int(stride) if stride else self.block
        if not 1 <= self.stride <= self.block:
            raise ValueError('stride {0} must be in [1, block {1}]'.format(self.stride, self.block))
        self.rng = np.random.default_rng(seed)
        self.mixture = np.asarray(mixtures, dtype=np.int64).reshape(-1).copy()
        if self.mixture.shape[0] != n or self.mixture.min() < 0 or self.mixture.max() >= self.num_mixtures:
            raise ValueError('mixtures must be one index per lane in [0, {0})'.format(self.num_mixtures))
        self.betas = agent.net.betas.detach().cpu().numpy().astype(np.float32)
        self.state = np.zeros((n, buffer.state_width), dtype=np.float32)
        self.prev_action = self.rng.integers(0, agent.num_actions, size=n).astype(np.int64)
        self.prev_reward = np.zeros(n, dtype=np.float32)
        self.prev_reward_int = np.zeros(n, dtype=np.float32)
        self.fresh = np.ones(n, dtype=bool)
        self.episode_len = np.zeros(n, dtype=np.int64)
        self.recent = [deque(maxlen=self.burn_in + 1) for _ in range(n)]
        self.pending = [[] for _ in range(n)]
        self.counters = {name: 0 for name in ('episodes', 'transitions', 'perfect_games', 'windows')}
        self.intrinsic_sum, self.intrinsic_n = 0.0, 0
        self.obs = vec.reset_all()
        if self.obs is None:
            self.obs = vec.observe()
        self.board = self._boards()

    def set_epsilons(self, epsilons):
        self.epsilons = np.asarray(epsilons, dtype=np.float64).reshape(-1)

    def _boards(self):
        return self.vec.board_codes() if self.agent.net.intrinsic['input'] == 'board' else None

    def _inputs(self, board, obs):
        codes = None if board is None else torch.as_tensor(board, device=self.agent.device)
        return self.agent.net.intrinsic_input(codes, torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.agent.device))

    # ---------------------------------------------------------------- the loop

    def step(self):
        """One lockstep advance of every lane. Returns the rows banked (one per lane)."""
        n = self.vec.n
        starting = np.flatnonzero(self.fresh)
        if starting.size:
            inputs = self._inputs(None if self.board is None else self.board[starting], self.obs[starting])
            self.novelty.begin_episode(starting, inputs)
        actions, next_state, mu = self.agent.act(self.obs, self.prev_action, self.prev_reward, self.prev_reward_int,
                                                 self.mixture, self.state, self.fresh, self.epsilons)
        carried = self.state * (~self.fresh)[:, None]
        previous, previous_board = self.obs, self.board
        self.obs, rewards, done, info = self.vec.step(actions)
        self.board = self._boards()
        alive = np.flatnonzero(~done)
        reward_int = np.zeros(n, dtype=np.float32)
        if alive.size:
            inputs = self._inputs(None if self.board is None else self.board[alive], self.obs[alive])
            reward_int[alive] = self.novelty.reward(alive, inputs)
        self.intrinsic_sum += float(reward_int[alive].sum())
        self.intrinsic_n += int(alive.size)
        augmented = rewards.astype(np.float32) + self.betas[self.mixture] * reward_int
        for lane in range(n):
            row = self.buffer.add_row(previous[lane], actions[lane], augmented[lane], done[lane],
                                      self.prev_action[lane], self.prev_reward[lane], self.prev_reward_int[lane], mu[lane],
                                      None if previous_board is None else previous_board[lane], reward_int[lane])
            self._bank(lane, row, carried[lane], bool(done[lane]))
        self.counters['transitions'] += n
        finished = np.flatnonzero(done)
        if finished.size:
            self.counters['episodes'] += int(finished.size)
            self.counters['perfect_games'] += int(np.count_nonzero(info['perfect']))
        self.state = np.asarray(next_state, dtype=np.float32)
        self.prev_action = np.asarray(actions, dtype=np.int64).copy()
        self.prev_reward = np.asarray(rewards, dtype=np.float32).copy()
        self.prev_reward_int = reward_int
        self.fresh = np.asarray(done, dtype=bool).copy()
        if finished.size:
            self.state[finished] = 0.0
            self.prev_action[finished] = self.rng.integers(0, self.agent.num_actions, size=finished.size)
            self.prev_reward[finished] = 0.0
            self.prev_reward_int[finished] = 0.0
            if self.assign == 'episode':
                self.mixture[finished] = self.rng.integers(0, self.num_mixtures, size=finished.size)
        return n

    # ---------------------------------------------------------------- the windows (R2D2's rule, plus the index)

    def _bank(self, lane, row, state, done):
        recent = self.recent[lane]
        recent.append((row, state))
        pending = self.pending[lane]
        position = int(self.episode_len[lane])
        if position % self.stride == 0:
            burn = min(self.burn_in, position)
            first_row, first_state = recent[-1 - burn]
            pending.append({'start': first_row, 'burn': burn, 'state': first_state.copy(), 'length': 0,
                            'ahead': 0, 'closed': False, 'mixture': int(self.mixture[lane])})
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
                               min(window['ahead'], self.lookahead), window['state'], window['mixture'])
        self.counters['windows'] += 1

    # ---------------------------------------------------------------- reporting

    def snapshot(self):
        out = dict(self.counters)
        out['pending_windows'] = int(sum(len(p) for p in self.pending))
        out['intrinsic_mean'] = self.intrinsic_sum / self.intrinsic_n if self.intrinsic_n else 0.0
        out['memory_mean'] = float(self.novelty.memory.sizes().mean())
        out.update(self.novelty.last)
        self.intrinsic_sum, self.intrinsic_n = 0.0, 0
        return out
