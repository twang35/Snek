"""NGU's replay: R2D2's `SequenceReplay` (rows once, windows over them, the stored `(h, c)`, PER) with the
columns Group F adds (`plans/algoExploration/f-ngu-agent57.md` §1-2):

| per row | |
|---|---|
| `reward` | **augmented**: `r^e + β_j r^i` at the lane's mixture, the quantity the Q-network regresses (F1) |
| `reward_int` | `r^i` alone, so the augmented reward could be recomputed at another β (CMR > 0 is not built; stored for the reading) |
| `prev_reward_int` | `r^i_{t−1}`, the cell input at the row |
| `mu` | the behaviour probability of the action taken, Retrace's μ(a_t\\|x_t) |
| `board` | the board codes (`VecSnake.board_codes`, uint8, 100 bytes), the intrinsic modules' input; width 0 under `input: obs` |

| per window | |
|---|---|
| `mixture` | the index j the lane held: the window trains Q at that index and no other (CMR 0, §0) |
"""

import os

import numpy as np

from algos.r2d2.replay import SequenceReplay, BUFFER_FILENAME


class NguReplay(SequenceReplay):

    def __init__(self, windows, obs_len, lanes, state_width, board_len=100, **kw):
        super().__init__(windows, obs_len, lanes, state_width, **kw)
        cap = self.row_capacity
        self.board_len = int(board_len)
        self.reward_int = np.zeros(cap, dtype=np.float32)
        self.prev_reward_int = np.zeros(cap, dtype=np.float32)
        self.mu = np.ones(cap, dtype=np.float32)
        self.board = np.zeros((cap, self.board_len), dtype=np.uint8)
        self.mixture = np.zeros(self.windows_capacity, dtype=np.int64)

    # ---------------------------------------------------------------- writing

    def add_row(self, obs, action, reward, done, prev_action, prev_reward, prev_reward_int=0.0, mu=1.0, board=None,
                reward_int=0.0):
        slot = super().add_row(obs, action, reward, done, prev_action, prev_reward)
        self.reward_int[slot] = float(reward_int)
        self.prev_reward_int[slot] = float(prev_reward_int)
        self.mu[slot] = float(mu)
        if self.board_len:
            self.board[slot] = board
        return slot

    def add_window(self, start, burn, length, ahead, state, mixture=0):
        slot = super().add_window(start, burn, length, ahead, state)
        self.mixture[slot] = int(mixture)
        return slot

    # ---------------------------------------------------------------- reading

    def sample(self, batch_size):
        drawn = super().sample(batch_size)
        if drawn is None:
            return None
        batch, slots, weights = drawn
        B, T = batch['action'].shape
        reward_int = np.zeros((B, T), dtype=np.float32)
        prev_reward_int = np.zeros((B, T), dtype=np.float32)
        mu = np.ones((B, T), dtype=np.float32)
        board = np.zeros((B, T, self.board_len), dtype=np.uint8)
        for b, slot in enumerate(slots):
            rows = self.window_rows(slot)
            first = self.burn_in - int(self.burn[slot])
            span = slice(first, first + rows.size)
            reward_int[b, span] = self.reward_int[rows]
            prev_reward_int[b, span] = self.prev_reward_int[rows]
            mu[b, span] = self.mu[rows]
            if self.board_len:
                board[b, span] = self.board[rows]
        batch.update({'reward_int': reward_int, 'prev_reward_int': prev_reward_int, 'mu': mu, 'board': board,
                      'mixture': self.mixture[slots].copy()})
        return batch, slots, weights

    # ---------------------------------------------------------------- persistence

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, BUFFER_FILENAME)
        staging = path + '.partial.npz'
        rows, w = self.rows, self.n_windows
        np.savez(staging, obs=self.obs[:rows], action=self.action[:rows], reward=self.reward[:rows],
                 done=self.done[:rows], prev_action=self.prev_action[:rows], prev_reward=self.prev_reward[:rows],
                 reward_int=self.reward_int[:rows], prev_reward_int=self.prev_reward_int[:rows], mu=self.mu[:rows],
                 board=self.board[:rows], rows=rows, row_write=self.row_write,
                 start=self.start[:w], burn=self.burn[:w], length=self.length[:w], ahead=self.ahead[:w],
                 state=self.state[:w], mixture=self.mixture[:w], n_windows=w, win_write=self.win_write,
                 priorities=self.tree.nodes[self.tree.size:self.tree.size + w], max_priority=self.max_priority,
                 lanes=self.lanes, row_capacity=self.row_capacity, board_len=self.board_len)
        os.replace(staging, path)
        return path

    def load(self, directory):
        path = os.path.join(directory, BUFFER_FILENAME)
        if not os.path.exists(path):
            return False
        with np.load(path) as data:
            if 'board_len' not in data or int(data['board_len']) != self.board_len:
                raise ValueError('saved buffer is not an ngu buffer at board width {0}'.format(self.board_len))
        loaded = super().load(directory)
        if not loaded:
            return False
        with np.load(path) as data:
            rows, w = int(data['rows']), int(data['n_windows'])
            self.reward_int[:rows] = data['reward_int']
            self.prev_reward_int[:rows] = data['prev_reward_int']
            self.mu[:rows] = data['mu']
            self.board[:rows] = data['board']
            self.mixture[:w] = data['mixture']
        return True
