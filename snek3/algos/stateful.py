"""`StatefulPolicy`: a `policy_fn` that carries a recurrent state per lane, behind the seam the
measurement engine drives.

`vectorized/engine.py`'s seam is a plain callable `(m, obs_len) float32 -> (m,) int64`, and it is stateless:
the engine calls it once per resident job on that job's rows, and **lanes migrate between jobs and reset
between episodes without telling the policy.** A recurrent policy needs, per lane, a hidden state that
persists across steps and is zeroed when the lane's episode restarts -- and R2D2's net also takes the previous
action and the previous reward as inputs, the first of which the policy chose itself and the second of which
only the engine knows. So the protocol (`plans/algoExploration/e-memory.md` §1) is the same callable **plus**
one method:

    policy.begin(rows, fresh, prev_reward)   # before the call: which lanes, which of them were reset since
                                             # the last step, and each lane's reward from its previous step
    actions = policy(obs)                    # the call, on exactly those rows

The engine duck-types on the method name (`getattr(policy_fn, 'begin', None)`), so a plain callable is
treated as today and `vectorized/` still imports no torch: the arrays that cross the seam are numpy. **The
policy owns the state.** It keeps the recurrent state, the previous action and the previous reward indexed by
absolute lane row, zeroes the `fresh` rows in `begin`, reads and writes the rows it is handed, and the engine
holds nothing for it.

**A caller that never calls `begin`** -- a hand-written loop over one lane -- gets rows `0..m-1`, a fresh state
on its first call and a carried one after, and a previous reward of 0. `watch.py` and `record_gif.py` do call
`begin`, so the reward reaches the net there; the fallback exists so a stateful policy is never *wrong* under
an old caller, only reward-blind.
"""

import numpy as np
import torch


class StatefulPolicy(object):
    """`step_fn(obs, prev_action, prev_reward, state) -> (actions, state)`, all torch tensors on `device`:
    `obs (m, obs_len)`, `prev_action (m,)` long with -1 for "none yet", `prev_reward (m,)`, `state (m, width)`.
    `state_width` is the flat recurrent state's width (a GRU's hidden, an LSTM's 2 x hidden, 0 for none)."""

    def __init__(self, step_fn, state_width, device='cpu'):
        self.step_fn = step_fn
        self.state_width = int(state_width)
        self.device = device
        self.state = torch.zeros((0, self.state_width), dtype=torch.float32, device=device)
        self.prev_action = torch.full((0,), -1, dtype=torch.int64, device=device)
        self.prev_reward = torch.zeros((0,), dtype=torch.float32, device=device)
        self._rows = None
        self._began = False

    # ------------------------------------------------------------ the protocol

    def _grow(self, needed):
        have = self.state.shape[0]
        if needed <= have:
            return
        extra = needed - have
        self.state = torch.cat([self.state, torch.zeros((extra, self.state_width), dtype=torch.float32,
                                                        device=self.device)])
        self.prev_action = torch.cat([self.prev_action, torch.full((extra,), -1, dtype=torch.int64,
                                                                   device=self.device)])
        self.prev_reward = torch.cat([self.prev_reward, torch.zeros((extra,), dtype=torch.float32,
                                                                    device=self.device)])

    def begin(self, rows, fresh, prev_reward):
        """Names the lanes the next call covers. `fresh` rows start a new episode: state zero, no previous
        action, reward 0."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        fresh = np.asarray(fresh, dtype=bool).reshape(-1)
        prev_reward = np.asarray(prev_reward, dtype=np.float32).reshape(-1)
        if fresh.shape[0] != rows.shape[0] or prev_reward.shape[0] != rows.shape[0]:
            raise ValueError('begin() needs one fresh flag and one previous reward per row: {0} rows, {1} '
                             'flags, {2} rewards'.format(rows.shape[0], fresh.shape[0], prev_reward.shape[0]))
        if rows.size:
            self._grow(int(rows.max()) + 1)
        index = torch.as_tensor(rows, device=self.device)
        reset = torch.as_tensor(fresh, device=self.device)
        self.prev_reward[index] = torch.as_tensor(prev_reward, device=self.device)
        if reset.any():
            cleared = index[reset]
            self.state[cleared] = 0.0
            self.prev_action[cleared] = -1
            self.prev_reward[cleared] = 0.0
        self._rows = index
        self._began = True

    def __call__(self, observations):
        observations = np.asarray(observations, dtype=np.float32)
        m = observations.shape[0]
        if self._rows is None:
            # The fallback for a caller without `begin`: rows 0..m-1, fresh once, reward unknown (0).
            rows = np.arange(m)
            self.begin(rows, np.full(m, not self._began, dtype=bool), np.zeros(m, dtype=np.float32))
        index = self._rows
        self._rows = None
        if index.shape[0] != m:
            raise ValueError('begin() named {0} rows but the call carries {1} observations'.format(
                index.shape[0], m))
        with torch.no_grad():
            obs = torch.as_tensor(observations, device=self.device)
            actions, state = self.step_fn(obs, self.prev_action[index], self.prev_reward[index],
                                          self.state[index])
            actions = actions.to(torch.int64)
            self.prev_action[index] = actions
            if self.state_width:
                self.state[index] = state
        return actions.cpu().numpy()


def one_hot_previous(prev_action, num_actions):
    """`(m,) long with -1 for none -> (m, num_actions)` float: the zero vector at an episode's first step."""
    known = (prev_action >= 0)
    one_hot = torch.nn.functional.one_hot(prev_action.clamp(min=0), int(num_actions)).to(torch.float32)
    return one_hot * known.unsqueeze(-1).to(torch.float32)
