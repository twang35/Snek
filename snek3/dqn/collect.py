"""Collecting experience: N games in lockstep, n-step returns, and forking.

**Every lane steps on every call.** That is the whole simplification over snek2, whose collector was
330 lines of round robin, an environment pool, a snapshot validator and a per-stream deque, all
because scalar environments advance one at a time. `VecSnake` advances all of them at once, so the
collector is a per-lane bookkeeping array and one `vec.step()`.

## Why fork at all

At epsilon ~0.003 only about 0.2% of steps take a non-greedy action, so at any endgame decision point
the buffer holds the consequence of the action the policy took and **never** the consequence of the
alternative. Measured over three snek2 arms at length >= 85: a mean of 2.06 safe actions per such
state, so ~1.06 safe actions per state are never tried.

That asymmetry matters for Q-learning specifically. An arm that dies from state `s` learns that
`Q(s, a_bad)` is low, but nothing ever raises `Q(s, a_good)` for the action it did not take, so the
argmax has no reason to flip. Forking supplies the missing half: from one state, both continuations
reach the buffer.

## How a fork works here

Lanes have two roles, fixed at construction:

| lanes | role |
|---|---|
| `[0, collect_envs)` | **primary.** Never reseated. These are the arm's own games |
| `[collect_envs, width)` | **secondary.** Reseatable. Between forks they play ordinary episodes |

When a primary lane is at a decision point — long enough, more than one safe move, and the coin
passes — a free secondary lane is **reseated** onto a copy of the primary's state via
`VecSnake.copy_rows` and given a forced first action, one the primary did not take. It runs until it
dies or hits `fork_max_steps`, then becomes free again.

**A reseated lane's own episode is abandoned, and that is not a loss.** Whatever it already stored
genuinely happened and stays in the buffer; only its partial n-step window is dropped, so no
transition is stitched across the discontinuity. A free secondary lane still plays and still stores,
so no lane is ever idle — snek2's branches shared a step budget with the main line and cost it ~1.9x
fewer episodes per gradient step. Here the budget is per transition, so the replay ratio is exact
whether forking is on or off.

## n-step returns

The window lives here, per lane, not in the buffer. `n=1` is the default and stores every transition
immediately; `n>1` holds a lane's last `n` steps and emits `(obs_t, a_t, sum gamma^k r_{t+k},
obs_{t+n}, gamma^n)`. A window is **flushed on an episode boundary, not carried across one** —
snek2's buffer held the window and had no episode check at all, which is why one shared window across
interleaved games fabricated transitions.

**A terminal step emits every transition still in the window**, each with `discount=0`, because the
episode's outcome is what a shorter tail is for and dropping it would discard exactly the deaths and
wins that matter most.
"""

import numpy as np

from dqn.agent import SAFETY_RANGE
from vectorized import config as C

# The shortest snake a fork gate can sensibly name: one segment longer than the opening.
MIN_GATE_LENGTH = C.START_SEGMENTS + 1


class ForkConfig(object):
    """The fork knobs, validated once. `branches == 1` means forking is off.

    Rejects rather than clamps, for the same reason `SNEK_MIN_EPSILON` does: this project reads the
    startup `hyperparameter override:` lines to confirm an arm got its config, and an override that
    quietly does something else is worse than one that refuses.
    """

    def __init__(self, branches=4, prob=0.5, min_length=85, max_steps=60):
        self.branches = int(branches)
        self.prob = float(prob)
        self.min_length = int(min_length)
        self.max_steps = int(max_steps)
        if self.branches < 1:
            raise ValueError('SNEK_FORK_BRANCHES={0} must be at least 1 (1 disables forking)'
                             .format(self.branches))
        if self.branches > 1:
            if not 0.0 <= self.prob <= 1.0:
                raise ValueError('SNEK_FORK_PROB={0} is not a probability'.format(self.prob))
            if not MIN_GATE_LENGTH <= self.min_length <= C.PERFECT_SCORE:
                raise ValueError(
                    'SNEK_FORK_MIN_LENGTH={0} is outside [{1}, {2}] — the opening length and the '
                    'whole board'.format(self.min_length, MIN_GATE_LENGTH, C.PERFECT_SCORE))
            if self.max_steps < 0:
                raise ValueError('SNEK_FORK_MAX_STEPS={0} must be at least 0 (0 runs a branch to '
                                 'its terminal state)'.format(self.max_steps))
            blinded = sorted(set(C.ZERO_OBS_INDICES) & set(range(*SAFETY_RANGE)))
            if blinded:
                # Silent otherwise: with those indices zeroed every move reads fatal, so no state is
                # ever eligible and the arm looks like a forking arm while collecting nothing extra.
                raise ValueError(
                    'SNEK_ZERO_OBS blanks the safe-move flags at {0}, which forking reads to find '
                    'its branch points — every move would read fatal and no fork would ever '
                    'happen. Drop the ablation or drop SNEK_FORK_BRANCHES.'.format(blinded))

    @property
    def enabled(self):
        return self.branches > 1


class Collector(object):
    """Advances every lane one step per `step()` call and banks the transitions it produces.

    `width` must be `collect_envs * fork.branches`, so each primary lane has `branches - 1` secondary
    lanes' worth of fork slots available — shared across primaries rather than reserved per primary,
    because the demand for slots far exceeds the supply (roughly 76 eligible states per episode at
    length >= 85) and reserving them would leave slots idle beside a primary that is not in its
    endgame yet.
    """

    def __init__(self, vec, agent, buffer, discount=0.99, n_step=1, collect_envs=1,
                 fork=None, guided_fraction=0.0, seed=None):
        self.vec = vec
        self.agent = agent
        self.buffer = buffer
        self.discount = float(discount)
        self.n_step = int(n_step)
        self.collect_envs = int(collect_envs)
        self.fork = fork if fork is not None else ForkConfig(branches=1)
        self.guided_fraction = float(guided_fraction)
        if self.n_step < 1:
            raise ValueError('SNEK_N_STEP_UPDATE={0} must be at least 1'.format(self.n_step))
        expected = self.collect_envs * self.fork.branches
        if vec.n != expected:
            raise ValueError(
                'width {0} does not match collect_envs {1} x fork branches {2} = {3}'.format(
                    vec.n, self.collect_envs, self.fork.branches, expected))
        # A dedicated Generator: fork coins and shield coins must not perturb the stream the env
        # draws food from, or an arm's decisions would depend on how many food cells were rejected.
        self.rng = np.random.default_rng(seed)

        self.primaries = np.arange(self.collect_envs)
        self.secondaries = np.arange(self.collect_envs, vec.n)
        # Per lane: the last `n_step` (obs, action, reward) triples, oldest first.
        self.windows = [[] for _ in range(vec.n)]
        self.pending = np.full(vec.n, -1, dtype=np.int64)
        self.branching = np.zeros(vec.n, dtype=bool)
        self.branch_age = np.zeros(vec.n, dtype=np.int64)
        self.guided = self.rng.random(vec.n) < self.guided_fraction
        self.counters = {name: 0 for name in
                         ('forks', 'retired', 'truncated', 'terminated', 'eligible',
                          'skipped_full', 'episodes', 'transitions', 'perfect_games')}

        self.obs = vec.reset_all()
        if self.obs is None:
            self.obs = vec.observe()

    # ---------------------------------------------------------------- the loop

    def step(self, epsilon):
        """One lockstep advance of every lane. Returns the number of transitions banked.

        The transition count, not the lane count: they differ whenever `n_step > 1` (a lane's first
        `n-1` steps bank nothing) or an episode ends (a terminal step banks the whole window). The
        training loop drives its gradient budget off this number, which is what keeps the replay
        ratio exact whether forking is on or off.
        """
        actions = self.agent.act(self.obs, epsilon, self.guided)
        self._maybe_fork(actions)

        forced = self.pending >= 0
        if forced.any():
            # A branch's forced first action is taken **without consulting the policy** — that is
            # the entire point of a branch, and it is why the fork above runs before this line.
            actions = np.where(forced, self.pending, actions)
            self.pending[:] = -1

        previous = self.obs
        self.obs, rewards, done, info = self.vec.step(actions)
        banked = self._bank(previous, actions, rewards, done)
        self._settle_episodes(done, info)
        self._retire_branches(done)
        return banked

    # ---------------------------------------------------------------- n-step windows

    def _bank(self, previous, actions, rewards, done):
        banked = 0
        for lane in range(self.vec.n):
            window = self.windows[lane]
            window.append((previous[lane], int(actions[lane]), float(rewards[lane])))
            if done[lane]:
                # Every transition still in the window is emitted, each bootstrapping off nothing.
                # A shorter tail is exactly what an episode's outcome needs, and dropping it would
                # discard the deaths and the wins — the transitions that matter most.
                for start in range(len(window)):
                    self._emit(window, start, self.obs[lane], 0.0)
                    banked += 1
                del window[:]
            elif len(window) == self.n_step:
                self._emit(window, 0, self.obs[lane], self.discount ** self.n_step)
                banked += 1
                del window[0]
        self.counters['transitions'] += banked
        return banked

    def _emit(self, window, start, next_obs, discount):
        """Stores `window[start]` with the discounted reward of everything from `start` on.

        `next_obs` on a terminal transition is the lane's **post-reset** observation, because
        `VecSnake` auto-resets inside `step()` and a second `observe()` would double the cost of the
        most expensive call in the loop. It cannot matter: `discount` is 0 there, so the value it
        would bootstrap is multiplied out. `tests/test_collect.py` pins that discount at 0 for every
        terminal transition, which is the property this rests on.
        """
        obs, action, _ = window[start]
        total = 0.0
        for offset in range(start, len(window)):
            total += (self.discount ** (offset - start)) * window[offset][2]
        self.buffer.add(obs, action, total, next_obs, discount)

    # ---------------------------------------------------------------- episodes and the shield

    def _settle_episodes(self, done, info):
        finished = np.flatnonzero(done)
        if not finished.size:
            return
        self.counters['episodes'] += int(finished.size)
        self.counters['perfect_games'] += int(np.count_nonzero(info['perfect']))
        # Redrawn once per episode, never per step: an episode is guided end to end or not at all,
        # or the buffer holds a trajectory that was half shielded and cannot be reasoned about.
        self.guided[finished] = self.rng.random(finished.size) < self.guided_fraction

    def set_guided_fraction(self, fraction):
        """Follows the schedule between evals. Takes effect at each lane's next episode boundary.

        Not applied to episodes already in flight, for the same reason the draw is per episode.
        """
        self.guided_fraction = float(fraction)

    # ---------------------------------------------------------------- forking

    def _free_slots(self):
        return self.secondaries[~self.branching[self.secondaries]]

    def _maybe_fork(self, actions):
        if not self.fork.enabled:
            return
        # The same block, and the same threshold, the shield reads. One definition in `agent.py`,
        # so a fork point and a shielded draw can never disagree about which moves are safe.
        safe = self.obs[:, SAFETY_RANGE[0]:SAFETY_RANGE[1]] > 0.5
        lengths = self.vec.length
        for lane in self.primaries:
            # Not on a lane whose snake is still short: exploration is nearly free early, and the
            # asymmetry this feature fixes is an endgame one.
            if lengths[lane] < self.fork.min_length:
                continue
            options = np.flatnonzero(safe[lane])
            if options.size < 2:
                continue
            self.counters['eligible'] += 1
            if self.rng.random() >= self.fork.prob:
                continue
            slots = self._free_slots()
            if not slots.size:
                self.counters['skipped_full'] += 1
                continue
            alternatives = [action for action in options.tolist() if action != int(actions[lane])]
            if not alternatives:
                continue
            # Shuffled before truncating, or a slot shortage would take alternatives in action order
            # every time and systematically over-collect one turn direction.
            self.rng.shuffle(alternatives)
            for slot, action in zip(slots.tolist(), alternatives):
                self._fork(lane, slot, action)

    def _fork(self, parent, slot, action):
        self.vec.copy_rows([parent], [slot])
        # The slot's own episode is abandoned. Its stored transitions stay in the buffer — they
        # happened — but its partial window goes, so nothing is stitched across the discontinuity.
        del self.windows[slot][:]
        # The observation too: the transition this slot is about to store must be anchored at the
        # *fork point*, which is the parent's state, not whatever the slot was looking at.
        self.obs[slot] = self.obs[parent]
        # A fork is the same episode taking a different action, so it inherits the shield flag:
        # "guided end to end or not at all" holds for the whole trajectory tree.
        self.guided[slot] = self.guided[parent]
        self.pending[slot] = action
        self.branching[slot] = True
        self.branch_age[slot] = 0
        self.counters['forks'] += 1

    def _retire_branches(self, done):
        if not self.fork.enabled:
            return
        active = self.secondaries[self.branching[self.secondaries]]
        if not active.size:
            return
        self.branch_age[active] += 1
        died = active[done[active]]
        capped = active[(self.branch_age[active] >= self.fork.max_steps)] \
            if self.fork.max_steps else np.empty(0, dtype=np.int64)
        # Truncation is not a bug: the last stored transition bootstraps off the branch's current
        # state exactly as any non-terminal one does. The causal window this targets is short — a
        # snek2 diagnostic put the fatal decision a median of 2 steps and at most 29 before the death
        # — so a cap buys many more distinct branch points for the same budget.
        capped = np.setdiff1d(capped, died)
        for lane in np.concatenate([died, capped]).tolist():
            self.branching[lane] = False
        self.counters['terminated'] += int(died.size)
        self.counters['truncated'] += int(capped.size)
        self.counters['retired'] += int(died.size + capped.size)

    # ---------------------------------------------------------------- reporting

    def snapshot(self):
        """The counters, plus what share of the work went to branches. For the eval row."""
        out = dict(self.counters)
        out['live_branches'] = int(np.count_nonzero(self.branching))
        out['free_slots'] = int(self._free_slots().size)
        return out
