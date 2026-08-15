"""Collects experience from several branches of one game, so endgame alternatives get explored.

At epsilon ~0.003 only about 0.2% of steps take a non-greedy action, so at any endgame decision
point the replay buffer holds the consequence of the action the policy took and **never** the
consequence of the alternative. Measured over three arms at length >= 85: a mean of 2.06 safe
actions per such state, so ~1.06 safe actions per state are never tried.

That asymmetry matters for Q-learning specifically. An arm that dies from state `s` does learn that
`Q(s, a_bad)` is low, but nothing ever raises `Q(s, a_good)` for the action it did not take, so the
argmax has no reason to flip. Forking supplies the missing half of the comparison: from one state,
both continuations end up in the buffer.

**The shape of it.** When the main line reaches a state with more than one non-fatal action and the
snake is long enough, the game is snapshotted and the other safe actions are handed to branch
environments restored from that snapshot. Every live branch — the main line included — then advances
in round robin, **one counted environment step per call**, so the 1 collect step : 1 gradient step
ratio the training loop relies on is unchanged. A branch runs until it terminates or hits a step cap,
then retires; the main line never retires.

What this costs: the main line advances more slowly per `global_step`, because it shares the step
budget. Confined to the endgame by the length gate, that works out to roughly 1.9x fewer main-line
episodes per gradient step at a cap of 4. Read a forking arm at matched `global_step` against a
matched control, and use the `main_steps` counter for the games-played axis.

**On by default since 2026-08-14** (4 branches) — every arm from batch 17 on passed
`SNEK_FORK_BRANCHES=4` explicitly, so a default of 1 described no run that existed. `SNEK_FORK_BRANCHES=1`
still turns it off: `validate_config` returns False and `training.py` calls `PyDriver` exactly as before.

A resume starts with the main line only. Live branches are in memory and are not checkpointed, the
same discontinuity the replay buffer's own trailing window has always had.
"""
import collections
import itertools
import random

import numpy as np
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import trajectory

from prioritized_replay_buffer import DEFAULT_STREAM
from shielded_policy import SAFETY_OBS_END, SAFETY_OBS_START
from snake_constants import ACTIONS, PERFECT_SCORE, START_SEGMENTS

# The shortest snake a snapshot can describe, and so the lowest sensible length gate.
MIN_GATE_LENGTH = START_SEGMENTS + 1


def validate_config(branches, fork_prob, fork_min_length, fork_max_steps, zero_obs_indices=()):
    """Checks the fork knobs, returning False when the feature is off and True when it is on.

    Rejects rather than clamps, for the reason `SNEK_MIN_EPSILON` does: an override that quietly
    does something else is worse than one that refuses, because this project reads the startup
    `hyperparameter override:` lines to confirm an arm got its config.
    """
    if branches < 1:
        raise SystemExit('SNEK_FORK_BRANCHES={0} must be at least 1 (1 disables forking).'
                         .format(branches))
    if branches == 1:
        return False
    if not 0.0 <= fork_prob <= 1.0:
        raise SystemExit('SNEK_FORK_PROB={0} is not a probability. Use a value in [0.0, 1.0].'
                         .format(fork_prob))
    if not MIN_GATE_LENGTH <= fork_min_length <= PERFECT_SCORE:
        raise SystemExit('SNEK_FORK_MIN_LENGTH={0} is outside [{1}, {2}] — the opening length and '
                         'the whole board.'.format(fork_min_length, MIN_GATE_LENGTH, PERFECT_SCORE))
    if fork_max_steps < 0:
        raise SystemExit('SNEK_FORK_MAX_STEPS={0} must be at least 0 (0 runs a branch to its '
                         'terminal state).'.format(fork_max_steps))
    blinded = sorted(set(zero_obs_indices) & set(range(SAFETY_OBS_START, SAFETY_OBS_END)))
    if blinded:
        # Silent otherwise: with those indices zeroed every move reads fatal, so no state is ever
        # eligible and the arm looks like a forking arm while collecting nothing extra.
        raise SystemExit(
            'SNEK_ZERO_OBS blanks the safe-move flags at {0}, which forking reads to find its '
            'branch points — every move would read fatal and no fork would ever happen. Drop the '
            'ablation or drop SNEK_FORK_BRANCHES.'.format(blinded))
    return True


class _Branch:
    """One live continuation: its own environment, replay stream and position in the game."""

    def __init__(self, env, stream, time_step, policy_state, guided, is_main,
                 pending_action=None):
        self.env = env
        self.stream = stream
        self.time_step = time_step
        self.policy_state = policy_state
        self.guided = guided
        self.is_main = is_main
        self.pending_action = pending_action
        self.steps = 0
        self.retired = False


class ForkingCollector:
    """Drop-in replacement for `PyDriver.run()` in the training loop, with branches.

    The per-step body deliberately mirrors `py_driver.PyDriver.run` — the same
    `action_step._replace(state=policy_state)` substitution, the same
    `trajectory.from_transition`, and the same rule that a boundary step does not count toward the
    step budget — so that turning forking on changes which *games* are played and nothing about how
    a transition is built.
    """

    def __init__(self, main_env, branch_envs, policy, replay_buffer, max_branches, fork_prob,
                 fork_min_length, fork_max_steps, guided_flag=None, seed=None):
        if len(branch_envs) < max_branches - 1:
            raise ValueError('need {0} branch environments for {1} live branches, got {2}'.format(
                max_branches - 1, max_branches, len(branch_envs)))
        self._policy = policy
        self._buffer = replay_buffer
        self._max_branches = max_branches
        self._fork_prob = fork_prob
        self._fork_min_length = fork_min_length
        self._fork_max_steps = fork_max_steps
        self._guided_flag = guided_flag
        # A dedicated Random instance, so fork coins never perturb the global stream food placement
        # draws from. Reproducibility is not the point — cpprb's sampling RNG is unseedable — but an
        # arm's fork decisions should not depend on how many food cells were rejected.
        self._rng = random.Random(seed)
        self._pool = list(branch_envs)
        # Ids are allocated from a counter that never repeats. That is what makes a missed
        # close_stream a small leak rather than a splice onto a dead branch's tail.
        self._next_stream = itertools.count(1)
        # Seeded from the shield's own Variable rather than assumed False. The first call happens
        # mid-episode on a resume, and `_apply_guided` writes this value back before the policy
        # reads it — so a guess here would silently *overwrite* the arm's real guided state and,
        # if it guessed False, unshield the episode in progress.
        initial_guided = bool(guided_flag.numpy()) if guided_flag is not None else False
        self._main = _Branch(main_env, DEFAULT_STREAM, None, (), initial_guided, is_main=True)
        self._schedule = collections.deque([self._main])
        self._counters = collections.Counter()
        self._live_high_water = 1

    # ------------------------------------------------------------------ the training-loop entry

    def run(self, time_step):
        """Advances exactly one counted environment step, then returns the *main* line's TimeStep.

        The main line's, whichever branch actually moved. `training.py` threads the return value
        into the next call, so returning a branch's TimeStep would make the loop start following
        that branch and the main env's own state would drift out of sync with it.
        """
        self._main.time_step = time_step
        while True:
            branch = self._schedule.popleft()
            counted = self._step_once(branch)
            if branch.retired:
                self._retire(branch)
            else:
                self._schedule.append(branch)
            self._live_high_water = max(self._live_high_water, len(self._schedule))
            if counted:
                if not branch.is_main:
                    self._counters['branch_steps'] += 1
                else:
                    self._counters['main_steps'] += 1
                return self._main.time_step

    def counters(self):
        """Snapshot of the bookkeeping, for the eval row. Cumulative over the process."""
        counted = self._counters['main_steps'] + self._counters['branch_steps']
        return {
            'forks': self._counters['forks'],
            'retired': self._counters['retired'],
            'truncated': self._counters['truncated'],
            'terminated': self._counters['terminated'],
            'skipped_full': self._counters['skipped_full'],
            'eligible': self._counters['eligible'],
            'main_steps': self._counters['main_steps'],
            'branch_steps': self._counters['branch_steps'],
            'branch_share': (self._counters['branch_steps'] / counted) if counted else 0.0,
            'live_high_water': self._live_high_water,
            'live_now': len(self._schedule),
        }

    # ------------------------------------------------------------------ one step

    def _step_once(self, branch):
        """One environment step for `branch`. Returns whether it counted toward the step budget.

        A boundary step — the one that crosses an episode end — is free, exactly as in `PyDriver`,
        where `num_steps += np.sum(~traj.is_boundary())`. It is also the step that stores the
        branch's death or win: with a two-frame window, the terminal transition's item is
        `[terminal_traj, boundary_traj]`, so a branch that vanished at its terminal step would
        contribute no terminal transition at all — and terminal outcomes are the point of forking.
        """
        action_step = self._action_step_for(branch)
        next_time_step = branch.env.step(action_step.action)
        traj = trajectory.from_transition(
            branch.time_step, action_step._replace(state=branch.policy_state), next_time_step)
        self._buffer.add(traj, stream=branch.stream)

        was_boundary = bool(np.any(traj.is_boundary()))
        branch.time_step = next_time_step
        branch.policy_state = action_step.state
        branch.steps += 1

        if was_boundary and not branch.is_main:
            # The episode ended on this branch's previous turn and its terminal item is now
            # complete. The environment auto-reset into a fresh episode, which is discarded: a
            # branch is one continuation, and letting it start a second episode would make it an
            # unbounded extra collect stream.
            branch.retired = True
            self._counters['terminated'] += 1
        elif (not branch.is_main and self._fork_max_steps
                and branch.steps >= self._fork_max_steps and not was_boundary):
            # Truncation, not a bug: the last stored item bootstraps off this branch's current
            # state the same way any non-terminal transition does. The causal window this feature
            # targets is short — the frozen diagnostics put the fatal decision a median of 2 steps
            # and at most 29 before the death — so a cap buys many more distinct branch points for
            # the same budget.
            branch.retired = True
            self._counters['truncated'] += 1

        return not was_boundary

    def _action_step_for(self, branch):
        """The PolicyStep for this branch's next move, forking first if the state qualifies.

        A branch's forced first action is consumed here **without consulting the policy** — that is
        the whole point of a branch, and `test_a_branch_takes_its_forced_action_first` is what holds
        it.

        No fork is attempted on that step either, though today that is belt and braces rather than
        load-bearing: `_maybe_fork` only forks the main line, and the main line never carries a
        pending action, so the two rules cannot both be reachable. Moving the call would be an
        equivalent mutation, so no test claims to cover it.
        """
        if branch.pending_action is not None:
            action = np.int32(branch.pending_action)
            branch.pending_action = None
            return policy_step.PolicyStep(action=action, state=branch.policy_state, info=())

        self._apply_guided(branch)
        action_step = self._policy.action(branch.time_step, branch.policy_state)
        self._read_back_guided(branch)
        self._maybe_fork(branch, int(np.asarray(action_step.action).item()))
        return action_step

    # ------------------------------------------------------------------ forking

    def _maybe_fork(self, parent, chosen_action):
        """Snapshots `parent` and hands its untaken safe actions to new branches.

        Called before the parent steps, so the snapshot is of the decision point itself.

        Only the main line forks. A branch of a branch would deepen the correlation between
        collected trajectories rather than widen coverage, and the demand for branch slots already
        exceeds the supply — roughly 76 eligible states per episode at length >= 85 against a
        handful of slots.
        """
        if not parent.is_main:
            return
        # Not on the boundary step. Once the episode has ended, the game holds a corpse — a
        # self-collision death leaves the head on top of the body, and an out-of-bounds death leaves
        # it off the board — and this step's action is the throwaway one that triggers the reset, so
        # it is not a decision point at all. Snapshotting here is what a smoke run crashed on inside
        # 10k steps, and `validate_snapshot` is what caught it.
        if bool(parent.time_step.is_last()):
            return
        if parent.env.snake_length() < self._fork_min_length:
            return

        observation = np.asarray(parent.time_step.observation)
        safe = [action for action in range(len(ACTIONS))
                if observation[SAFETY_OBS_START + action] > 0.5]
        if len(safe) < 2:
            return
        self._counters['eligible'] += 1

        if self._rng.random() >= self._fork_prob:
            return
        room = self._max_branches - (len(self._schedule) + 1)
        if room <= 0:
            self._counters['skipped_full'] += 1
            return

        alternatives = [action for action in safe if action != chosen_action]
        # Shuffled before truncating, or a cap that admits fewer branches than there are
        # alternatives would take them in ACTIONS order every time and systematically over-collect
        # 'left' over 'right'.
        self._rng.shuffle(alternatives)
        for action in alternatives[:room]:
            self._fork(parent, action)

    def _fork(self, parent, action):
        snapshot = parent.env.snapshot()
        env = self._pool.pop()
        env.restore_from_snapshot(snapshot, parent.time_step)
        stream = next(self._next_stream)
        # A fork is the same episode taking a different action, so it inherits the parent's shield
        # flag: "guided end to end or not at all" holds for the whole trajectory tree.
        child = _Branch(env, stream, parent.time_step, parent.policy_state, parent.guided,
                        is_main=False, pending_action=action)
        self._buffer.fork_stream(parent.stream, stream)
        self._schedule.append(child)
        self._counters['forks'] += 1

    def _retire(self, branch):
        self._buffer.close_stream(branch.stream)
        self._pool.append(branch.env)
        self._counters['retired'] += 1

    # ------------------------------------------------------------------ the shield's flag

    def _apply_guided(self, branch):
        """Points the shield's per-episode flag at this branch before the policy reads it.

        `ShieldedEpsilonGreedyPolicy` keeps one Variable for the whole policy, which was sound while
        a single environment stepped it. With branches interleaved, the main line starting a new
        episode would silently reassign the flag under every live branch — so each branch carries its
        own value and writes it back here, extending "guided end to end or not at all" from an
        episode to a whole trajectory tree.

        The `is_first` short-circuit is defensive only, and no test claims otherwise: the real
        policy's `_resolve_guided` redraws unconditionally on a FIRST step and ignores whatever the
        Variable held, so assigning first would be overwritten. It is kept so that this code does
        not silently start fighting the schedule if that redraw ever becomes conditional.
        """
        if self._guided_flag is None or bool(branch.time_step.is_first()):
            return
        self._guided_flag.assign(branch.guided)

    def _read_back_guided(self, branch):
        """Records what the policy drew, so this branch keeps it for the rest of its episode."""
        if self._guided_flag is None or not bool(branch.time_step.is_first()):
            return
        branch.guided = bool(self._guided_flag.numpy())
