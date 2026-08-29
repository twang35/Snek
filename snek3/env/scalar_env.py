"""One game behind a plain reset/step API. No framework, no TF-Agents.

**The signature matches `vectorized.vec_env.VecSnake.step` deliberately** — `(obs, reward, done,
info)`, with the same `info` keys — so the two implementations can be driven by one loop and
compared elementwise. This env is the parity *reference*: if it and `VecSnake` disagree, `VecSnake`
is wrong.

**`done` is the bootstrap mask, and there is no `discount` field.** snek2 returned a TimeStep whose
terminal discount had to be zeroed by hand, and for a long time it was not: every episode's final
transition trained toward `reward + 0.9975 * V(terminal)` instead of toward `reward`, quietly
discounting the death penalty by whatever value the network happened to assign a terminal
observation. An algorithm here computes `target = r + gamma * (1 - done) * V(s')` itself, which
makes that bug unrepresentable rather than fixed by a constant.

`discount` is still threaded into `Game`, because the potential-based shaping term needs the agent's
gamma to compute `c * (gamma * Phi(s') - Phi(s))`.
"""

import numpy as np

from env import constants
from env.constants import ACTION_INDEX_TO_NAME, OBS_ERA, OBS_LEN
from env.game import Game

# The observation's blocks, in order, as `(name, width)`. The sum is the vector length and each
# entry's offset is its index range — so this table *is* the layout in docs/environment.md.
#
# Kept as data rather than as arithmetic inside a spec function so a test can pin each block to its
# range by comparing against the function in `env.observations` that produces it. snek2's earlier
# test compared against hardcoded literals and an ordering bug passed it, because two blocks
# coincidentally held the same values.
OBS_BLOCKS = (
    ('food', 6),                 # 0-5    [is closer, 1/(distance+1)] per action
    ('body_and_wall', 3),        # 6-8    is the move safe. The only place legality is stated
    ('head_with_tail_groups', 6),  # 9-14 [can reach tail, lg(open regions)] per action
    ('safe_to_chase_food', 3),   # 15-17  head, food and tail in one region
    ('perfect_game_move', 3),    # 18-20  nonzero in <0.03% of states; not meaningfully trained
    ('starve_budget', 1),        # 21
    ('board_fill', 1),           # 22     rank 1 of 30 by saliency in every snek2 arm measured
    ('hugging_wall', 3),         # 23-25
    ('not_following_tail', 3),   # 26-28  a *fatal* move also reads 1 here
    ('food_space', 1),           # 29     sits at 1 in ~99.95% of states
)


def observation_length():
    return sum(width for _, width in OBS_BLOCKS)


def block_ranges():
    """`{name: (start, stop)}`, stop exclusive."""
    out, at = {}, 0
    for name, width in OBS_BLOCKS:
        out[name] = (at, at + width)
        at += width
    return out


if observation_length() != OBS_LEN:
    raise ImportError('OBS_BLOCKS sums to {0} but OBS_LEN is {1}'.format(
        observation_length(), OBS_LEN))


class SnakeEnv:
    """A single game. `reset()` then `step(action)` until `done`."""

    def __init__(self, discount=1.0, display=False, limit_fps=False, policy_name=''):
        self._game = Game(display=display, limit_fps=limit_fps, policy_name=policy_name,
                          discount=discount)
        self.obs_era = OBS_ERA
        self.high_score = 0
        self.total_steps = 0

    # ------------------------------------------------------------------ specs
    @staticmethod
    def observation_spec():
        return {'shape': (OBS_LEN,), 'dtype': np.float32}

    @staticmethod
    def action_spec():
        return {'shape': (), 'dtype': np.int64, 'minimum': 0,
                'maximum': constants.NUM_ACTIONS - 1}

    # ------------------------------------------------------------------ loop
    def reset(self):
        self._game.reset()
        return self._observation()

    def step(self, action):
        """`action` is an index into `constants.ACTIONS` — a relative turn, not a heading."""
        if self._game.finished or self._game.perfect_game:
            return self.reset(), 0.0, False, self._info(done=False)

        self.total_steps += 1
        done, reward = self._game.step(ACTION_INDEX_TO_NAME[int(action)])
        self._game.render()

        if self._game.current_score > self.high_score:
            self.high_score = self._game.current_score

        return self._observation(), float(reward), bool(done), self._info(done)

    def _observation(self):
        return np.asarray(self._game.get_observation(), dtype=np.float32)

    def _info(self, done):
        game = self._game
        return {'done': bool(done),
                'score': int(game.current_score),
                'steps': int(game.current_step),
                'perfect': bool(game.perfect_game),
                'starved': bool(game.starved),
                'died': bool(game.finished and not game.starved)}

    # ------------------------------------------------------------------ state
    def get_score(self):
        return self._game.current_score

    def snake_length(self):
        """Head plus segments — an integer, so a length gate is an exact `>=`.

        Observation index 22 carries the same number scaled by `PERFECT_SCORE`, but reading it from
        the game keeps a gate working under `SNEK_ZERO_OBS` and needs no scaling constant.
        """
        return len(self._game.snake_group)

    def snapshot(self):
        return self._game.snapshot()

    def restore_from_snapshot(self, snapshot):
        """Make this env a copy of whatever `snapshot` describes, mid-episode.

        A restored game produces a byte-identical observation, which `tests/test_game_snapshot.py`
        asserts directly — that exactness is what the forking collector depends on.
        """
        self._game.restore_snapshot(snapshot)
        return self._observation()

    def set_display(self, enabled):
        self._game.set_display(enabled)

    @property
    def game(self):
        """The underlying `Game`, for the renderer and the recorder only."""
        return self._game
