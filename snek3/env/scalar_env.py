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

# The observation layout lives in `env.constants`, beside `OBS_LEN`, so the length and the blocks
# that sum to it cannot drift apart and so `dqn/` can read a block range without importing pygame.
# Re-exported here because this module is where the observation *spec* is asked for.
OBS_BLOCKS = constants.OBS_BLOCKS
observation_length = constants.observation_length
block_ranges = constants.block_ranges


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
