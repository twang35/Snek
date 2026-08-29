"""Pins `vectorized/` against snek2's own constants and lookup tables.

The vectorised env compiles the reference's string-keyed direction maps into integer arrays and its
padded 2-D grid into a flat bitboard. Those translations are the kind of thing that is right when
written and silently wrong after someone changes `GRID_LENGTH` or reorders `DIRECTIONS` — so every
one of them is asserted here against the thing it was derived from, rather than trusted.

This is the same pattern as `runner.EVAL_RELEVANT_ENV`, which carries a test that fails if it drifts
from `eval_wave`'s copy.
"""

import numpy as np

import snake_constants as sc
import state_helpers
from vectorized import config as C
from vectorized import vec_env as V


def test_reward_and_rule_constants_come_from_snake_constants():
    """Every value must be the *same object's* value, not a copy that happens to match today.

    Half of these are read from `SNEK_*` environment variables, so a hardcoded copy would not merely
    drift — it would be wrong the first time an arm was measured under different shaping, silently,
    in a field nobody checks by hand.
    """
    assert C.FOOD_REWARD == sc.FOOD_REWARD
    assert C.DEATH_REWARD == sc.DEATH_REWARD
    assert C.STARVE_REWARD == sc.STARVE_REWARD
    assert C.PERFECT_GAME_REWARD == sc.PERFECT_GAME_REWARD
    assert C.FOOD_DISTANCE_REWARD == sc.FOOD_DISTANCE_REWARD
    assert C.CHASE_SAFE_SHAPING == sc.CHASE_SAFE_SHAPING
    assert C.CHASE_SAFE_GATE == sc.CHASE_SAFE_GATE
    assert C.FREE_SPACE_SHAPING == sc.FREE_SPACE_SHAPING
    assert C.FREE_SPACE_GATE == sc.FREE_SPACE_GATE
    assert C.MIN_STARVE_BUDGET == sc.MIN_STARVE_BUDGET
    assert C.MAX_STARVE_BUDGET == sc.MAX_STARVE_BUDGET
    assert C.STARVE_MULT == sc.MAX_STEPS_BEFORE_STARVE_SIZE_MULTIPLIER
    assert C.MAX_GROUPS_FOR_SCALE == sc.MAX_GROUPS_FOR_SCALE
    assert C.PERFECT_SCORE == sc.PERFECT_SCORE
    assert C.MAX_POSSIBLE_SCORE == sc.MAX_POSSIBLE_SCORE
    assert C.START_SEGMENTS == sc.START_SEGMENTS
    assert C.START_TILE == tuple(sc.START_TILE)
    assert C.ZERO_OBS_INDICES == tuple(sc.ZERO_OBS_INDICES)


def test_observation_scales_match_state_helpers():
    assert C.STARVE_OBS_SCALE == state_helpers.STARVE_OBS_SCALE
    assert C.GROUPS_OBS_SCALE == state_helpers.GROUPS_OBS_SCALE


def test_obs_len_matches_the_environments_own_spec():
    """The one assertion here that reaches into tf_agents, because it is the definitive pin.

    `observation_spec` reads nothing off `self`, so it can be called unbound — which avoids building
    a `Game` (and therefore pygame) just to ask how long the vector is.
    """
    from snake_environment import SnakeEnvironment
    spec_len = int(SnakeEnvironment.observation_spec(None).shape[0])
    assert C.OBS_LEN == spec_len, 'vectorized expects {0}, spec says {1}'.format(C.OBS_LEN, spec_len)


def test_turn_table_matches_current_direction_maps():
    """`TURN[dir][action]` must reproduce the reference's string-keyed relative-turn map."""
    for d_index, direction in enumerate(sc.DIRECTIONS):
        for a_index, action in enumerate(sc.ACTIONS):
            expected = sc.CURRENT_DIRECTION_MAPS[direction][action]
            got = sc.DIRECTIONS[V.TURN[d_index, a_index]]
            assert got == expected, (
                'facing {0}, turning {1}: table says {2}, reference says {3}'.format(
                    direction, action, got, expected))


def test_delta_matches_move_vectors_under_the_flat_layout():
    """A flat-index delta must equal what moving by the reference's (dx, dy) does to `flat`."""
    for d_index, direction in enumerate(sc.DIRECTIONS):
        dx, dy = sc.MOVE_VECTORS[direction]
        for x in range(1, sc.SCREENTILES[0]):
            for y in range(1, sc.SCREENTILES[1]):
                assert V.flat(x + dx, y + dy) - V.flat(x, y) == V.DELTA[d_index]


def test_flat_and_unflat_round_trip_over_the_whole_board():
    for x in range(-1, sc.SCREENTILES[0] + 2):
        for y in range(-1, sc.SCREENTILES[1] + 2):
            f = V.flat(x, y)
            assert 0 <= f < C.NCELL
            assert V.unflat(f) == (x, y)


def test_playable_matches_out_of_bounds():
    """`PLAYABLE` is the bitboard's only notion of the wall, so it must equal the reference's."""
    for x in range(-1, sc.SCREENTILES[0] + 2):
        for y in range(-1, sc.SCREENTILES[1] + 2):
            assert bool(V.PLAYABLE[V.flat(x, y)]) == (not state_helpers.out_of_bounds((x, y))), (
                'disagreement at tile {0}'.format((x, y)))


def test_the_wall_ring_is_intact_which_is_what_makes_the_shifts_safe():
    """A dilation shifts by +-1 across a row boundary, and only the wall ring stops a wrap.

    If the first or last column of the padded grid were ever playable, an open cell could shift into
    the next row and the flood fill would leak between rows — silently, as a wrong observation.
    """
    grid = V.PLAYABLE[:C.NCELL].reshape(C.GRID, C.GRID)
    assert not grid[0].any() and not grid[-1].any(), 'top/bottom wall ring is not clear'
    assert not grid[:, 0].any() and not grid[:, -1].any(), 'left/right wall ring is not clear'
    assert grid.sum() == C.PERFECT_SCORE, 'playable area is {0}, expected {1}'.format(
        grid.sum(), C.PERFECT_SCORE)


def test_packed_width_is_whole_words_so_packbits_has_no_ragged_tail():
    assert C.PAD % 64 == 0
    assert C.PAD == C.WORDS * 64
    assert C.NCELL <= C.PAD
    probe = np.zeros((2, C.PAD), dtype=bool)
    probe[0, 0] = probe[0, C.NCELL - 1] = probe[1, 65] = True
    packed = np.packbits(probe, axis=1, bitorder='little').view(np.uint64)
    assert packed.shape == (2, C.WORDS)
    assert packed[0, 0] & np.uint64(1)
    assert packed[1, 1] & np.uint64(2)


def test_body_buffer_can_hold_a_full_board():
    """`CAP` bounds the circular body buffer; a perfect game fills `PERFECT_SCORE` cells."""
    assert C.CAP >= C.PERFECT_SCORE
