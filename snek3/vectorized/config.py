"""Every constant the vectorised env needs, imported from `env.constants` rather than copied.

**Importing is the point.** Half of these are read from `SNEK_*` environment variables — the reward
terms, both shaping coefficients and their gates — so a hardcoded copy would not merely *drift*, it
would be wrong the first time an arm was measured with a different reward config, silently and in a
field (`avg_reward`) nobody checks by hand. `tests/test_vec_config.py` pins every name here against
`env.constants` so a rename fails loudly instead of falling back to a default.

Unlike snek2's equivalent this costs no pygame import, because `env.constants` has none.

The geometry guards at the bottom are the load-bearing part. The bitboard packs a row stride of
`GRID` into `uint64` words and relies on the wall ring to make the +-1 shifts safe, so a change to
the board size has to fail here rather than produce a subtly wrong flood fill.
"""

import math

from env import constants as _c

# ---------------------------------------------------------------- geometry

# The playable board is `SCREENTILES + 1` per side, stored padded by a one-cell wall ring, so the
# row stride is playable + 2. The connectivity block relies on that ring: shifting a bitboard by
# one crosses a row boundary, and the first and last column being wall is what keeps an open cell
# from wrapping.
PLAY = _c.SCREENTILES[0] + 1                  # 10
GRID = PLAY + 2                               # 12, the padded row stride
NCELL = GRID * GRID                           # 144

# Bitboard width, rounded up to whole uint64 words so `np.packbits(...).view(np.uint64)` yields an
# exact number of words with no ragged tail to mask around.
WORDS = (NCELL + 63) // 64                    # 3
PAD = WORDS * 64                              # 192

# Circular body buffer. The real snake can reach PERFECT_SCORE cells; the next power of two above
# that keeps the modulo cheap and leaves headroom for a snapshot restored mid-growth.
CAP = 128

PERFECT_SCORE = _c.PERFECT_SCORE              # 100
START_SEGMENTS = _c.START_SEGMENTS            # 4
START_TILE = tuple(_c.START_TILE)             # (5, 3)
MAX_POSSIBLE_SCORE = _c.MAX_POSSIBLE_SCORE    # 95

# ---------------------------------------------------------------- game rules

MIN_STARVE_BUDGET = _c.MIN_STARVE_BUDGET
MAX_STARVE_BUDGET = _c.MAX_STARVE_BUDGET
STARVE_MULT = _c.STARVE_SIZE_MULTIPLIER
MAX_GROUPS_FOR_SCALE = _c.MAX_GROUPS_FOR_SCALE

# Spelled with `math.log2` rather than the game's `log2plus1` so they do not depend on import
# order — same reason `env.observations` spells them out.
STARVE_OBS_SCALE = math.log2(MAX_STARVE_BUDGET + 1)
GROUPS_OBS_SCALE = math.log2(MAX_GROUPS_FOR_SCALE + 1)

# ---------------------------------------------------------------- rewards

FOOD_REWARD = _c.FOOD_REWARD
DEATH_REWARD = _c.DEATH_REWARD
STARVE_REWARD = _c.STARVE_REWARD
PERFECT_GAME_REWARD = _c.PERFECT_GAME_REWARD
FOOD_DISTANCE_REWARD = _c.FOOD_DISTANCE_REWARD

CHASE_SAFE_SHAPING = _c.CHASE_SAFE_SHAPING
CHASE_SAFE_GATE = _c.CHASE_SAFE_GATE
FREE_SPACE_SHAPING = _c.FREE_SPACE_SHAPING
FREE_SPACE_GATE = _c.FREE_SPACE_GATE

ZERO_OBS_INDICES = tuple(_c.ZERO_OBS_INDICES)

OBS_LEN = _c.OBS_LEN
NUM_ACTIONS = _c.NUM_ACTIONS

# ---------------------------------------------------------------- guards

if _c.SCREENTILES[0] != _c.SCREENTILES[1]:
    raise ImportError(
        'the vectorised env assumes a square board; SCREENTILES is {0}'.format(_c.SCREENTILES))
if PERFECT_SCORE != PLAY * PLAY:
    raise ImportError(
        'PERFECT_SCORE {0} != playable area {1}x{1}; the vectorised env derives one from the '
        'other'.format(PERFECT_SCORE, PLAY))
if NCELL > PAD:
    raise ImportError('NCELL {0} exceeds the packed width {1}'.format(NCELL, PAD))


def describe():
    """One line naming the reward/shaping config, for an eval report header.

    Worth printing rather than assuming: these come from the environment, so two runs of the same
    checkpoint can legitimately produce different `avg_reward`.
    """
    return ('grid {0}x{0}, max score {1}, food {2}, death {3}, starve {4}, perfect {5}, '
            'dist {6}, chase_safe c={7} gate={8}, free_space c={9} gate={10}'.format(
                PLAY, MAX_POSSIBLE_SCORE, FOOD_REWARD, DEATH_REWARD, STARVE_REWARD,
                PERFECT_GAME_REWARD, FOOD_DISTANCE_REWARD, CHASE_SAFE_SHAPING, CHASE_SAFE_GATE,
                FREE_SPACE_SHAPING, FREE_SPACE_GATE))
