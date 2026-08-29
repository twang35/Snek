"""Every constant the vectorised env needs, taken from `snake_constants` rather than copied.

**Importing is the point.** Half of these are read from `SNEK_*` environment variables — the reward
terms, both shaping coefficients and their gates, `PERFECT_GAME_REWARD` — so a hardcoded copy would
not merely *drift*, it would be wrong the first time an arm was measured with a different reward
config, silently and in a field (`avg_reward`) nobody checks by hand. `tests/test_vectorized_config.py`
pins every name here against `snake_constants` so a rename fails loudly instead of falling back to a
default.

The cost is one `pygame` import at module load, because `snake_constants` builds a `pygame.Rect`.
That is import-time only and never touches the step loop, which is the property that matters — and
the eval driver imports TensorFlow anyway, which is far heavier. Nothing here creates a `Game`, so no
display is opened; callers still set `SDL_VIDEODRIVER`/`SDL_AUDIODRIVER` to `dummy` first, per the
project's no-audio rule.

The geometry assertions at the bottom are the load-bearing part. The bitboard packs a row stride of
`GRID` into `uint64` words and relies on the wall ring to make the +-1 shifts safe, so a change to
`GRID_LENGTH` has to fail here rather than produce a subtly wrong flood fill.
"""

import math
import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import snake_constants as _sc

# `OBS_ERA` is deliberately *not* imported here. It lives in `snake_environment`, which pulls in
# tf_agents and TensorFlow, and this module is meant to be importable with nothing but numpy and
# pygame so the env can be tested and benchmarked without a 3-second TF import. The eval driver
# imports the era where it belongs — next to `policy_arch.assert_restorable`, which is the only
# thing that acts on it.

# ---------------------------------------------------------------- geometry

# Playable board is SCREENTILES + 1 per side, stored padded by a one-cell wall ring, so the row
# stride is playable + 2. `count_groups` relies on that ring: shifting a bitboard by one crosses a
# row boundary, and the first and last column being wall is what keeps an open cell from wrapping.
PLAY = _sc.SCREENTILES[0] + 1                 # 10
GRID = PLAY + 2                               # 12, the padded row stride
NCELL = GRID * GRID                           # 144

# Bitboard width, rounded up to whole uint64 words so `np.packbits(...).view(np.uint64)` yields an
# exact number of words with no ragged tail to mask around.
WORDS = (NCELL + 63) // 64                    # 3
PAD = WORDS * 64                              # 192

# Circular body buffer. The real snake can reach PERFECT_SCORE cells; the next power of two above
# that keeps the modulo cheap and leaves headroom for a snapshot restored mid-growth.
CAP = 128

PERFECT_SCORE = _sc.PERFECT_SCORE             # 100
START_SEGMENTS = _sc.START_SEGMENTS           # 4
START_TILE = tuple(_sc.START_TILE)            # (5, 3)
MAX_POSSIBLE_SCORE = _sc.MAX_POSSIBLE_SCORE   # 95

# ---------------------------------------------------------------- game rules

MIN_STARVE_BUDGET = _sc.MIN_STARVE_BUDGET
MAX_STARVE_BUDGET = _sc.MAX_STARVE_BUDGET
STARVE_MULT = _sc.MAX_STEPS_BEFORE_STARVE_SIZE_MULTIPLIER
MAX_GROUPS_FOR_SCALE = _sc.MAX_GROUPS_FOR_SCALE

# Both spelled with math.log2 rather than the game's log2plus1, so they do not depend on import
# order — same reason state_helpers spells them out.
STARVE_OBS_SCALE = math.log2(MAX_STARVE_BUDGET + 1)
GROUPS_OBS_SCALE = math.log2(MAX_GROUPS_FOR_SCALE + 1)

# ---------------------------------------------------------------- rewards

FOOD_REWARD = _sc.FOOD_REWARD
DEATH_REWARD = _sc.DEATH_REWARD
STARVE_REWARD = _sc.STARVE_REWARD
PERFECT_GAME_REWARD = _sc.PERFECT_GAME_REWARD
FOOD_DISTANCE_REWARD = _sc.FOOD_DISTANCE_REWARD

CHASE_SAFE_SHAPING = _sc.CHASE_SAFE_SHAPING
CHASE_SAFE_GATE = _sc.CHASE_SAFE_GATE
FREE_SPACE_SHAPING = _sc.FREE_SPACE_SHAPING
FREE_SPACE_GATE = _sc.FREE_SPACE_GATE

ZERO_OBS_INDICES = tuple(_sc.ZERO_OBS_INDICES)

OBS_LEN = 30

# ---------------------------------------------------------------- guards

# A change to the board size invalidates the packed-bitboard layout below, and the failure would be
# a subtly wrong flood fill rather than an exception. Fail at import instead.
if _sc.SCREENTILES[0] != _sc.SCREENTILES[1]:
    raise ImportError(
        'the vectorised env assumes a square board; SCREENTILES is {0}'.format(_sc.SCREENTILES))
if PERFECT_SCORE != PLAY * PLAY:
    raise ImportError(
        'PERFECT_SCORE {0} != playable area {1}x{1}; the vectorised env derives one from the '
        'other'.format(PERFECT_SCORE, PLAY))
if NCELL > PAD:
    raise ImportError('NCELL {0} exceeds the packed width {1}'.format(NCELL, PAD))


def describe():
    """One line naming the reward/shaping config, for the eval driver's report header.

    Worth printing rather than assuming: these come from the environment, so two runs of the same
    checkpoint can legitimately produce different `avg_reward`.
    """
    return ('grid {0}x{0}, max score {1}, food {2}, death {3}, starve {4}, perfect {5}, '
            'dist {6}, chase_safe c={7} gate={8}, free_space c={9} gate={10}'.format(
                PLAY, MAX_POSSIBLE_SCORE, FOOD_REWARD, DEATH_REWARD, STARVE_REWARD,
                PERFECT_GAME_REWARD, FOOD_DISTANCE_REWARD, CHASE_SAFE_SHAPING, CHASE_SAFE_GATE,
                FREE_SPACE_SHAPING, FREE_SPACE_GATE))
