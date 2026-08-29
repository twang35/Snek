"""Game rules, rewards and paths. **No pygame** — see `env/render.py` for anything drawable.

The split is load-bearing rather than tidy: `vectorized/` and the whole eval path import this
module and must not pull in pygame, because `pygame.init()` opens a real CoreAudio stream per
process (10 idle workers drove `coreaudiod` to 15% CPU in snek2).

Everything tunable is read from the environment exactly once, at import, and **nothing here is
mutated at runtime.** Eval shards are subprocesses that inherit the environment, so an env var
reaches them where an assignment into this module would not.
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _flag(name, default='0'):
    return os.environ.get(name, default) not in ('0', '', 'false', 'False')


def _num(name, default, cast=float):
    return cast(os.environ.get('SNEK_' + name, default))


DEBUG_LOGGING = _flag('SNEK_DEBUG')

# ----------------------------------------------------------------- the board
GRID_LENGTH = 9                       # inclusive, so the playable board is 10x10
SCREENTILES = (GRID_LENGTH, GRID_LENGTH)
START_TILE = (5, 3)
START_SEGMENTS = 4

PERFECT_SCORE = (SCREENTILES[0] + 1) * (SCREENTILES[1] + 1)   # 100 cells
# Highest food-eaten count reachable before a perfect game triggers. A score of 95 *is* a filled
# board; that has been misread as a near-miss.
MAX_POSSIBLE_SCORE = PERFECT_SCORE - START_SEGMENTS - 1

# Steps the snake may go without food: 10 per segment, floored so a short snake can cross the
# board and capped so a long one cannot stall forever. The cap is what makes the starve
# observation go flat for every length from 50 up.
STARVE_SIZE_MULTIPLIER = 10
MIN_STARVE_BUDGET = 100
MAX_STARVE_BUDGET = 500

# Design cap for normalising lg(open regions) into [0, 1]. Not a game rule — nothing clamps the
# true count, so a rare board runs slightly past 1.0 rather than losing information the way
# clamping the raw count would. There is no closed form; two independent methods (a heuristic
# swept over 422,608 candidate moves, and a hand-built adversarial comb body) both topped out at
# 13, so 16 is headroom over either.
MAX_GROUPS_FOR_SCALE = 16

# ----------------------------------------------------------------- actions
DIRECTIONS = ['left', 'right', 'up', 'down']
MOVE_VECTORS = {'left': (-1, 0), 'right': (1, 0), 'up': (0, -1), 'down': (0, 1)}

# **Relative turns, not compass directions.** Everything "per action" in the observation is
# ordered by this list.
ACTIONS = ['left', 'right', 'forward']
NUM_ACTIONS = len(ACTIONS)
ACTION_INDEX_TO_NAME = dict(enumerate(ACTIONS))

CURRENT_DIRECTION_MAPS = {
    'left':  {'forward': 'left',  'left': 'down',  'right': 'up'},
    'right': {'forward': 'right', 'left': 'up',    'right': 'down'},
    'up':    {'forward': 'up',    'left': 'left',  'right': 'right'},
    'down':  {'forward': 'down',  'left': 'right', 'right': 'left'},
}

# ----------------------------------------------------------------- observation
OBS_LEN = 30

# **Bump this whenever the vector's *meaning* changes, even at constant length.** A checkpoint
# restores whenever the length matches and nothing checks the values still mean what they meant:
# snek2 repurposed two indices at constant length on 2026-08-02 and every hall-of-fame checkpoint
# restored silently and played like a beginner, 90.3% to scoring 0, 0, 1.
#
# `b09c616` is the snek2 commit this layout comes from, kept verbatim so a converted snek2
# champion's `arch.json` matches. See docs/environment.md for the index layout.
OBS_ERA = 'b09c616'

# The observation's blocks, in order, as `(name, width)`. The sum is the vector length and each
# entry's offset is its index range — so this table *is* the layout in docs/environment.md.
#
# **It lives here, beside `OBS_LEN`, and not in `env/scalar_env.py` where it started.** Two reasons.
# Keeping the length and the blocks that sum to it in one module is what lets the consistency check
# below fire at the earliest possible import rather than whenever something happens to touch the
# scalar env. And `dqn/agent.py` needs one block's range for the exploration shield: `scalar_env`
# imports `env.game` and therefore pygame, which `dqn/` may not, so reading the layout from there
# would have broken the layering invariant `tests/test_module_layering.py` asserts.
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



def _parse_zero_obs(raw):
    """`SNEK_ZERO_OBS=26-29,12` -> {12, 26, 27, 28, 29}. Indices and inclusive ranges."""
    if not raw:
        return frozenset()
    out = set()
    for part in (p.strip() for p in raw.split(',')):
        if not part:
            continue
        if '-' in part.lstrip('-'):
            low, high = part.split('-', 1)
            out.update(range(int(low), int(high) + 1))
        else:
            out.add(int(part))
    return frozenset(out)


# Observation indices forced to 0.0, for ablations. **Zeroes rather than deletes**, so the vector
# length is unchanged, an ablated arm and a full arm share one spec, and their checkpoints stay
# mutually loadable — deleting a block makes the comparison one between two environments.
#
# A zeroed input is not a deleted one: the network keeps weights for it with nothing to learn
# from. That is the right null for "does this signal help", and it is the `game_over` situation of
# invariant 3 — so never repurpose an index that spent a run zeroed.
ZERO_OBS_INDICES = _parse_zero_obs(os.environ.get('SNEK_ZERO_OBS'))

# ----------------------------------------------------------------- rewards
#
# **A reward is a sum of terms**, so nothing may identify an outcome by comparing against one of
# these. `is_perfect_score(score)` is the single definition of a win.
FOOD_REWARD = 1.0
DEATH_REWARD = -5.0
STARVE_REWARD = -0.5

# **Coupled to DISCOUNT and not independently tunable.** A terminal reward is a potential: with
# `k` steps per meal, progress only raises value when `W > 1/(1 - gamma^k)`, which is 34-58 at
# this game's numbers and gamma=0.9975. snek2 cut it to 10, missed by 3-6x, and the agents
# correctly learned to avoid finishing.
DEFAULT_PERFECT_GAME_REWARD = 100.0
PERFECT_GAME_REWARD = _num('PERFECT_GAME_REWARD', DEFAULT_PERFECT_GAME_REWARD)

# Subtracted on an ordinary move that *increases* Manhattan distance to the food — a penalty for
# moving away rather than a bonus for approaching, which are the same up to a constant. 0.0 is a
# clean ablation: the step still computes the distance and subtracts exactly 0.0, so control flow
# and the food stream are untouched. It does shift `avg_reward`, which the bootstrap epsilon phase
# thresholds on.
DEFAULT_FOOD_DISTANCE_REWARD = 0.001
FOOD_DISTANCE_REWARD = _num('FOOD_DISTANCE_REWARD', DEFAULT_FOOD_DISTANCE_REWARD)

# Potential-based shaping: the coefficient `c` in `F = c * (gamma * Phi(s') - Phi(s))`.
#
# `c` is not free. The discounted shaping telescopes to exactly `-c * Phi(s0)`, so `c` is
# invisible in the return and has to be calibrated against the per-transition reward instead:
# genuine potential flips run 2.5-3.6 per meal for a struggling policy, and holding the budget at
# ~25% of FOOD_REWARD per meal gives 0.10.
#
# The gate is a snake length below which the potential is identically 0. It keeps the invariance
# — the theorem holds for any bounded function of state — and makes the telescope exactly 0, since
# an opening board is below it.
DEFAULT_CHASE_SAFE_SHAPING = 0.0
DEFAULT_CHASE_SAFE_GATE = 85
CHASE_SAFE_SHAPING = _num('CHASE_SAFE_SHAPING', DEFAULT_CHASE_SAFE_SHAPING)
CHASE_SAFE_GATE = _num('CHASE_SAFE_GATE', DEFAULT_CHASE_SAFE_GATE, int)

DEFAULT_FREE_SPACE_SHAPING = 0.0
DEFAULT_FREE_SPACE_GATE = 85
FREE_SPACE_SHAPING = _num('FREE_SPACE_SHAPING', DEFAULT_FREE_SPACE_SHAPING)
FREE_SPACE_GATE = _num('FREE_SPACE_GATE', DEFAULT_FREE_SPACE_GATE, int)

# ----------------------------------------------------------------- checkpoints
#
# Below this average score no checkpoint is written. 40 is well clear of anything useful: across
# 232 snek2 checkpoints measured at 100 episodes, every one that reached 30% perfect games scored
# at least 49.8 on max(avg_score, trailing).
#
# snek3 keeps this for disk rather than for eviction — a policy-only checkpoint is ~45 KB and
# there is no `max_to_keep` rotation, which is what destroyed snek2's b5c 17.0% peak.
MIN_CHECKPOINT_SCORE = _num('MIN_CHECKPOINT_SCORE', 40.0)

# ----------------------------------------------------------------- paths
POLICY_DIR = os.path.join(_ROOT, 'savedPolicies')
RUNS_DIR = os.path.join(_ROOT, 'runs')
EVALS_DIR = os.path.join(_ROOT, 'evals')
HOF_DIR = os.path.join(_ROOT, 'hallOfFame')
GIFS_DIR = os.path.join(_ROOT, 'gifs')
