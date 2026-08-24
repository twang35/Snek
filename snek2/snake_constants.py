import os
import pygame

# Verbose logging. Off by default: a 2M-step run printed a loss line every 200 steps, five
# lines per eval, and a banner per perfect game, which is tens of thousands of lines nobody
# reads. `SNEK_DEBUG=1` restores all of it verbatim for when a run is actually being
# debugged. Read from the environment rather than passed around because the noisiest prints
# live in the game and environment classes, which run in separate worker processes.
DEBUG_LOGGING = os.environ.get('SNEK_DEBUG', '0') not in ('0', '', 'false', 'False')

# screen that game appears on 0 or 1
SCREEN_TO_DISPLAY = 0

FOOD_REWARD = 1.0
# Distance shaping, subtracted on any ordinary move that increases the head's distance to the
# food — so it is a penalty for moving away rather than a bonus for approaching, which are the
# same thing up to a constant. See Snake.step() for the two cases it deliberately skips.
#
# Read from the environment so an arm can turn it off without editing files, like every other
# hyperparameter. **Not** through `tuned()` in snek2.py: this value is consumed inside
# `Snake.step`, which runs in the parallel env worker processes, and `from snake_constants import
# *` binds a copy at import — an assignment in the parent would never reach a worker. Same reason
# TILE_PIXELS and ZERO_OBS_INDICES are read here. `run_config` still records it, so
# runs/<policy>.md shows what the arm actually ran with.
#
# **0.0 is the ablation, and it is clean.** The step keeps computing the distance and subtracts
# exactly 0.0, so control flow, the food stream and every observation are untouched — the only
# difference is the reward. Note it shifts `avg_reward`, which is what the bootstrap epsilon phase
# thresholds on, so a shaping-off arm reaches the refinement phase slightly earlier for
# measurement reasons on top of any real effect. `avg_score` is a food count and is unaffected.
DEFAULT_FOOD_DISTANCE_REWARD = 0.001
FOOD_DISTANCE_REWARD = float(os.environ.get('SNEK_FOOD_DISTANCE_REWARD',
                                            DEFAULT_FOOD_DISTANCE_REWARD))
# Potential-based shaping on "are the head, the food and the tail in one region" — the coefficient
# `c` in `F = c * (gamma * Phi(s') - Phi(s))`. Read from the environment for the same reason
# FOOD_DISTANCE_REWARD is: the term is consumed inside `Snake.step`, which runs in the parallel env
# worker processes, and `from snake_constants import *` binds a copy at import, so an assignment in
# the parent would never reach a worker.
#
# **0.0 is off and is a clean ablation.** The whole block is skipped, and `count_groups` draws no
# randomness, so skipping it cannot shift the food stream — at 0.0 the reward is bit-identical to a
# build without the term. Default 0.0, so every existing arm and every historical number is
# unaffected.
#
# **`c` is not a free parameter.** Measured 2026-08-14: the discounted shaping telescopes to exactly
# `-c * Phi(s0)`, so `c` is invisible in the return and has to be calibrated against the
# per-transition reward instead. Genuine Phi flips run 2.5-3.6 per meal for a struggling policy, and
# holding the budget at ~25% of FOOD_REWARD per meal gives 0.10. See
# ../plans/chase-safe-reward-shaping.md.
DEFAULT_CHASE_SAFE_SHAPING = 0.0
CHASE_SAFE_SHAPING = float(os.environ.get('SNEK_CHASE_SAFE_SHAPING',
                                          DEFAULT_CHASE_SAFE_SHAPING))
# Snake length below which the potential is identically 0, so no shaping is paid there at all.
# Defaults to 85 because that is the variant Phase 0 selected: a record policy spends only ~10.8%
# of its steps at or above it while a struggling one spends 41.2%, so the dose lands on the endgame
# and the early game — which needs no help — is left untouched. 85 also matches FORK_MIN_LENGTH, so
# the shaped transitions are the ones the forking collector already oversamples.
#
# **The gate keeps the invariance**: the theorem holds for any bounded function of state, and a
# length gate is one. It also makes the telescope exactly 0, since a length-5 opening board is below
# any gate, so `Phi(s0) = 0`.
#
# `SNEK_CHASE_SAFE_GATE=0` selects the ungated form (variant A in the plan). Irrelevant when
# CHASE_SAFE_SHAPING is 0.0, since the whole term is skipped.
DEFAULT_CHASE_SAFE_GATE = 85
CHASE_SAFE_GATE = int(os.environ.get('SNEK_CHASE_SAFE_GATE', DEFAULT_CHASE_SAFE_GATE))
# Potential-based shaping on "is the free space a single connected piece" — the coefficient `c` in
# `F = c * (gamma * Phi(s') - Phi(s))`, with Phi = 1 / (number of open regions), the tail cell freed
# first. Read from the environment for the same reason CHASE_SAFE_SHAPING is: it is consumed inside
# `Snake.step`, which runs in the parallel env workers, where a parent assignment never reaches a
# `from snake_constants import *` copy.
#
# **Why 1/count and not a size-weighted measure.** In a perfect game every open cell must eventually
# be filled, so a *single* permanently stranded cell loses the game — the number of pieces is the
# fatal quantity, not how the space is divided. `largest / total` reads 0.95 for a 19+1 split and so
# barely reacts to the first break; `1 / count` cliffs to 0.5 the moment the board stops being one
# piece, which is the intent.
#
# Complements CHASE_SAFE_SHAPING rather than replacing it. Chase-safety is a *local, binary* "can I
# eat this food and still reach my tail"; this is a *global* "has the board fragmented at all". They
# disagree on the cases that matter — a board can be one eat-trap from death while 98% of its free
# space is in one piece, and vice versa — so an arm may run both, and their PBRS terms simply add.
#
# **0.0 is off and is a clean ablation**, on the same argument as CHASE_SAFE_SHAPING: the block is
# skipped and `count_groups` draws no randomness, so the food stream is untouched. Default 0.0, so
# every existing arm and historical number is unaffected.
DEFAULT_FREE_SPACE_SHAPING = 0.0
FREE_SPACE_SHAPING = float(os.environ.get('SNEK_FREE_SPACE_SHAPING', DEFAULT_FREE_SPACE_SHAPING))
# Snake length below which this potential is identically 0, matching CHASE_SAFE_GATE's default of 85.
# Gated for the same reason and with the same invariance argument: a length gate is a bounded function
# of state, so the shaping stays policy-invariant, and a length-5 opening board sits below it, so
# Phi(s0) = 0 and the episode's discounted shaping telescopes to 0. `SNEK_FREE_SPACE_GATE=0` ungates.
DEFAULT_FREE_SPACE_GATE = 85
FREE_SPACE_GATE = int(os.environ.get('SNEK_FREE_SPACE_GATE', DEFAULT_FREE_SPACE_GATE))
DEATH_REWARD = -5.0  # maybe avoid deaths more?
STARVE_REWARD = -0.5
global PERFECT_GAME_REWARD
# What a filled board pays. Read from the environment for the same reason FOOD_DISTANCE_REWARD is —
# `Snake.step` pays it inside the parallel env worker processes, and `from snake_constants import *`
# binds a copy at import, so an assignment in the parent would never reach a worker.
#
# **Changing it changes the objective, not just the scale, and it changes the shape of the return.**
# At `gamma=0.9975` the discounted return from a state is `F + gamma^(T-t) * W`, where `F` is the
# remaining food discounted to now. `F` is largest at the *start* of an episode (~20.8, all 95 meals
# still to come) and smallest at the end (~3.6, four meals left). So at `W=100` the maximum return is
# just before the win (~102.6, measured 104.4) while at `W=10` it moves to the *opening* (~20.8) —
# `v_max` has to be re-derived rather than divided by 10, and this is why `SNEK_V_MAX` is not
# adjusted automatically here.
#
# It also rescales the urgency to finish: delaying the win 100 steps costs `W*(1 - 0.9975^100)`,
# so 22 reward at `W=100` and 2.2 at `W=10`. Endgame hunting speed is the measured elite-vs-mediocre
# discriminator, so a smaller `W` is a real change of what the agent is asked to do. See
# hyperparamTuning/findings.md.
#
# **Nothing identifies a perfect game by this value.** `state_helpers.is_perfect_score(score)` is the
# single definition and `tests/test_perfect_game_counting.py` has an `ast` tripwire that fails if a
# comparison against this constant reappears — which is what makes the knob safe to turn at all.
DEFAULT_PERFECT_GAME_REWARD = 100
PERFECT_GAME_REWARD = float(os.environ.get('SNEK_PERFECT_GAME_REWARD',
                                           DEFAULT_PERFECT_GAME_REWARD))
global PERFECT_GAME_WAIT_MS
PERFECT_GAME_WAIT_MS = 500

# Below this average score, an eval's checkpoint is not written at all. Set from
# SNEK_MIN_CHECKPOINT_SCORE in snek2.py. `max_to_keep` is a rolling window, so a dead arm
# that keeps training evicts the good checkpoints behind it — see the gate in training.py.
#
# 40 is well clear of anything useful: across 232 checkpoints measured at 100 episodes, every
# one that reached 30% perfect games had max(avg_score, trailing) of at least 49.8.
global MIN_CHECKPOINT_SCORE
MIN_CHECKPOINT_SCORE = 40.0
# Steps the snake may go without food: 10 per segment, floored at 100 so a short snake can cross
# the board, capped at 500 so a long one cannot stall forever. The floor and the cap were inline
# magic numbers inside steps_until_starve(); naming them matters because the cap is what made the
# old starve observation go flat for every length from 50 up.
MAX_STEPS_BEFORE_STARVE_SIZE_MULTIPLIER = 10
MIN_STARVE_BUDGET = 100
MAX_STARVE_BUDGET = 500

# Design cap for normalizing lg(open regions) into [0, 1], the same way MAX_STARVE_BUDGET does
# for the starve observation. Unlike the starve budget this is not a game rule - nothing clamps
# the true region count, and a rare board could exceed it, in which case the normalized value
# runs slightly past 1.0 rather than losing information the way clamping the raw count would.
#
# There is no closed-form maximum: the region count a single connected snake body can carve
# out of a 10x10 board depends on its exact shape, and no simple formula bounds it tightly.
# Measured instead. A heuristic player was swept across 180 episodes and 422,608 candidate
# moves (the same per-action lookahead group_obs itself computes) and never exceeded 13. A
# hand-built adversarial body - a comb of five full-height teeth, each notched twice to split
# its neighbouring corridor into three pieces - reached 13 with a 70-cell body, well under the
# ~100-cell board. Both independent methods topped out at the same number, so 16 is chosen for
# real headroom over either without compressing the range that matters for everyday values.
MAX_GROUPS_FOR_SCALE = 16

# Pixels per tile, and so the window size: the board is 10x10 tiles, so the historical 10
# gives a 100x100 window. That is small enough that macOS shows none of the title bar text,
# which is the whole problem when several watch.py windows are open at once — so watch.py
# raises it to 15 (a 150x150 window). Purely cosmetic: observations are built from tile
# positions, never pixels, so this cannot change what a policy sees or scores.
#
# Read from the environment rather than assigned by the caller, because `from snake_constants
# import *` binds a copy at import time — setting snake_constants.TILE_SIZE afterwards would
# never reach Snake.py — and because everything below is derived from it.
TILE_PIXELS = int(os.environ.get('SNEK_TILE_PIXELS', 10))
DISPLAY_SCALE = TILE_PIXELS / 10.0

# size of each grid
TILE_SIZE = (TILE_PIXELS, TILE_PIXELS)

# number of grid of the screen
# GRID_LENGTH = 15
GRID_LENGTH = 9
SCREENTILES = (GRID_LENGTH, GRID_LENGTH)

TILE_RECT = pygame.Rect(0, 0, TILE_SIZE[0], TILE_SIZE[1])
SCREENSIZE = ((SCREENTILES[0] + 1) * TILE_SIZE[0], (SCREENTILES[1] + 1) * TILE_SIZE[1])
SCREENRECT = pygame.Rect(0, 0, SCREENSIZE[0], SCREENSIZE[1])

# position of snake at start
# START_TILE = (5, 5)
START_TILE = (5, 3)
# length of snake at start
START_SEGMENTS = 4
# START_SEGMENTS = 20

# Both radii deliberately exceed half a tile, so the circles clip against their tile-sized
# surface and paint as solid squares. They scale with the tile size to keep that: left at 13
# and 17 in a 15px tile they would render as circles with visible gaps between segments,
# and a head noticeably smaller than its body.
SNAKE_HEAD_RADIUS = int(13 * DISPLAY_SCALE)
SNAKE_SEGMENT_RADIUS = int(17 * DISPLAY_SCALE)
FOOD_RADIUS = SNAKE_SEGMENT_RADIUS

POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'savedPolicies') + '/'

# Progress graphs are written here, one per policy name, rewritten each eval.
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')

# Eval charts live here instead of RUNS_DIR, one per policy, rewritten by each eval. **Nothing
# moves them any more** (2026-08-24): starting an eval used to sweep every chart at this level into
# EVALS_ARCHIVE_DIR, which cost more than the tidiness was worth -- see CLAUDE.md. Each arm
# overwrites its own file by name, so the folder accumulates and stays correct.
# Observation indices to force to 0.0, from SNEK_ZERO_OBS as a comma-separated list of indices
# and inclusive ranges — `SNEK_ZERO_OBS=26-29` zeroes the following-tail block and food-space.
#
# **This is how an observation gets ablated without changing the MDP's shape.** Deleting a block
# changes the vector length, which stops every existing checkpoint loading and makes the
# with/without comparison a comparison between two environments — the confound that has already
# cost this project the ability to attribute batch 10's result. Zeroing keeps the length, so an
# ablated arm and a full arm train on the same 30-value spec, their checkpoints are mutually
# loadable, and the only difference is the information content of those indices.
#
# A zeroed input is not quite a deleted one: the network still has weights for it, they just have
# nothing to learn from. That is the right null for "does this signal help", which is the question,
# and it is exactly the `game_over` situation documented in CLAUDE.md — so do not later repurpose
# an index that spent a run zeroed.
def _parse_zero_obs(raw):
    if not raw:
        return frozenset()
    indices = set()
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part.lstrip('-'):
            low, high = part.split('-', 1)
            indices.update(range(int(low), int(high) + 1))
        else:
            indices.add(int(part))
    return frozenset(indices)


ZERO_OBS_INDICES = _parse_zero_obs(os.environ.get('SNEK_ZERO_OBS'))

EVALS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evals')
# Historical only: nothing writes here since 2026-08-24. Kept because the directory still holds
# every chart the old archiving swept up, and it is in CLAUDE.md's never-delete table.
EVALS_ARCHIVE_DIR = os.path.join(EVALS_DIR, 'archive')

CAPTION = 'MiniSnake'
FPS_LIMIT = 15
SCORE_SLOW_THRESHOLD = 248
SCORE_THRESHOLD_FPS = 10

# SCREENTILES used to be recomputed here as
# `(SCREENSIZE[0] / TILE_SIZE[0]) - 1`, a round trip through SCREENSIZE that returned the
# same tile count but as a *float*. That made `random.randint(0, 9.0)` in Food.__init__
# raise a DeprecationWarning on Python 3.10 and a TypeError on 3.12+, so it blocked any
# interpreter upgrade. The tuple defined above from GRID_LENGTH is already correct and
# integral, so the recomputation is gone.

PERFECT_SCORE = (SCREENTILES[0] + 1) * (SCREENTILES[1] + 1)
# highest food-eaten count reachable before a perfect game triggers
MAX_POSSIBLE_SCORE = PERFECT_SCORE - START_SEGMENTS - 1

BACKGROUND_COLOR = (255, 255, 255)
SNAKE_HEAD_COLOR = (150, 0, 0)
SNAKE_SEGMENT_COLOR = (255, 0, 0)
FOOD_COLOR = (0, 255, 0)
BLOCK_COLOR = (0, 0, 150)
COLORKEY_COLOR = (255, 255, 0)

# The HUD is drawn on top of the board, so it grows sub-linearly on purpose. Scaling 12pt with
# the window covered the board in labels and *still* ran the policy name off the right edge —
# the 100x100 window's own problem, reproduced larger. Floored at the historical 12, which is
# also what a 150px window gets: at 12pt 'Policy: <arm name>' fits across it with ~5px spare.
HUD_FONT_SIZE = max(12, int(TILE_PIXELS * 0.45))
HUD_LINE_HEIGHT = HUD_FONT_SIZE + int(4 * DISPLAY_SCALE)
HUD_ORIGIN = (int(20 * DISPLAY_SCALE), int(20 * DISPLAY_SCALE))

# Death and win messages are meant to fill the window, so those do scale 1:1.
PERFECT_FONT_SIZE = int(25 * DISPLAY_SCALE)
STARVE_FONT_SIZE = int(60 * DISPLAY_SCALE)
DEATH_FONT_SIZE = int(100 * DISPLAY_SCALE)

SCORE_COLOR = (0, 0, 0)
SCORE_POS = HUD_ORIGIN
SCORE_PREFIX = 'Score: '
STEP_COLOR = (0, 0, 0)
STEP_POS = (HUD_ORIGIN[0], HUD_ORIGIN[1] + HUD_LINE_HEIGHT)
STEP_PREFIX = 'Steps: '
POLICY_COLOR = (0, 0, 0)
POLICY_POS = (HUD_ORIGIN[0], HUD_ORIGIN[1] + 2 * HUD_LINE_HEIGHT)
POLICY_PREFIX = 'Policy: '

# Background patch repainted before each label is redrawn, so the previous value doesn't show
# through. One line tall and the rest of the window wide: the old fixed (50, 100) was narrower
# than a policy name at any scale, which left the tail of a long one smeared across the board.
HUD_ERASE_SIZE = (SCREENSIZE[0] - HUD_ORIGIN[0], HUD_LINE_HEIGHT)

DIRECTIONS = ['left', 'right', 'up', 'down']
MOVE_VECTORS = {'left': (-1, 0),
                'right': (1, 0),
                'up': (0, -1),
                'down': (0, 1)
                }
MOVE_VECTORS_PIXELS = {'left': (-TILE_SIZE[0], 0),
                       'right': (TILE_SIZE[0], 0),
                       'up': (0, -TILE_SIZE[1]),
                       'down': (0, TILE_SIZE[1])
                       }

TF_ACTION_TO_ACTIONS = {0: 'left',
                        1: 'right',
                        2: 'forward'}
ACTIONS = ['left', 'right', 'forward']
# Used to map relative direction -> cardinal direction
CURRENT_DIRECTION_MAPS = {
    'left': {
        'forward': 'left',
        'left': 'down',
        'right': 'up'
    },
    'right': {
        'forward': 'right',
        'left': 'up',
        'right': 'down'
    },
    'up': {
        'forward': 'up',
        'left': 'left',
        'right': 'right'
    },
    'down': {
        'forward': 'down',
        'left': 'right',
        'right': 'left'
    }
}
