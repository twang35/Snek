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
FOOD_DISTANCE_REWARD = 0.001
DEATH_REWARD = -5.0  # maybe avoid deaths more?
STARVE_REWARD = -0.5
global PERFECT_GAME_REWARD
PERFECT_GAME_REWARD = 100
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
MAX_STEPS_BEFORE_STARVE_SIZE_MULTIPLIER = 10

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
