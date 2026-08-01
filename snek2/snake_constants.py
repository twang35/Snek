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
PERFECT_GAME_WAIT_MS = 5000

# Below this average score, an eval's checkpoint is not written at all. Set from
# SNEK_MIN_CHECKPOINT_SCORE in snek2.py. `max_to_keep` is a rolling window, so a dead arm
# that keeps training evicts the good checkpoints behind it — see the gate in training.py.
#
# 40 is well clear of anything useful: across 232 checkpoints measured at 100 episodes, every
# one that reached 30% perfect games had max(avg_score, trailing) of at least 49.8.
global MIN_CHECKPOINT_SCORE
MIN_CHECKPOINT_SCORE = 40.0
MAX_STEPS_BEFORE_STARVE_SIZE_MULTIPLIER = 10

CLOSER_TO_FOOD_REWARD_SCORE_LIMIT = 10
CLOSER_TO_FOOD_REWARD_STEP_LIMIT = 20000

# size of each grid
TILE_SIZE = (10, 10)

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

SNAKE_HEAD_RADIUS = 13
SNAKE_SEGMENT_RADIUS = 17
FOOD_RADIUS = SNAKE_SEGMENT_RADIUS

POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'savedPolicies') + '/'

# Progress graphs are written here, one per policy name, rewritten each eval.
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')

CAPTION = 'MiniSnake'
FPS_LIMIT = 15
SCORE_SLOW_THRESHOLD = 248
SCORE_THRESHOLD_FPS = 10

SCREENTILES = (
    (SCREENSIZE[0] / TILE_SIZE[0]) - 1,
    (SCREENSIZE[1] / TILE_SIZE[1]) - 1
)

PERFECT_SCORE = (SCREENTILES[0] + 1) * (SCREENTILES[1] + 1)
# highest food-eaten count reachable before a perfect game triggers
MAX_POSSIBLE_SCORE = PERFECT_SCORE - START_SEGMENTS - 1

BACKGROUND_COLOR = (255, 255, 255)
SNAKE_HEAD_COLOR = (150, 0, 0)
SNAKE_SEGMENT_COLOR = (255, 0, 0)
FOOD_COLOR = (0, 255, 0)
BLOCK_COLOR = (0, 0, 150)
COLORKEY_COLOR = (255, 255, 0)

SCORE_COLOR = (0, 0, 0)
SCORE_POS = (20, 20)
SCORE_PREFIX = 'Score: '
STEP_COLOR = (0, 0, 0)
STEP_POS = (20, 40)
STEP_PREFIX = 'Steps: '
POLICY_COLOR = (0, 0, 0)
POLICY_POS = (20, 60)
POLICY_PREFIX = 'Policy: '

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
