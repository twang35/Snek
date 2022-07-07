import pygame

# screen that game appears on 0 or 1
SCREEN_TO_DISPLAY = 0

FOOD_REWARD = 1.0
FOOD_DISTANCE_REWARD = 0.001
DEATH_REWARD = -10.0  # maybe avoid deaths more?
STARVE_REWARD = -0.5
MAX_STEPS_BEFORE_STARVE_SIZE_MULTIPLIER = 10

CLOSER_TO_FOOD_REWARD_SCORE_LIMIT = 10
CLOSER_TO_FOOD_REWARD_STEP_LIMIT = 20000

# size of each grid
TILE_SIZE = (10, 10)

# number of grid of the screen
GRID_LENGTH = 15
# GRID_LENGTH = 7
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

CAPTION = 'MiniSnake'
FPS_LIMIT = 15
SCORE_SLOW_THRESHOLD = 248
SCORE_THRESHOLD_FPS = 10

SCREENTILES = (
    (SCREENSIZE[0] / TILE_SIZE[0]) - 1,
    (SCREENSIZE[1] / TILE_SIZE[1]) - 1
)

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
