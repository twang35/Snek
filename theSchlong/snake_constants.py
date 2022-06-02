import pygame

# screen that game appears on 0 or 1
SCREEN_TO_DISPLAY = 0

# size of each grid
TILE_SIZE = (10, 10)

# number of grid of the screen
SCREENTILES = (15, 15)

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

MAX_STEPS_BEFORE_STARVE = 1000

CAPTION = 'MiniSnake'
SCORE_SLOW_THRESHOLD = 250
FPS = 15

MOVE_RATE = 1  # how many frame per move
DIFFICULTY_INCREASE_RATE = 0
MOVE_THRESHOLD = 1  # when moverate counts up to this the snake moves
BLOCK_SPAWN_RATE = 2

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
