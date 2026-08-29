"""Everything drawable: pixel geometry, colours, fonts, HUD layout.

**Every pygame constant in snek3 is here**, and `env/game.py` — which needs the module itself
for sprites and surfaces — is the only other importer of it.

`env/constants.py` holds the game rules and imports no pygame, so `vectorized/` and the whole eval
path never touch a display or an audio device. Nothing outside `env/` should import this module.

**Never call `pygame.init()`** — it starts every subsystem including `pygame.mixer`, which opens a
real CoreAudio stream per process (10 idle snek2 workers drove `coreaudiod` to 15% CPU).
`env.game.Game` inits `display` and `font` by name.

**Read these through the module** (`render.SCORE_COLOR`), never via `from env.render import *`. A
star import binds a copy, so a later assignment never reaches the reader — which is exactly the trap
snek2's GIF recorder had to work around by patching `Snake`'s own globals instead of the constants
module. Patching `env.render` is the supported way to change how a frame looks.
"""

import os

# Before the pygame import, so no process here can open an audio device.
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pygame

from env.constants import SCREENTILES

# Pixels per tile, and so the window size: the board is 10x10 tiles, so 10 gives a 100x100 window.
# That is small enough that macOS shows none of the title bar text, which matters when several
# watch.py windows are open — so watch.py raises it to 15.
#
# **Purely cosmetic**: observations are built from tile positions, never pixels. Verified in snek2 by
# a fixed-seed hash coming out identical at 10, 20, 25 and 40 pixels per tile.
#
# **Must be set in the environment before this module is imported**, because everything below is
# derived from it.
TILE_PIXELS = int(os.environ.get('SNEK_TILE_PIXELS', 10))
DISPLAY_SCALE = TILE_PIXELS / 10.0

TILE_SIZE = (TILE_PIXELS, TILE_PIXELS)
TILE_RECT = pygame.Rect(0, 0, TILE_SIZE[0], TILE_SIZE[1])
SCREENSIZE = ((SCREENTILES[0] + 1) * TILE_SIZE[0], (SCREENTILES[1] + 1) * TILE_SIZE[1])
SCREENRECT = pygame.Rect(0, 0, SCREENSIZE[0], SCREENSIZE[1])

MOVE_VECTORS_PIXELS = {'left':  (-TILE_SIZE[0], 0),
                       'right': (TILE_SIZE[0], 0),
                       'up':    (0, -TILE_SIZE[1]),
                       'down':  (0, TILE_SIZE[1])}

# Both radii deliberately exceed half a tile, so the circles clip against their tile-sized surface
# and paint as solid squares. They scale with the tile size to keep that: left at 13 and 17 in a
# 15px tile they render as circles with visible gaps between segments.
SNAKE_HEAD_RADIUS = int(13 * DISPLAY_SCALE)
SNAKE_SEGMENT_RADIUS = int(17 * DISPLAY_SCALE)
FOOD_RADIUS = SNAKE_SEGMENT_RADIUS

BACKGROUND_COLOR = (255, 255, 255)
SNAKE_HEAD_COLOR = (150, 0, 0)
SNAKE_SEGMENT_COLOR = (255, 0, 0)
FOOD_COLOR = (0, 255, 0)
BLOCK_COLOR = (0, 0, 150)
COLORKEY_COLOR = (255, 255, 0)

CAPTION = 'MiniSnake'
SCREEN_TO_DISPLAY = 0

# Frame rate. A display flip costs ~5.2 ms — a round trip to the OS window server, not our drawing
# code, which is 2-4 us — and the game flips once per game step. That is why training never draws.
FPS_LIMIT = 15
SCORE_SLOW_THRESHOLD = 248
SCORE_THRESHOLD_FPS = 10
PERFECT_GAME_WAIT_MS = 500

# The HUD is drawn on top of the board, so it grows sub-linearly on purpose: scaling 12pt with the
# window covered the board in labels and still ran the policy name off the right edge. Floored at 12,
# which is what a 150px window gets — at 12pt 'Policy: <arm name>' fits with ~5px spare.
HUD_FONT_SIZE = max(12, int(TILE_PIXELS * 0.45))
HUD_LINE_HEIGHT = HUD_FONT_SIZE + int(4 * DISPLAY_SCALE)
HUD_ORIGIN = (int(20 * DISPLAY_SCALE), int(20 * DISPLAY_SCALE))

# Death and win messages are meant to fill the window, so these scale 1:1.
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

# Background patch repainted before each label is redrawn, so the previous value does not show
# through. One line tall and the rest of the window wide: a fixed narrow patch left the tail of a
# long policy name smeared across the board.
HUD_ERASE_SIZE = (SCREENSIZE[0] - HUD_ORIGIN[0], HUD_LINE_HEIGHT)

# The HUD is blitted *before* the sprites, so score and step readouts get buried as the board
# fills. A recorder that wants a readable header suppresses these and composites its own outside
# the board — see `record_gif.py`.

_FONTS = {}


def font(size):
    """Cached `pygame.font.Font` by size.

    Constructing a Font parses the font file — ~122 us each, and a frame needed three of them. That
    was 366 us of a 5,300 us frame, second only to the display flip.
    """
    got = _FONTS.get(size)
    if got is None:
        got = pygame.font.Font(None, size)
        _FONTS[size] = got
    return got


def hud_font():
    return font(HUD_FONT_SIZE)


def message_font(text, max_size):
    """Largest font at or below `max_size` whose rendering of `text` fits across the window.

    End-of-game messages are sized to fill the window, but 'DED' at 100pt is ~150px wide against a
    100px board, so both D's were clipped off the edges — and scaling with the window reproduced
    that at every size. Runs once per episode, and every size it tries lands in the same cache.
    """
    limit = int(SCREENSIZE[0] * 0.95)
    size = max_size
    while size > 8 and font(size).size(text)[0] > limit:
        size = int(size * 0.9)
    return font(size)
