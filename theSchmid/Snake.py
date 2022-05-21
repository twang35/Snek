from enum import Enum

import pygame
from pygame.locals import *
import random
import numpy as np
import math

# ---------- constants ---------- #
# screen that game appears on 0 or 1
SCREEN_TO_DISPLAY = 1

# size of each grid
TILE_SIZE = (10, 10)

# number of grid of the screen
SCREENTILES = (15, 15)
# SCREENTILES = (4, 4)

TILE_RECT = pygame.Rect(0, 0, TILE_SIZE[0], TILE_SIZE[1])
SCREENSIZE = ((SCREENTILES[0] + 1) * TILE_SIZE[0], (SCREENTILES[1] + 1) * TILE_SIZE[1])
SCREENRECT = pygame.Rect(0, 0, SCREENSIZE[0], SCREENSIZE[1])

# position of snake at start
# START_TILE = (5, 5)
START_TILE = (5, 3)
# lenght of snake at start
START_SEGMENTS = 4
# START_SEGMENTS = 20

SNAKE_HEAD_RADIUS = 13
SNAKE_SEGMENT_RADIUS = 17
FOOD_RADIUS = SNAKE_SEGMENT_RADIUS

CAPTION = 'MiniSnake'
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


# ----------- game objects ----------- #
class snake_segment(pygame.sprite.Sprite):
    def __init__(self, tilepos, segment_groups, color=SNAKE_SEGMENT_COLOR, radius=SNAKE_SEGMENT_RADIUS):
        pygame.sprite.Sprite.__init__(self)
        self.image = self.image = pygame.Surface(TILE_SIZE).convert()
        self.image.fill(COLORKEY_COLOR)
        self.image.set_colorkey(COLORKEY_COLOR)
        pygame.draw.circle(self.image, color, TILE_RECT.center, radius)

        self.tilepos = tilepos

        self.rect = self.image.get_rect()
        self.rect.topleft = (tilepos[0] * TILE_SIZE[0], tilepos[1] * TILE_SIZE[1])

        self.segment_groups = segment_groups
        for group in segment_groups:
            group.add(self)

        self.front_segment = None
        self.behind_segment = None

        self.movedir = 'left'

    # this function adds a segment at the end of the snake
    def add_segment(self, tail):
        x = tail.tilepos[0]
        y = tail.tilepos[1]
        if tail.movedir == 'left':
            x += 1
        elif tail.movedir == 'right':
            x -= 1
        elif tail.movedir == 'up':
            y += 1
        elif tail.movedir == 'down':
            y -= 1
        tail.behind_segment = snake_segment((x, y), tail.segment_groups)
        tail.behind_segment.movedir = tail.movedir


    def update(self):
        pass

    def move(self):
        # todo: reformat to move only head and tail
        self.tilepos = (
            self.tilepos[0] + MOVE_VECTORS[self.movedir][0],
            self.tilepos[1] + MOVE_VECTORS[self.movedir][1]
        )
        self.rect.move_ip(MOVE_VECTORS_PIXELS[self.movedir])
        if self.behind_segment != None:
            self.behind_segment.move()
            self.behind_segment.movedir = self.movedir


class snake_head(snake_segment):
    def __init__(self, tilepos, movedir, segment_groups):
        snake_segment.__init__(self, tilepos, segment_groups, color=SNAKE_HEAD_COLOR, radius=SNAKE_HEAD_RADIUS)
        self.movedir = movedir
        self.movecount = 0

    def update(self):
        self.move()
        self.movecount = 0

    def get_positions(self):
        seg = self
        positions = []
        while True:
            position = seg.tilepos
            positions.append((position[0], position[1]))
            if seg.behind_segment == None:
                break
            else:
                # looping until we get the last segment of the snake
                seg = seg.behind_segment
        return positions


class food(pygame.sprite.Sprite):
    def __init__(self, takenupgroup):
        pygame.sprite.Sprite.__init__(self)
        self.image = self.image = pygame.Surface(TILE_SIZE).convert()
        self.image.fill(COLORKEY_COLOR)
        self.image.set_colorkey(COLORKEY_COLOR)
        pygame.draw.circle(self.image, FOOD_COLOR, TILE_RECT.center, FOOD_RADIUS)

        self.rect = self.image.get_rect()
        while True:
            self.position = (
                random.randint(0, SCREENTILES[0]),
                random.randint(0, SCREENTILES[1])
            )

            self.rect.topleft = (
                self.position[0] * TILE_SIZE[0],
                self.position[1] * TILE_SIZE[1]
            )
            continue_loop = False
            for sprt in takenupgroup:
                if self.rect.colliderect(sprt):
                    continue_loop = True  # collision, food cant go here
            if continue_loop:
                continue
            else:
                break  # no collision, food can go here


class KitchenSink:
    def __init__(self, snake_group, snake, head, tail, current_food, current_score, game_grid, game_over):
        self.snake_group = snake_group
        self.snake = snake
        self.head = head
        self.tail = tail
        self.current_food = current_food
        self.current_score = current_score
        self.game_grid = game_grid
        self.game_over = game_over


class Game():
    def __init__(self):
        pygame.init()

    def reset(self):
        # show screen
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 0, SCREEN_TO_DISPLAY, 0)
        pygame.display.set_caption(CAPTION)
        self.bg = pygame.Surface(SCREENSIZE).convert()
        self.bg.fill(BACKGROUND_COLOR)
        self.screen.blit(self.bg, (0, 0))

        self.snakegroup = pygame.sprite.Group()
        self.snakeheadgroup = pygame.sprite.Group()
        self.foodgroup = pygame.sprite.Group()
        self.takenupgroup = pygame.sprite.Group()

        self.all = pygame.sprite.RenderUpdates()
        column = SCREENSIZE[0] / TILE_SIZE[0]
        row = SCREENSIZE[1] / TILE_SIZE[1]
        self.grid = np.zeros((int(row), int(column)))
        self.snake = snake_head(START_TILE, 'right', [self.snakegroup, self.all, self.takenupgroup])
        self.snakeheadgroup.add(self.snake)
        self.head = self.snake
        self.tail = self.snake
        for index in range(START_SEGMENTS):
            self.add_segment()

        # weird but true

        self.currentfood = 'no food'

        self.currentscore = 0
        self.currentstep = 0

        # turn screen to white
        pygame.display.flip()

        # mainloop
        self.quit = False
        self.clock = pygame.time.Clock()
        self.lose = False


    def add_segment(self):
        self.snake.add_segment(self.tail)
        self.tail.behind_segment.front_segment = self.tail
        self.tail = self.tail.behind_segment


    def step(self, direction):
        currentmovedir = self.snake.movedir
        if direction == "up":
            tomove = 'up'
            dontmove = 'down'
        elif direction == "down":
            tomove = 'down'
            dontmove = 'up'
        elif direction == "left":
            tomove = 'left'
            dontmove = 'right'
        elif direction == "right":
            tomove = 'right'
            dontmove = 'left'
        else:
            tomove = currentmovedir
            dontmove = 'left'
        if not currentmovedir == dontmove:
            self.snake.movedir = tomove

        # clearing
        self.all.clear(self.screen, self.bg)

        # updates snake position
        self.all.update()

        if self.currentfood == 'no food':
            self.currentfood = food(self.takenupgroup)
            self.foodgroup.add(self.currentfood)
            self.takenupgroup.add(self.currentfood)
            self.all.add(self.currentfood)

        column = SCREENSIZE[0] / TILE_SIZE[0]
        row = SCREENSIZE[1] / TILE_SIZE[1]
        self.grid = np.zeros((int(row + 2), int(column + 2)))
        self.grid[[0, -1], :] = 4
        self.grid[:, [0, -1]] = 4

        BodyPositions = self.snake.get_positions()

        # 2 is food
        self.grid[self.currentfood.position[1] + 1, self.currentfood.position[0] + 1] = 1

        for i in range(0, len(BodyPositions)):
            # 3 is body part
            position = BodyPositions[i]
            if i == 0:
                # head
                self.grid[position[1] + 1, position[0] + 1] = 2
            else:
                self.grid[position[1] + 1, position[0] + 1] = 3

        # checks out of bounds
        pos = self.snake.rect.topleft
        if pos[0] < 0:
            quit.lose = True
            self.lose = True
        if pos[0] >= SCREENSIZE[0]:
            quit.lose = True
            self.lose = True
        if pos[1] < 0:
            quit.lose = True
            self.lose = True
        if pos[1] >= SCREENSIZE[1]:
            quit.lose = True
            self.lose = True

        # collisions
        # head -> tail
        col = pygame.sprite.groupcollide(self.snakeheadgroup, self.snakegroup, False, False)
        for head in col:
            for body_part in col[head]:
                # self.snake is actually snake_head sprite (which resembles a LinkedList)
                if not body_part is self.snake:
                    self.quit = True
                    self.lose = True
        # head -> food
        col = pygame.sprite.groupcollide(self.snakeheadgroup, self.foodgroup, False, True)
        if len(col) > 0:
            self.currentfood = 'no food'
            self.add_segment()
            self.currentscore += 1

        self.currentstep += 1

        # game over
        if self.lose is True:
            f = pygame.font.Font(None, 100)
            failmessage = f.render('FAIL', True, (0, 0, 0))
            failrect = failmessage.get_rect()
            failrect.center = SCREENRECT.center
            self.screen.blit(failmessage, failrect)
            pygame.display.flip()
            pygame.time.wait(2000)

        return self.getKitchenSink()

    def render(self):
        # score
        d = self.screen.blit(self.bg, SCORE_POS, pygame.Rect(SCORE_POS, (50, 100)))
        f = pygame.font.Font(None, 12)
        scoreimage = f.render(SCORE_PREFIX + str(self.currentscore), True, SCORE_COLOR)
        d2 = self.screen.blit(scoreimage, SCORE_POS)
        # steps
        d3 = self.screen.blit(self.bg, STEP_POS, pygame.Rect(STEP_POS, (50, 100)))
        f = pygame.font.Font(None, 12)
        stepimage = f.render(STEP_PREFIX + str(self.currentstep), True, STEP_COLOR)
        d4 = self.screen.blit(stepimage, STEP_POS)

        # drawing
        dirty = self.all.draw(self.screen)
        dirty.append(d)
        dirty.append(d2)
        dirty.append(d3)
        dirty.append(d4)

        # updating screen
        pygame.display.update(dirty)

        # waiting
        self.clock.tick(FPS)
        # print("\n", self.grid, "\n")
        return self.getKitchenSink()

    def getKitchenSink(self):
        return KitchenSink(self.snakegroup, self.snake, self.head, self.tail, self.currentfood, self.currentscore,
                           self.grid, self.lose)