import pygame
import random
import numpy as np

# ---------- constants ---------- #
# screen that game appears on 0 or 1
SCREEN_TO_DISPLAY = 0

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

        self.movedir = 'right'
        self.last_move_dir = 'right'

    def update(self):
        pass

    def move(self):
        self.tilepos = (
            self.tilepos[0] + MOVE_VECTORS[self.movedir][0],
            self.tilepos[1] + MOVE_VECTORS[self.movedir][1]
        )
        self.rect.move_ip(MOVE_VECTORS_PIXELS[self.movedir])
        self.last_move_dir = self.movedir
        if self.behind_segment != None:
            self.behind_segment.move()
            self.behind_segment.movedir = self.movedir


class SnakeHead(snake_segment):
    def __init__(self, tilepos, movedir, segment_groups):
        snake_segment.__init__(self, tilepos, segment_groups, color=SNAKE_HEAD_COLOR, radius=SNAKE_HEAD_RADIUS)
        self.movedir = movedir
        self.last_move_dir = movedir
        self.movecount = 0

    def update(self):
        self.move()
        self.movecount += 1

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
        self.head = None
        self.tail = None
        self.current_step = 0
        self.lose = False
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
        self.snake = SnakeHead(START_TILE, 'right', [self.snakegroup, self.all, self.takenupgroup])
        self.snakeheadgroup.add(self.snake)
        self.head = self.snake
        self.tail = self.snake
        for index in range(START_SEGMENTS):
            self.add_segment()

        # weird but true

        self.currentfood = 'no food'

        self.currentscore = 0
        self.current_step = 0
        self.last_food_step = 0

        # turn screen to white
        pygame.display.flip()

        # mainloop
        self.clock = pygame.time.Clock()
        self.lose = False

    # this function adds a segment at the end of the snake
    def add_segment(self):
        # get tile_pos for new segment
        x = self.tail.tilepos[0]
        y = self.tail.tilepos[1]
        if self.tail.last_move_dir == 'left':
            x += 1
        elif self.tail.last_move_dir == 'right':
            x -= 1
        elif self.tail.last_move_dir == 'up':
            y += 1
        elif self.tail.last_move_dir == 'down':
            y -= 1
        # set up new tail segment
        self.tail.behind_segment = snake_segment((x, y), self.tail.segment_groups)
        self.tail.behind_segment.movedir = self.tail.last_move_dir
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

        # update all sprites positions
        self.all.update()

        if self.currentfood == 'no food' and not self.perfect_game():
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

        # 1 is food
        self.grid[self.currentfood.position[1] + 1, self.currentfood.position[0] + 1] = 1

        for i in range(0, len(BodyPositions)):
            position = BodyPositions[i]
            if i == 0:
                # 2 is head
                self.grid[position[1] + 1, position[0] + 1] = 2
            else:
                if not (position[1] < 0 or position[0] < 0):
                    # 3 is body part
                    self.grid[position[1] + 1, position[0] + 1] = 3

        # checks out of bounds
        pos = self.snake.rect.topleft
        if pos[0] < 0:
            self.lose = True
        if pos[0] >= SCREENSIZE[0]:
            self.lose = True
        if pos[1] < 0:
            self.lose = True
        if pos[1] >= SCREENSIZE[1]:
            self.lose = True

        # collisions
        # head -> body
        col = pygame.sprite.groupcollide(self.snakeheadgroup, self.snakegroup, False, False)
        for head in col:
            for body_part in col[head]:
                # self.snake is actually snake_head sprite (which resembles a LinkedList)
                if not body_part is self.snake:
                    self.lose = True
        # head -> food
        col = pygame.sprite.groupcollide(self.snakeheadgroup, self.foodgroup, False, True)
        if len(col) > 0:
            self.currentfood = 'no food'
            self.add_segment()
            self.currentscore += 1
            self.last_food_step = self.current_step

        self.current_step += 1

        # game over
        if self.perfect_game():
            self.lose = True
            f = pygame.font.Font(None, 25)
            fail_message = f.render('PERFECT GAME!!!', True, (0, 0, 0))
            fail_rect = fail_message.get_rect()
            fail_rect.center = SCREENRECT.center
            self.screen.blit(fail_message, fail_rect)
            pygame.display.flip()
            pygame.time.wait(5000)

        elif self.lose is True or (self.current_step - self.last_food_step) > MAX_STEPS_BEFORE_STARVE:
            self.lose = True
            f = pygame.font.Font(None, 100)
            fail_message = f.render('FAIL', True, (0, 0, 0))
            fail_rect = fail_message.get_rect()
            fail_rect.center = SCREENRECT.center
            self.screen.blit(fail_message, fail_rect)
            pygame.display.flip()
            pygame.time.wait(2000)

        return self.get_kitchen_sink()

    def perfect_game(self):
        return (self.currentscore + START_SEGMENTS + 1) == ((SCREENTILES[0]+1) * (SCREENTILES[1]+1))

    def render(self):
        # clearing
        self.all.clear(self.screen, self.bg)

        # score
        d = self.screen.blit(self.bg, SCORE_POS, pygame.Rect(SCORE_POS, (50, 100)))
        f = pygame.font.Font(None, 12)
        scoreimage = f.render(SCORE_PREFIX + str(self.currentscore), True, SCORE_COLOR)
        d2 = self.screen.blit(scoreimage, SCORE_POS)
        # steps
        d3 = self.screen.blit(self.bg, STEP_POS, pygame.Rect(STEP_POS, (50, 100)))
        f = pygame.font.Font(None, 12)
        stepimage = f.render(STEP_PREFIX + str(self.current_step), True, STEP_COLOR)
        d4 = self.screen.blit(stepimage, STEP_POS)

        # drawing
        dirty = self.all.draw(self.screen)
        dirty.append(d)
        dirty.append(d2)
        dirty.append(d3)
        dirty.append(d4)

        # updating screen
        pygame.display.update(dirty)

        # slow down when close to finished
        if self.currentscore >= SCORE_SLOW_THRESHOLD:
            self.clock.tick(FPS)

        # print("\n", self.grid, "\n")
        return self.get_kitchen_sink()

    def get_kitchen_sink(self):
        return KitchenSink(self.snakegroup, self.snake, self.head, self.tail, self.currentfood, self.currentscore,
                           self.grid, self.lose)
