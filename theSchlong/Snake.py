import random
import numpy as np

from snake_constants import *
from state_helpers import *


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


class Game():
    def __init__(self, display=True):
        # show screen
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 0, SCREEN_TO_DISPLAY, 0)

        self.current_score = 0
        self.head = None
        self.tail = None
        self.current_step = 0
        self.finished = False
        self.perfect_game = False
        self.display = display
        pygame.init()

    def reset(self):
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

        self.current_score = 0
        self.current_step = 0
        self.last_food_step = 0

        # turn screen to white
        pygame.display.flip()

        # mainloop
        self.clock = pygame.time.Clock()
        self.finished = False
        self.perfect_game = False

    def get_observation(self):
        observations = []
        observations.extend(food_observations(self.grid, self.head.tilepos, self.tail.tilepos, self.currentfood))
        observations.extend(body_and_wall_collisions(self.grid, self.head.tilepos, self.tail.tilepos))
        return np.array(observations)

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
        if self.currentfood != 'no food' and self.current_score < 30:
            old_moves_to_food = distance_to_food(self.head.tilepos, self.currentfood.position)
        else:
            old_moves_to_food = 0

        reward = 0.0
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

        if self.currentfood == 'no food' and not self.perfect_game:
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
            self.finished = True
            reward = DEATH_REWARD
        if pos[0] >= SCREENSIZE[0]:
            self.finished = True
            reward = DEATH_REWARD
        if pos[1] < 0:
            self.finished = True
            reward = DEATH_REWARD
        if pos[1] >= SCREENSIZE[1]:
            self.finished = True
            reward = DEATH_REWARD

        # collisions
        # head -> body
        col = pygame.sprite.groupcollide(self.snakeheadgroup, self.snakegroup, False, False)
        for head in col:
            for body_part in col[head]:
                # self.snake is actually snake_head sprite (which resembles a LinkedList)
                if not body_part is self.snake:
                    self.finished = True
                    reward = DEATH_REWARD
        # head -> food
        col = pygame.sprite.groupcollide(self.snakeheadgroup, self.foodgroup, False, True)
        if len(col) > 0:
            self.currentfood = 'no food'
            self.add_segment()
            self.current_score += 1
            self.last_food_step = self.current_step
            reward = FOOD_REWARD

        self.current_step += 1

        # game over
        if self.check_perfect_game():
            self.finished = True
            self.perfect_game = True

        elif self.finished is True or (self.current_step - self.last_food_step) > MAX_STEPS_BEFORE_STARVE:
            self.finished = True

        if self.currentfood != 'no food':
            moves_to_food = distance_to_food(self.head.tilepos, self.currentfood.position)
            if moves_to_food < old_moves_to_food:
                reward += FOOD_DISTANCE_REWARD
            else:
                reward -= FOOD_DISTANCE_REWARD
            # reward += (GRID_LENGTH - moves_to_food)/50

        return self.finished, reward

    def check_perfect_game(self):
        return (self.current_score + START_SEGMENTS + 1) == ((SCREENTILES[0] + 1) * (SCREENTILES[1] + 1))

    def render(self):
        if not self.display:
            return

        # unfreezes the window if stuck
        pygame.event.pump()

        if self.perfect_game:
            f = pygame.font.Font(None, 25)
            fail_message = f.render('PERFECT GAME!!!', True, (0, 0, 0))
            fail_rect = fail_message.get_rect()
            fail_rect.center = SCREENRECT.center
            self.screen.blit(fail_message, fail_rect)
            pygame.display.flip()
            pygame.time.wait(5000)
            return

        if self.finished:
            f = pygame.font.Font(None, 100)
            fail_message = f.render('DED', True, (0, 0, 0))
            fail_rect = fail_message.get_rect()
            fail_rect.center = SCREENRECT.center
            self.screen.blit(fail_message, fail_rect)
            pygame.display.flip()
            pygame.time.wait(10)
            return

        # score
        d = self.screen.blit(self.bg, SCORE_POS, pygame.Rect(SCORE_POS, (50, 100)))
        f = pygame.font.Font(None, 12)
        score_image = f.render(SCORE_PREFIX + str(self.current_score), True, SCORE_COLOR)
        d2 = self.screen.blit(score_image, SCORE_POS)
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
        if self.current_score >= SCORE_SLOW_THRESHOLD:
            self.clock.tick(FPS)
