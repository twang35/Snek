import random

from state_helpers import *


# ----------- game objects ----------- #
class SnakeSegment(pygame.sprite.Sprite):
    def __init__(self, tile_pos, segment_groups, color=SNAKE_SEGMENT_COLOR, radius=SNAKE_SEGMENT_RADIUS):
        pygame.sprite.Sprite.__init__(self)
        self.image = self.image = pygame.Surface(TILE_SIZE).convert()
        self.image.fill(COLORKEY_COLOR)
        self.image.set_colorkey(COLORKEY_COLOR)
        pygame.draw.circle(self.image, color, TILE_RECT.center, radius)

        self.tile_pos = tile_pos

        self.rect = self.image.get_rect()
        self.rect.topleft = (tile_pos[0] * TILE_SIZE[0], tile_pos[1] * TILE_SIZE[1])

        self.segment_groups = segment_groups
        for group in segment_groups:
            group.add(self)

        self.front_segment = None
        self.behind_segment = None

        self.move_dir = 'right'
        self.last_move_dir = 'right'

    def update(self):
        pass

    def move(self):
        self.tile_pos = (
            self.tile_pos[0] + MOVE_VECTORS[self.move_dir][0],
            self.tile_pos[1] + MOVE_VECTORS[self.move_dir][1]
        )
        self.rect.move_ip(MOVE_VECTORS_PIXELS[self.move_dir])
        self.last_move_dir = self.move_dir
        if self.behind_segment is not None:
            self.behind_segment.move()
            self.behind_segment.move_dir = self.move_dir


class SnakeHead(SnakeSegment):
    def __init__(self, tile_pos, move_dir, segment_groups):
        SnakeSegment.__init__(self, tile_pos, segment_groups, color=SNAKE_HEAD_COLOR, radius=SNAKE_HEAD_RADIUS)
        self.move_dir = move_dir
        self.last_move_dir = move_dir
        self.move_count = 0

    def update(self):
        self.move()
        self.move_count += 1

    def get_positions(self):
        seg = self
        positions = []
        while True:
            position = seg.tile_pos
            positions.append((position[0], position[1]))
            if seg.behind_segment is None:
                break
            else:
                # looping until we get the last segment of the snake
                seg = seg.behind_segment
        return positions


class Food(pygame.sprite.Sprite):
    def __init__(self, taken_up_group):
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
            for sprt in taken_up_group:
                if self.rect.colliderect(sprt):
                    continue_loop = True  # collision, food can't go here
            if continue_loop:
                continue
            else:
                break  # no collision, food can go here


class Game:
    def __init__(self, display=True, limit_fps=False):
        self.display = display
        self.limit_fps = limit_fps
        # show screen
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 0, SCREEN_TO_DISPLAY, 0)

        self.snake_group = pygame.sprite.Group()
        self.snake_head_group = pygame.sprite.Group()
        self.food_group = pygame.sprite.Group()
        self.taken_up_group = pygame.sprite.Group()

        self.bg = None
        self.all = None
        self.grid = None
        self.snake = None
        self.current_food = None

        self.current_score = 0
        self.head = None
        self.tail = None
        self.total_steps = 0
        self.current_step = 0
        self.last_food_step = 0

        self.finished = False
        self.starved = False
        self.perfect_game = False
        self.clock = pygame.time.Clock()
        pygame.init()

    def reset(self):
        pygame.display.set_caption(CAPTION)
        self.bg = pygame.Surface(SCREENSIZE).convert()
        self.bg.fill(BACKGROUND_COLOR)
        self.screen.blit(self.bg, (0, 0))

        self.snake_group = pygame.sprite.Group()
        self.snake_head_group = pygame.sprite.Group()
        self.food_group = pygame.sprite.Group()
        self.taken_up_group = pygame.sprite.Group()

        self.all = pygame.sprite.RenderUpdates()
        column = SCREENSIZE[0] / TILE_SIZE[0]
        row = SCREENSIZE[1] / TILE_SIZE[1]
        self.grid = np.zeros((int(row), int(column)))
        self.snake = SnakeHead(START_TILE, 'right', [self.snake_group, self.all, self.taken_up_group])
        self.snake_head_group.add(self.snake)
        self.head = self.snake
        self.tail = self.snake
        for index in range(START_SEGMENTS):
            self.add_segment()

        # weird but true
        self.current_food = 'no food'

        self.current_score = 0
        self.current_step = 0
        self.last_food_step = 0

        # turn screen to white
        pygame.display.flip()

        # mainloop
        self.finished = False
        self.starved = False
        self.perfect_game = False

    def get_observation(self):
        return np.array(get_observations(self.grid,
                                         self.head.tile_pos,
                                         self.tail.tile_pos,
                                         self.current_food,
                                         self.current_step,
                                         self.last_food_step,
                                         len(self.snake_group),
                                         self.finished))

    # this function adds a segment at the end of the snake
    def add_segment(self):
        # get tile_pos for new segment
        x = self.tail.tile_pos[0]
        y = self.tail.tile_pos[1]
        if self.tail.last_move_dir == 'left':
            x += 1
        elif self.tail.last_move_dir == 'right':
            x -= 1
        elif self.tail.last_move_dir == 'up':
            y += 1
        elif self.tail.last_move_dir == 'down':
            y -= 1
        # set up new tail segment
        self.tail.behind_segment = SnakeSegment((x, y), self.tail.segment_groups)
        self.tail.behind_segment.move_dir = self.tail.last_move_dir
        self.tail.behind_segment.front_segment = self.tail
        self.tail = self.tail.behind_segment

    def step(self, direction):
        start_time = time()
        if self.current_food != 'no food':
            old_moves_to_food = distance_to_food(self.head.tile_pos, self.current_food.position)
        else:
            old_moves_to_food = 0

        reward = 0.0
        current_move_dir = self.snake.move_dir
        if direction == "up":
            to_move = 'up'
            dont_move = 'down'
        elif direction == "down":
            to_move = 'down'
            dont_move = 'up'
        elif direction == "left":
            to_move = 'left'
            dont_move = 'right'
        elif direction == "right":
            to_move = 'right'
            dont_move = 'left'
        else:
            to_move = current_move_dir
            dont_move = 'left'
        if not current_move_dir == dont_move:
            self.snake.move_dir = to_move

        # clearing
        self.all.clear(self.screen, self.bg)

        # updates snake position
        self.all.update()

        if self.current_food == 'no food' and not self.perfect_game:
            self.current_food = Food(self.taken_up_group)  # check that score was updated
            self.food_group.add(self.current_food)
            self.taken_up_group.add(self.current_food)
            self.all.add(self.current_food)

        column = SCREENSIZE[0] / TILE_SIZE[0]
        row = SCREENSIZE[1] / TILE_SIZE[1]
        self.grid = np.zeros((int(row + 2), int(column + 2)))
        self.grid[[0, -1], :] = 4
        self.grid[:, [0, -1]] = 4

        body_positions = self.snake.get_positions()

        # 1 is food
        self.grid[self.current_food.position[1] + 1, self.current_food.position[0] + 1] = 1

        for i in range(0, len(body_positions)):
            position = body_positions[i]
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
        collisions = pygame.sprite.groupcollide(self.snake_head_group, self.snake_group, False, False)
        # noinspection PyTypeChecker
        if len(collisions.get(self.head)) > 1:
            self.finished = True
            reward = DEATH_REWARD
        # head -> food
        col = pygame.sprite.groupcollide(self.snake_head_group, self.food_group, False, True)
        if len(col) > 0:
            self.current_food = 'no food'  # check that score was updated and persists
            self.add_segment()
            self.current_score += 1
            self.last_food_step = self.current_step
            reward = FOOD_REWARD

        self.current_step += 1
        self.total_steps += 1

        # game over
        if self.check_perfect_game():
            self.finished = True
            self.perfect_game = True

        elif not self.finished and steps_until_starve(self.current_step,
                                                      self.last_food_step,
                                                      len(self.snake_group))[0] <= 0:
            self.finished = True
            self.starved = True
            reward = STARVE_REWARD

        if self.current_food != 'no food' and self.current_score < CLOSER_TO_FOOD_REWARD_SCORE_LIMIT:
            moves_to_food = distance_to_food(self.head.tile_pos, self.current_food.position)
            if moves_to_food < old_moves_to_food:
                reward += FOOD_DISTANCE_REWARD
            else:
                reward -= FOOD_DISTANCE_REWARD

        restart_and_print_time('game step', start_time)

        return self.finished, reward

    def check_perfect_game(self):
        return (self.current_score + START_SEGMENTS + 1) == ((SCREENTILES[0] + 1) * (SCREENTILES[1] + 1))

    def render(self):
        if not self.display:
            return

        # unfreezes the window if stuck
        pygame.event.pump()

        if self.perfect_game:
            print('PERFECT GAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111!!!!!!!!')
            f = pygame.font.Font(None, 25)
            fail_message = f.render('PERFECT GAME!!!', True, (0, 0, 0))
            fail_rect = fail_message.get_rect()
            fail_rect.center = SCREENRECT.center
            self.screen.blit(fail_message, fail_rect)
            pygame.display.flip()
            pygame.time.wait(5000)
            return

        if self.finished:
            if self.starved:
                f = pygame.font.Font(None, 60)
                death_reason = 'NO FUD'
            else:
                death_reason = 'DED'
                f = pygame.font.Font(None, 100)
            fail_message = f.render(death_reason, True, (0, 0, 0))
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

        if self.limit_fps:
            self.clock.tick(FPS_LIMIT)

        # slow down when close to finished
        if self.current_score >= SCORE_SLOW_THRESHOLD:
            self.clock.tick(SCORE_THRESHOLD_FPS)
