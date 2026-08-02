import random
import snake_constants

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
    def __init__(self, display=True, limit_fps=False, policy_name=''):
        self.display = display
        self.limit_fps = limit_fps
        self.policy_name = policy_name
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
        # Window title. The arm name leads, because macOS truncates from the right and the
        # arm is what tells two watch.py windows apart. Settable from outside — watch.py
        # appends the checkpoint step it loaded — and reset() re-applies it every episode, so
        # an override sticks rather than being overwritten on the next reset.
        self.caption = '{0} — {1}'.format(policy_name, CAPTION) if policy_name else CAPTION
        self.clock = pygame.time.Clock()
        # Init only the subsystems this game uses. Bare pygame.init() also starts
        # pygame.mixer, which opens a real CoreAudio output stream per process even
        # under SDL_VIDEODRIVER=dummy. Nothing here plays sound, but 10 idle workers
        # holding those streams pushed coreaudiod to 15% CPU, and evals commonly run
        # several 10-worker processes at once. pygame.time/sprite/draw need no init.
        pygame.display.init()
        pygame.font.init()
        self._fonts = {}

    def set_display(self, enabled):
        self.display = enabled

    def reset(self):
        pygame.display.set_caption(self.caption)
        self.bg = pygame.Surface(SCREENSIZE).convert()
        self.bg.fill(BACKGROUND_COLOR)
        self.screen.blit(self.bg, (0, 0))

        self.snake_group = pygame.sprite.Group()
        self.snake_head_group = pygame.sprite.Group()
        self.food_group = pygame.sprite.Group()
        self.taken_up_group = pygame.sprite.Group()

        self.all = pygame.sprite.RenderUpdates()
        self.snake = SnakeHead(START_TILE, 'right', [self.snake_group, self.all, self.taken_up_group])
        self.snake_head_group.add(self.snake)
        self.head = self.snake
        self.tail = self.snake
        for index in range(START_SEGMENTS):
            self.add_segment()

        self.current_food = Food(self.taken_up_group)  # check that score was updated
        self.food_group.add(self.current_food)
        self.taken_up_group.add(self.current_food)
        self.all.add(self.current_food)

        self.current_score = 0
        self.current_step = 0
        self.last_food_step = 0

        # The first observation of an episode is read straight after reset(), so the grid has
        # to be built here too. reset() used to build its own — a bare (10, 10) of zeros with
        # no border, no snake and no food — while step() builds a (12, 12) with wall cells.
        # Every episode's opening observation was therefore computed against an empty,
        # wrongly-shaped grid. It happened to produce the same 20 numbers because the snake
        # always starts mid-board with all three moves open, but it is not robust: with the
        # tail near an edge the smaller array raises IndexError out of get_grid_value(),
        # which bounds-checks against SCREENTILES + 1 and so assumes the (12, 12) form.
        self._rebuild_grid()

        # turn screen to white
        pygame.display.flip()

        # mainloop
        self.finished = False
        self.starved = False
        self.perfect_game = False

    def get_observation(self):
        # Each segment lands on the cell its predecessor just left, so the tail's next cell is
        # wherever the segment ahead of it stands now. group_obs needs that as well as the
        # current tail: the tail moves on the same step the head does, and asking about the cell
        # it is leaving rather than the one it is taking is what made head_with_tail go silent in
        # coiled endgames. front_segment is None only for a length-1 snake, which cannot happen
        # while START_SEGMENTS is at least 1, but the fallback keeps this honest.
        ahead_of_tail = self.tail if self.tail.front_segment is None else self.tail.front_segment
        return np.array(get_observations(self.grid,
                                         self.head.tile_pos,
                                         self.tail.tile_pos,
                                         ahead_of_tail.tile_pos,
                                         self.head.move_dir,
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

    def _rebuild_grid(self):
        """Rewrites self.grid from the current snake and food positions.

        Cell values: 0 empty, 1 food, 2 head, 3 body, 4 wall. The array is two cells wider
        and taller than the board so the outer ring can hold the walls, which is why every
        lookup offsets by +1 — see get_grid_value() in state_helpers.

        Shared by reset() and step() on purpose. These were two separate copies that
        disagreed on the shape, and the observation code assumes the bordered form.
        """
        column = SCREENSIZE[0] // TILE_SIZE[0]
        row = SCREENSIZE[1] // TILE_SIZE[1]
        self.grid = np.zeros((row + 2, column + 2))
        self.grid[[0, -1], :] = 4
        self.grid[:, [0, -1]] = 4

        # 1 is food
        if self.current_food != 'no food':
            self.grid[self.current_food.position[1] + 1, self.current_food.position[0] + 1] = 1

        body_positions = self.snake.get_positions()
        for i in range(0, len(body_positions)):
            position = body_positions[i]
            if i == 0:
                # 2 is head
                self.grid[position[1] + 1, position[0] + 1] = 2
            else:
                if not (position[1] < 0 or position[0] < 0):
                    # 3 is body part
                    self.grid[position[1] + 1, position[0] + 1] = 3

    def step(self, relative_direction):
        # Counted up front so `current_step` already includes this step when it is compared
        # against `last_food_step` below. These used to be incremented near the end, after
        # the food block had set `last_food_step = current_step`, which left
        # `current_step - last_food_step == 1` on the step immediately after eating and so
        # docked one step from every starve budget.
        self.current_step += 1
        self.total_steps += 1

        if self.current_food != 'no food':
            old_moves_to_food = distance_to_food(self.head.tile_pos, self.current_food.position)
        else:
            old_moves_to_food = 0

        reward = 0.0

        # remap relative direction to cardinal directions
        self.snake.move_dir = CURRENT_DIRECTION_MAPS[self.snake.move_dir][relative_direction]

        # clearing
        self.all.clear(self.screen, self.bg)

        # updates snake position
        self.all.update()

        # head -> food
        col = pygame.sprite.groupcollide(self.snake_head_group, self.food_group, False, True)
        ate_food = len(col) > 0
        if ate_food:
            self.current_food = 'no food'
            self.add_segment()
            self.current_score += 1
            self.last_food_step = self.current_step
            reward = FOOD_REWARD
            if self.check_perfect_game():
                self.finished = True
                self.perfect_game = True
                reward = snake_constants.PERFECT_GAME_REWARD
        if self.current_food == 'no food' and not self.perfect_game:
            self.current_food = Food(self.taken_up_group)  # check that score was updated
            self.food_group.add(self.current_food)
            self.taken_up_group.add(self.current_food)
            self.all.add(self.current_food)

        self._rebuild_grid()

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

        # game over
        if self.check_perfect_game():
            self.finished = True
            self.perfect_game = True
            reward = snake_constants.PERFECT_GAME_REWARD
        elif not self.finished and steps_until_starve(self.current_step,
                                                      self.last_food_step,
                                                      len(self.snake_group))[0] <= 0:
            self.finished = True
            self.starved = True
            reward = STARVE_REWARD

        # Distance shaping, and only for an ordinary move. Skipped when this step ate,
        # because `old_moves_to_food` measures the food that was just consumed while
        # `current_food` is already its randomly placed replacement — comparing the two
        # is meaningless, and since the head had to be adjacent to eat, old_moves_to_food
        # is always 1 and the replacement is almost never that close. The penalty
        # therefore fired on 96.8% of food-eating steps (measured over 4825), quietly
        # making every FOOD_REWARD 0.999. Skipped when the episode ended for the same
        # reason: on a death the head is off the board, so the comparison is noise on top
        # of DEATH_REWARD.
        if not ate_food and not self.finished and self.current_food != 'no food':
            moves_to_food = distance_to_food(self.head.tile_pos, self.current_food.position)
            if moves_to_food > old_moves_to_food:
                reward -= FOOD_DISTANCE_REWARD

        return self.finished, reward

    def check_perfect_game(self):
        return (self.current_score + START_SEGMENTS + 1) == PERFECT_SCORE

    def _font(self, size):
        """Cached pygame.font.Font by size.

        Constructing a Font parses the font file: ~122us each, and render() built three of
        them per frame. That was 366us of a 5300us frame — second only to the display flip.
        """
        font = self._fonts.get(size)
        if font is None:
            font = pygame.font.Font(None, size)
            self._fonts[size] = font
        return font

    def _hud_font(self):
        return self._font(HUD_FONT_SIZE)

    def _message_font(self, text, max_size):
        """Largest font at or below max_size whose rendering of `text` fits across the window.

        The end-of-game messages are sized to fill the window, but 'DED' at 100pt is already
        ~150px wide against a 100px board, so both D's were clipped off the edges — and scaling
        with the window reproduced that at every size. Shrinking to fit keeps the message
        full-width without cutting it in half. Runs once per episode, on the death frame, and
        every size it tries lands in the same font cache.
        """
        limit = int(SCREENSIZE[0] * 0.95)
        size = max_size
        while size > 8 and self._font(size).size(text)[0] > limit:
            size = int(size * 0.9)
        return self._font(size)

    def render(self):
        if self.perfect_game and snake_constants.DEBUG_LOGGING:
            print('PERFECT GAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!!1!!1!!!!11111!!!!111!!!!!!!!')

        if not self.display:
            return

        # unfreezes the window if stuck
        pygame.event.pump()

        if self.perfect_game:
            f = self._message_font('PERFECT GAME!!!', PERFECT_FONT_SIZE)
            fail_message = f.render('PERFECT GAME!!!', True, (0, 0, 0))
            fail_rect = fail_message.get_rect()
            fail_rect.center = SCREENRECT.center
            self.screen.blit(fail_message, fail_rect)
            pygame.display.flip()
            pygame.time.wait(snake_constants.PERFECT_GAME_WAIT_MS)
            return

        if self.finished:
            if self.starved:
                death_reason = 'NO FUD'
                f = self._message_font(death_reason, STARVE_FONT_SIZE)
            else:
                death_reason = 'DED'
                f = self._message_font(death_reason, DEATH_FONT_SIZE)
            fail_message = f.render(death_reason, True, (0, 0, 0))
            fail_rect = fail_message.get_rect()
            fail_rect.center = SCREENRECT.center
            self.screen.blit(fail_message, fail_rect)
            pygame.display.flip()
            pygame.time.wait(10)
            return

        # score
        f = self._hud_font()
        d = self.screen.blit(self.bg, SCORE_POS, pygame.Rect(SCORE_POS, HUD_ERASE_SIZE))
        score_image = f.render(SCORE_PREFIX + str(self.current_score), True, SCORE_COLOR)
        d2 = self.screen.blit(score_image, SCORE_POS)
        # steps
        d3 = self.screen.blit(self.bg, STEP_POS, pygame.Rect(STEP_POS, HUD_ERASE_SIZE))
        step_image = f.render(STEP_PREFIX + str(self.current_step), True, STEP_COLOR)
        d4 = self.screen.blit(step_image, STEP_POS)
        # policy name
        d5 = self.screen.blit(self.bg, POLICY_POS, pygame.Rect(POLICY_POS, HUD_ERASE_SIZE))
        policy_image = f.render(POLICY_PREFIX + self.policy_name, True, POLICY_COLOR)
        d6 = self.screen.blit(policy_image, POLICY_POS)

        # drawing
        dirty = self.all.draw(self.screen)
        dirty.append(d)
        dirty.append(d2)
        dirty.append(d3)
        dirty.append(d4)
        dirty.append(d5)
        dirty.append(d6)

        # updating screen
        pygame.display.update(dirty)

        if self.limit_fps:
            # Read through the module, not the star-imported name: `from snake_constants import
            # *` binds a copy at import time, so watch.py setting snake_constants.FPS_LIMIT
            # would have had no effect here. PERFECT_GAME_WAIT_MS above is qualified for the
            # same reason.
            self.clock.tick(snake_constants.FPS_LIMIT)

        # slow down when close to finished
        if self.current_score >= SCORE_SLOW_THRESHOLD:
            self.clock.tick(SCORE_THRESHOLD_FPS)
