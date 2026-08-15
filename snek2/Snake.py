import collections
import random
import snake_constants

from state_helpers import *

# Everything needed to rebuild a game elsewhere, as plain data. See Game.snapshot().
GameSnapshot = collections.namedtuple('GameSnapshot', [
    'body',                 # ordered body cells, head first
    'head_move_dir',
    'tail_last_move_dir',   # currently dead — see Game._build_snake()
    'food',                 # (x, y), or None for the 'no food' sentinel
    'current_score', 'current_step', 'last_food_step',
    'finished', 'starved', 'perfect_game',
])


def direction_between(start, end):
    """The move name that steps from `start` to `end`, which must be adjacent.

    Raises rather than returning None, because every caller here is asserting adjacency as
    much as it is asking for the direction.
    """
    delta = (end[0] - start[0], end[1] - start[1])
    for name, vector in MOVE_VECTORS.items():
        if vector == delta:
            return name
    raise ValueError('cells {0} and {1} are not adjacent'.format(start, end))


def validate_snapshot(snapshot):
    """Rejects a snapshot that could not have come from a live game.

    Cheap next to a restore, and the alternative is silent: a body with a repeated cell
    restores fine and then reports a self-collision on its very first groupcollide, which
    reads as a policy that suddenly plays badly rather than as corrupt state.

    Note a *finished* game is not restorable — on a death the head is left off the board, so
    its snapshot fails the bounds check here. Nothing needs to restore one.
    """
    body = snapshot.body
    if not body:
        raise ValueError('a snapshot needs at least one body cell')
    # A live snake never overlaps itself, so a repeated cell means the snapshot is of a *finished*
    # game — a self-collision death leaves the head sitting on the body. Measured over 300 random
    # episodes: 149 deaths by self-collision, 151 off the board, and **zero** overlapping bodies in
    # 2743 live steps. This check earned its place on its first smoke run by catching the forking
    # collector snapshotting a game that had already ended.
    if len(set(body)) != len(body):
        raise ValueError('snapshot body has a repeated cell, so this game has already ended: {0}'
                         .format(body))
    for cell in body:
        if not (0 <= cell[0] <= SCREENTILES[0] and 0 <= cell[1] <= SCREENTILES[1]):
            raise ValueError('snapshot body cell {0} is off the board'.format(cell))
    for ahead, behind in zip(body, body[1:]):
        direction_between(behind, ahead)
    if snapshot.food is not None and tuple(snapshot.food) in set(body):
        raise ValueError('snapshot food {0} sits on the body'.format(snapshot.food))
    if snapshot.head_move_dir not in MOVE_VECTORS:
        raise ValueError('unknown head_move_dir {0}'.format(snapshot.head_move_dir))


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
    def __init__(self, taken_up_group, position=None):
        pygame.sprite.Sprite.__init__(self)
        self.image = self.image = pygame.Surface(TILE_SIZE).convert()
        self.image.fill(COLORKEY_COLOR)
        self.image.set_colorkey(COLORKEY_COLOR)
        pygame.draw.circle(self.image, FOOD_COLOR, TILE_RECT.center, FOOD_RADIUS)

        self.rect = self.image.get_rect()
        # An explicit position is how a snapshot restores its food, and it deliberately
        # short-circuits the sampling loop below rather than seeding it. That loop draws from
        # the module-global `random`, so a restore that went through it would consume RNG the
        # original game never consumed and shift every later food placement in the process.
        # With no position the draws below are unchanged, in the same order.
        if position is not None:
            self.position = (position[0], position[1])
            self.rect.topleft = (self.position[0] * TILE_SIZE[0], self.position[1] * TILE_SIZE[1])
            return
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
    def __init__(self, display=True, limit_fps=False, policy_name='', discount=1.0):
        self.display = display
        self.limit_fps = limit_fps
        self.policy_name = policy_name
        # The discount the *agent* trains with, threaded in from SnakeEnvironment rather than
        # re-read from the environment here, because snek2.py holds its default (`tuned('DISCOUNT',
        # 0.99)`) and a second copy of that number would drift silently. Only the potential-based
        # shaping term uses it: with gamma = 1 the shaped rewards telescope in undiscounted terms
        # while the agent discounts, which leaves a residual per-step bonus for being chase-safe —
        # a different and unprincipled term. The 1.0 default is what the frozen diagnostics that
        # build a Game directly get, and they all run with shaping off.
        self.shaping_discount = discount
        # Phi(s) for the board as it currently stands, cached between steps so the flood fill runs
        # once per step rather than twice. reset() and restore_snapshot() both set it.
        self.chase_safe_potential = 0.0
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

    def _new_groups(self):
        """Fresh, empty sprite groups.

        Shared by reset() and restore_snapshot() rather than written twice, for the reason in
        _rebuild_grid's docstring: this file has already paid once for two builders that
        disagreed with each other.
        """
        self.snake_group = pygame.sprite.Group()
        self.snake_head_group = pygame.sprite.Group()
        self.food_group = pygame.sprite.Group()
        self.taken_up_group = pygame.sprite.Group()
        self.all = pygame.sprite.RenderUpdates()

    def reset(self):
        pygame.display.set_caption(self.caption)
        self.bg = pygame.Surface(SCREENSIZE).convert()
        self.bg.fill(BACKGROUND_COLOR)
        self.screen.blit(self.bg, (0, 0))

        self._new_groups()
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

        # After _rebuild_grid(), because the potential reads the grid. Below any sensible gate at
        # the opening length of 5, so this is 0 for the gated form — which is what makes the
        # episode's discounted shaping telescope to exactly 0.
        self.chase_safe_potential = self._chase_safe_potential()

    def _chase_safe_potential(self):
        """Phi(s) for the current board: chase-safety, gated by snake length. 0.0 when shaping is off.

        Gated because Phase 0 chose the length-gated variant — see CHASE_SAFE_GATE. The gate is
        tested *before* the flood fill, so a below-gate board costs nothing but a comparison, which
        is most of the episode.

        Returns 0.0 whenever CHASE_SAFE_SHAPING is 0.0 so the ablation skips `count_groups`
        entirely. That cannot change the food stream: `count_groups` draws no randomness.
        """
        if not CHASE_SAFE_SHAPING:
            return 0.0
        if len(self.snake_group) < CHASE_SAFE_GATE:
            return 0.0
        return float(chase_safe_state(self.grid, self.head.tile_pos, self.tail.tile_pos,
                                      self.current_food))

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
                                         len(self.snake_group)))

    def snapshot(self):
        """Everything needed to rebuild this game elsewhere, as plain data.

        `copy.deepcopy` cannot do this. Every sprite owns a pygame Surface, and Surface, Clock
        and Font all refuse to pickle; `self.screen` is worse, because
        pygame.display.set_mode() returns one process-global surface, so two Games share it and
        no copy could own its own.

        What is left out is either derived or write-only. `self.grid` is rebuilt from the snake
        and food on every step, each sprite's image and rect come from its tile_pos, and
        `total_steps` and SnakeHead.move_count are written but never read anywhere.
        """
        return GameSnapshot(
            body=tuple(self.snake.get_positions()),
            head_move_dir=self.head.move_dir,
            tail_last_move_dir=self.tail.last_move_dir,
            food=None if self.current_food == 'no food' else tuple(self.current_food.position),
            current_score=self.current_score,
            current_step=self.current_step,
            last_food_step=self.last_food_step,
            finished=self.finished,
            starved=self.starved,
            perfect_game=self.perfect_game)

    def restore_snapshot(self, snapshot):
        """Makes this game an independent copy of whatever `snapshot` describes.

        Independent in the only sense that matters: it shares no sprite, group or grid with the
        game the snapshot came from, so the two play on without touching each other. They do
        share `self.screen`, because pygame allows one display surface per process — harmless
        under SDL_VIDEODRIVER=dummy, which is the only way training ever runs, and each game
        keeps its own `bg` to blit through.
        """
        validate_snapshot(snapshot)
        if self.bg is None:
            # A pooled Game that has never been reset() has no background, and step() blits
            # through it on every call via all.clear().
            self.bg = pygame.Surface(SCREENSIZE).convert()
            self.bg.fill(BACKGROUND_COLOR)

        self._new_groups()
        self._build_snake(snapshot.body, snapshot.head_move_dir, snapshot.tail_last_move_dir)

        if snapshot.food is None:
            self.current_food = 'no food'
        else:
            self.current_food = Food(self.taken_up_group, position=snapshot.food)
            self.food_group.add(self.current_food)
            self.taken_up_group.add(self.current_food)
            self.all.add(self.current_food)

        self.current_score = snapshot.current_score
        self.current_step = snapshot.current_step
        self.last_food_step = snapshot.last_food_step
        self.finished = snapshot.finished
        self.starved = snapshot.starved
        self.perfect_game = snapshot.perfect_game

        self._rebuild_grid()

        # **The potential is per-episode state, and the snapshot does not carry it**, so a forked
        # branch would otherwise compute its first F against the value left behind by whatever
        # branch last used this pooled env — an arbitrary constant of up to +/-c in the branch's
        # return, which breaks the telescope on exactly the transitions the term exists to improve.
        # Live rather than hypothetical: the base config runs FORK_BRANCHES=4 with forks at length
        # >= 85, and `ForkingCollector._fork` pops envs off a reused pool.
        #
        # Recomputed rather than added to GameSnapshot: Phi is a pure function of grid, head, tail
        # and food, all of which are restored exactly above, so this is byte-identical to carrying
        # the value and it leaves GameSnapshot, validate_snapshot and test_game_snapshot alone.
        self.chase_safe_potential = self._chase_safe_potential()

    def _build_snake(self, body, head_move_dir, tail_last_move_dir):
        """Builds a snake of arbitrary shape from ordered body cells, head first.

        The general form of what reset() does. reset() can only make the fixed opening shape,
        because add_segment() appends one cell at a time using the tail's last_move_dir.

        **Each segment's move_dir is derived here, and it is the load-bearing field.** In
        SnakeSegment.move() a segment advances with its *own* move_dir and only then inherits
        its parent's, so a body whose directions are wrong walks apart on the very first step —
        leaving them at the constructor's 'right' default mismatches immediately. The invariant
        is that segment i's move_dir points at the cell of the segment ahead of it, which makes
        it exactly `direction_between(body[i], body[i - 1])`.

        `tail_last_move_dir` is restored for faithfulness but is **dead state today**: step()
        moves the whole snake before the food block calls add_segment(), so the only reader of
        `last_move_dir` always sees a value written by that move, never the one restored here.
        Kept because a future reordering of step() would make it live again, and a snapshot that
        silently dropped a field would be worse than one carrying a redundant one.
        """
        groups = [self.snake_group, self.all, self.taken_up_group]
        self.snake = SnakeHead(body[0], head_move_dir, groups)
        self.snake_head_group.add(self.snake)
        self.head = self.snake

        previous = self.snake
        for cell in body[1:]:
            segment = SnakeSegment(cell, groups)
            segment.move_dir = direction_between(cell, previous.tile_pos)
            segment.last_move_dir = segment.move_dir
            segment.front_segment = previous
            previous.behind_segment = segment
            previous = segment

        # No `previous.behind_segment = None` here on purpose: every segment in this loop is
        # freshly constructed and SnakeSegment.__init__ already leaves both links None, so the
        # assignment is dead. It was here, and no mutation of it could fail a test.
        # `test_the_segment_chain_is_linked_both_ways_and_ends_at_the_tail` is what holds the
        # invariant, so a change to the constructor still gets caught.
        self.tail = previous
        self.tail.last_move_dir = tail_last_move_dir

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
                                                      len(self.snake_group)) <= 0:
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

        # Potential-based shaping (Ng, Harada and Russell 1999): F = c * (gamma * Phi(s') - Phi(s)).
        # This form leaves the optimal policy unchanged for any bounded Phi, which is the reason to
        # prefer it here — the marker it rests on is correlational, so a term that *could* move the
        # optimum would be a real risk. It can only make learning faster or slower.
        #
        # Its position at the end of step() is load-bearing. By here _rebuild_grid() has run, so
        # self.grid is the post-move board; every branch that can set self.finished has run; and on
        # a step that ate, the replacement food is already placed, so Phi(s') is measured against
        # the **new** food. That last point is what the per-action flag at obs[15 + a] cannot give.
        #
        # `Phi(terminal) = 0` is required by the theory, and the branch is also what keeps this off
        # an unusable grid: on a death the head is off the board.
        #
        # A perfect game therefore pays -c at the winning step. Required, and negligible against
        # PERFECT_GAME_REWARD = 100.
        if CHASE_SAFE_SHAPING:
            new_potential = 0.0 if self.finished else self._chase_safe_potential()
            reward += CHASE_SAFE_SHAPING * (self.shaping_discount * new_potential
                                            - self.chase_safe_potential)
            self.chase_safe_potential = new_potential

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
