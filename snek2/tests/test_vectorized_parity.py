"""Parity between `vectorized.vec_env.VecSnake` and the live `Snake.Game` it reimplements.

**The argument these tests exist to establish.** If the vectorised observation is bit-identical to
the reference's on every step under the same food, then a greedy policy's argmax is identical, so the
action sequence is identical, so the episode is identical, so any measurement taken over those
episodes is identical *by construction*. That is why parity is asserted elementwise on the
observation rather than statistically on a win rate: a statistical check at n=500 and ~99% perfect
can only bound a systematic bias to about a point, and would never see a divergence that fires on one
board topology in ten thousand.

**Why food is forced rather than seeded.** The reference rejection-samples the module-global `random`
an unpredictable number of times per placement — it retries until the draw misses the snake, which
near a full board is 20-50 draws — so no seeding discipline can align the two streams. Every test
here steps the reference first, reads where it actually put the food, and forces the vectorised env to
match. Distribution equality is a separate question, tested in `test_vectorized_env.py`.

**Why several reference games run at once.** A single-env `VecSnake` pays full batch overhead for one
board, so the harness runs `LANES` independent `Game`s against one `LANES`-wide env. That is faster
*and* covers more board shapes per step, which matters because the connectivity code is only stressed
by coiled endgame boards that short episodes never reach.
"""

import numpy as np

import snake_constants as sc
import state_helpers
from Snake import Game, GameSnapshot, direction_between
from vectorized import config as C
from vectorized import vec_env as V

LANES = 8
DISCOUNT = 0.9975


# ------------------------------------------------------- constructed hard boards

def _coiled_body(rng, length):
    """A self-avoiding walk of `length` cells, biased to coil rather than sprawl.

    **Why this exists.** No heuristic policy in this file gets past about length 79, and the
    observation code's hardest work — region enumeration, `head_with_tail`, `safe_to_chase_food` —
    only becomes interesting once the snake has cut the free space into pieces, which is the last
    fifteen cells of a perfect game. Parity that stops at 79 leaves the endgame untested, and the
    endgame is what a 99%-perfect checkpoint spends its time in.

    The walk is Warnsdorff's rule (step to the neighbour with the fewest free neighbours) with
    random tie-breaks. Plain random self-avoiding walks almost never reach length 90 on a 10x10
    board; Warnsdorff reaches it nearly every attempt, and the tie-breaks keep the shapes varied
    instead of returning the same serpentine every call.
    """
    span = sc.SCREENTILES[0] + 1
    while True:
        free = np.ones((span, span), dtype=bool)
        cell = (int(rng.integers(span)), int(rng.integers(span)))
        free[cell] = False
        path = [cell]
        while len(path) < length:
            options = []
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = cell[0] + dx, cell[1] + dy
                if 0 <= nx < span and 0 <= ny < span and free[nx, ny]:
                    onward = sum(
                        1 for ax, ay in ((-1, 0), (1, 0), (0, -1), (0, 1))
                        if 0 <= nx + ax < span and 0 <= ny + ay < span and free[nx + ax, ny + ay])
                    options.append((onward, nx, ny))
            if not options:
                break
            fewest = min(o[0] for o in options)
            best = [o for o in options if o[0] == fewest]
            _, nx, ny = best[int(rng.integers(len(best)))]
            cell = (nx, ny)
            free[cell] = False
            path.append(cell)
        if len(path) == length:
            # Head at the walk's tip, so the body reads head-first like `get_positions()`.
            return list(reversed(path)), free


def starve_budget_for(length):
    return max(C.MIN_STARVE_BUDGET, min(C.STARVE_MULT * length, C.MAX_STARVE_BUDGET))


def _coiled_snapshot(rng, length, since_food=None):
    """A legal `GameSnapshot` of a coiled snake, with food on a free cell.

    `tail_last_move_dir` is set to the direction along the body at the tail. `validate_snapshot`
    does not check it and no observation reads it, so any legal direction would do — but pointing it
    along the body is the value a real game would most often carry, and a wrong-looking constant
    here would be the first thing a future reader distrusted.

    `since_food` overrides how long it has been since the last meal, and it exists because the
    starve path is otherwise unreachable. The budget is 100-500 steps, a coiled endgame board
    resolves in a few dozen, and the growth run eats every few steps — so no regime here starves by
    accident. Setting it to `budget - 1` makes the next step starve unless that step happens to eat,
    which is what gives the starve reward and termination any coverage at all. It was a surviving
    mutant (`starve_reward_off_by_one`) that showed this was missing.
    """
    body, free = _coiled_body(rng, length)
    cells = np.flatnonzero(free)
    food = None
    if cells.size:
        pick = int(cells[int(rng.integers(cells.size))])
        food = (pick // free.shape[1], pick % free.shape[1])
    # Vary how starved the snake is, including right up against the budget, so index 21 and the
    # starve termination are both exercised rather than always reading "just ate".
    budget = starve_budget_for(length)
    if since_food is None:
        since_food = int(rng.integers(0, budget))
    step = max(500, since_food + 1)
    return GameSnapshot(
        body=tuple(body),
        head_move_dir=direction_between(body[1], body[0]),
        tail_last_move_dir=direction_between(body[-1], body[-2]),
        food=food,
        current_score=length - sc.START_SEGMENTS - 1,
        current_step=step,
        last_food_step=step - since_food,
        finished=False,
        starved=False,
        perfect_game=False)


def _flat_tile(tile):
    return V.flat(int(tile[0]), int(tile[1]))


def _food_flat(game):
    """The reference's current food as a flat index, or -1 for its 'no food' sentinel."""
    if game.current_food == 'no food':
        return -1
    pos = game.current_food.position
    return V.flat(int(pos[0]), int(pos[1]))


def _push_reference_state(vec, games, rows=None):
    """Copy the named reference games' boards into the vectorised env, so both are identical.

    Goes through `snapshot()` rather than poking at sprites, so it exercises the same data a
    checkpoint restore or a forked collector would carry. `rows` is passed through to `set_state`
    for the reason documented there: pushing a row re-derives its shaping potential, so a row that
    did not restart must be left alone or the shaping tests lose their teeth.
    """
    rows = list(range(len(games))) if rows is None else list(rows)
    snaps = [games[r].snapshot() for r in rows]
    width = max(len(s.body) for s in snaps)
    bodies = np.zeros((len(snaps), width), dtype=np.int64)
    lengths, dirs, foods, steps, last_food, scores = [], [], [], [], [], []
    for row, snap in enumerate(snaps):
        for k, cell in enumerate(snap.body):
            bodies[row, k] = _flat_tile(cell)
        lengths.append(len(snap.body))
        dirs.append(sc.DIRECTIONS.index(snap.head_move_dir))
        foods.append(-1 if snap.food is None else V.flat(int(snap.food[0]), int(snap.food[1])))
        steps.append(snap.current_step)
        last_food.append(snap.last_food_step)
        scores.append(snap.current_score)
    vec.set_state(bodies, lengths, dirs, foods, steps, last_food, scores, rows=rows)


class _Lockstep:
    """Drives `LANES` reference games and one vectorised env through identical action sequences."""

    def __init__(self, seed=0, epsilon=0.05, lanes=LANES, board=None):
        """`board` is an optional `rng -> GameSnapshot` factory used in place of `Game.reset()`.

        It is how the endgame gets covered: a factory returning coiled length-90 boards restarts
        every lane into a hard position, and since such a position resolves within a few dozen steps
        the run spends nearly all its time there rather than climbing from length 5.
        """
        self.lanes = lanes
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.board = board
        self.games = [Game(display=False, limit_fps=False, policy_name='parity',
                           discount=DISCOUNT) for _ in range(lanes)]
        for g in self.games:
            self._restart(g)
        self.vec = V.VecSnake(lanes, seed=seed, shaping_discount=DISCOUNT)
        _push_reference_state(self.vec, self.games)
        self.obs_mismatches = []
        self.reward_mismatches = []
        self.flag_mismatches = []
        self.states = 0
        self.steps = 0
        self.episodes = 0
        self.perfect_games = 0
        self.max_length = 0
        # Summed so a test can prove that forcing shaping on actually changed a reward; without
        # that, a vec env ignoring the shaping term entirely would pass the shaping-parity tests.
        self.reward_sum = 0.0
        self.length_hist = np.zeros(C.PERFECT_SCORE + 2, dtype=np.int64)

    def _restart(self, game):
        """Reset, then optionally overwrite with a constructed board.

        `reset()` first even when a factory is supplied, because `restore_snapshot` assumes a game
        that has been reset at least once — a never-reset Game has no background surface, and
        `step` blits through it.
        """
        game.reset()
        if self.board is not None:
            game.restore_snapshot(self.board(self.rng))

    def _choose(self, obs):
        """A deterministic survival policy, plus seeded exploration for board diversity.

        Prefers a legal move, then one that keeps the tail reachable, then one that closes on the
        food. Random play dies at length ~5 and would never exercise the connectivity block, which is
        the whole reason these tests are slow enough to need a policy at all.
        """
        score = obs[:, 6:9] * 100.0 + obs[:, 9:14:2] * 10.0 + obs[:, 0:6:2]
        actions = np.argmax(score, axis=1)
        explore = self.rng.random(self.lanes) < self.epsilon
        if explore.any():
            legal = obs[:, 6:9] > 0.5
            noise = self.rng.random((self.lanes, 3)) + legal * 10.0
            actions = np.where(explore, np.argmax(noise, axis=1), actions)
        return actions.astype(np.int64)

    def run(self, max_states):
        while self.states < max_states:
            ref_obs = np.stack([g.get_observation() for g in self.games]).astype(np.float32)
            vec_obs = self.vec.observe()
            for lane in range(self.lanes):
                self.states += 1
                bad = np.flatnonzero(ref_obs[lane] != vec_obs[lane])
                if bad.size:
                    self.obs_mismatches.append({
                        'lane': lane, 'indices': bad.tolist(),
                        'ref': ref_obs[lane][bad].tolist(),
                        'vec': vec_obs[lane][bad].tolist(),
                        'length': int(self.vec.length[lane]),
                        'step': int(self.vec.step_count[lane])})

            actions = self._choose(vec_obs)

            # Reference first, so its food placement is known before the vec env needs it.
            ref_done, ref_reward, ref_perfect, ref_starved = [], [], [], []
            for lane, game in enumerate(self.games):
                finished, reward = game.step(sc.ACTIONS[int(actions[lane])])
                ref_done.append(bool(finished))
                ref_reward.append(float(reward))
                ref_perfect.append(bool(game.perfect_game))
                ref_starved.append(bool(game.starved))
            forced = np.array([_food_flat(g) for g in self.games], dtype=np.int64)
            # A row that did not eat ignores `forced_food`; passing its unchanged food is harmless.
            forced = np.where(forced < 0, 0, forced)

            _, vec_reward, vec_done, info = self.vec.step(
                actions, forced_food=forced, autoreset=False)
            self.steps += self.lanes

            for lane in range(self.lanes):
                if not np.isclose(vec_reward[lane], ref_reward[lane], rtol=0, atol=1e-9):
                    self.reward_mismatches.append({
                        'lane': lane, 'ref': ref_reward[lane], 'vec': float(vec_reward[lane]),
                        'length': int(self.vec.length[lane])})
                if (bool(vec_done[lane]) != ref_done[lane]
                        or bool(info['perfect'][lane]) != ref_perfect[lane]
                        or bool(info['starved'][lane]) != ref_starved[lane]):
                    self.flag_mismatches.append({
                        'lane': lane,
                        'ref': (ref_done[lane], ref_perfect[lane], ref_starved[lane]),
                        'vec': (bool(vec_done[lane]), bool(info['perfect'][lane]),
                                bool(info['starved'][lane]))})
                if ref_perfect[lane]:
                    self.perfect_games += 1
                self.reward_sum += float(vec_reward[lane])

            finished = np.flatnonzero(vec_done)
            if finished.size:
                for lane in finished:
                    length = int(self.vec.length[lane])
                    self.max_length = max(self.max_length, length)
                    self.length_hist[min(length, C.PERFECT_SCORE + 1)] += 1
                    self.episodes += 1
                    self._restart(self.games[lane])
                # Reset the reference first, then adopt its fresh board — stricter than mirroring
                # the reset here, since it re-checks `set_state` on every episode boundary. Only the
                # lanes that restarted are pushed; see `_push_reference_state`.
                _push_reference_state(self.vec, self.games, rows=finished)
            else:
                self.max_length = max(self.max_length, int(self.vec.length.max()))
        return self

# ------------------------------------------------------------------ shared runs

_GROWTH = None
_ENDGAME = None


def _growth_run():
    """A full-game run from `reset()`: covers the climb from length 5, where most steps happen."""
    global _GROWTH
    if _GROWTH is None:
        _GROWTH = _Lockstep(seed=7).run(24000)
    return _GROWTH


def _endgame_run():
    """Constructed coiled boards at lengths 80-99, where the connectivity code does its real work.

    Lengths are drawn per board rather than fixed, so one run spans the whole band. A length-99
    board wins on its first step and a length-80 one survives a few dozen, which is why the episode
    count here is enormous next to the growth run's.
    """
    global _ENDGAME
    if _ENDGAME is None:
        rng_lengths = np.random.default_rng(101)

        def board(rng):
            return _coiled_snapshot(rng, int(rng_lengths.integers(80, 100)))

        _ENDGAME = _Lockstep(seed=23, board=board).run(12000)
    return _ENDGAME


def _both():
    return (('growth', _growth_run()), ('endgame', _endgame_run()))


# --------------------------------------------------------------- L1: observations

def test_observation_parity_elementwise():
    """All 30 indices, every state, both regimes. This is the assertion the eval driver rests on.

    If the observation is identical then a greedy policy's argmax is identical, so the whole episode
    is identical and any measurement over it is identical *by construction* — which is why parity is
    asserted here rather than by comparing win rates, where n=500 at ~99% could not see a divergence
    that fires on one board topology in ten thousand.
    """
    for name, run in _both():
        assert run.obs_mismatches == [], '{0}: {1} of {2} states disagreed; first: {3}'.format(
            name, len(run.obs_mismatches), run.states, run.obs_mismatches[0])
    assert _growth_run().states >= 20000
    assert _endgame_run().states >= 8000


def test_the_parity_runs_actually_cover_the_hard_boards():
    """A parity result is worthless if every board was an open one.

    The connectivity block short-circuits fatal moves and the region count only becomes interesting
    once the snake has fragmented the free space, so a suite that never got past length ~40 would
    pass every assertion above while exercising almost none of the code it claims to cover. These
    bounds are the coverage the elementwise result is conditional on.
    """
    growth, endgame = _growth_run(), _endgame_run()
    assert growth.max_length >= 55, 'growth run topped out at {0}'.format(growth.max_length)
    assert growth.episodes >= 10, 'growth run completed only {0} episodes'.format(growth.episodes)
    reached = np.flatnonzero(endgame.length_hist)
    assert reached.max() >= C.PERFECT_SCORE - 1, (
        'endgame run never reached a near-full board; longest was {0}'.format(reached.max()))
    # The endgame run draws lengths uniformly across 80-99 for topology diversity, so most of its
    # boards die rather than win; the win path gets its own regime below rather than being asserted
    # out of a run whose job is coverage of shapes.
    assert endgame.perfect_games >= 20, (
        'only {0} perfect games; the win path is barely covered'.format(endgame.perfect_games))


# --------------------------------------------------- L2: rewards and terminations

def test_reward_parity():
    for name, run in _both():
        assert run.reward_mismatches == [], (
            '{0}: {1} of {2} steps disagreed on reward; first: {3}'.format(
                name, len(run.reward_mismatches), run.steps, run.reward_mismatches[0]))


def test_termination_and_outcome_flag_parity():
    """`done`, `perfect` and `starved` must agree — and `perfect` comes off the score.

    Never off the reward. Three counters in this project once compared a final reward with
    `PERFECT_GAME_REWARD`, and the moment a shaping term shipped a perfect game paid 99.9 instead of
    100, every counter read 0%, and eight arms trained handicapped for 300k steps because the
    exploration schedule reads that same number.
    """
    for name, run in _both():
        assert run.flag_mismatches == [], '{0}: {1} disagreements; first: {2}'.format(
            name, len(run.flag_mismatches), run.flag_mismatches[0])


def test_perfect_game_is_decided_by_score_not_reward():
    """The env's own flag must be the project-wide rule, not a threshold of its own."""
    vec = V.VecSnake(3, seed=0)
    vec.score[:] = [C.MAX_POSSIBLE_SCORE, C.MAX_POSSIBLE_SCORE - 1, 0]
    flag = np.array([state_helpers.is_perfect_score(int(v)) for v in vec.score])
    assert flag.tolist() == [True, False, False]
    # And the win reward is nowhere in that decision: a shaping term that shifts the payout by a
    # fraction must not be able to change the answer.
    assert state_helpers.is_perfect_score(C.MAX_POSSIBLE_SCORE)


def test_starvation_parity():
    """Starving is a real termination mode and no other regime here reaches it.

    The budget is 100-500 steps while a coiled board resolves in a few dozen and the growth run eats
    constantly, so this needs a board constructed one step short of the budget. Without it, an
    `off-by-one` in `STARVE_REWARD` survived the whole suite.
    """
    run = _Lockstep(seed=41, board=lambda r: _coiled_snapshot(
        r, 40, since_food=starve_budget_for(40) - 1)).run(4000)
    assert run.obs_mismatches == [], run.obs_mismatches[0]
    assert run.reward_mismatches == [], run.reward_mismatches[0]
    assert run.flag_mismatches == [], run.flag_mismatches[0]
    starved = int(run.length_hist.sum())
    assert starved >= 100, 'only {0} episodes ended in this regime'.format(starved)


def test_win_path_parity():
    """A regime that wins constantly, so the perfect-game reward and flag are hammered.

    Separate from the endgame run on purpose: that run draws lengths across 80-99 to vary the board
    topology, which means most of its boards die instead of winning. Coverage of shapes and coverage
    of the win are different jobs and reading one out of the other is how a bound ends up asserting
    whichever number the run happened to produce.
    """
    run = _Lockstep(seed=43, board=lambda r: _coiled_snapshot(r, 98)).run(4000)
    assert run.obs_mismatches == [], run.obs_mismatches[0]
    assert run.reward_mismatches == [], run.reward_mismatches[0]
    assert run.flag_mismatches == [], run.flag_mismatches[0]
    assert run.perfect_games >= 200, 'only {0} wins'.format(run.perfect_games)


# ------------------------------------------------------------- L2b: shaping parity

def _with_shaping(names, run_kwargs, states=4000):
    """Run a lockstep with shaping forced on, patching both engines' bound copies.

    Both modules have to be patched, and the reason is a real trap rather than caution: `Snake.py`
    does `from state_helpers import *`, which does `from snake_constants import *`, so `Snake` holds
    its *own* bound copies of these names. Setting them on `snake_constants` alone would leave the
    reference reading the old values and the test would pass while comparing nothing — the same
    binding trap `SNEK_TILE_PIXELS` is documented for.

    Gates are dropped well below the defaults so the term is live for most of an episode instead of
    only in the last few steps, which a short run would never reach.
    """
    import Snake
    saved = {}
    try:
        for module in (Snake, C):
            for key, value in names.items():
                saved[(id(module), key)] = getattr(module, key)
                setattr(module, key, value)
        return _Lockstep(**run_kwargs).run(states)
    finally:
        for module in (Snake, C):
            for key in names:
                setattr(module, key, saved[(id(module), key)])


def test_chase_safe_shaping_parity():
    """Potential-based shaping must match, and it is off by default so it needs forcing on."""
    for label, kwargs in (('growth', {'seed': 11}),
                          ('endgame', {'seed': 12,
                                       'board': lambda r: _coiled_snapshot(r, 88)})):
        run = _with_shaping({'CHASE_SAFE_SHAPING': 0.1, 'CHASE_SAFE_GATE': 10}, kwargs)
        assert run.reward_mismatches == [], (
            '{0}: chase-safe shaping disagreed on {1} of {2} steps; first: {3}'.format(
                label, len(run.reward_mismatches), run.steps, run.reward_mismatches[0]))
        assert run.obs_mismatches == [], '{0}: observations disagreed under shaping'.format(label)


def test_free_space_shaping_parity():
    """The second potential term, on the same argument as the first."""
    for label, kwargs in (('growth', {'seed': 13}),
                          ('endgame', {'seed': 14,
                                       'board': lambda r: _coiled_snapshot(r, 88)})):
        run = _with_shaping({'FREE_SPACE_SHAPING': 0.1, 'FREE_SPACE_GATE': 10}, kwargs)
        assert run.reward_mismatches == [], (
            '{0}: free-space shaping disagreed on {1} of {2} steps; first: {3}'.format(
                label, len(run.reward_mismatches), run.steps, run.reward_mismatches[0]))


def test_both_shaping_terms_together_are_in_parity():
    """Both at once, because the reference adds them in one expression and a sign error in the
    second could be cancelled by reading the first — which a one-term-at-a-time test would miss."""
    run = _with_shaping({'CHASE_SAFE_SHAPING': 0.1, 'CHASE_SAFE_GATE': 10,
                         'FREE_SPACE_SHAPING': 0.05, 'FREE_SPACE_GATE': 10},
                        {'seed': 15})
    assert run.reward_mismatches == [], run.reward_mismatches[0]


def test_the_shaping_tests_would_notice_if_shaping_were_silently_off():
    """Guards the guard: if forcing the coefficients on did not actually change any reward, the two
    tests above would pass against a vec env that ignored shaping entirely."""
    plain = _Lockstep(seed=11).run(1200)
    shaped = _with_shaping({'CHASE_SAFE_SHAPING': 0.1, 'CHASE_SAFE_GATE': 10},
                           {'seed': 11}, states=1200)
    assert plain.reward_sum != shaped.reward_sum, (
        'forcing chase-safe shaping on changed no reward, so the parity tests are vacuous')


# ------------------------------------------------------------------- L3: mutation

def _mutants():
    """Hand-broken variants of the vectorised env, each of which parity must reject.

    A passing parity suite is not evidence until the suite is shown capable of failing. Every entry
    is a plausible mistake rather than a random perturbation: a transposed direction table, a
    dropped neighbour in the dilation, an off-by-one reward, a rescaled observation.
    """
    def turn_left_right_swapped():
        V.TURN = V.TURN[:, [1, 0, 2]]

    def turn_forward_becomes_left():
        V.TURN = V.TURN[:, [0, 1, 0]]

    def delta_axes_transposed():
        V.DELTA = np.array([-GRID_, GRID_, -1, 1], dtype=np.int64)

    def one_neighbour_becomes_a_diagonal():
        # Transposing NB would be an *equivalent* mutant, not a broken one: NB is consumed as a set
        # of offsets, so reordering it cannot change any answer. That version survived, correctly.
        V.NB = np.array([-1, 1, -GRID_, GRID_ + 1], dtype=np.int64)

    def dilation_drops_one_direction():
        original = V._dilate

        def broken(reach, open_):
            return (reach | V._shl(reach, 1) | V._shr(reach, 1)
                    | V._shl(reach, GRID_)) & open_          # the -GRID neighbour is gone
        V._dilate = broken
        return original

    def flood_stops_early():
        original = V._flood

        def broken(seed, open_, max_iter=4):
            return original(seed, open_, max_iter=max_iter)
        V._flood = broken

    def wall_ring_becomes_playable():
        V.PLAYABLE = V.PLAYABLE.copy()
        V.OPEN_TEMPLATE = V.OPEN_TEMPLATE.copy()
        V.PLAYABLE[V.flat(-1, 4)] = True
        V.OPEN_TEMPLATE[V.flat(-1, 4)] = True

    def region_count_always_one():
        original = V._enumerate

        def broken(pk, head_nb, tail_nb, food_bit, max_regions=None):
            count, ht, hf, htf = original(pk, head_nb, tail_nb, food_bit)
            return np.ones_like(count), ht, hf, htf
        V._enumerate = broken

    def food_reward_off_by_one():
        C.FOOD_REWARD = C.FOOD_REWARD + 1

    def death_reward_off_by_one():
        C.DEATH_REWARD = C.DEATH_REWARD + 1

    def starve_reward_off_by_one():
        C.STARVE_REWARD = C.STARVE_REWARD + 1

    def perfect_game_reward_off_by_one():
        C.PERFECT_GAME_REWARD = C.PERFECT_GAME_REWARD + 1

    def distance_shaping_sign_flipped():
        C.FOOD_DISTANCE_REWARD = -C.FOOD_DISTANCE_REWARD - 0.01

    def starve_budget_multiplier_raised():
        C.STARVE_MULT = C.STARVE_MULT + 1

    def starve_observation_rescaled():
        C.STARVE_OBS_SCALE = C.STARVE_OBS_SCALE * 1.01

    def group_observation_rescaled():
        C.GROUPS_OBS_SCALE = C.GROUPS_OBS_SCALE * 1.01

    def perfect_score_off_by_one():
        C.MAX_POSSIBLE_SCORE = C.MAX_POSSIBLE_SCORE - 1

    return [
        ('turn_left_right_swapped', turn_left_right_swapped),
        ('turn_forward_becomes_left', turn_forward_becomes_left),
        ('delta_axes_transposed', delta_axes_transposed),
        ('one_neighbour_becomes_a_diagonal', one_neighbour_becomes_a_diagonal),
        ('dilation_drops_one_direction', dilation_drops_one_direction),
        ('flood_stops_early', flood_stops_early),
        ('wall_ring_becomes_playable', wall_ring_becomes_playable),
        ('region_count_always_one', region_count_always_one),
        ('food_reward_off_by_one', food_reward_off_by_one),
        ('death_reward_off_by_one', death_reward_off_by_one),
        ('starve_reward_off_by_one', starve_reward_off_by_one),
        ('perfect_game_reward_off_by_one', perfect_game_reward_off_by_one),
        ('distance_shaping_sign_flipped', distance_shaping_sign_flipped),
        ('starve_budget_multiplier_raised', starve_budget_multiplier_raised),
        ('starve_observation_rescaled', starve_observation_rescaled),
        ('group_observation_rescaled', group_observation_rescaled),
        ('perfect_score_off_by_one', perfect_score_off_by_one),
    ]


GRID_ = C.GRID

_MUTABLE_V = ('TURN', 'DELTA', 'NB', 'PLAYABLE', 'OPEN_TEMPLATE', '_dilate', '_flood', '_enumerate')
_MUTABLE_C = ('FOOD_REWARD', 'DEATH_REWARD', 'STARVE_REWARD', 'PERFECT_GAME_REWARD',
              'FOOD_DISTANCE_REWARD', 'STARVE_MULT', 'STARVE_OBS_SCALE', 'GROUPS_OBS_SCALE',
              'MAX_POSSIBLE_SCORE')


# The regimes a mutant is hunted in. Each covers a termination the others do not: growth covers
# ordinary play and death, the coiled board covers the endgame, the starving board covers the starve
# reward, and the near-full board covers the win. A mutant is only as detectable as the regime set.
_MUTANT_REGIMES = (
    {'seed': 31},
    {'seed': 32, 'board': lambda r: _coiled_snapshot(r, 92)},
    {'seed': 33, 'board': lambda r: _coiled_snapshot(
        r, 40, since_food=starve_budget_for(40) - 1)},
    {'seed': 34, 'board': lambda r: _coiled_snapshot(r, 98)},
)


def _run_under_mutant(apply_):
    """Apply a mutant, run a short lockstep in both regimes, restore, return the kill evidence.

    A raise counts as a kill. `_flood` refuses to return a non-converged answer, so breaking the
    dilation surfaces as `RuntimeError` rather than as a wrong number — which is the behaviour that
    module wants, and it is still a detection.
    """
    saved_v = {k: getattr(V, k) for k in _MUTABLE_V}
    saved_c = {k: getattr(C, k) for k in _MUTABLE_C}
    try:
        apply_()
        # Short-circuits on the first regime that detects the mutant. Most die in the growth run,
        # and running all four regardless tripled this test's wall clock for no extra information —
        # one detection is a kill. The control below finds nothing, so it still runs all four.
        for kwargs in _MUTANT_REGIMES:
            try:
                run = _Lockstep(**kwargs).run(1200)
            except Exception:
                return 'raised'
            found = (len(run.obs_mismatches) + len(run.reward_mismatches)
                     + len(run.flag_mismatches))
            if found:
                return found
        return 0
    finally:
        for k, v in saved_v.items():
            setattr(V, k, v)
        for k, v in saved_c.items():
            setattr(C, k, v)


def test_every_mutant_is_killed():
    """The suite's own credibility check: each broken variant must be detected.

    Run last in spirit if not in order — a mutant that survives means the corresponding assertion
    above is decorative, and that is worth more than any individual parity number.
    """
    mutants = _mutants()
    assert len(mutants) >= 12, 'only {0} mutants defined'.format(len(mutants))
    survivors = [name for name, apply_ in mutants if _run_under_mutant(apply_) == 0]
    assert survivors == [], 'these mutants were not detected: {0}'.format(survivors)


def test_the_unmutated_env_is_clean_on_the_same_short_runs():
    """The control for the mutation test: the same two short runs must find nothing.

    Without this, a mutation suite that killed everything would be indistinguishable from a harness
    that reported mismatches unconditionally.
    """
    assert _run_under_mutant(lambda: None) == 0
