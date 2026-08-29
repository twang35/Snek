"""`VecSnake` — N snake games advancing in lockstep, in branchless numpy, with no pygame.

Parity target is snek2 itself: `Snake.Game.step` for the mechanics and rewards, and
`state_helpers.get_observations` for the 30-value observation. `tests/test_vectorized_parity.py`
asserts both elementwise against a live `Game`, so this file is a *reimplementation*, never an
approximation — if it and `Snake.py` disagree, this file is wrong.

Representation
--------------
``body[n, CAP]``   circular buffer of *flat padded* cell indices, head at ``hp``. Cell (x, y) is
                   ``(y + 1) * GRID + (x + 1)``, so the tail is an O(1) lookup rather than a walk.
``open_[n, PAD]``  bool, True iff the cell is "open" in the reference's sense — grid value 0 or 1,
                   empty or food. Width is padded to whole uint64 words, not NCELL.
connectivity       runs on the packed ``(M, WORDS)`` uint64 form, so one dilation round is ~17 numpy
                   ops on three words per board instead of a pass over 144 bytes. The wall ring is
                   what makes the +-1 shifts safe; see `config.GRID`.

Why a bitboard dilation rather than a per-cell walk
---------------------------------------------------
The same reason `state_helpers.count_groups` uses one: a round costs a handful of operations
regardless of how many cells the region holds. Measured against the scalar version, the flood fill is
~19% of an env step and an *incremental* scheme that cuts the dilation rounds 6.1x wins only 1.27x of
wall clock, because a round was already nearly free. Batching is what pays here — see
`plans/snek3-pytorch-rewrite.md`.
"""

import numpy as np

from vectorized import config as C

GRID = C.GRID
NCELL = C.NCELL
WORDS = C.WORDS
PAD = C.PAD
CAP = C.CAP

U1 = np.uint64(1)

# DIRECTIONS order is ['left', 'right', 'up', 'down'] and these are the flat-index deltas that
# correspond, given the row stride. Derived rather than written so a stride change cannot desync them.
DELTA = np.array([-1, 1, -GRID, GRID], dtype=np.int64)

# TURN[current_dir][action] -> new dir, with actions ordered as ACTIONS = ['left', 'right',
# 'forward'] — relative turns, not compass directions. Mirrors CURRENT_DIRECTION_MAPS; pinned
# against it in tests/test_vectorized_config.py rather than trusted.
TURN = np.array([[3, 2, 0],       # facing left  -> left=down,  right=up,    forward=left
                 [2, 3, 1],       # facing right -> left=up,    right=down,  forward=right
                 [0, 1, 2],       # facing up    -> left=left,  right=right, forward=up
                 [1, 0, 3]],      # facing down  -> left=right, right=left,  forward=down
                dtype=np.int64)
NB = DELTA.copy()                 # the four orthogonal neighbour offsets

_yy, _xx = np.divmod(np.arange(NCELL), GRID)
PLAYABLE = np.zeros(PAD, dtype=bool)
PLAYABLE[:NCELL] = (_xx >= 1) & (_xx <= C.PLAY) & (_yy >= 1) & (_yy <= C.PLAY)
OPEN_TEMPLATE = PLAYABLE.copy()

# Compaction thresholds. Below COMPACT_MIN rows the bookkeeping costs more than the rounds it saves;
# COMPACT_SHRINK is how far the live set must fall before it is worth re-indexing. Both measured.
COMPACT_MIN = 64
COMPACT_SHRINK = 0.75


def flat(x, y):
    """Tile (x, y) -> flat padded cell index. The inverse of `unflat`."""
    return (y + 1) * GRID + (x + 1)


def unflat(f):
    return (f % GRID - 1, f // GRID - 1)


# ------------------------------------------------------------------ bit helpers

def _shl(x, k):
    """cell i -> cell i + k, across a (M, WORDS) uint64 bitboard."""
    out = x << np.uint64(k)
    out[:, 1:] |= x[:, :-1] >> np.uint64(64 - k)
    return out


def _shr(x, k):
    out = x >> np.uint64(k)
    out[:, :-1] |= x[:, 1:] << np.uint64(64 - k)
    return out


def _any(x):
    """(M, WORDS) -> (M,) bool: is any bit set in this row."""
    out = x[:, 0] != 0
    for w in range(1, WORDS):
        out |= x[:, w] != 0
    return out


def _dilate(reach, open_):
    g = reach | _shl(reach, 1)
    g |= _shr(reach, 1)
    g |= _shl(reach, GRID)
    g |= _shr(reach, GRID)
    g &= open_
    return g


def _flood(seed, open_, max_iter=4 * NCELL):
    """Grow `seed` through `open_` until nothing changes. Returns the reached set.

    **Compaction is not an optimisation here, it is the difference between this being worth doing and
    not.** The naive form runs *every* board for the batch's *maximum* dilation count, and the
    distribution is wildly skewed: the median board converges in ~15 rounds while the worst board in
    a 1024-board batch needs ~125, so a diverse batch measured 24.98 us against 3.43 us for a batch
    of identical boards — i.e. without this the vectorised observation costs the same as the scalar
    one and the whole exercise is worth ~3x instead of ~50x. Dropping a row from the working set the
    moment it stops growing recovers 5.1x at 1024 envs and 8.5x at 16384.

    `tests/test_vectorized_env.py` asserts the compacted and naive paths agree, because a bug here
    is a wrong observation rather than a slow one.
    """
    reach = seed & open_
    if open_.shape[0] < COMPACT_MIN:
        for _ in range(max_iter):
            g = _dilate(reach, open_)
            if np.array_equal(g, reach):
                return reach
            reach = g
        raise RuntimeError('flood fill did not converge in {0} rounds'.format(max_iter))

    out = reach
    active = None                       # None means "out is the full batch, in order"
    r, o = reach, open_
    for _ in range(max_iter):
        g = _dilate(r, o)
        changed = _any(g ^ r)
        live = int(changed.sum())
        if live == 0:
            if active is None:
                return g
            out[active] = g
            return out
        if live < COMPACT_SHRINK * r.shape[0]:
            done = ~changed
            if active is None:
                out = g.copy()
                active = np.flatnonzero(changed)
            else:
                out[active[done]] = g[done]
                active = active[changed]
            r = g[changed]
            o = o[changed]
        else:
            r = g
    raise RuntimeError('flood fill did not converge in {0} rounds'.format(max_iter))


def _flood_naive(seed, open_, max_iter=4 * NCELL):
    """The uncompacted fill, kept solely so a test can assert the fast path agrees with it."""
    reach = seed & open_
    for _ in range(max_iter):
        g = _dilate(reach, open_)
        if np.array_equal(g, reach):
            return reach
        reach = g
    raise RuntimeError('flood fill did not converge')


def _bits(indices):
    """(M, k) flat cell indices -> (M, WORDS) uint64 mask with those cells set."""
    idx = np.atleast_2d(indices)
    out = np.zeros((idx.shape[0], WORDS), dtype=np.uint64)
    mrows = np.arange(idx.shape[0])
    for j in range(idx.shape[1]):
        f = idx[:, j]
        out[mrows, f >> 6] |= U1 << (f & 63).astype(np.uint64)
    return out


def _enumerate(pk, head_nb, tail_nb, food_bit, max_regions=NCELL):
    """Partition each board's open cells into regions; report what each region touches.

    Returns `(count, head_and_tail, head_and_food, head_and_tail_and_food)`. Seeds at the lowest
    unassigned open cell, floods, clears, repeats — the vectorised form of `count_groups`' outer
    loop, and compacted for the same reason `_flood` is: the loop runs for the batch's *maximum*
    region count while a typical board has 1-4.

    The three flags are per *region*, which is the whole point and the thing a union of regions
    cannot answer. `safe_to_chase_food` asks whether one region holds the food while touching both
    the head and the tail — testing the food against the head's regions alone is wrong, because the
    head can neighbour two regions and reaching the food through one while the only escape is through
    the other is exactly the trap the flag exists to name. See `state_helpers.group_obs`.
    """
    M = pk.shape[0]
    count = np.zeros(M, dtype=np.int64)
    ht = np.zeros(M, dtype=bool)
    hf = np.zeros(M, dtype=bool)
    htf = np.zeros(M, dtype=bool)
    rows = np.arange(M)
    rem = pk.copy()
    P, H, T, F = pk, head_nb, tail_nb, food_bit
    for _ in range(max_regions):
        alive = _any(rem)
        live = int(alive.sum())
        if live == 0:
            return count, ht, hf, htf
        if live < COMPACT_SHRINK * rem.shape[0]:
            sel = np.flatnonzero(alive)
            rows, rem, P, H, T, F = rows[sel], rem[sel], P[sel], H[sel], T[sel], F[sel]
            alive = None
        m = rem.shape[0]
        mrows = np.arange(m)
        first = np.argmax(rem != 0, axis=1)
        val = rem[mrows, first]
        seed = np.zeros_like(rem)
        seed[mrows, first] = val & (~val + U1)      # lowest set bit: an arbitrary unvisited cell
        reach = _flood(seed, P)
        th = _any(reach & H)
        tt = _any(reach & T)
        fd = _any(reach & F)
        if alive is None:
            count[rows] += 1
            ht[rows] |= th & tt
            hf[rows] |= th & fd
            htf[rows] |= th & tt & fd
        else:
            count[rows] += alive
            ht[rows] |= th & tt & alive
            hf[rows] |= th & fd & alive
            htf[rows] |= th & tt & fd & alive
        rem &= ~reach
    raise RuntimeError('region enumeration exceeded {0} regions'.format(max_regions))


class VecSnake:
    """`num_envs` games in lockstep. One `step` is one agent decision in every game.

    Autoreset: a finished game is reset **in place**, so the observation returned alongside
    `done=True` already belongs to the next episode. That is the standard contract and it is
    harmless for evaluation, which never bootstraps — but a future training use must gate on `done`
    rather than reading the returned observation as terminal.

    `forced_food` on `step`/`reset_all` is how the parity tests drive this in lockstep with a real
    `Game`: the reference rejection-samples the module-global `random` an unpredictable number of
    times, so aligning RNGs is not possible and forcing the placement is both exact and simpler.
    """

    def __init__(self, num_envs, seed=0, shaping_discount=1.0):
        self.n = int(num_envs)
        if self.n < 1:
            raise ValueError('num_envs must be >= 1, got {0}'.format(num_envs))
        self.rng = np.random.default_rng(seed)
        # The agent's gamma, needed because potential-based shaping pays
        # `c * (gamma * Phi(s') - Phi(s))`. Threaded in rather than read from a constant for the same
        # reason `Game.shaping_discount` is: it belongs to the agent, not the game.
        self.shaping_discount = float(shaping_discount)
        n = self.n
        self.rows = np.arange(n)
        self.body = np.zeros((n, CAP), dtype=np.int64)
        self.hp = np.zeros(n, dtype=np.int64)          # head pointer into the circular buffer
        self.length = np.zeros(n, dtype=np.int64)
        self.head_dir = np.zeros(n, dtype=np.int64)
        self.food = np.full(n, -1, dtype=np.int64)     # -1 is the 'no food' sentinel
        self.step_count = np.zeros(n, dtype=np.int64)
        self.last_food_step = np.zeros(n, dtype=np.int64)
        self.score = np.zeros(n, dtype=np.int64)
        self.open_ = np.zeros((n, PAD), dtype=bool)
        # Per-episode shaping state. Not part of a snapshot in the reference either — see
        # Game.restore_snapshot, which recomputes it.
        self.chase_safe_potential = np.zeros(n, dtype=np.float64)
        self.free_space_potential = np.zeros(n, dtype=np.float64)
        self.episodes_started = 0
        self.reset_all()

    # ------------------------------------------------------------------ reset

    def reset_all(self, forced_food=None):
        self._reset_rows(self.rows, forced_food=forced_food)

    def _reset_rows(self, idx, forced_food=None):
        if idx.size == 0:
            return
        self.open_[idx] = OPEN_TEMPLATE
        # The reference builds the snake head-first at START_TILE running right, with
        # START_SEGMENTS body cells trailing behind it.
        cells = [flat(C.START_TILE[0] - k, C.START_TILE[1]) for k in range(C.START_SEGMENTS + 1)]
        self.body[idx] = 0
        for k, cell in enumerate(cells):
            self.body[idx, C.START_SEGMENTS - k] = cell
            self.open_[idx, cell] = False
        self.hp[idx] = C.START_SEGMENTS
        self.length[idx] = C.START_SEGMENTS + 1
        self.head_dir[idx] = 1                          # 'right'
        self.step_count[idx] = 0
        self.last_food_step[idx] = 0
        self.score[idx] = 0
        if forced_food is None:
            self.food[idx] = self._sample_food(idx)
        else:
            self.food[idx] = np.asarray(forced_food, dtype=np.int64)
        self.open_[idx, self.food[idx]] = True
        self.episodes_started += int(idx.size)
        # Phi(s0) for the fresh board. The reference sets this in reset() for the same reason: the
        # first step's shaping must be measured against the starting board, not against zero, or the
        # episode's shaping does not telescope to zero.
        self._refresh_potentials(idx)

    def reset_rows(self, idx):
        """Restart the games in `idx`, leaving every other row untouched.

        The eval driver needs this because it controls *when* an episode starts: a measurement of N
        episodes must start exactly N of them, so a lane whose quota is spent has to stay finished
        while its siblings keep playing. `autoreset=True` restarts every finished row
        unconditionally, which would over-count.
        """
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size:
            self._reset_rows(idx)

    def _sample_food(self, idx):
        """Uniform over free playable cells, per row in `idx`.

        The reference rejection-samples `random.randint` over the whole board until it misses the
        snake, which is uniform over free cells — so this matches in *distribution* but consumes a
        different amount of randomness. That is why the parity tests force placement instead of
        seeding. It also means this does not slow down near a full board, where the reference needs
        ~20-50 retries per draw.
        """
        free = self.open_[idx, :NCELL] & PLAYABLE[None, :NCELL]
        r = self.rng.random(free.shape)
        r[~free] = -1.0
        return np.argmax(r, axis=1).astype(np.int64)

    # ------------------------------------------------------------ state access

    def tail_cells(self):
        """`(tail, next_tail)`: the cell the tail occupies now, and the one ahead of it.

        Both are needed and they are not interchangeable. `tail` is the cell being *vacated*, which
        is what decides legality and what `following_tail_obs` asks about; `next_tail` is where the
        tail lands on any move that does not eat, which is what "can the head still reach the tail"
        is actually about. Using the first for both was a real defect in the scalar code — it made
        `head_with_tail` go silent in exactly the coiled endgames it exists for.
        """
        tail_idx = (self.hp - self.length + 1) % CAP
        next_idx = (self.hp - self.length + 2) % CAP
        return self.body[self.rows, tail_idx], self.body[self.rows, next_idx]

    def heads(self):
        return self.body[self.rows, self.hp]

    def body_list(self, i):
        """Env `i`'s body as flat cell indices, head first — the order `GameSnapshot.body` uses."""
        length = int(self.length[i])
        hp = int(self.hp[i])
        return [int(self.body[i, (hp - k) % CAP]) for k in range(length)]

    def set_state(self, bodies, lengths, dirs, foods, steps, last_food_steps, scores, rows=None):
        """Overwrite game state. `bodies` is (m, K) flat indices, head first, padded.

        Exists for the parity tests, which drive this from a real `Game`'s snapshot so both engines
        start from an identical board — including the coiled endgame boards that random play never
        reaches and that stress the connectivity code hardest.

        `rows` restricts the write to a subset, and it is not a convenience. Rewriting a row calls
        `_refresh_potentials`, which derives the shaping potential from the board rather than
        carrying it forward — so a harness that re-pushed every row on every episode boundary would
        silently repair any drift between the incremental update and a fresh derivation, which is
        precisely the bug the shaping-parity tests are there to find. Restoring only the rows that
        actually restarted leaves the others carrying their own history.
        """
        rows = self.rows if rows is None else np.asarray(rows, dtype=np.int64)
        bodies = np.asarray(bodies, dtype=np.int64)
        lengths = np.asarray(lengths, dtype=np.int64)
        self.open_[rows] = OPEN_TEMPLATE
        self.body[rows] = 0
        for k in range(bodies.shape[1]):
            cell = bodies[:, k]
            live = k < lengths
            # Body element k counts back from the head, so placing the head at CAP-1 keeps the
            # buffer contiguous and makes hp a constant.
            self.body[rows, (CAP - 1 - k) % CAP] = np.where(live, cell, 0)
            self.open_[rows[live], cell[live]] = False
        self.hp[rows] = CAP - 1
        self.length[rows] = lengths
        self.head_dir[rows] = np.asarray(dirs, dtype=np.int64)
        self.food[rows] = np.asarray(foods, dtype=np.int64)
        self.step_count[rows] = np.asarray(steps, dtype=np.int64)
        self.last_food_step[rows] = np.asarray(last_food_steps, dtype=np.int64)
        self.score[rows] = np.asarray(scores, dtype=np.int64)
        has = self.food[rows] >= 0
        self.open_[rows[has], self.food[rows][has]] = True
        self._refresh_potentials(rows)

    def starve_budget(self):
        return np.maximum(C.MIN_STARVE_BUDGET,
                          np.minimum(C.STARVE_MULT * self.length, C.MAX_STARVE_BUDGET))

    # ------------------------------------------------------------------ step

    def step(self, actions, forced_food=None, autoreset=True, observe=True, groups_mode='full'):
        """Advance every game by one agent decision.

        Returns `(obs, reward, done, info)`. `info` carries `ate / died / perfect / starved`, all
        (n,) bool — `perfect` is decided from the **score**, never from the reward, which is the one
        rule this project has paid for twice over (`state_helpers.is_perfect_score`).

        The reward follows the reference's *assignment* precedence rather than accumulating: a death
        overwrites the food reward, and a perfect game overwrites a death. Then distance shaping
        subtracts and the potential terms add.
        """
        rows = self.rows
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n,):
            raise ValueError('expected {0} actions, got shape {1}'.format(self.n, actions.shape))
        head = self.body[rows, self.hp]
        new_dir = TURN[self.head_dir, actions]
        new_head = head + DELTA[new_dir]
        tail_idx = (self.hp - self.length + 1) % CAP
        tail = self.body[rows, tail_idx]

        food = self.food
        has_food = food >= 0
        open_new_head = self.open_[rows, new_head]
        ate = has_food & (new_head == food)

        food_safe = np.where(has_food, food, 0)
        dist_before = (np.abs(head % GRID - food_safe % GRID)
                       + np.abs(head // GRID - food_safe // GRID))

        # --- advance the snake
        self.step_count += 1
        self.head_dir = new_dir
        self.hp = (self.hp + 1) % CAP
        self.body[rows, self.hp] = new_head
        not_ate = ~ate
        # Free the vacated tail *before* occupying the new head, so a move onto the tail's own cell
        # leaves that cell occupied rather than open. A move that eats does not move the tail at all
        # — the reference's add_segment() refills the cell it came from.
        self.open_[rows[not_ate], tail[not_ate]] = True
        self.open_[rows, new_head] = False
        self.length += ate
        self.score += ate
        self.last_food_step = np.where(ate, self.step_count, self.last_food_step)
        self.food = np.where(ate, -1, food)

        # --- terminal conditions
        off_board = ~PLAYABLE[new_head]
        # The tail's cell is safe: it vacates on the same step the head arrives. `open_new_head` was
        # read before the move, so it still describes the pre-move board.
        self_hit = (~off_board) & (~open_new_head) & (new_head != tail)
        died = off_board | self_hit
        perfect = self.score == C.MAX_POSSIBLE_SCORE
        finished = died | perfect
        starve_left = self.starve_budget() - (self.step_count - self.last_food_step)
        starved = (~finished) & (starve_left <= 0)
        finished = finished | starved

        reward = np.zeros(self.n, dtype=np.float64)
        reward[ate] = C.FOOD_REWARD
        reward[died] = C.DEATH_REWARD
        reward[perfect] = C.PERFECT_GAME_REWARD
        reward[starved] = C.STARVE_REWARD

        # --- replacement food. Only a step that ate has none, and such a step cannot have died.
        needs_food = ate & (~perfect)
        if forced_food is not None:
            self.food = np.where(needs_food, np.asarray(forced_food, dtype=np.int64), self.food)
            placed = np.flatnonzero(needs_food)
        elif needs_food.any():
            placed = np.flatnonzero(needs_food)
            self.food[placed] = self._sample_food(placed)
        else:
            placed = np.empty(0, dtype=np.int64)
        if placed.size:
            self.open_[placed, self.food[placed]] = True

        # --- distance shaping: ordinary moves only. Skipped when this step ate, because
        # `dist_before` measures the food just consumed while `self.food` is already its replacement,
        # and skipped on a terminal step because a dead head is off the board.
        shaped = not_ate & (~finished) & (self.food >= 0)
        food_now = np.where(self.food >= 0, self.food, 0)
        dist_after = (np.abs(new_head % GRID - food_now % GRID)
                      + np.abs(new_head // GRID - food_now // GRID))
        reward[shaped & (dist_after > dist_before)] -= C.FOOD_DISTANCE_REWARD

        # --- potential-based shaping, F = c * (gamma * Phi(s') - Phi(s)), measured on the post-move
        # board with the replacement food already placed, and Phi(terminal) = 0 as the theory needs.
        if C.CHASE_SAFE_SHAPING or C.FREE_SPACE_SHAPING:
            reward += self._shaping_reward(finished)

        info = {'done': finished, 'ate': ate, 'died': died,
                'perfect': perfect, 'starved': starved}

        if autoreset and finished.any():
            self._reset_rows(np.flatnonzero(finished))

        obs = self.observe(groups_mode=groups_mode) if observe else None
        return obs, reward, finished, info

    # -------------------------------------------------------------- shaping

    def _refresh_potentials(self, idx):
        """Recompute Phi for the rows in `idx` from their current board."""
        if C.CHASE_SAFE_SHAPING:
            self.chase_safe_potential[idx] = self._chase_safe_now()[idx]
        if C.FREE_SPACE_SHAPING:
            self.free_space_potential[idx] = self._free_space_now()[idx]

    def _shaping_reward(self, finished):
        out = np.zeros(self.n, dtype=np.float64)
        if C.CHASE_SAFE_SHAPING:
            new = np.where(finished, 0.0, self._chase_safe_now())
            out += C.CHASE_SAFE_SHAPING * (self.shaping_discount * new
                                           - self.chase_safe_potential)
            self.chase_safe_potential = new
        if C.FREE_SPACE_SHAPING:
            new = np.where(finished, 0.0, self._free_space_now())
            out += C.FREE_SPACE_SHAPING * (self.shaping_discount * new
                                           - self.free_space_potential)
            self.free_space_potential = new
        return out

    def _packed(self, open_=None):
        src = self.open_ if open_ is None else open_
        return np.packbits(src, axis=1, bitorder='little').view(np.uint64)

    def _chase_safe_now(self):
        """`state_helpers.chase_safe_state`, vectorised: head, food and tail in *one* region.

        The state form of observation indices 15-17, and not substitutable for them: on a step that
        ate, the food has already been replaced by the time this is read, while the per-action flag
        was computed against the food that was consumed.

        Gated on length before the flood fill, exactly as `Game._chase_safe_potential` is, so a
        below-gate board costs one comparison — which is most of an episode.
        """
        out = np.zeros(self.n, dtype=np.float64)
        gate = (self.length >= C.CHASE_SAFE_GATE) & (self.food >= 0)
        idx = np.flatnonzero(gate)
        if idx.size == 0:
            return out
        pk = self._packed()[idx]
        head = self.body[idx, self.hp[idx]]
        tail_idx = (self.hp[idx] - self.length[idx] + 1) % CAP
        tail = self.body[idx, tail_idx]
        head_nb = _bits(head[:, None] + NB[None, :])
        tail_nb = _bits(tail[:, None] + NB[None, :])
        food_bit = _bits(self.food[idx][:, None])
        _, _, _, htf = _enumerate(pk, head_nb, tail_nb, food_bit)
        out[idx] = htf.astype(np.float64)
        return out

    def _free_space_now(self):
        """`state_helpers.free_space_pieces`, vectorised: 1 / open-region count, tail cell freed.

        The tail is freed first for the reason `update_grid` documents — it vacates on the next
        ordinary move, so a region reachable only through it is not really sealed. Counting the tail
        as a wall made the region count wrong on 40% of steps past score 80.
        """
        out = np.zeros(self.n, dtype=np.float64)
        gate = self.length >= C.FREE_SPACE_GATE
        idx = np.flatnonzero(gate)
        if idx.size == 0:
            return out
        open_ = self.open_[idx].copy()
        tail_idx = (self.hp[idx] - self.length[idx] + 1) % CAP
        tail = self.body[idx, tail_idx]
        open_[np.arange(idx.size), tail] = True
        pk = self._packed(open_)
        zero = np.zeros((idx.size, WORDS), dtype=np.uint64)
        count, _, _, _ = _enumerate(pk, zero, zero, zero)
        out[idx] = np.where(count > 0, 1.0 / np.maximum(count, 1), 0.0)
        return out

    # ---------------------------------------------------------- observation

    def observe(self, groups_mode='full'):
        """The 30-value observation for every game. Layout is `state_helpers.get_observations`'.

        `groups_mode` exists for benchmarking only: 'full' is the real observation, 'fast' skips
        region enumeration (so indices 10/12/14 read 0 while 9/11/13 and 15-17 stay exact), and
        'none' zeroes 9-17 entirely. **Only 'full' is parity-correct**; the others measure what the
        connectivity block costs. See `plans/snek3-pytorch-rewrite.md`.
        """
        n = self.n
        rows = self.rows
        r2 = rows[:, None]
        open_ = self.open_
        head = self.body[rows, self.hp]
        length = self.length
        food = self.food
        has_food = food >= 0
        tail, next_tail = self.tail_cells()

        obs = np.zeros((n, C.OBS_LEN), dtype=np.float32)

        new_dir = TURN[self.head_dir]                        # (n, 3)
        new_head = head[:, None] + DELTA[new_dir]            # (n, 3)
        open_nh = open_[r2, new_head]
        is_tail = new_head == tail[:, None]
        food_safe = np.where(has_food, food, 0)
        eats = has_food[:, None] & (new_head == food_safe[:, None])
        # The tail's cell counts as legal: it vacates as the snake advances.
        legal = open_nh | is_tail

        # --- 0-5: [is closer, 1/(distance+1)] per action. All six are 0 only when there is no food,
        #     which happens on the winning step.
        fx, fy = food_safe % GRID, food_safe // GRID
        d0 = np.abs(head % GRID - fx) + np.abs(head // GRID - fy)
        dn = np.abs(new_head % GRID - fx[:, None]) + np.abs(new_head // GRID - fy[:, None])
        # Landing on the food reads [1, 1] — the reference's log2plus1(0) == 1 branch.
        reciprocal = np.where(eats, 1.0, 1.0 / (dn + 1.0))
        closer = np.where(eats, 1.0, (dn < d0[:, None]).astype(np.float32))
        gate = has_food[:, None]
        obs[:, 0:6:2] = np.where(gate, closer, 0.0)
        obs[:, 1:6:2] = np.where(gate, reciprocal, 0.0)

        # --- 6-8: is the move survivable. 1 means safe.
        obs[:, 6:9] = legal

        # --- 9-17: connectivity
        ht, groups, chase = self._connectivity(new_head, tail, next_tail, eats, is_tail, legal,
                                               has_food, food_safe, groups_mode)
        obs[:, 9:14:2] = ht
        obs[:, 10:15:2] = groups
        obs[:, 15:18] = chase

        # --- 18-20: does this move win. Zero unless the snake is exactly one food short, so this
        #     fires only on the final move of a game — nonzero in 0.000-0.025% of states.
        obs[:, 18:21] = eats & (length == C.PERFECT_SCORE - 1)[:, None]

        # --- 21-22: starve budget left (lg-compressed), and board fill (linear)
        remaining = np.maximum(0, self.starve_budget() - (self.step_count - self.last_food_step))
        obs[:, 21] = np.log2(remaining + 1.0) / C.STARVE_OBS_SCALE
        obs[:, 22] = length / C.PERFECT_SCORE

        # --- 23-25: is the post-move head hugging a wall or body on its left or right. Checked
        #     against the board *after* the move, so the cell the tail vacates reads as open — which
        #     only matters in a tight coil. 0 for a fatal move.
        left_dir = TURN[new_dir][:, :, 0]
        right_dir = TURN[new_dir][:, :, 1]
        left_pos = new_head + DELTA[left_dir]
        right_pos = new_head + DELTA[right_dir]
        vacates = ~eats
        left_open = open_[r2, left_pos] | ((left_pos == tail[:, None]) & vacates)
        right_open = open_[r2, right_pos] | ((right_pos == tail[:, None]) & vacates)
        obs[:, 23:26] = legal & ((~left_open) | (~right_open))

        # --- 26-28: the move does NOT land on the cell the tail is vacating. 1 is good, and a fatal
        #     move also reads 1 — the flag only asks "is this the tail's cell". Combine with 6-8.
        obs[:, 26:29] = ~is_tail

        # --- 29: room around the food. 0 sealed in, 0.5 a two-cell pocket, 1 roomier or no food.
        #     Decided locally rather than with a fourth flood fill, which is exact here: one open
        #     neighbour whose only other opening is the food itself means the region is two cells.
        nb = food_safe[:, None] + NB[None, :]
        open_nb = open_[r2, nb]
        n_open = open_nb.sum(axis=1)
        pick = np.argmax(open_nb, axis=1)
        neighbour = nb[rows, pick]
        beyond = neighbour[:, None] + NB[None, :]
        # Subtract one for the food's own cell, which is always among the neighbour's open cells.
        beyond_open = open_[r2, beyond].sum(axis=1) - 1
        space = np.where(n_open == 0, 0.0,
                         np.where(n_open > 1, 1.0, np.where(beyond_open > 0, 1.0, 0.5)))
        obs[:, 29] = np.where(has_food, space, 1.0)

        # Ablation applied last, so the indices it names are the ones in the layout above.
        for index in C.ZERO_OBS_INDICES:
            if 0 <= index < C.OBS_LEN:
                obs[:, index] = 0.0
        return obs

    def _connectivity(self, new_head, tail, next_tail, eats, is_tail, legal,
                      has_food, food_safe, mode='full'):
        """`(head_with_tail, lg(regions) scaled, safe_to_chase_food)`, each (n, 3).

        One post-move board per action, flattened to `n * 3` boards and solved in one batch. A fatal
        move is zeroed rather than answered: it reports what a hypothetical survivor of that move
        would see, and measured in play `head_with_tail` claimed 1 on 5,289 of 14,642
        body-collision actions.
        """
        n = self.n
        if mode == 'none':
            zero = np.zeros((n, 3), dtype=np.float32)
            return zero, zero, zero

        base = self._packed()
        pk = np.repeat(base, 3, axis=0)                  # (n*3, WORDS), C-order (env, action)
        M = pk.shape[0]
        mrows = np.arange(M)

        tail_f = np.repeat(tail, 3)
        eats_f = eats.reshape(-1)
        # A move that eats does not move the tail; anything else advances it one cell.
        post_tail = np.where(eats_f, tail_f, np.repeat(next_tail, 3))
        head_f = new_head.reshape(-1)
        legal_f = legal.reshape(-1)
        is_tail_f = is_tail.reshape(-1)

        # Post-move board: free the vacated tail unless the move eats, then occupy the new head.
        keep = ~eats_f
        tw = (tail_f >> 6)[keep]
        tb = (tail_f & 63)[keep].astype(np.uint64)
        pk[mrows[keep], tw] |= U1 << tb
        pk[mrows, head_f >> 6] &= ~(U1 << (head_f & 63).astype(np.uint64))
        pk[~legal_f] = 0                                  # fatal move: no board, no regions

        head_nb = _bits(head_f[:, None] + NB[None, :])
        tail_nb = _bits(post_tail[:, None] + NB[None, :])
        food_bit = _bits(np.repeat(food_safe, 3)[:, None])
        food_bit[~np.repeat(has_food, 3)] = 0

        if mode == 'full':
            count, ht_region, head_food, head_tail_food = _enumerate(pk, head_nb, tail_nb, food_bit)
            groups = (np.log2(count + 1.0) / C.GROUPS_OBS_SCALE).astype(np.float32)
        else:
            # 'fast': the two flags that do not need region identity, from two floods. Exact for
            # 9/11/13 and 15-17; leaves the region count at zero. Benchmarking only.
            reach_head = _flood(head_nb, pk)
            ht_region = _any(reach_head & tail_nb)
            reach_food = _flood(food_bit, pk)
            head_food = _any(reach_food & head_nb)
            head_tail_food = head_food & _any(reach_food & tail_nb)
            groups = np.zeros(M, dtype=np.float32)

        # Stepping onto the cell the tail is vacating is always safe, and no region test can see it:
        # that cell ends up holding the head rather than being open. Dropping this clause measured
        # 1,481 spurious disagreements in 15,700 actions.
        ht = legal_f & (ht_region | is_tail_f)
        has_food_f = np.repeat(has_food, 3)
        # A move that eats leaves no food cell to reach, so the question collapses to whether the
        # tail survives. Following the tail has no region containing the vacated cell, so take what
        # the head can see from there.
        chase = np.where(~has_food_f, False,
                         np.where(eats_f, ht,
                                  np.where(is_tail_f, head_food, head_tail_food))) & legal_f
        return (ht.reshape(n, 3).astype(np.float32),
                groups.reshape(n, 3),
                chase.reshape(n, 3).astype(np.float32))
