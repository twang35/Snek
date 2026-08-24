"""Unit tests for `vectorized/vec_env.py` internals that parity cannot isolate.

`test_vectorized_parity.py` compares the whole engine against the live `Snake.Game`, which is the
strongest evidence available — but it is end-to-end, so a bug that both paths of a fast/slow pair
share, or one that only appears at a batch width parity never runs at, would pass it. These tests
attack the vectorised implementation from inside: the compacted flood fill against its own naive
form, the region count against an independent BFS, the food sampler's distribution, and above all
**row independence**, which is the failure mode a batched rewrite actually has.
"""

import numpy as np

from vectorized import config as C
from vectorized import vec_env as V


def _random_boards(m, rng, fill=0.45):
    """`m` random open-cell bitboards, wall ring respected."""
    grid = rng.random((m, C.PAD)) > fill
    grid &= V.PLAYABLE[None, :]
    return np.packbits(grid, axis=1, bitorder='little').view(np.uint64)


def _seed_bits(m, rng):
    cells = np.flatnonzero(V.PLAYABLE[:C.NCELL])
    pick = cells[rng.integers(0, cells.size, size=(m, 1))]
    return V._bits(pick)


# ------------------------------------------------------------------- flood fill

def test_compacted_flood_agrees_with_the_naive_flood():
    """The fast path drops a row from the working set the moment it stops growing.

    That reindexes `reach` and `open_` mid-loop and writes finished rows back through an `active`
    index, which is exactly the kind of bookkeeping that is right for the batch that was tested and
    wrong for the next one. A disagreement here is a wrong observation, not a slow one.

    Both batch widths matter: `_flood` takes a completely different branch below `COMPACT_MIN`, so
    testing only one width leaves half the function unexecuted.
    """
    rng = np.random.default_rng(0)
    for m in (1, 8, V.COMPACT_MIN - 1, V.COMPACT_MIN, 200):
        for fill in (0.2, 0.45, 0.8):
            open_ = _random_boards(m, rng, fill)
            seed = _seed_bits(m, rng)
            fast = V._flood(seed, open_)
            slow = V._flood_naive(seed, open_)
            assert np.array_equal(fast, slow), (
                'flood paths disagree at m={0}, fill={1}'.format(m, fill))


def test_the_compacted_flood_is_exercised_and_not_silently_taking_the_slow_branch():
    """Guards the guard above: if every width fell through to the naive loop, it proves nothing."""
    rng = np.random.default_rng(1)
    open_ = _random_boards(200, rng)
    calls = []
    original = V._dilate

    def counting(reach, open_):
        calls.append(reach.shape[0])
        return original(reach, open_)
    V._dilate = counting
    try:
        V._flood(_seed_bits(200, rng), open_)
    finally:
        V._dilate = original
    assert len(calls) > 3, 'the fill converged too fast to say anything'
    assert min(calls) < max(calls), (
        'the working set never shrank, so compaction did not run: widths {0}'.format(sorted(set(calls))))


def test_flood_raises_rather_than_returning_a_non_converged_answer():
    """A truncated fill is a plausible-looking wrong number, which is worse than a crash.

    Both branches must raise, so this is checked below and above `COMPACT_MIN`.
    """
    rng = np.random.default_rng(2)
    for m in (4, 200):
        open_ = _random_boards(m, rng, fill=0.05)      # nearly everything open, so the fill is long
        seed = _seed_bits(m, rng)
        try:
            V._flood(seed, open_, max_iter=2)
        except RuntimeError:
            continue
        raise AssertionError('a 2-round fill of an open board did not raise at m={0}'.format(m))


def test_flood_does_not_leak_across_the_wall_ring():
    """The +-1 shift crosses a row boundary, and only the wall ring stops it wrapping.

    Floods a single cell on an entirely open board and checks the reach never touches a wall cell.
    Without the ring this would silently connect the right edge of one row to the left of the next,
    which reads as a snake that thinks it can escape a sealed pocket.
    """
    open_ = np.packbits(np.tile(V.PLAYABLE, (1, 1)), axis=1, bitorder='little').view(np.uint64)
    reach = V._flood(V._bits(np.array([[V.flat(0, 0)]])), open_)
    bits = np.unpackbits(reach.view(np.uint8), bitorder='little')[:C.PAD].astype(bool)
    assert not (bits & ~V.PLAYABLE).any(), 'the fill reached a wall cell'
    assert bits.sum() == C.PERFECT_SCORE, (
        'an open board should flood to all {0} cells, reached {1}'.format(
            C.PERFECT_SCORE, bits.sum()))


# ------------------------------------------------------------ region enumeration

def _bfs_region_count(open_row):
    """Independent region count for one board, by plain BFS over a boolean grid."""
    bits = np.unpackbits(open_row.view(np.uint8), bitorder='little')[:C.NCELL].astype(bool)
    seen = np.zeros(C.NCELL, dtype=bool)
    count = 0
    for start in range(C.NCELL):
        if not bits[start] or seen[start]:
            continue
        count += 1
        stack = [start]
        seen[start] = True
        while stack:
            cell = stack.pop()
            for step in (-1, 1, -C.GRID, C.GRID):
                nxt = cell + step
                if 0 <= nxt < C.NCELL and bits[nxt] and not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
    return count


def test_region_count_matches_an_independent_bfs():
    """`_enumerate` seeds at the lowest open cell and re-floods; BFS is a different algorithm.

    Comparing the bitboard partition against a plain grid BFS is the only check here that does not
    share code with the thing it is testing, which is what makes it worth its cost.
    """
    rng = np.random.default_rng(3)
    open_ = _random_boards(60, rng, fill=0.5)
    zero = np.zeros((open_.shape[0], C.WORDS), dtype=np.uint64)
    count, _, _, _ = V._enumerate(open_, zero, zero, zero)
    expected = np.array([_bfs_region_count(open_[i]) for i in range(open_.shape[0])])
    assert np.array_equal(count, expected), (
        'region counts differ; first bad row {0}'.format(
            int(np.flatnonzero(count != expected)[0])))


# --------------------------------------------------------------- row independence

def test_a_wide_batch_matches_the_same_games_run_one_at_a_time():
    """**The failure mode a vectorised rewrite actually has.**

    Every step here is written branchlessly over the whole batch, so a mis-shaped mask, a stale
    row index or a broadcast that should have been a per-row gather produces a board that depends on
    its *neighbours*. Parity against the reference would still catch that, but only if it happened to
    run at the width that triggers it — so this compares one 16-wide env against sixteen 1-wide
    envs driven by identical actions and identical forced food, and requires the observations, the
    rewards and the flags to be equal at every step.
    """
    rng = np.random.default_rng(4)
    n = 16
    wide = V.VecSnake(n, seed=99, shaping_discount=0.9975)
    singles = [V.VecSnake(1, seed=99, shaping_discount=0.9975) for _ in range(n)]

    def sync(rows):
        """Copy the wide env's rows into the matching single-row envs.

        Needed at the start because the two samplers consume randomness at different rates, so
        equal seeds do not mean equal boards — and needed again after every death, because
        `autoreset=False` leaves a dead row's state undefined and this test must not stop at the
        first one. Bailing out on the first death is what an earlier version did: it compared two
        steps and passed in 0.1s, which is worse than no test.
        """
        for i in rows:
            body = np.array([wide.body_list(i)], dtype=np.int64)
            singles[i].set_state(
                body, [len(body[0])], [int(wide.head_dir[i])], [int(wide.food[i])],
                [int(wide.step_count[i])], [int(wide.last_food_step[i])], [int(wide.score[i])])

    sync(range(n))
    compared = 0
    terminations = 0
    for _ in range(400):
        wide_obs = wide.observe()
        for i, one in enumerate(singles):
            assert np.array_equal(one.observe()[0], wide_obs[i]), (
                'row {0} observation depends on batch width'.format(i))
        compared += n

        actions = rng.integers(0, 3, size=n)
        # Forced food removes the one legitimate source of divergence: the samplers draw from
        # different streams, so an unforced meal would put the food in different cells.
        cells = np.flatnonzero(V.PLAYABLE[:C.NCELL])
        forced = cells[rng.integers(0, cells.size, size=n)]

        _, wr, wd, wi = wide.step(actions, forced_food=forced, autoreset=False)
        for i, one in enumerate(singles):
            _, sr, sd, si = one.step(actions[i:i + 1], forced_food=forced[i:i + 1],
                                     autoreset=False)
            assert np.isclose(sr[0], wr[i]), 'row {0} reward differs'.format(i)
            assert bool(sd[0]) == bool(wd[i]), 'row {0} done differs'.format(i)
            for key in ('ate', 'died', 'perfect', 'starved'):
                assert bool(si[key][0]) == bool(wi[key][i]), (
                    'row {0} flag {1} differs'.format(i, key))

        dead = np.flatnonzero(wd)
        if dead.size:
            terminations += int(dead.size)
            wide._reset_rows(dead)
            sync(dead)

    assert compared >= 400 * n, 'only {0} row-steps compared'.format(compared)
    assert terminations >= 10, (
        'only {0} terminations, so the comparison barely covered an episode end'.format(
            terminations))


# ------------------------------------------------------------------ state round-trip

def test_set_state_round_trips_through_body_list():
    """`set_state` writes a circular buffer head-first from CAP-1 down; `body_list` reads it back."""
    vec = V.VecSnake(4, seed=5)
    bodies = []
    for row in range(4):
        length = 6 + 3 * row
        cells = [V.flat(x, row) for x in range(length)]
        bodies.append(cells)
    width = max(len(b) for b in bodies)
    padded = np.zeros((4, width), dtype=np.int64)
    for row, cells in enumerate(bodies):
        padded[row, :len(cells)] = cells
    vec.set_state(padded, [len(b) for b in bodies], [0, 1, 2, 3],
                  [V.flat(0, 8)] * 4, [10] * 4, [5] * 4, [1] * 4)
    for row, cells in enumerate(bodies):
        assert vec.body_list(row) == cells, 'row {0}: {1}'.format(row, vec.body_list(row))


def test_set_state_leaves_the_rows_it_was_not_given_alone():
    """The `rows` argument is load-bearing for the shaping tests, so it is pinned here.

    If it silently wrote every row, the parity harness's episode-boundary restore would re-derive
    every lane's shaping potential and the shaping-parity tests would lose their ability to detect a
    drift between the incremental update and a fresh derivation.
    """
    vec = V.VecSnake(4, seed=6)
    before = [vec.body_list(i) for i in range(4)]
    cells = [V.flat(x, 5) for x in range(7)]
    vec.set_state(np.array([cells]), [len(cells)], [0], [V.flat(0, 8)], [3], [1], [2], rows=[2])
    assert vec.body_list(2) == cells
    for row in (0, 1, 3):
        assert vec.body_list(row) == before[row], 'row {0} was disturbed'.format(row)


def test_the_body_buffer_survives_wrapping_past_cap():
    """The buffer is circular, so a long episode wraps the head pointer through zero.

    `set_state` parks the head at CAP-1 precisely so the buffer is contiguous at the start, which
    means the wrap happens on step 1 — but a run of several CAP lengths is what proves the modulo is
    applied at both the write and the read.
    """
    vec = V.VecSnake(4, seed=7)
    for _ in range(3 * C.CAP):
        vec.step(np.zeros(4, dtype=np.int64))
    for row in range(4):
        cells = vec.body_list(row)
        assert len(cells) == int(vec.length[row])
        assert len(set(cells)) == len(cells), 'row {0} body has a repeated cell'.format(row)
        for ahead, behind in zip(cells, cells[1:]):
            assert abs(ahead - behind) in (1, C.GRID), (
                'row {0} body is not a connected path'.format(row))


# ----------------------------------------------------------------- food sampling

def test_food_is_sampled_uniformly_over_free_cells_and_never_on_the_snake():
    """The reference rejection-samples until it misses the snake, which is uniform over free cells.

    Parity forces placement, so this is the only test that checks the sampler at all. Uniformity is
    checked loosely — the point is to catch a sampler biased toward low indices, which is what an
    `argmax` over masked randomness degenerates to if the mask is wrong.
    """
    vec = V.VecSnake(512, seed=8)
    counts = np.zeros(C.NCELL, dtype=np.int64)
    for _ in range(40):
        vec.reset_all()
        occupied = np.zeros(512, dtype=bool)
        for row in range(512):
            cell = int(vec.food[row])
            assert cell >= 0 and V.PLAYABLE[cell], 'food off the board at {0}'.format(cell)
            occupied[row] = cell in set(vec.body_list(row))
            counts[cell] += 1
        assert not occupied.any(), 'food landed on the snake'
    # Every reset starts the snake on the same cells, so those can never hold food. Excluding them
    # is not a loosened bound — including them made this test assert that the sampler places food
    # inside the snake, which the assertion above forbids.
    reachable = V.PLAYABLE[:C.NCELL].copy()
    for cell in vec.body_list(0):
        reachable[cell] = False
    assert reachable.sum() == C.PERFECT_SCORE - (C.START_SEGMENTS + 1), (
        'expected the opening snake to cover {0} cells'.format(C.START_SEGMENTS + 1))
    assert counts[~reachable].sum() == 0, 'food was placed on a wall or on the opening snake'
    free = counts[reachable]
    assert (free > 0).all(), '{0} free cells never received food'.format(int((free == 0).sum()))
    # 512*40 draws over ~95 free cells is ~215 each; a factor of three either way is generous and
    # still rules out a sampler that concentrates on a corner, which is what an argmax over masked
    # randomness degenerates to when the mask is wrong.
    assert free.max() < 3 * free.mean() and free.min() > free.mean() / 3, (
        'food distribution is lopsided: min {0}, mean {1:.1f}, max {2}'.format(
            free.min(), free.mean(), free.max()))


# ------------------------------------------------------------------- observe modes

def test_only_the_full_groups_mode_is_parity_correct_and_the_others_say_so():
    """`fast` and `none` exist for benchmarking. Pinning what they do keeps them from being
    mistaken for measurement paths — the eval driver must never use them."""
    vec = V.VecSnake(8, seed=9)
    for _ in range(30):
        vec.step(np.ones(8, dtype=np.int64))
    full = vec.observe(groups_mode='full')
    none = vec.observe(groups_mode='none')
    assert np.array_equal(none[:, 9:18], np.zeros((8, 9), dtype=np.float32))
    others = [i for i in range(C.OBS_LEN) if not 9 <= i < 18]
    assert np.array_equal(none[:, others], full[:, others]), (
        'groups_mode changed an index outside the group block'
    )
