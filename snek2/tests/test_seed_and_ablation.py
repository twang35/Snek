"""Tests for SNEK_SEED (derive_seed / seed_process) and SNEK_ZERO_OBS (ZERO_OBS_INDICES).

The seeding tests are mostly about one failure: every parallel worker getting the same stream.
Food placement uses the *global* `random` module and ParallelPyEnvironment runs one constructor
per worker process, so a shared seed makes all ten workers deal identical food. A 10-episode eval
would silently become one episode counted ten times, with confidence intervals to match. Nothing
raises, so it has to be a test.
"""
import importlib
import os
import random

import numpy as np

from under_the_hood import derive_seed, seed_process


# ------------------------------------------------------------------ derive_seed

def test_derive_seed_is_none_when_seeding_is_off():
    # The default path for every arm up to batch 10. Must stay a no-op.
    assert derive_seed(None, 0) is None
    assert derive_seed(None, 7) is None


def test_derive_seed_differs_per_stream():
    # The whole point: worker 1 and worker 2 must not share a food sequence.
    streams = [derive_seed(42, s) for s in range(11)]
    assert len(set(streams)) == 11, streams


def test_derive_seed_differs_per_base_seed():
    assert derive_seed(1, 0) != derive_seed(2, 0)


def test_derive_seed_does_not_collide_across_adjacent_seed_and_stream():
    # A naive `seed + stream` would make (1, 2) and (2, 1) identical, so two arms one seed apart
    # would share worker streams. The prime multiplier is what prevents it.
    assert derive_seed(1, 2) != derive_seed(2, 1)
    assert derive_seed(5, 3) != derive_seed(3, 5)
    pairs = {derive_seed(s, w) for s in range(1, 6) for w in range(0, 11)}
    assert len(pairs) == 5 * 11


def test_derive_seed_fits_numpy_seed_range():
    # numpy.random.seed rejects anything outside [0, 2**32-1]; large base seeds are plausible
    # since a user may paste a timestamp.
    for seed in (0, 1, 12345, 2 ** 31 - 1, 1754000000):
        value = derive_seed(seed, 10)
        assert 0 <= value < 2 ** 31, (seed, value)


def test_derive_seed_is_stable_across_calls():
    assert derive_seed(99, 4) == derive_seed(99, 4)


# ----------------------------------------------------------------- seed_process

def test_seed_process_is_a_noop_without_a_seed():
    assert seed_process(None, 0) is None


def test_seed_process_makes_the_global_random_module_repeatable():
    # Food placement reads this module, so it is the one that decides whether two workers play the
    # same game.
    seed_process(7, 1)
    first = [random.randint(0, 8) for _ in range(20)]
    seed_process(7, 1)
    assert [random.randint(0, 8) for _ in range(20)] == first


def test_seed_process_gives_different_workers_different_food_sequences():
    """The failure this file exists for, stated in the terms that matter.

    Two workers seeded from the same base seed but different streams must not produce the same
    sequence of food placements. If they do, a parallel eval measures one episode N times.
    """
    def sequence(stream):
        seed_process(7, stream)
        return [random.randint(0, 8) for _ in range(40)]

    worker_sequences = [sequence(stream) for stream in range(1, 11)]
    assert len({tuple(s) for s in worker_sequences}) == 10, 'workers share a food sequence'


def test_seed_process_also_seeds_numpy():
    seed_process(3, 0)
    first = np.random.rand(10).tolist()
    seed_process(3, 0)
    assert np.random.rand(10).tolist() == first


# ------------------------------------------------------------ ZERO_OBS_INDICES

def parse(raw):
    """Re-reads snake_constants with SNEK_ZERO_OBS set, since it is parsed at import."""
    previous = os.environ.get('SNEK_ZERO_OBS')
    if raw is None:
        os.environ.pop('SNEK_ZERO_OBS', None)
    else:
        os.environ['SNEK_ZERO_OBS'] = raw
    try:
        import snake_constants
        importlib.reload(snake_constants)
        return set(snake_constants.ZERO_OBS_INDICES)
    finally:
        if previous is None:
            os.environ.pop('SNEK_ZERO_OBS', None)
        else:
            os.environ['SNEK_ZERO_OBS'] = previous
        import snake_constants
        importlib.reload(snake_constants)


def test_zero_obs_is_empty_by_default():
    # Ablation off is the normal case and must never quietly blank an input.
    assert parse(None) == set()
    assert parse('') == set()


def test_zero_obs_accepts_a_single_index():
    assert parse('29') == {29}


def test_zero_obs_accepts_a_list():
    assert parse('26,27,28') == {26, 27, 28}


def test_zero_obs_accepts_an_inclusive_range():
    # The documented form for ablating both 2026-08-03 blocks at once. Inclusive on both ends:
    # 26-29 is four indices, not three.
    assert parse('26-29') == {26, 27, 28, 29}


def test_zero_obs_mixes_ranges_and_singles():
    assert parse('0,6-8,29') == {0, 6, 7, 8, 29}


def test_zero_obs_tolerates_whitespace_and_empty_entries():
    assert parse(' 26 - 29 , , 0 ') == {26, 27, 28, 29, 0}


# --------------------------------------------- ablation through get_observations

def test_ablation_zeroes_the_new_blocks_without_changing_length():
    """Checks the ablation end to end, and that it does not change the vector's length.

    Length is the point: deleting a block would change the observation spec and stop every
    existing checkpoint loading, making an ablation a comparison between two environments rather
    than between two information sets.
    """
    import snake_constants
    import state_helpers
    import test_observation_spec as fixtures

    full = state_helpers.get_observations(**fixtures.coiled_snake())

    os.environ['SNEK_ZERO_OBS'] = '26-29'
    try:
        importlib.reload(snake_constants)
        importlib.reload(state_helpers)
        ablated = state_helpers.get_observations(**fixtures.coiled_snake())
    finally:
        os.environ.pop('SNEK_ZERO_OBS', None)
        importlib.reload(snake_constants)
        importlib.reload(state_helpers)

    assert len(ablated) == len(full) == 30, 'ablation must not resize the vector'
    assert ablated[26:30] == [0, 0, 0, 0], ablated[26:30]
    assert ablated[:26] == full[:26], 'no index outside the ablated range may change'
    assert full[26:30] != [0, 0, 0, 0], 'the fixture must have something to ablate'
