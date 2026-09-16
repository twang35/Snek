"""The fixed-path snake: a closed tour of the board that the opening body already lies on, driven
through `VecSnake` without an observation, and never losing."""

import numpy as np

from env import constants
from tools import fixed_path
from vectorized import config as C
from vectorized import vec_env
from vectorized.vec_env import VecSnake


def test_the_tour_visits_every_cell_once_and_closes():
    tour = fixed_path.cycle()
    assert len(tour) == C.PLAY * C.PLAY
    assert len(set(tour)) == len(tour)
    for a, b in zip(tour, tour[1:] + tour[:1]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_the_opening_body_lies_along_the_tour_tail_to_head():
    tour = fixed_path.cycle()
    at = {cell: i for i, cell in enumerate(tour)}
    x0, y0 = C.START_TILE
    tail_first = [(x0 - k, y0) for k in range(C.START_SEGMENTS, -1, -1)]
    positions = [at[cell] for cell in tail_first]
    assert positions == list(range(positions[0], positions[0] + len(positions)))
    # and the head's next cell is the one it is already facing, so the first action is `forward`
    assert tour[(at[(x0, y0)] + 1) % len(tour)] == (x0 + 1, y0)


def test_the_first_action_of_a_fresh_game_is_forward_on_every_lane():
    vec = VecSnake(8, seed=1)
    actions = fixed_path.FixedPath().actions(vec)
    assert actions.shape == (8,)
    assert (actions == constants.ACTIONS.index('forward')).all()


def test_every_game_is_perfect_and_the_mean_matches_the_closed_form():
    sample = fixed_path.play(fixed_path.FixedPath(), 60, seed=3, needs_obs=False)
    assert sample['perfect'].all()
    assert not sample['died'].any() and not sample['starved'].any()
    assert (sample['score'] == C.MAX_POSSIBLE_SCORE).all()
    # 60 games, per-game sd ~200: a 3-sigma band on the mean is about +-80 steps.
    assert abs(sample['steps'].mean() - fixed_path.expected_steps()) < 120


def test_the_closed_form_is_the_sum_over_lengths():
    # Length 5 to 99, each meal costing (101 - L) / 2 on average: (2 + 3 + ... + 96) / 2.
    assert fixed_path.expected_steps() == sum(range(2, 97)) / 2.0 == 2327.5


def test_a_lane_whose_head_is_in_the_wall_ring_still_gets_an_action():
    # A dead lane sits in the ring until it is reset; the table must not raise for it.
    table = fixed_path.next_cell_table()
    ring = vec_env.flat(-1, 3)
    assert table[ring] == -1
    vec = VecSnake(2, seed=0)
    vec.body[0, vec.hp[0]] = ring
    actions = fixed_path.FixedPath().actions(vec)
    assert actions.shape == (2,) and set(np.unique(actions)) <= {0, 1, 2}


def test_summarise_reports_the_perfect_games_only():
    sample = {'steps': np.array([1000, 3000, 50]), 'score': np.array([95, 95, 3]),
              'perfect': np.array([True, True, False]), 'starved': np.array([False, False, False]),
              'died': np.array([False, False, True])}
    row = fixed_path.summarise(sample, 'x')
    assert row['perfect_games'] == 2 and row['died'] == 1
    assert row['steps_mean'] == 2000.0 and row['steps_min'] == 1000 and row['steps_max'] == 3000
    assert row['steps_per_food'] == round(2000.0 / C.MAX_POSSIBLE_SCORE, 2)


def test_an_entry_from_an_earlier_era_is_not_loadable_here(tmp_path):
    import json
    import os
    old = tmp_path / 'old'
    old.mkdir()
    (old / 'arch.json').write_text(json.dumps({'obs_era': 'b09c616', 'obs_len': 30}))
    new = tmp_path / 'new'
    new.mkdir()
    (new / 'arch.json').write_text(json.dumps({'obs_era': constants.BASE_OBS_ERA + '-hist8'}))
    assert not fixed_path.loadable_here(str(old))
    assert fixed_path.loadable_here(str(new))
