"""Result rows: the interval, the summaries, and the round trip that makes a run resumable.

The row's job is to be *comparable*. Every one is full length under snek3's single-stage protocol,
so the things that can still go wrong are arithmetic: an interval that runs past 100%, a median
recomputed from summaries rather than from episodes, a resumed sample that loses its history.
"""

import pytest

from tools import eval_plan


def held(scores, perfect=None, rewards=None, seconds=1.0):
    perfect = [score == 95 for score in scores] if perfect is None else perfect
    rewards = [float(score) for score in scores] if rewards is None else rewards
    return {'scores': list(scores), 'perfect': list(perfect), 'rewards': list(rewards),
            'seconds': seconds, 'abandoned': False}


# ------------------------------------------------------------------ the interval

def test_the_interval_stays_inside_zero_and_one():
    """The reason it is Wilson's and not the normal approximation.

    At 499/500 the symmetric interval is 0.998 ± 0.0039, so its upper bound is 1.0019 — a reported
    confidence interval whose top is 100.2% perfect. Wilson's cannot leave [0, 1] by construction,
    and these are the two rates snek3 actually measures near.
    """
    for successes, trials in ((500, 500), (499, 500), (0, 500), (1, 500)):
        low, high = eval_plan.wilson_interval(successes, trials)
        assert 0.0 <= low <= high <= 1.0, (successes, trials, low, high)


def test_the_interval_brackets_the_point_estimate():
    for successes in (0, 1, 250, 499, 500):
        low, high = eval_plan.wilson_interval(successes, 500)
        assert low <= successes / 500 <= high, successes


def test_a_larger_sample_gives_a_tighter_interval():
    # The property every comparison in this project rests on, and the reason stage B is 500 episodes
    # rather than 100.
    narrow = eval_plan.wilson_interval(99, 100)
    wide = eval_plan.wilson_interval(2970, 3000)
    assert (wide[1] - wide[0]) < (narrow[1] - narrow[0])


def test_no_trials_is_not_a_division_by_zero():
    assert eval_plan.wilson_interval(0, 0) == (0.0, 0.0)


# ------------------------------------------------------------------ the row

def test_a_row_reports_the_rate_the_episodes_carry():
    row = eval_plan.build_row(7000, held([95] * 98 + [40, 12]))
    assert row['step'] == 7000
    assert row['episodes'] == 100
    assert row['perfect_games'] == 98
    assert row['perfect_percent'] == 98.0
    assert row['min_score'] == 12 and row['max_score'] == 95


def test_the_perfect_count_comes_from_the_flags_not_the_scores():
    """A perfect game is identified by its score, and the *flag* is where that decision landed.

    The row must not re-derive it: `env`/`vectorized` already applied `is_perfect_score`, and a
    second definition here is the second definition that went silent in snek2 when a shaping term
    changed what a win paid. Handing in flags that disagree with the scores proves which one is
    being read.
    """
    row = eval_plan.build_row(1, held([95, 95, 95], perfect=[True, False, False]))
    assert row['perfect_games'] == 1


def test_the_median_is_the_median_of_the_episodes():
    # Not derivable from the summaries, which is the whole reason the per-episode lists are stored.
    row = eval_plan.build_row(1, held([0, 0, 0, 95, 95]))
    assert row['median_score'] == 0.0
    assert row['avg_score'] == 38.0


def test_the_stage_a_screen_is_carried_through():
    # So stage A and stage B can be checked against each other on the same weights. A systematic gap
    # between a 100-episode screen and a 500-episode measurement would mean the two measurement
    # paths differ, which no single row can show.
    assert eval_plan.build_row(1, held([95])) ['stage_a_percent'] is None
    assert eval_plan.build_row(1, held([95]), stage_a_percent=96.0)['stage_a_percent'] == 96.0


def test_a_row_carries_no_comparability_caveats():
    """Every row is full length, so there is nothing to check before comparing two of them.

    snek2's rows carried `selected_by`, `abandoned` and a nullable `min_achievable`, and half of
    reading two of them was working out whether they were comparable at all. Their absence here is
    the protocol, so it is pinned: a field reappearing means the single-stage design has quietly
    grown a second stage.
    """
    row = eval_plan.build_row(1, held([95] * 10))
    for field in ('selected_by', 'abandoned', 'min_achievable', 'graph_surrounding'):
        assert field not in row, field


# ------------------------------------------------------------------ the round trip

def test_a_row_rebuilds_the_sample_it_was_built_from():
    original = held([95, 40, 95, 0], perfect=[True, False, True, False],
                    rewards=[100.5, 33.25, 100.5, -5.0])
    restored = eval_plan.held_from_row(eval_plan.build_row(3000, original))
    assert restored['scores'] == original['scores']
    assert restored['perfect'] == original['perfect']
    assert restored['rewards'] == pytest.approx(original['rewards'])


def test_a_rebuilt_sample_produces_an_identical_row():
    # What resumability means: a topped-up row and a row measured in one pass are the same row.
    first = eval_plan.build_row(10, held([95, 12, 95, 95, 3]))
    second = eval_plan.build_row(10, eval_plan.held_from_row(first))
    assert {k: v for k, v in second.items() if k != 'seconds'} == \
           {k: v for k, v in first.items() if k != 'seconds'}
