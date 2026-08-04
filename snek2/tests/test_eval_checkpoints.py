"""Tests for the pure helpers in eval_checkpoints.py.

Only the file-and-arithmetic helpers are covered here. Anything that builds an agent or a
ParallelPyEnvironment needs TensorFlow and 20 subprocesses, which does not belong in a unit
test; those paths are exercised by running a real eval with a throwaway EVAL_OUT_SUFFIX.

Importing eval_checkpoints pulls in TensorFlow, which is slow but unavoidable — the helpers
live in the same module as the eval loop on purpose, so the docstring that explains the
protocol sits next to the code implementing it.
"""
import json
import os

import eval_checkpoints
from snake_constants import RUNS_DIR


def write_result_file(policy_name, suffix, rows, complete=True):
    """Puts a result file where load_finished_results() will look for it."""
    path = os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy_name, suffix))
    with open(path, 'w') as handle:
        json.dump({'policy_name': policy_name, 'complete': complete, 'results': rows}, handle)
    return path


def row(step, episodes=100, perfect=70):
    return {'step': step, 'episodes': episodes, 'perfect_games': perfect,
            'perfect_percent': round(100.0 * perfect / episodes, 1)}


# ---------------------------------------------------------------- resume_suffixes

def test_resume_off_by_default():
    # Unset is the normal case and must never silently skip work.
    assert eval_checkpoints.resume_suffixes(None, '_top20') == []


def test_resume_off_for_falsey_spellings():
    for spec in ('', '0', 'false', 'False'):
        assert eval_checkpoints.resume_suffixes(spec, '_top20') == [], spec


def test_resume_1_means_this_runs_own_output_file():
    # The case that matters: relaunch the identical command with EVAL_RESUME=1.
    assert eval_checkpoints.resume_suffixes('1', '_top20') == ['_top20']
    assert eval_checkpoints.resume_suffixes('true', '') == ['']


def test_resume_takes_an_explicit_suffix_list():
    assert eval_checkpoints.resume_suffixes('_a,_b', '_top20') == ['_a', '_b']


def test_resume_list_tolerates_whitespace_and_empty_entries():
    assert eval_checkpoints.resume_suffixes(' _a , , _b ', '_x') == ['_a', '_b']


# --------------------------------------------------- resolve_screen_episodes

def test_screening_is_on_by_default():
    # The default is the decision this pins: a close-out with nothing set should screen.
    assert eval_checkpoints.resolve_screen_episodes(None, 100) == (20, None)


def test_screening_default_survives_an_empty_env_var():
    # `EVAL_SCREEN_EPISODES=` in a launch script must not be read as int('').
    assert eval_checkpoints.resolve_screen_episodes('', 100) == (20, None)


def test_screening_can_be_turned_off_explicitly():
    assert eval_checkpoints.resolve_screen_episodes('0', 100) == (0, None)


def test_screening_honours_an_explicit_length():
    assert eval_checkpoints.resolve_screen_episodes('40', 100) == (40, None)


def test_screening_default_stands_down_on_a_short_run():
    # EVAL_EPISODES=20 to sanity-check one checkpoint must not fail because the default screen is
    # also 20. It reports and carries on flat.
    screen, note = eval_checkpoints.resolve_screen_episodes(None, 20)
    assert screen == 0 and note and 'screening off' in note


def test_screening_rejects_an_explicit_length_that_cannot_confirm():
    try:
        eval_checkpoints.resolve_screen_episodes('100', 100)
    except SystemExit as error:
        assert 'must be below' in str(error)
    else:
        raise AssertionError('a screen as long as the full measurement should be rejected')


# ------------------------------------------------------- load_finished_results

def test_load_returns_nothing_when_there_is_no_file():
    rows, steps = eval_checkpoints.load_finished_results('no-such-policy-xyz', [''], 100)
    assert rows == [] and steps == set()


def test_load_skips_only_checkpoints_measured_to_the_full_episode_count():
    # The point of the >= test: a killed run leaves partial rows behind, and topping those up
    # would mean pooling summary statistics. Re-measuring is cheaper than getting that wrong.
    policy = 'unittest-resume-a'
    path = write_result_file(policy, '_t', [row(100, episodes=100), row(200, episodes=20),
                                            row(300, episodes=140)])
    try:
        rows, steps = eval_checkpoints.load_finished_results(policy, ['_t'], 100)
        assert steps == {100, 300}, 'the 20-episode row must be re-measured, not skipped'
        assert [r['step'] for r in rows] == [100, 300]
    finally:
        os.remove(path)


def test_load_returns_rows_in_step_order_whatever_the_file_order():
    policy = 'unittest-resume-b'
    path = write_result_file(policy, '_t', [row(500), row(100), row(300)])
    try:
        rows, _ = eval_checkpoints.load_finished_results(policy, ['_t'], 100)
        assert [r['step'] for r in rows] == [100, 300, 500]
    finally:
        os.remove(path)


def test_load_takes_a_repeated_step_once_from_the_first_file_listed():
    # These are alternative records of one frozen checkpoint, not extra samples: loading both
    # would put the same step in `results` twice and double-count it in the pooled rate.
    policy = 'unittest-resume-c'
    first = write_result_file(policy, '_first', [row(100, perfect=80)])
    second = write_result_file(policy, '_second', [row(100, perfect=20), row(200)])
    try:
        rows, steps = eval_checkpoints.load_finished_results(policy, ['_first', '_second'], 100)
        assert steps == {100, 200}
        assert len(rows) == 2, 'step 100 must appear once'
        assert [r for r in rows if r['step'] == 100][0]['perfect_games'] == 80, 'first file wins'
    finally:
        os.remove(first)
        os.remove(second)


def test_load_reads_an_incomplete_file():
    # The reason resume exists at all is a run that was killed part-way, so its file says
    # complete: false. Refusing to read that would make the feature useless.
    policy = 'unittest-resume-d'
    path = write_result_file(policy, '_t', [row(100), row(200)], complete=False)
    try:
        _, steps = eval_checkpoints.load_finished_results(policy, ['_t'], 100)
        assert steps == {100, 200}
    finally:
        os.remove(path)


def test_load_ignores_a_missing_file_among_present_ones():
    policy = 'unittest-resume-e'
    path = write_result_file(policy, '_t', [row(100)])
    try:
        _, steps = eval_checkpoints.load_finished_results(policy, ['_absent', '_t'], 100)
        assert steps == {100}
    finally:
        os.remove(path)


# -------------------------------------------------------------------- build_row

def held(perfect_flags, scores=None, rewards=None, seconds=1.0):
    n = len(perfect_flags)
    return {'perfect': list(perfect_flags),
            'scores': list(scores if scores is not None else range(n)),
            'rewards': list(rewards if rewards is not None else [1.0] * n),
            'seconds': seconds}


def test_build_row_counts_perfect_games_and_the_rate():
    r = eval_checkpoints.build_row(500, held([True] * 7 + [False] * 3))
    assert r['step'] == 500
    assert r['episodes'] == 10 and r['perfect_games'] == 7 and r['perfect_percent'] == 70.0


def test_build_row_recomputes_from_the_whole_sample_after_a_top_up():
    # The point of keeping raw episodes: a screen of 3/10 topped up with 14/20 must read 17/30,
    # and the median must come from all 30 rather than being averaged from two summaries.
    screen = held([True] * 3 + [False] * 7, scores=[1] * 10)
    screen['perfect'].extend([True] * 14 + [False] * 6)
    screen['scores'].extend([99] * 20)
    screen['rewards'].extend([1.0] * 20)
    r = eval_checkpoints.build_row(1, screen)
    assert r['episodes'] == 30 and r['perfect_games'] == 17
    assert r['perfect_percent'] == 56.7
    assert r['median_score'] == 99.0, 'median must come from all 30 episodes'
    assert r['min_score'] == 1.0 and r['max_score'] == 99.0


def test_build_row_carries_the_selection_metadata_through():
    r = eval_checkpoints.build_row(7, held([True]), {'selected_by': 'threshold90',
                                                     'single_eval': 90.0, 'surrounding': 61.5})
    assert r['selected_by'] == 'threshold90'
    assert r['graph_single_eval'] == 90.0 and r['graph_surrounding'] == 61.5


def test_build_row_defaults_metadata_for_an_explicit_step():
    r = eval_checkpoints.build_row(7, held([True]))
    assert r['selected_by'] == 'explicit'
    assert r['graph_single_eval'] is None and r['graph_surrounding'] is None


def test_build_row_ci_widens_as_the_sample_shrinks():
    wide = eval_checkpoints.build_row(1, held([True] * 7 + [False] * 3))
    narrow = eval_checkpoints.build_row(1, held([True] * 70 + [False] * 30))
    assert wide['perfect_percent'] == narrow['perfect_percent'] == 70.0
    assert (wide['perfect_ci95'][1] - wide['perfect_ci95'][0]) > \
           (narrow['perfect_ci95'][1] - narrow['perfect_ci95'][0])


# --------------------------------------------------------------- skips_screening

def test_graph_100_percent_goes_straight_to_full_length():
    assert eval_checkpoints.skips_screening({'selected_by': 'threshold90', 'single_eval': 100.0})


def test_graph_90_percent_is_screened_first():
    # 90% is still in select_top_checkpoints' mandatory tier, so it is measured — just not at full
    # length unless it earns a confirmation slot. This is the line the whole split turns on.
    assert not eval_checkpoints.skips_screening({'selected_by': 'threshold90', 'single_eval': 90.0})


def test_lower_graph_points_are_screened_first():
    for single in (60.0, 70.0, 80.0):
        assert not eval_checkpoints.skips_screening({'single_eval': single}), single


def test_an_explicitly_named_step_goes_straight_to_full_length():
    # Naming a step is a request to measure it. The docs promise explicit steps bypass the
    # selection thresholds; leaving one at 20 episodes would quietly break that.
    assert eval_checkpoints.skips_screening({'selected_by': 'explicit'})
    assert eval_checkpoints.skips_screening({'selected_by': 'explicit', 'single_eval': None})
    assert eval_checkpoints.skips_screening(None)


# ---------------------------------------------------------------- plan_stages

def graph(single):
    return {'selected_by': 'threshold90', 'single_eval': single}


def batch10ish(n_full=140, n_rest=480):
    """Roughly b10d's shape: a large uncapped 100% tier and a much larger 90% tier."""
    steps = list(range(1, n_full + n_rest + 1))
    selected_by = {s: graph(100.0 if i < n_full else 90.0) for i, s in enumerate(steps)}
    return steps, selected_by


def test_plan_splits_the_full_tier_from_the_screened_one():
    steps, selected_by = batch10ish(3, 7)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20, 30, 100, 10)
    assert plan['full'] == [1, 2, 3]
    assert plan['screened'] == [4, 5, 6, 7, 8, 9, 10]
    assert plan['confirmed'] == 7, 'confirm count is capped by the screened pool, not padded'


def test_plan_prices_each_stage_correctly():
    steps, selected_by = batch10ish(2, 10)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20, 3, 100, 10)
    # 2 full x 100 + 10 screened x 20 + 3 confirmed x 80 = 200 + 200 + 240
    assert plan['episodes_planned'] == 640
    assert plan['flat_episodes'] == 1200
    # one measurement per full, per screen, and per confirmation
    assert plan['measurements_planned'] == 2 + 10 + 3


def test_plan_rounds_episodes_up_to_whole_rounds():
    # 12 workers cannot run exactly 100 or exactly 20 episodes: it runs 108 and 24.
    steps, selected_by = batch10ish(1, 1)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20, 1, 100, 12)
    # 1 x 108 (full) + 1 x 24 (screen) + 1 x 84 (80 rounded up) = 216
    assert plan['episodes_planned'] == 216


def test_plan_counts_resumed_rows_as_already_done_work():
    steps, selected_by = batch10ish(1, 1)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20, 1, 100, 10, resumed=5)
    assert plan['measurements_planned'] == 5 + 1 + 1 + 1
    assert plan['episodes_planned'] == 5 * 100 + 100 + 20 + 80
    assert plan['flat_episodes'] == 7 * 100


def test_plan_with_screening_off_is_one_flat_pass():
    steps, selected_by = batch10ish(3, 7)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 0, 30, 100, 10)
    assert plan['full'] == steps and plan['screened'] == [] and plan['confirmed'] == 0
    assert plan['episodes_planned'] == plan['flat_episodes'] == 1000


def test_plan_handles_an_arm_that_is_all_100_percent():
    # A very strong arm can leave nothing to screen. There is then no stage 3, and the plan is the
    # flat one — correctly, since every checkpoint is being measured at full length anyway.
    steps, selected_by = batch10ish(6, 0)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20, 30, 100, 10)
    assert plan['screened'] == [] and plan['confirmed'] == 0
    assert plan['episodes_planned'] == plan['flat_episodes'] == 600


def test_plan_handles_an_arm_with_no_100_percent_checkpoints():
    steps, selected_by = batch10ish(0, 8)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20, 2, 100, 10)
    assert plan['full'] == [] and len(plan['screened']) == 8
    assert plan['episodes_planned'] == 8 * 20 + 2 * 80


def test_default_confirm_count_is_100_and_still_pays_for_itself():
    """Pins the confirm count, and the reason it can be this high.

    100 rather than 30 because 30 recovered b10d's best non-100% checkpoint only 57% of the time
    against 97% at 100 — a coin flip on the headline number. The guard on the ratio is the other
    half of that decision: the count can be raised until screening stops being worth doing, and
    below about 2x a flat pass is simpler and gives every checkpoint a real measurement, so a
    future increase should have to justify itself here.
    """
    assert eval_checkpoints.DEFAULT_CONFIRM_COUNT == 100
    steps, selected_by = batch10ish(146, 514)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20,
                                        eval_checkpoints.DEFAULT_CONFIRM_COUNT, 100, 10)
    assert plan['confirmed'] == 100
    ratio = plan['flat_episodes'] / plan['episodes_planned']
    assert ratio > 2.0, 'screening must still beat a flat pass by a clear margin: %.2fx' % ratio


def test_plan_on_a_real_arm_shape_beats_a_flat_pass():
    # b10d's actual shape. The saving is smaller than the 3.6x of a pure screen-everything
    # protocol, because the uncapped 100% tier is 146 checkpoints at full length.
    steps, selected_by = batch10ish(146, 514)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20, 30, 100, 10)
    ratio = plan['flat_episodes'] / plan['episodes_planned']
    assert 2.3 < ratio < 2.5, ratio


# ------------------------------------------------------------ equal_effort_pooled

def sample(perfect_flags):
    return {'perfect': list(perfect_flags), 'scores': [1] * len(perfect_flags),
            'rewards': [1.0] * len(perfect_flags), 'seconds': 1.0}


def test_equal_effort_pooled_truncates_every_checkpoint_to_the_same_prefix():
    # The deep checkpoint is a perfect 100/100 and the shallow one 5/20. Pooling the rows would
    # give 105/120 = 87.5%; equal effort gives 25/40 = 62.5%, which is the arm-level rate.
    samples = {1: sample([True] * 100), 2: sample([True] * 5 + [False] * 15)}
    perfect, episodes, count = eval_checkpoints.equal_effort_pooled(samples, 20)
    assert (perfect, episodes, count) == (25, 40, 2)


def test_equal_effort_pooled_uses_the_prefix_not_the_best_part():
    # Order matters: the first 20 must be taken as they came, not sorted or sampled favourably.
    samples = {1: sample([False] * 20 + [True] * 80)}
    assert eval_checkpoints.equal_effort_pooled(samples, 20) == (0, 20, 1)


def test_equal_effort_pooled_skips_a_checkpoint_with_too_few_episodes():
    # A checkpoint interrupted mid-screen would otherwise be pooled at a different weight.
    samples = {1: sample([True] * 20), 2: sample([True] * 7)}
    assert eval_checkpoints.equal_effort_pooled(samples, 20) == (20, 20, 1)


def test_equal_effort_pooled_counts_the_full_tier_too():
    # The 100%-tier checkpoints never have a screening stage, so a snapshot taken at the end of
    # one could not include them. Truncating can.
    samples = {1: sample([True] * 100), 2: sample([True] * 100), 3: sample([True] * 10 + [False] * 10)}
    perfect, episodes, count = eval_checkpoints.equal_effort_pooled(samples, 20)
    assert count == 3 and episodes == 60 and perfect == 50


# --------------------------------------------------------------- pick_finalists

def screened(step, perfect, episodes=20, surrounding=None):
    return {'step': step, 'episodes': episodes, 'perfect_games': perfect,
            'perfect_percent': round(100.0 * perfect / episodes, 1),
            'graph_surrounding': surrounding}


def test_pick_finalists_takes_the_best_screen_rates():
    rows = [screened(1, 10), screened(2, 18), screened(3, 4), screened(4, 15)]
    assert [r['step'] for r in eval_checkpoints.pick_finalists(rows, 2)] == [2, 4]


def test_pick_finalists_returns_fewer_than_asked_rather_than_padding():
    rows = [screened(1, 10), screened(2, 18)]
    assert len(eval_checkpoints.pick_finalists(rows, 30)) == 2


def test_pick_finalists_ranks_on_the_rate_not_the_raw_count():
    # A checkpoint measured over more episodes must not win on volume alone.
    rows = [screened(1, 12, episodes=20), screened(2, 30, episodes=100)]
    assert [r['step'] for r in eval_checkpoints.pick_finalists(rows, 1)] == [1]


def test_pick_finalists_breaks_screen_ties_on_the_surrounding_rate():
    # A 20-episode screen only has 21 possible values, so ties are the common case, and the
    # surrounding graph rate is the measured-better tie-break (+0.48 vs +0.10).
    rows = [screened(1, 15, surrounding=40.0),
            screened(2, 15, surrounding=75.0),
            screened(3, 15, surrounding=60.0)]
    assert [r['step'] for r in eval_checkpoints.pick_finalists(rows, 2)] == [2, 3]


def test_pick_finalists_tolerates_a_missing_surrounding_rate():
    rows = [screened(1, 15, surrounding=None), screened(2, 15, surrounding=10.0)]
    assert [r['step'] for r in eval_checkpoints.pick_finalists(rows, 1)] == [2]


def test_pick_finalists_accepts_a_set_of_excluded_steps():
    # main() passes `set(resumed) | set(full_steps)`, not a dict, so the membership test has to
    # work for both.
    rows = [screened(1, 20), screened(2, 15), screened(3, 14)]
    picked = eval_checkpoints.pick_finalists(rows, 2, already_full={1})
    assert [r['step'] for r in picked] == [2, 3]


def test_pick_finalists_excludes_steps_already_measured_at_full_length():
    # A resumed row already has the measurement this stage would buy, so spending a slot on it
    # would spend it on finished work.
    rows = [screened(1, 20), screened(2, 15), screened(3, 14)]
    picked = eval_checkpoints.pick_finalists(rows, 2, already_full={1: rows[0]})
    assert [r['step'] for r in picked] == [2, 3]


# ------------------------------------------------------------- wilson_interval

def test_wilson_interval_brackets_the_point_estimate():
    low, high = eval_checkpoints.wilson_interval(70, 100)
    assert low < 0.70 < high


def test_wilson_interval_stays_in_range_at_the_extremes():
    # Wilson's interval is asymmetric and never reaches the boundary exactly, which is the
    # point of using it here: 0/100 still gets a non-zero upper bound rather than collapsing
    # to a useless [0, 0] the way the normal approximation does.
    low, high = eval_checkpoints.wilson_interval(0, 100)
    assert low == 0.0 and 0.0 < high < 0.1
    low, high = eval_checkpoints.wilson_interval(100, 100)
    assert 0.9 < low < 1.0 and high <= 1.0
    assert eval_checkpoints.wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_tightens_with_more_episodes():
    # This is the whole argument for spending 500 episodes on a finalist instead of 100.
    narrow = eval_checkpoints.wilson_interval(350, 500)
    wide = eval_checkpoints.wilson_interval(70, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0]) / 2
