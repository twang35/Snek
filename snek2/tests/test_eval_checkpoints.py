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
