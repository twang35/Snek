"""The stage-A summary: the numbers every progress check reads instead of recomputing.

Two of these carry warnings in `CLAUDE.md`, and both are tested against the incident that produced
the warning. **`zero_since` answers "is it dead now" and `dead_since` does not** — a snek2 arm carried
`dead_since=275000` while going on to a 36% best-30 window. And **`strong_eval_fraction` is a
fraction of an arm's own evals**, so it only compares at a common horizon.
"""

import pytest

from tools import run_report


def evals(pairs, epsilon=0.01):
    """`(step, trailing, perfect)` triples as eval rows."""
    return [{'step': step, 'avg_score': trailing, 'trailing_avg_score': trailing,
             'min_score': 0, 'max_score': 95, 'avg_reward': 0.0,
             'perfect_percent': perfect, 'epsilon': epsilon}
            for step, trailing, perfect in pairs]


# ------------------------------------------------------------------ strong_eval_fraction

def test_the_fraction_counts_evals_at_or_above_the_threshold():
    assert run_report.strong_eval_fraction([80.0, 80.0]) == 100.0
    assert run_report.strong_eval_fraction([79.9, 80.0]) == 50.0
    assert run_report.strong_eval_fraction([]) == 0.0


def test_the_threshold_is_inclusive():
    # At exactly 80 an eval counts. Not a detail: it is the modal value of a good arm's evals, so an
    # exclusive comparison would shift every arm's headline metric.
    assert run_report.strong_eval_fraction([run_report.STRONG_EVAL_THRESHOLD]) == 100.0


def test_the_fraction_is_of_the_arms_own_evals():
    """Which is why it compares only at a common horizon, and the test says so explicitly.

    The same arm, read at 10 evals and at 20 after a decline, gives two different answers — that is
    intended, and it is exactly why a comparison has to fix the horizon first.
    """
    good = [90.0] * 10
    assert run_report.strong_eval_fraction(good) == 100.0
    assert run_report.strong_eval_fraction(good + [0.0] * 10) == 50.0


# ------------------------------------------------------------------ the death fields

def test_zero_since_is_the_current_stretch_and_dead_since_is_the_earliest():
    """The distinction that cost a real judgement call.

    This arm dies early, recovers, and is healthy now. `dead_since` records the history and stays
    set; `zero_since` is None because the latest eval is above threshold. Reading `dead_since` as
    "is it dead" would retire a working arm.
    """
    rows = evals([(step * 1000, 0.0, 0.0) for step in range(1, 41)]
                 + [(step * 1000, 50.0, 60.0) for step in range(41, 81)])
    summary = run_report.build_summary(rows)
    assert summary['dead_since'] == 1000
    assert summary['zero_since'] is None


def test_zero_since_marks_the_start_of_an_unbroken_current_stretch():
    rows = evals([(1000, 50.0, 60.0), (2000, 50.0, 60.0)]
                 + [(step * 1000, 0.0, 0.0) for step in range(3, 43)])
    summary = run_report.build_summary(rows)
    assert summary['zero_since'] == 3000


def test_a_stretch_shorter_than_the_window_sets_zero_since_but_not_dead_since():
    # `zero_since` has no window — one sub-threshold eval starts a stretch — while `dead_since`
    # needs 30 consecutive. So a brief dip is visible without being called a wall.
    rows = evals([(1000, 50.0, 60.0), (2000, 0.0, 0.0), (3000, 0.0, 0.0)])
    summary = run_report.build_summary(rows)
    assert summary['zero_since'] == 2000
    assert summary['dead_since'] is None


def test_a_healthy_arm_has_neither():
    rows = evals([(step * 1000, 90.0, 95.0) for step in range(1, 41)])
    summary = run_report.build_summary(rows)
    assert summary['dead_since'] is None and summary['zero_since'] is None


# ------------------------------------------------------------------ the rest of the summary

def test_the_peak_reports_the_step_it_happened_at():
    rows = evals([(1000, 10.0, 0.0), (2000, 90.0, 0.0), (3000, 20.0, 0.0)])
    summary = run_report.build_summary(rows)
    assert summary['peak_trailing'] == {'value': 90.0, 'step': 2000}
    assert summary['trailing_now'] == 20.0


def test_the_best_window_is_only_reported_once_there_are_enough_evals():
    # Fewer than the window and there is no window to be best, so it stays 0 rather than reporting
    # a partial window as if it were full — which would flatter every young arm.
    rows = evals([(step * 1000, 50.0, 100.0) for step in range(1, 10)])
    assert run_report.build_summary(rows)['best_perfect30']['value'] == 0.0
    rows = evals([(step * 1000, 50.0, 100.0) for step in range(1, 31)])
    assert run_report.build_summary(rows)['best_perfect30']['value'] == 100.0


def test_the_best_window_is_a_maximum_and_the_summary_says_where():
    rows = evals([(step * 1000, 50.0, 0.0) for step in range(1, 31)]
                 + [(step * 1000, 50.0, 90.0) for step in range(31, 61)])
    summary = run_report.build_summary(rows)
    assert summary['best_perfect30']['value'] == 90.0
    assert summary['best_perfect30']['step'] == 60000


def test_an_arm_with_no_evals_summarises_to_nothing():
    # Not an error and not a summary full of zeroes, which would read as a dead arm.
    assert run_report.build_summary([]) == {}


# ------------------------------------------------------------------ history I/O

def test_a_resumed_eval_replaces_the_row_at_its_step():
    """Resuming re-evaluates at the step the previous run ended on.

    Appending instead would put two points at the same x and draw a vertical segment in the graph.
    """
    rows = evals([(1000, 10.0, 0.0), (2000, 20.0, 0.0)])
    run_report.merge_eval_row(rows, {'step': 2000, 'avg_score': 99.0, 'trailing_avg_score': 99.0,
                                     'perfect_percent': 50.0, 'epsilon': 0.01})
    assert len(rows) == 2
    assert rows[-1]['avg_score'] == 99.0


def test_rows_stay_in_step_order_however_they_arrive():
    rows = []
    for step in (3000, 1000, 2000):
        run_report.merge_eval_row(rows, {'step': step, 'avg_score': 0.0,
                                         'trailing_avg_score': 0.0, 'perfect_percent': 0.0,
                                         'epsilon': 0.0})
    assert [row['step'] for row in rows] == [1000, 2000, 3000]


def test_history_round_trips_and_carries_its_summary(tmp_path):
    path = str(tmp_path / 'runs' / 'a_evals.json')
    rows = evals([(step * 1000, 50.0, 90.0) for step in range(1, 41)])
    summary = run_report.save_history(path, rows, resume_steps=[20000])
    loaded, resumes = run_report.load_history(path)
    assert loaded == rows and resumes == [20000]
    assert summary['evals'] == 40 and summary['strong_eval_fraction'] == 100.0


def test_a_corrupt_history_is_reported_and_treated_as_empty(tmp_path, capsys):
    # Losing a graph is bad; losing a training run to a graph is worse. So this must not raise.
    path = tmp_path / 'a_evals.json'
    path.write_text('{not json')
    assert run_report.load_history(str(path)) == ([], [])
    assert 'could not read' in capsys.readouterr().out


def test_an_absent_history_is_empty(tmp_path):
    assert run_report.load_history(str(tmp_path / 'nope.json')) == ([], [])


# ------------------------------------------------------------------ the markdown report

def test_the_report_leads_with_the_summary_line(tmp_path):
    path = str(tmp_path / 'a.md')
    rows = evals([(step * 1000, 50.0, 90.0) for step in range(1, 41)])
    run_report.write_run_report(path, 'b1a-thing', {'lr': '1e-7'}, rows, 'b1a-thing.png')
    text = open(path).read()
    assert text.startswith('# b1a-thing')
    assert 'sef **100.0**' in text
    assert '| lr | 1e-7 |' in text
    assert '![b1a-thing](b1a-thing.png)' in text


def test_a_long_eval_series_is_elided_rather_than_printed_whole(tmp_path):
    # A 3M-step arm has ~3,000 evals. Inlining them makes the file unreadable and duplicates the
    # JSON beside it.
    path = str(tmp_path / 'a.md')
    rows = evals([(step * 1000, 50.0, 90.0) for step in range(1, 500)])
    run_report.write_run_report(path, 'a', {}, rows)
    lines = open(path).read().splitlines()
    assert sum(1 for line in lines if line.startswith('| 1')) < 30
    assert any('...' in line for line in lines)


def test_a_report_for_an_arm_below_threshold_says_how_long(tmp_path):
    path = str(tmp_path / 'a.md')
    rows = evals([(1000, 50.0, 60.0)] + [(step * 1000, 0.0, 0.0) for step in range(2, 42)])
    run_report.write_run_report(path, 'a', {}, rows)
    text = open(path).read()
    assert 'Below threshold since step 2,000' in text
    assert 'Not a verdict' in text


def test_the_stage_b_section_flags_its_own_maximum_as_selected(tmp_path):
    """The maximum over stage B is a selected high, and the report has to say so where it is read.

    snek2's 99.0%/500 champion re-measured at 97.5% over 1,000 fresh episodes, and its four best
    hall-of-fame entries fell a mean 1.4 pp. A promotion reads this file.
    """
    path = str(tmp_path / 'a.md')
    stage_b = [{'step': 1000, 'episodes': 500, 'perfect_percent': 97.0, 'perfect_ci95': [95, 98]},
               {'step': 2000, 'episodes': 500, 'perfect_percent': 99.0, 'perfect_ci95': [97, 100]}]
    run_report.write_run_report(path, 'a', {}, evals([(1000, 50.0, 90.0)]), stage_b_rows=stage_b)
    text = open(path).read()
    assert 'best **99.0%** @2,000' in text
    assert 'selected high' in text
    assert '**1** row(s) at >=98%' in text
