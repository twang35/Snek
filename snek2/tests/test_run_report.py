"""Tests for run_report.build_summary, and for the primary metric it computes.

`strong_eval_fraction` became the primary cross-arm metric on 2026-08-04 because it has the
lowest between-seed variance of the candidates measured on batch 11. These pin its definition,
since a change to it silently invalidates every comparison made with it.
"""
import run_report


def rows(*specs):
    """(step, trailing, perfect) triples -> eval rows."""
    return [{'step': s, 'trailing_avg_score': t, 'perfect_percent': p, 'epsilon': 0.002}
            for s, t, p in specs]


# ------------------------------------------------- strong_eval_fraction

def test_fraction_counts_evals_at_or_above_the_threshold():
    # 80 is inclusive: a 10-episode eval reads in steps of 10, so an exclusive bound would
    # silently mean ">= 90" and count a third as many evals.
    assert run_report.strong_eval_fraction([70, 80, 90, 100]) == 75.0


def test_fraction_is_zero_when_nothing_qualifies():
    assert run_report.strong_eval_fraction([0, 10, 70]) == 0.0


def test_fraction_is_100_when_everything_qualifies():
    assert run_report.strong_eval_fraction([80, 90, 100]) == 100.0


def test_fraction_handles_no_evals():
    assert run_report.strong_eval_fraction([]) == 0.0


def test_fraction_is_a_share_not_a_count():
    # The denominator is the arm's own eval count, which is why two arms are only comparable at
    # a common step horizon. Doubling the run with mediocre evals must halve the figure.
    strong = [100] * 10
    assert run_report.strong_eval_fraction(strong) == 100.0
    assert run_report.strong_eval_fraction(strong + [0] * 10) == 50.0


def test_fraction_penalises_a_long_decline_that_best30_ignores():
    # The reason for the switch, as a fixture. Two arms with the same best 30-eval window: one
    # holds it, one collapses afterwards. best_perfect30 calls them equal; this does not.
    held = rows(*[(1000 * i, 90.0, 100) for i in range(60)])
    collapsed = rows(*([(1000 * i, 90.0, 100) for i in range(30)]
                       + [(1000 * (30 + i), 90.0, 0) for i in range(30)]))
    a, b = run_report.build_summary(held), run_report.build_summary(collapsed)
    assert a['best_perfect30'] == b['best_perfect30']
    assert a['strong_eval_fraction'] == 100.0
    assert b['strong_eval_fraction'] == 50.0


# ----------------------------------------------------------- build_summary

def test_summary_exposes_the_new_metric_alongside_the_old_one():
    # best_perfect30 stays: every arm through batch 11 is recorded on it, so dropping it would
    # make the historical table unreadable.
    summary = run_report.build_summary(rows(*[(1000 * i, 50.0, 80) for i in range(30)]))
    assert summary['strong_eval_fraction'] == 100.0
    assert summary['best_perfect30']['value'] == 80.0


def test_summary_is_empty_for_no_rows():
    assert run_report.build_summary([]) == {}


def test_best_perfect30_needs_a_full_window():
    # Fewer than 30 evals must not report a partial window as a best-30, which would read high
    # on any arm in its first 30k steps.
    summary = run_report.build_summary(rows(*[(1000 * i, 50.0, 100) for i in range(29)]))
    assert summary['best_perfect30']['value'] == 0.0
    assert summary['strong_eval_fraction'] == 100.0


def test_zero_since_is_the_current_stretch_not_the_earliest():
    # The documented trap: dead_since is history, zero_since answers "is it dead now".
    # b8d carried dead_since=275000 while going on to a 36% best-30 window.
    recovered = rows(*([(1000 * i, 0.0, 0) for i in range(30)]
                       + [(1000 * (30 + i), 50.0, 40) for i in range(5)]))
    summary = run_report.build_summary(recovered)
    assert summary['dead_since'] == 0
    assert summary['zero_since'] is None


def test_zero_since_reports_the_start_of_an_ongoing_stretch():
    dying = rows(*([(1000 * i, 50.0, 40) for i in range(5)]
                   + [(1000 * (5 + i), 0.0, 0) for i in range(10)]))
    assert run_report.build_summary(dying)['zero_since'] == 5000
