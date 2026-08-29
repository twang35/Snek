"""Comparing two result files, and the question that actually matters: is the gap noise?

**The naive reading of an A/B is wrong and this module exists to prevent it.** Two independent
100-episode measurements of the *same* policy differ by a standard deviation of
`sqrt(2 p (1-p) / n)` — 2.2 pp at p=0.975 — so per-row disagreements of 3 or 4 points *are* what
agreement looks like. What separates agreement from a real difference is the mean across rows, whose
standard error shrinks as `sd / sqrt(rows)`.

So the tests below are mostly about the null: synthetic files built to agree must report a mean
difference inside a couple of standard errors, and a file built with a real shift must not.
"""

import json
import math
import os
import random

import pytest

from tools import compare_results


def a_file(path, steps, rate, episodes=100, rng=None, shift=0.0, snek2_style=False):
    """A result file whose rows are binomial draws at `rate + shift`, so the null is exact."""
    rng = rng or random.Random(0)
    rows = []
    for step in steps:
        perfect = sum(1 for _ in range(episodes) if rng.random() < rate + shift)
        rows.append({'step': step, 'episodes': episodes, 'perfect_games': perfect,
                     'perfect_percent': round(100.0 * perfect / episodes, 1)})
    key = 'results' if snek2_style else 'rows'
    with open(path, 'w') as handle:
        json.dump({key: rows}, handle)
    return rows


# ------------------------------------------------------------------ the expected spread

def test_the_predicted_sd_is_the_sd_of_a_difference_of_two_rates():
    # Two independent estimates, so the variances add. Getting this wrong by dropping the factor of
    # two would make every honest A/B look like a 1.4x overspread.
    predicted = compare_results.expected_sd(0.975, 100, 100)
    assert predicted == pytest.approx(100 * math.sqrt(2 * 0.975 * 0.025 / 100), rel=1e-9)
    assert 2.1 < predicted < 2.3


def test_unequal_episode_counts_are_handled():
    # Comparing a 100-episode file against a 500-episode one is a real case: snek2's close-out ran
    # at 100 and snek3's stage B runs at 500.
    assert compare_results.expected_sd(0.975, 100, 500) < compare_results.expected_sd(0.975, 100, 100)
    assert compare_results.expected_sd(0.975, 500, 500) < compare_results.expected_sd(0.975, 100, 500)


def test_a_degenerate_rate_predicts_no_spread():
    assert compare_results.expected_sd(1.0, 100, 100) == 0.0
    assert compare_results.expected_sd(0.0, 100, 100) == 0.0


# ------------------------------------------------------------------ the null

def test_two_files_drawn_from_the_same_rate_read_as_agreeing(tmp_path, capsys):
    """The phase-2 gate, in miniature. 400 rows at 97.5%, no shift, two independent samples.

    The assertion is on the mean in standard errors, not on the per-row spread — which is the whole
    point of the tool.
    """
    steps = list(range(1000, 1000 + 400))
    a, b = str(tmp_path / 'a.json'), str(tmp_path / 'b.json')
    a_file(a, steps, 0.975, rng=random.Random(1))
    a_file(b, steps, 0.975, rng=random.Random(2))
    stats = compare_results.compare(a, b, show=0)
    capsys.readouterr()
    assert stats['rows'] == 400
    assert abs(stats['mean_difference']) < 3 * stats['standard_error']
    # And the observed spread must land near the prediction, or the model behind the gate is wrong.
    assert 0.8 < stats['observed_sd'] / stats['predicted_sd'] < 1.25


def test_a_real_shift_is_detected(tmp_path, capsys):
    """A fixture whose subject cannot violate it is not a fixture.

    A 4 pp shift over 400 rows must come out many standard errors away, or the test above is only
    asserting that the tool runs.
    """
    steps = list(range(1000, 1000 + 400))
    a, b = str(tmp_path / 'a.json'), str(tmp_path / 'b.json')
    a_file(a, steps, 0.975, rng=random.Random(1))
    a_file(b, steps, 0.975, rng=random.Random(2), shift=-0.04)
    stats = compare_results.compare(a, b, show=0)
    capsys.readouterr()
    assert stats['mean_difference'] > 3.0
    assert stats['mean_difference'] / stats['standard_error'] > 5.0


# ------------------------------------------------------------------ the file formats

def test_a_snek2_file_and_a_snek3_file_both_read(tmp_path, capsys):
    # snek2 keys its rows `results` and snek3 keys them `rows`. Reading both is the entire reason
    # this tool can close the phase-2 gate.
    steps = list(range(1000, 1020))
    a, b = str(tmp_path / 'a.json'), str(tmp_path / 'b.json')
    a_file(a, steps, 0.97, rng=random.Random(3))
    a_file(b, steps, 0.97, rng=random.Random(4), snek2_style=True)
    stats = compare_results.compare(a, b, show=0)
    capsys.readouterr()
    assert stats['rows'] == 20


def test_only_the_shared_steps_are_compared_and_the_rest_are_reported(tmp_path, capsys):
    a, b = str(tmp_path / 'a.json'), str(tmp_path / 'b.json')
    a_file(a, [1000, 2000, 3000], 0.97)
    a_file(b, [2000, 3000, 4000, 5000], 0.97)
    stats = compare_results.compare(a, b, show=0)
    output = capsys.readouterr().out
    assert stats['rows'] == 2
    assert '1 only in A, 2 only in B' in output


def test_no_shared_steps_is_an_error_rather_than_an_empty_report(tmp_path):
    # An empty comparison printing "mean difference +0.000" would read as perfect agreement.
    a, b = str(tmp_path / 'a.json'), str(tmp_path / 'b.json')
    a_file(a, [1000], 0.97)
    a_file(b, [2000], 0.97)
    with pytest.raises(SystemExit):
        compare_results.compare(a, b)


def test_a_file_with_neither_key_is_refused(tmp_path):
    path = str(tmp_path / 'a.json')
    with open(path, 'w') as handle:
        json.dump({'measurements': []}, handle)
    with pytest.raises(SystemExit):
        compare_results.rows_of(path)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        compare_results.rows_of(str(tmp_path / 'nope.json'))
