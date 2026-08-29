"""Fixtures for `hyperparamTuning/perDiagnostics/init_optimism.py`.

`ladder_steps` decides *where* the wash-out is sampled, and every way of getting it wrong produces a
plausible-looking table rather than an error. The measurement it serves — a value offset with a
~280k-step half-life over a 2M-step arm — is decided almost entirely in the first 100k steps, so a
ladder that drifts toward even spacing would put one rung inside the transient and fourteen in the tail
and report "washed out immediately" on the same checkpoints. It also has to end on the newest
checkpoint, because `excess` is defined against that row.

`score_at` is pinned because the whole harm argument reads two columns side by side: a value offset that
is still present while the arm is already scoring is the evidence that the offset is common-mode. An
off-by-one alignment there would swap "learned while miscalibrated" for "learned after calibrating".
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'hyperparamTuning', 'perDiagnostics'))

import init_optimism as IO


EVERY_1K = list(range(1000, 2000001, 1000))


def test_ladder_is_log_spaced_not_even():
    """The gaps must *grow*. An evenly spaced ladder is the mutant this exists to catch."""
    rungs = IO.ladder_steps(EVERY_1K, 15)
    gaps = [b - a for a, b in zip(rungs, rungs[1:])]
    assert gaps[0] > 0
    # Geometric spacing over three orders of magnitude makes the last gap enormously wider than the
    # first; an arithmetic ladder would make them equal.
    assert gaps[-1] > 100 * gaps[0], gaps
    assert all(later >= earlier for earlier, later in zip(gaps, gaps[1:])), gaps


def test_ladder_spends_its_early_rungs_inside_the_transient():
    """At least a third of the rungs below 100k on a 2M arm — the point of log spacing."""
    rungs = IO.ladder_steps(EVERY_1K, 15)
    early = [step for step in rungs if step <= 100000]
    assert len(early) >= 5, rungs


def test_ladder_always_ends_on_the_newest_checkpoint():
    """`excess` is taken against the last row, so the last row has to be the arm's final checkpoint."""
    for count in (2, 3, 7, 15, 40):
        rungs = IO.ladder_steps(EVERY_1K, count)
        assert rungs[-1] == EVERY_1K[-1], (count, rungs[-3:])


def test_ladder_only_returns_real_checkpoints_ascending():
    steps = [8000, 13000, 27000, 91000, 350000, 1200000, 2011000]
    rungs = IO.ladder_steps(steps, 15)
    assert all(step in steps for step in rungs), rungs
    assert rungs == sorted(rungs), rungs


def test_ladder_drops_duplicates_rather_than_repeating_a_checkpoint():
    """A sparse early series gets *fewer* rungs, not the same one restored several times.

    Restoring one checkpoint twice would print two identical rows, which reads as a value that stopped
    moving — the exact conclusion the script is measuring.
    """
    sparse = [1000, 2000, 3000, 500000, 1000000]
    rungs = IO.ladder_steps(sparse, 15)
    assert len(rungs) == len(set(rungs)), rungs
    assert len(rungs) <= len(sparse), rungs


def test_ladder_handles_degenerate_inputs():
    assert IO.ladder_steps([], 10) == []
    assert IO.ladder_steps([5000], 10) == [5000]
    assert IO.ladder_steps(EVERY_1K, 1) == [EVERY_1K[-1]]


def test_score_at_takes_the_nearest_eval_either_side():
    curve = [(0, 0.0), (10000, 45.0), (20000, 80.0)]
    assert IO.score_at(curve, 10000) == 45.0
    assert IO.score_at(curve, 11000) == 45.0        # nearer 10000 than 20000
    assert IO.score_at(curve, 16000) == 80.0        # nearer 20000
    assert IO.score_at(curve, 999999) == 80.0       # past the end, clamps to the last eval


def test_score_at_is_nan_without_a_curve():
    value = IO.score_at([], 10000)
    assert value != value, value                    # nan, so the column prints '-' rather than 0.0


def test_washout_takes_the_first_rung_inside_tolerance():
    """First, not last: the question is when the offset *stopped*, and `excess` re-crosses zero later.

    Both b36a and b36c overshoot slightly and come back, so a version keying on the last crossing would
    report the final checkpoint for every arm.
    """
    rows = [
        {'policy': 'init-standard', 'step': 0, 'excess': 0.0, 'score': 1.0},
        {'policy': 'arm', 'step': 100000, 'excess': 20.0, 'score': 50.0},
        {'policy': 'arm', 'step': 900000, 'excess': 1.0, 'score': 94.0},
        {'policy': 'arm', 'step': 1400000, 'excess': -1.8, 'score': 93.0},
        {'policy': 'arm', 'step': 2000000, 'excess': 0.0, 'score': 90.0},
    ]
    washed = IO.washout(rows, 2.0)
    assert list(washed) == ['arm'], washed          # the synthetic init row is never a wash-out
    assert washed['arm']['step'] == 900000, washed['arm']


def test_washout_reports_nothing_for_an_arm_that_never_settles():
    rows = [{'policy': 'arm', 'step': step, 'excess': 30.0, 'score': 10.0}
            for step in (10000, 100000, 1000000)]
    assert IO.washout(rows, 2.0) == {}
