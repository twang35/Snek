"""Fixtures for `hyperparamTuning/perDiagnostics/plasticity_analysis.py`.

`drawdown_events` is what questions 4, 5 and 6 all rest on: it decides what counts as a drawdown, how
deep it was, when it happened and whether the arm ever recovered. Every one of those is a judgement
that a plausible implementation gets subtly wrong — events that overlap so one collapse is counted
twice, a trough taken from before its own peak, a final unrecovered decline dropped because no new
high closes it. None of those raise; they just change the answer. So the detector is pinned against
curves whose events are known by construction.

`flat_stretches` has the same property in reverse: it is the control for question 6, and a version
that returns overlapping windows would weight one long flat region as many.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'hyperparamTuning', 'perDiagnostics'))

import numpy as np

import plasticity_analysis as PA


def curve_of(pairs):
    return sorted(pairs)


def test_a_single_fall_and_recovery_is_one_event():
    curve = curve_of([(0, 10.0), (1000, 60.0), (2000, 20.0), (3000, 40.0), (4000, 70.0)])
    events = PA.drawdown_events(curve, min_depth=15.0)
    assert len(events) == 1, events
    event = events[0]
    assert (event['peak_step'], event['peak']) == (1000, 60.0)
    assert (event['trough_step'], event['trough']) == (2000, 20.0)
    assert event['depth'] == 40.0
    assert event['recovered'] is True
    assert event['recovery_step'] == 4000       # the step that made a new high, not the trough


def test_a_fall_shallower_than_the_threshold_is_not_an_event():
    curve = curve_of([(0, 50.0), (1000, 60.0), (2000, 49.0), (3000, 70.0)])
    assert PA.drawdown_events(curve, min_depth=15.0) == []
    # ... and one exactly at the threshold is, which pins the comparison direction.
    curve = curve_of([(0, 50.0), (1000, 60.0), (2000, 45.0), (3000, 70.0)])
    assert len(PA.drawdown_events(curve, min_depth=15.0)) == 1


def test_a_decline_that_never_recovers_is_still_reported_and_marked():
    """The arm the whole investigation is about: peaks, collapses, ends there.

    A detector that only emits an event when a new high closes it would drop exactly the case question
    4 asks about, and would do so silently.
    """
    curve = curve_of([(0, 20.0), (1000, 80.0), (2000, 30.0), (3000, 35.0), (4000, 33.0)])
    events = PA.drawdown_events(curve, min_depth=15.0)
    assert len(events) == 1
    assert events[0]['recovered'] is False
    assert events[0]['recovery_step'] is None
    assert events[0]['trough_step'] == 2000
    assert events[0]['depth'] == 50.0


def test_two_collapses_separated_by_a_new_high_are_two_events_and_do_not_overlap():
    curve = curve_of([(0, 10.0), (1000, 60.0), (2000, 20.0), (3000, 65.0), (4000, 25.0),
                      (5000, 70.0)])
    events = PA.drawdown_events(curve, min_depth=15.0)
    assert len(events) == 2, events
    assert [e['peak_step'] for e in events] == [1000, 3000]
    assert [e['trough_step'] for e in events] == [2000, 4000]
    # Non-overlap is the property that makes "depth" and "when" single-valued per event.
    assert events[0]['recovery_step'] <= events[1]['peak_step']


def test_the_trough_is_never_taken_from_before_its_own_peak():
    """A deep early trough must not be attached to a later peak's event.

    The bug this pins reads a running minimum that is never reset at a new high, so the second event
    inherits the first's trough — reporting a 55 pp drawdown where there was a 20 pp one.
    """
    curve = curve_of([(0, 60.0), (1000, 5.0), (2000, 65.0), (3000, 45.0), (4000, 80.0)])
    events = PA.drawdown_events(curve, min_depth=15.0)
    assert len(events) == 2, events
    second = events[1]
    assert second['peak_step'] == 2000
    assert second['trough_step'] == 3000, 'the trough must be after the peak it belongs to'
    assert second['depth'] == 20.0


def test_a_flat_curve_and_an_empty_curve_produce_nothing():
    assert PA.drawdown_events([]) == []
    assert PA.drawdown_events(curve_of([(0, 50.0), (1000, 50.0), (2000, 50.0)])) == []


def test_row_at_refuses_a_row_that_is_too_far_away():
    trained = [{'step': 100000}, {'step': 900000}]
    assert PA.row_at(trained, 105000, 15000)['step'] == 100000
    # 300000 is 200000 from the nearest row: a silent match there would put a metric on the wrong side
    # of a peak, which is the whole comparison.
    assert PA.row_at(trained, 300000, 15000) is None
    assert PA.row_at([], 100, 15000) is None


def test_near_for_widens_with_the_ladder_stride():
    # A 50k ladder cannot be held to a 15k tolerance — that dropped every event of the four
    # desktop-trained arms as unmatched while their metrics were sitting right there.
    assert PA.near_for({'stride': 10000}) == PA.NEAR
    assert PA.near_for({'stride': 50000}) == 30000
    assert PA.near_for({}) == PA.NEAR


def test_flat_stretches_are_maximal_and_do_not_overlap():
    steps = list(range(0, 2000001, 100000))
    # Flat at 60/61 for the first 1M, then a jump straight to 80. The jump has to clear the band in a
    # single sample: a gentle ramp through 65 and 75 is *inside* a 6 pp window of either plateau, so
    # the stretches legitimately swallow part of it and the boundary stops being known.
    values = [60.0 + (step // 100000 % 2) if step <= 1000000 else 80.0 for step in steps]
    curve = list(zip(steps, values))
    trained = [{'step': step} for step in steps]
    stretches = PA.flat_stretches(trained, curve, tol=6.0, span=400000)
    assert len(stretches) == 2, stretches
    assert stretches[0]['start'] == 0 and stretches[0]['end'] == 1000000
    assert stretches[1]['start'] == 1100000 and stretches[1]['end'] == 2000000
    for stretch in stretches:
        assert stretch['range'] <= 6.0
        assert stretch['span'] >= 400000
    assert stretches[0]['end'] <= stretches[1]['start'], 'stretches must not overlap'


def test_a_flat_stretch_shorter_than_the_span_is_not_reported():
    steps = [0, 100000, 200000]
    curve = list(zip(steps, [50.0, 50.0, 50.0]))
    assert PA.flat_stretches([{'step': s} for s in steps], curve, span=400000) == []


def test_per_million_is_a_slope_in_the_units_it_claims():
    # A series rising 1.0 per 1M steps must read +1.0, whatever the sampling density.
    series = [(step, step / 1e6) for step in range(0, 3000001, 250000)]
    assert abs(PA.per_million(series) - 1.0) < 1e-9
    assert abs(PA.per_million([(s, -2.0 * v) for s, v in series]) + 2.0) < 1e-9
    assert PA.per_million(series[:2]) is None            # too short to be a trend
    assert PA.per_million([(5, 1.0), (5, 2.0), (5, 3.0)]) is None


def test_hidden_params_counts_the_hidden_stack_only():
    # 30 inputs -> 50 -> 100 -> 50, and the 3-wide Q head is deliberately excluded: the head is the
    # same size in every arm, so including it would blur the size axis the batch-20 sweep varied.
    assert PA.hidden_params([50, 100, 50], 30) == 30 * 50 + 50 * 100 + 100 * 50
    assert PA.hidden_params([320], 30) == 9600


def test_split_rows_separates_the_control_and_orders_the_rest():
    payload = {'rows': [{'step': 2000, 'fresh': False}, {'step': -1, 'fresh': True},
                        {'step': 1000, 'fresh': False}]}
    fresh, trained = PA.split_rows(payload)
    assert fresh['step'] == -1
    assert [r['step'] for r in trained] == [1000, 2000]


def test_per_point_normalises_by_depth_and_skips_missing_values():
    group = [{'depth': 20.0, 'd_dormant': 0.04}, {'depth': 10.0, 'd_dormant': 0.01},
             {'depth': 30.0, 'd_dormant': None}]
    assert abs(PA.per_point(group, 'd_dormant') - 0.0015) < 1e-12
    assert PA.per_point([{'depth': 0.0, 'd_dormant': 1.0}], 'd_dormant') is None


def test_load_payloads_prefers_the_denser_ladder_for_the_same_arm():
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        for name, rows in (('coarse', 3), ('dense', 30)):
            with open(os.path.join(directory, name + '.json'), 'w') as handle:
                json.dump({'policy': 'armA', 'fc_layer_params': [50],
                           'rows': [{'step': i * 1000, 'fresh': False} for i in range(rows)]},
                          handle)
        # Something that is not a payload at all must be ignored rather than crashing the load.
        with open(os.path.join(directory, 'notes.json'), 'w') as handle:
            json.dump({'unrelated': True}, handle)
        loaded = PA.load_payloads(directory)
        assert list(loaded) == ['armA']
        assert len(loaded['armA']['rows']) == 30
