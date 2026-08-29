"""Fixtures for `hyperparamTuning/perDiagnostics/input_sensitivity_over_time.py`.

Only the pure parts. Everything else in that script needs a restored checkpoint, and the load-bearing
guard there is a runtime one — it re-reads `global_step` after each restore and dies if it disagrees
with the requested step, because a silent restore failure would otherwise produce a whole plausible
series off one network.

The step spec is worth pinning because a ladder that quietly drops or shifts checkpoints changes
every churn number computed from it: churn is `1 - agreement` divided by the step gap, so a wrong gap
is a wrong answer rather than a missing one.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'hyperparamTuning', 'perDiagnostics'))

import input_sensitivity_over_time as sensitivity


def test_parse_steps_single_values():
    assert sensitivity.parse_steps('240000') == [240000]
    assert sensitivity.parse_steps('300000,240000') == [240000, 300000], 'must come back sorted'


def test_parse_steps_range_is_inclusive_of_both_ends():
    # An exclusive upper end would silently drop the last checkpoint of every window, which is the
    # one a before/after comparison leans on.
    assert sensitivity.parse_steps('200000-210000:5000') == [200000, 205000, 210000]


def test_parse_steps_range_default_stride_is_one_eval():
    assert sensitivity.parse_steps('200000-203000') == [200000, 201000, 202000, 203000]


def test_parse_steps_deduplicates_across_pieces():
    # Overlapping windows are the normal way these ladders get written by hand, and a duplicated
    # step would appear as a zero-gap transition -- a division by zero in the churn rate.
    assert sensitivity.parse_steps('200000-204000:2000,204000,202000') == [200000, 202000, 204000]


def test_parse_steps_ignores_empty_pieces_and_whitespace():
    assert sensitivity.parse_steps(' 240000 , ,300000 ') == [240000, 300000]


def test_bands_cover_every_sampled_length_exactly_once():
    # A length falling in no band vanishes from the per-band agreement; one falling in two is
    # double-counted. Either way the bands stop summing to the overall figure.
    for length in sensitivity.LENGTHS:
        hits = [name for name, low, high in sensitivity.BANDS if low <= length <= high]
        assert len(hits) == 1, 'length %d lands in %r' % (length, hits)


def test_endgame_band_is_inside_the_sampled_lengths():
    low, high = sensitivity.ENDGAME
    assert [L for L in sensitivity.LENGTHS if low <= L <= high], 'endgame band samples nothing'
