"""Fixtures for `hyperparamTuning/perDiagnostics/behaviour_profile.py`.

The index arithmetic is the part worth pinning. The root `CLAUDE.md` warns that not every block in the
observation is a per-action triple, and this script reads two kinds in the same expression: the
triples at 6-8, 15-17, 18-20, 23-25 and 26-28, and the **interleaved** pair at 9-14 where tail
reachability sits at `9 + 2a` and the region count at `10 + 2a`. Getting the stride wrong there
silently reports the region count as tail reachability, both of which are floats in [0, 1] that look
plausible in a results table.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'hyperparamTuning', 'perDiagnostics'))

import behaviour_profile as profile


def labelled_observation():
    """obs[i] == i, so a misread index is visible as the wrong number rather than a wrong-looking one."""
    return [float(i) for i in range(30)]


def test_triple_blocks_index_one_per_action():
    obs = labelled_observation()
    for action in range(3):
        assert profile.per_action(obs, profile.SAFE, action) == 6 + action
        assert profile.per_action(obs, profile.CHASE, action) == 15 + action
        assert profile.per_action(obs, profile.WIN, action) == 18 + action
        assert profile.per_action(obs, profile.HUG, action) == 23 + action
        assert profile.per_action(obs, profile.NOTTAIL, action) == 26 + action


def test_tail_and_region_block_is_interleaved_not_a_triple():
    # The whole point of the stride argument. With stride 1 these would read 9,10,11 and 10,11,12.
    obs = labelled_observation()
    assert [profile.per_action(obs, profile.TAIL_REACH, a, 2) for a in range(3)] == [9, 11, 13]
    assert [profile.per_action(obs, profile.REGIONS, a, 2) for a in range(3)] == [10, 12, 14]


def test_per_action_never_reads_outside_its_own_block():
    # Every index a per-action read can touch, across all blocks and actions, must stay inside the
    # 30-value vector and must not collide with the single-value inputs at 21, 22 and 29.
    obs = labelled_observation()
    singles = {21, 22, 29}
    touched = set()
    for first, stride in ((profile.SAFE, 1), (profile.CHASE, 1), (profile.WIN, 1),
                          (profile.HUG, 1), (profile.NOTTAIL, 1),
                          (profile.TAIL_REACH, 2), (profile.REGIONS, 2)):
        for action in range(3):
            index = int(profile.per_action(obs, first, action, stride))
            assert 0 <= index < 30, 'index %d out of range' % index
            touched.add(index)
    assert not touched & singles, 'per-action read collides with %r' % (touched & singles)


def test_step_record_reads_the_chosen_action_from_every_block():
    # Covers the call site, not just the helper: an earlier version had the stride only at the call
    # site, so dropping it there broke the measurement while every helper test still passed.
    obs = labelled_observation()
    expected = {
        0: {'safe': 6, 'tail_reach': 9, 'regions': 10, 'chase_safe': 15, 'hug': 23, 'nottail': 26},
        1: {'safe': 7, 'tail_reach': 11, 'regions': 12, 'chase_safe': 16, 'hug': 24, 'nottail': 27},
        2: {'safe': 8, 'tail_reach': 13, 'regions': 14, 'chase_safe': 17, 'hug': 25, 'nottail': 28},
    }
    for action, wanted in expected.items():
        record = profile.step_record(obs, action)
        for metric, index in wanted.items():
            assert record[metric] == index, ('action %d %s read %s, wanted obs[%d]'
                                             % (action, metric, record[metric], index))


def test_step_record_forward_flag_only_fires_on_the_forward_action():
    obs = labelled_observation()
    fired = [profile.step_record(obs, a)['forward'] for a in range(3)]
    assert sum(fired) == 1 and fired[profile.FORWARD_ACTION] == 1.0


def test_step_record_win_available_is_over_all_actions_not_the_chosen_one():
    # win_avail asks whether *any* move wins, so it must not depend on the action taken.
    obs = [0.0] * 30
    obs[profile.WIN + 1] = 1.0
    assert [profile.step_record(obs, a)['win_avail'] for a in range(3)] == [1.0, 1.0, 1.0]
    assert profile.step_record([0.0] * 30, 0)['win_avail'] == 0.0


def test_step_record_covers_every_reported_metric():
    # If a metric is added to the record but not to STEP_METRICS it never reaches the output file.
    assert set(profile.step_record(labelled_observation(), 0)) == set(profile.STEP_METRICS)


def test_bands_partition_every_reachable_length():
    # A length in no band is dropped from every metric; one in two bands is counted twice. The snake
    # starts at 5 and a full board is 100, so lengths below 10 are deliberately unbanded.
    for length in range(10, 100):
        hits = [name for name, low, high in profile.BANDS if low <= length <= high]
        assert len(hits) == 1, 'length %d lands in %r' % (length, hits)


def test_bands_are_contiguous_and_ordered():
    highs = [high for _, _, high in profile.BANDS]
    lows = [low for _, low, _ in profile.BANDS]
    assert lows == sorted(lows)
    for previous, following in zip(highs, lows[1:]):
        assert following == previous + 1, 'gap or overlap between %d and %d' % (previous, following)


def test_forward_action_matches_the_environment_mapping():
    from snake_constants import TF_ACTION_TO_ACTIONS
    assert TF_ACTION_TO_ACTIONS[profile.FORWARD_ACTION] == 'forward'
