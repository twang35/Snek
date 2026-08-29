"""Which checkpoints a wave measures — and, mostly, what it refuses.

The selector is where a wave's cost and its meaning are both decided, so the tests concentrate on
two things. **A selected step that is not on disk is an error**, because silently dropping it turns a
dispatch bug into a short result file, which reads exactly like a checkpoint that was never good
enough to select. And **the shard slices stride rather than block**, because episode cost is not
uniform along an arm.
"""

import os

import pytest

from dqn import net as network
from tools import checkpoints
from tools import step_selectors as selectors


def a_policy(tmp_path, steps):
    directory = str(tmp_path / 'policy')
    for step in steps:
        checkpoints.save(directory, step, network.QNet(30, [4], 3))
    return directory


# ---------------------------------------------------------------------- parsing

def test_the_known_selectors_parse():
    assert selectors.parse('all') == ('all', None)
    assert selectors.parse('screen') == ('screen', 95.0)
    assert selectors.parse('screen:98') == ('screen', 98.0)
    assert selectors.parse('above:98') == ('above', (98.0, None))
    assert selectors.parse('above:98:hof') == ('above', (98.0, 'hof'))
    assert selectors.parse('steps:runs/x.txt') == ('steps', 'runs/x.txt')


def test_the_default_screen_is_the_protocol_threshold():
    # 95/100 in stage A is what stage B selects on. A bare `screen` has to mean that, or a wave
    # launched without arguments measures something other than the protocol.
    assert selectors.parse('screen')[1] == float(selectors.DEFAULT_SCREEN) == 95.0


def test_a_retired_snek2_selector_is_refused_by_name():
    # The point of closing the list. `top50` and the tiered stages are gone, and a spec or a habit
    # that still names one must fail loudly rather than falling through to a default.
    for token in ('top50', 'full', 'confirm', 'flat'):
        with pytest.raises(selectors.SelectorError) as raised:
            selectors.parse(token)
        assert 'unknown selector' in str(raised.value)


def test_a_selector_missing_its_argument_is_refused():
    for token in ('above:', 'steps:'):
        with pytest.raises(selectors.SelectorError):
            selectors.parse(token)


# ---------------------------------------------------------------------- resolving

def test_all_is_every_checkpoint_present(tmp_path):
    directory = a_policy(tmp_path, [3000, 1000, 2000])
    steps, description = selectors.resolve(directory, 'all')
    assert steps == [1000, 2000, 3000]
    assert 'every checkpoint' in description


def test_a_step_list_is_read_with_comments_and_blanks_skipped(tmp_path):
    directory = a_policy(tmp_path, [1000, 2000, 3000])
    listing = tmp_path / 'steps.txt'
    listing.write_text('# a comment\n\n2000\n1000  # trailing\n\n')
    steps, description = selectors.resolve(directory, 'steps:{0}'.format(listing))
    assert steps == [1000, 2000]
    assert '2 steps' in description


def test_a_selected_step_with_no_checkpoint_is_an_error(tmp_path):
    """The refusal that matters most, so it is asserted rather than assumed.

    A wave that quietly skipped it would write a file one row short, and nothing downstream can tell
    a missing row from a checkpoint that never cleared the screen.
    """
    directory = a_policy(tmp_path, [1000, 2000])
    listing = tmp_path / 'steps.txt'
    listing.write_text('1000\n999999\n')
    with pytest.raises(selectors.SelectorError) as raised:
        selectors.resolve(directory, 'steps:{0}'.format(listing))
    assert '999999' in str(raised.value)


def test_duplicates_in_a_step_list_collapse(tmp_path):
    directory = a_policy(tmp_path, [1000, 2000])
    listing = tmp_path / 'steps.txt'
    listing.write_text('2000\n1000\n2000\n')
    assert selectors.resolve(directory, 'steps:{0}'.format(listing))[0] == [1000, 2000]


def test_a_policy_with_no_checkpoints_is_an_error(tmp_path):
    with pytest.raises(selectors.SelectorError):
        selectors.resolve(str(tmp_path), 'all')


def test_a_bad_line_in_a_step_list_names_its_line_number(tmp_path):
    directory = a_policy(tmp_path, [1000])
    listing = tmp_path / 'steps.txt'
    listing.write_text('1000\nnot-a-step\n')
    with pytest.raises(selectors.SelectorError) as raised:
        selectors.resolve(directory, 'steps:{0}'.format(listing))
    assert ':2:' in str(raised.value)


def test_screen_says_what_to_use_instead_when_there_is_no_stage_a_file(tmp_path):
    # The imported-arm case, which is how phase 2's A/B runs: there is no snek3 training history to
    # screen on, so the message has to point at `steps:` rather than just failing.
    directory = a_policy(tmp_path, [1000])
    with pytest.raises(selectors.SelectorError) as raised:
        selectors.resolve(directory, 'screen', policy='nothing-trained-here')
    assert 'steps:' in str(raised.value)


# ---------------------------------------------------------------------- sharding

def test_the_shards_partition_the_steps_exactly():
    steps = list(range(1000, 1000 + 97))
    slices = [selectors.slice_for(steps, shard, 7) for shard in range(7)]
    assert sorted(step for shard in slices for step in shard) == steps
    assert sum(len(shard) for shard in slices) == len(steps)


def test_the_shards_stride_rather_than_block():
    """Striding is a load-balancing decision, not an arbitrary one.

    A strong checkpoint plays a ~1,800-step perfect game and a weak one dies in 40, so contiguous
    blocks hand the shard covering the trained end of an arm several times the work of the shard
    covering the start — and a wave finishes when its slowest shard does. Asserted by requiring every
    shard to draw from both halves of the arm.
    """
    steps = list(range(1000, 1000 + 400))
    midpoint = steps[len(steps) // 2]
    for shard in range(4):
        mine = selectors.slice_for(steps, shard, 4)
        assert any(step < midpoint for step in mine), shard
        assert any(step >= midpoint for step in mine), shard


def test_shard_counts_differ_by_at_most_one():
    steps = list(range(50))
    sizes = {len(selectors.slice_for(steps, shard, 7)) for shard in range(7)}
    assert max(sizes) - min(sizes) <= 1


def test_a_shard_index_outside_the_range_is_refused():
    with pytest.raises(selectors.SelectorError):
        selectors.slice_for([1, 2, 3], 4, 4)
    with pytest.raises(selectors.SelectorError):
        selectors.slice_for([1, 2, 3], -1, 4)


def test_more_shards_than_steps_leaves_the_surplus_empty():
    # Not an error: the wave reports it and the empty shards exit at once.
    slices = [selectors.slice_for([1, 2], shard, 5) for shard in range(5)]
    assert [len(shard) for shard in slices] == [1, 1, 0, 0, 0]
