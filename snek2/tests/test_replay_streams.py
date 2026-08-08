"""Tests for TrajectoryPrioritizedReplayBuffer's per-stream windows.

A stored item is "the last `sequence_length` add() calls for this stream, in call order". There is
no episode id in the payload and no discontinuity check, so the *only* thing keeping a stored
transition real is that consecutive calls belong to the same continuation. The forking collector
round-robins several games through one buffer, so a single shared window would splice frame k of one
game onto frame k+1 of another and store a transition that never happened — with nothing raising,
and nothing downstream able to tell.

`test_two_streams_never_share_an_item` is the guard for that, and it is the reason this file exists.

Items are read back through `buffer._buffer.get_all_transitions()`. Reaching into the cpprb handle
from a test is deliberate: the alternative is production API that only tests would call.
"""
import collections

import numpy as np
from tf_agents.specs import array_spec

from prioritized_replay_buffer import DEFAULT_STREAM, TrajectoryPrioritizedReplayBuffer

# One scalar field, so an item's provenance is readable straight off the payload. The real buffer
# stores a whole Trajectory; nothing here depends on that, and a 30-value observation would just
# make the assertions harder to read.
#
# An ArraySpec with a *numpy* dtype, matching what snek2.py hands the buffer — it passes
# `tensor_spec.to_nest_array_spec(agent.collect_data_spec)`, and cpprb cannot interpret a
# tf.DType.
SPEC = array_spec.ArraySpec((), np.float32, 'tag')


def buffer_with(sequence_length=2, capacity=1000):
    return TrajectoryPrioritizedReplayBuffer(SPEC, capacity, sequence_length=sequence_length)


def tag(stream, index):
    """A frame that encodes which stream it came from and how far along it was.

    Streams are spaced by 100 so `divmod` recovers both, which is what lets a test assert that an
    item's frames are consecutive *and* from one stream rather than merely plausible.
    """
    return np.float32(stream * 100 + index)


def stored_items(buffer):
    """Every stored item as a list of (stream, index) pairs, oldest first."""
    transitions = buffer._buffer.get_all_transitions()
    return [[divmod(int(round(float(value))), 100) for value in row]
            for row in transitions['field0']]


# ------------------------------------------------------------------ the default path

def test_the_default_stream_behaves_exactly_as_before():
    # A caller that never mentions a stream must see the old behaviour: K adds give K - (n - 1)
    # items, each one the consecutive inputs.
    buffer = buffer_with()
    for index in range(5):
        buffer.add(tag(DEFAULT_STREAM, index))
    assert stored_items(buffer) == [[(0, 0), (0, 1)], [(0, 1), (0, 2)],
                                    [(0, 2), (0, 3)], [(0, 3), (0, 4)]]


def test_a_partial_window_stores_nothing():
    buffer = buffer_with(sequence_length=3)
    buffer.add(tag(0, 0))
    buffer.add(tag(0, 1))
    assert buffer.size == 0
    buffer.add(tag(0, 2))
    assert stored_items(buffer) == [[(0, 0), (0, 1), (0, 2)]]


def test_naming_the_default_stream_explicitly_is_the_same_thing():
    implicit, explicit = buffer_with(), buffer_with()
    for index in range(4):
        implicit.add(tag(0, index))
        explicit.add(tag(0, index), stream=DEFAULT_STREAM)
    assert stored_items(implicit) == stored_items(explicit)


# ------------------------------------------------------------------ the guard

def test_two_streams_never_share_an_item():
    """Interleave two branches and assert no item mixes them.

    This is the test that stands between the forking collector and a silently poisoned buffer.
    Verified to fail when `add` goes back to a single shared window: with one window the
    alternating adds below store [(1,0),(2,0)], [(2,0),(1,1)], … — every item a fabrication.
    """
    buffer = buffer_with()
    for index in range(6):
        buffer.add(tag(1, index), stream=1)
        buffer.add(tag(2, index), stream=2)

    items = stored_items(buffer)
    assert items, 'the fixture must store something'
    for item in items:
        streams = {stream for stream, _ in item}
        assert len(streams) == 1, 'item mixes streams: {0}'.format(item)
        indexes = [index for _, index in item]
        assert indexes == list(range(indexes[0], indexes[0] + len(indexes))), \
            'item is not consecutive within its stream: {0}'.format(item)


def test_streams_stay_contiguous_at_n_step_update_3():
    # sequence_length = n_step_update + 1, and window logic that happens to work at 2 can still
    # splice at 4. Three streams, so a bug that pairs neighbours also shows up.
    buffer = buffer_with(sequence_length=4)
    for index in range(8):
        for stream in (1, 2, 3):
            buffer.add(tag(stream, index), stream=stream)

    for item in stored_items(buffer):
        assert len({stream for stream, _ in item}) == 1, item
        indexes = [index for _, index in item]
        assert indexes == list(range(indexes[0], indexes[0] + 4)), item


def test_a_new_stream_starts_empty_unless_forked():
    # An unforked stream must not inherit anyone's frames, or its first item would begin in
    # another branch's history.
    buffer = buffer_with()
    buffer.add(tag(0, 0))
    buffer.add(tag(0, 1))
    before = len(stored_items(buffer))
    buffer.add(tag(9, 0), stream=9)
    assert len(stored_items(buffer)) == before, 'a fresh stream stored an item from one add'


# ------------------------------------------------------------------ fork_stream

def test_fork_stream_copies_the_parent_window():
    """A forked child's first add completes an item, using the parent's frames as its history.

    Without the copy a child needs `sequence_length` adds of its own, so at n_step_update=3 a
    branch that dies two steps after forking stores nothing at all.
    """
    buffer = buffer_with(sequence_length=3)
    buffer.add(tag(0, 0))
    buffer.add(tag(0, 1))
    assert buffer.size == 0

    buffer.fork_stream(DEFAULT_STREAM, 1)
    buffer.add(tag(1, 0), stream=1)
    assert stored_items(buffer) == [[(0, 0), (0, 1), (1, 0)]]


def test_fork_stream_does_not_alias_the_parent_window():
    """The child needs its own deque, not a reference to the parent's.

    Sharing the object would make both windows advance together, so the parent's next item would
    contain the child's frame — the same splice the per-stream design exists to prevent, just
    reintroduced one level down.
    """
    buffer = buffer_with()
    buffer.add(tag(0, 0))
    buffer.fork_stream(DEFAULT_STREAM, 1)

    buffer.add(tag(1, 0), stream=1)
    buffer.add(tag(0, 1))

    items = stored_items(buffer)
    assert [(0, 0), (1, 0)] in items, items
    assert [(0, 0), (0, 1)] in items, items
    for item in items:
        assert item in ([(0, 0), (1, 0)], [(0, 0), (0, 1)]), item


def test_forking_from_an_unknown_stream_is_an_empty_start():
    # The collector should never do this, but silently inventing frames would be worse than
    # starting empty, and an exception here would kill a run over bookkeeping.
    buffer = buffer_with()
    buffer.fork_stream(77, 78)
    buffer.add(tag(78, 0), stream=78)
    assert buffer.size == 0


# ------------------------------------------------------------------ close_stream

def test_closing_a_stream_drops_its_partial_window():
    buffer = buffer_with()
    buffer.add(tag(1, 0), stream=1)
    buffer.close_stream(1)
    buffer.add(tag(1, 1), stream=1)
    assert buffer.size == 0, 'a reopened stream continued a dead branch'


def test_closing_a_stream_keeps_everything_it_already_stored():
    # The point of the feature: a retired branch's transitions stay in the buffer and keep being
    # trained on. Only the unfinished tail of its window goes.
    buffer = buffer_with()
    for index in range(4):
        buffer.add(tag(1, index), stream=1)
    stored = stored_items(buffer)
    assert len(stored) == 3
    buffer.close_stream(1)
    assert stored_items(buffer) == stored
    assert buffer.size == 3


def test_closing_the_default_stream_is_refused():
    # It is the main collect line; dropping its window mid-episode would lose a real transition
    # every time a branch happened to retire.
    buffer = buffer_with()
    try:
        buffer.close_stream(DEFAULT_STREAM)
    except ValueError:
        return
    raise AssertionError('closing the default stream was allowed')


def test_closing_an_unknown_stream_is_harmless():
    buffer = buffer_with()
    buffer.close_stream(12345)


def test_closing_a_branch_does_not_disturb_the_main_line():
    """Retiring a branch must leave every other stream's window exactly where it was.

    Branches retire constantly — several per main-line episode — so a close that clipped the main
    line's window would drop a real main-line transition every single time, and the loss would be
    invisible: the buffer would simply contain slightly fewer items than adds.

    The main line is mid-window here (one add of a two-frame window), which is the state that can
    actually be damaged.
    """
    buffer = buffer_with(sequence_length=3)
    buffer.add(tag(0, 0))
    buffer.add(tag(0, 1))
    buffer.add(tag(1, 0), stream=1)
    assert buffer.size == 0

    buffer.close_stream(1)

    buffer.add(tag(0, 2))
    assert stored_items(buffer) == [[(0, 0), (0, 1), (0, 2)]], \
        'closing a branch cost the main line its pending frames'


# ------------------------------------------------------------------ still a working buffer

def test_sampling_and_priorities_still_work_across_streams():
    buffer = buffer_with()
    for index in range(6):
        buffer.add(tag(1, index), stream=1)
        buffer.add(tag(2, index), stream=2)
    experience, indexes, weights = buffer.sample(4, train_step=0)
    assert experience.shape == (4, 2)
    assert len(indexes) == 4
    assert abs(float(np.mean(weights)) - 1.0) < 1e-6, 'IS weights are rescaled to mean 1'
    buffer.update_priorities(indexes, np.ones(4, dtype=np.float32))
