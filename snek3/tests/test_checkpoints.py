"""Checkpoint files: naming, atomicity, and the step being where it says it is.

The step lives in the filename *and* in the payload, and they can only come apart one way that
matters: a checkpoint copied into `hallOfFame/` under a new name. That is exactly what happened in
snek2 — a hall-of-fame measurement reported against the wrong step — so `load` refuses the
disagreement rather than preferring either side.
"""

import os

import pytest

from dqn import net as network
from tools import checkpoints


def a_net():
    return network.QNet(30, [16], 3)


def test_a_step_round_trips_through_the_filename(tmp_path):
    written = checkpoints.save(str(tmp_path), 1234, a_net())
    assert os.path.basename(written) == 'ckpt-1234.pt'
    assert checkpoints.step_of(written) == 1234
    assert checkpoints.load(written)['step'] == 1234


def test_steps_are_listed_numerically_not_lexically(tmp_path):
    # The bug this exists for: sorting `ckpt-*` by name puts 1000000 before 999000, so "the newest
    # checkpoint" becomes an arbitrary one somewhere in the middle of an arm.
    for step in (999000, 1000000, 9000):
        checkpoints.save(str(tmp_path), step, a_net())
    assert checkpoints.steps(str(tmp_path)) == [9000, 999000, 1000000]
    assert checkpoints.latest_step(str(tmp_path)) == 1000000


def test_a_directory_with_no_checkpoints_is_empty_not_an_error(tmp_path):
    # A young or dead arm legitimately has none, and every caller distinguishes that from a failure.
    assert checkpoints.steps(str(tmp_path)) == []
    assert checkpoints.latest_step(str(tmp_path)) is None


def test_unrelated_files_are_ignored(tmp_path):
    for name in ('arch.json', 'ckpt-.pt', 'ckpt-12.pt.partial', 'ckpt-abc.pt', 'notes.md'):
        (tmp_path / name).write_text('x')
    assert checkpoints.steps(str(tmp_path)) == []
    # `.partial` in particular: `save` writes through that name, so a crashed write must not be
    # picked up as a checkpoint by the next reader.
    assert checkpoints.step_of('ckpt-12.pt.partial') is None


def test_no_partial_file_is_left_behind(tmp_path):
    checkpoints.save(str(tmp_path), 7000, a_net())
    assert sorted(os.listdir(str(tmp_path))) == ['ckpt-7000.pt']


def test_a_renamed_checkpoint_is_refused(tmp_path):
    """The hall-of-fame case: the payload's step and the filename's must agree.

    Refused rather than resolved in either direction. Preferring the filename reports a real
    measurement against a step that did not produce it; preferring the payload silently ignores the
    name a human chose. Both are worse than stopping.
    """
    written = checkpoints.save(str(tmp_path), 2739000, a_net())
    renamed = str(tmp_path / 'ckpt-1.pt')
    os.rename(written, renamed)
    with pytest.raises(checkpoints.CheckpointError) as raised:
        checkpoints.load(renamed)
    assert '2739000' in str(raised.value)


def test_a_path_with_no_step_in_its_name_still_loads(tmp_path):
    # A checkpoint under a hand-chosen filename is allowed — there is simply nothing to cross-check
    # it against, so the payload's own step is used.
    written = checkpoints.save(str(tmp_path), 55, a_net())
    other = str(tmp_path / 'best.pt')
    os.rename(written, other)
    assert checkpoints.load(other)['step'] == 55


def test_weights_survive_the_round_trip(tmp_path):
    source, destination = a_net(), a_net()
    written = checkpoints.save(str(tmp_path), 10, source)
    checkpoints.load(written, destination)
    for key, value in source.state_dict().items():
        assert destination.state_dict()[key].equal(value), key


def test_a_shape_mismatch_is_refused_by_the_loader(tmp_path):
    # `arch.json` is what stops this happening, but `strict=True` is the belt to its braces, and a
    # test that never checked would not notice the flag being dropped.
    written = checkpoints.save(str(tmp_path), 10, a_net())
    with pytest.raises(RuntimeError):
        checkpoints.load(written, network.QNet(30, [17], 3))


def test_extra_fields_travel_but_cannot_shadow_the_payload(tmp_path):
    written = checkpoints.save(str(tmp_path), 3, a_net(), extra={'imported_from': 'somewhere'})
    assert checkpoints.load(written)['imported_from'] == 'somewhere'
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.save(str(tmp_path), 4, a_net(), extra={'step': 999})
