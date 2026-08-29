"""`arch.json` — the sidecar that catches the failure `load_state_dict` cannot.

Torch's `strict=True` refuses a shape mismatch, so most of what snek2's sidecar guarded is now
covered by the framework. **`obs_era` is the part that is not.** On 2026-08-02 two observation
indices were repurposed at constant length: every champion restored cleanly, matched every shape,
and played like a beginner — 90.3% perfect became scores of 0, 0, 1. No amount of shape checking
sees that, which is why these tests concentrate on the era field and on the guards refusing rather
than on the happy path.
"""

import json
import os

import pytest

from env import constants
from tools import arch as arch_tools


def an_arch(**overrides):
    arch = arch_tools.build_arch([320], constants.NUM_ACTIONS, constants.OBS_LEN,
                                 constants.OBS_ERA)
    arch.update(overrides)
    return arch


def test_a_round_trip_through_json_preserves_the_signature(tmp_path):
    # Through the file, not through a dict copy: `fc_layer_params` is a tuple in every caller's head
    # and a list on disk, and a signature that hashed differently depending on which side it came
    # from would break lane grouping in the eval wave silently.
    directory = str(tmp_path / 'policy')
    written = an_arch()
    arch_tools.write_arch(directory, written)
    assert arch_tools.signature(arch_tools.read_arch(directory)) == arch_tools.signature(written)


def test_every_field_is_written(tmp_path):
    directory = str(tmp_path / 'policy')
    arch_tools.write_arch(directory, an_arch())
    with open(arch_tools.arch_path(directory)) as handle:
        assert sorted(json.load(handle)) == sorted(arch_tools.FIELDS)


def test_an_incomplete_arch_is_refused_rather_than_written(tmp_path):
    directory = str(tmp_path / 'policy')
    partial = an_arch()
    del partial['obs_era']
    with pytest.raises(arch_tools.ArchMismatch):
        arch_tools.write_arch(directory, partial)
    # And nothing was left behind: a half-written sidecar is worse than none, because the next
    # reader would find a file and trust it.
    assert not os.path.exists(arch_tools.arch_path(directory))


def test_rewriting_the_same_arch_is_fine_and_a_different_one_is_not(tmp_path):
    # Every training resume rewrites the sidecar it already wrote, so that has to be a no-op. A
    # *different* one means the directory now holds checkpoints from two architectures, and the
    # second half would load into the wrong network with no shape error anywhere.
    directory = str(tmp_path / 'policy')
    arch_tools.write_arch(directory, an_arch())
    arch_tools.write_arch(directory, an_arch())
    with pytest.raises(arch_tools.ArchMismatch):
        arch_tools.write_arch(directory, an_arch(fc_layer_params=[100, 200]))


def test_a_missing_sidecar_raises_rather_than_defaulting(tmp_path):
    # There is deliberately no "restore without one" path. snek2 had to tolerate sidecar-less
    # directories because ~100 of them predated the file; snek3 writes every one of its own.
    with pytest.raises(arch_tools.ArchMismatch):
        arch_tools.read_arch(str(tmp_path))


def test_a_matching_arch_is_restorable(tmp_path):
    directory = str(tmp_path / 'policy')
    arch_tools.write_arch(directory, an_arch())
    returned = arch_tools.assert_restorable(directory, constants.OBS_LEN, constants.OBS_ERA,
                                           constants.NUM_ACTIONS)
    assert returned['fc_layer_params'] == [320]


@pytest.mark.parametrize('field,value', [('obs_len', 26),
                                         ('obs_era', 'something-else'),
                                         ('num_actions', 4)])
def test_each_environment_field_refuses_on_its_own(tmp_path, field, value):
    """One test per field, because a guard that only fires when *several* disagree is no guard.

    `obs_era` is the one that matters most and the one an `assert` on shapes cannot replace: the
    length and the action count are unchanged, so torch would load these weights without a word.
    """
    directory = str(tmp_path / field)
    arch_tools.write_arch(directory, an_arch(**{field: value}))
    with pytest.raises(arch_tools.ArchMismatch) as raised:
        arch_tools.assert_restorable(directory, constants.OBS_LEN, constants.OBS_ERA,
                                     constants.NUM_ACTIONS)
    assert field in str(raised.value)


def test_the_era_message_says_why_a_clean_load_is_the_problem(tmp_path):
    # The message is load-bearing. Whoever hits this will be looking at a checkpoint that loads
    # perfectly and plays terribly, and the only way out is knowing that is the expected symptom.
    directory = str(tmp_path / 'policy')
    arch_tools.write_arch(directory, an_arch(obs_era='older'))
    with pytest.raises(arch_tools.ArchMismatch) as raised:
        arch_tools.assert_restorable(directory, constants.OBS_LEN, constants.OBS_ERA,
                                     constants.NUM_ACTIONS)
    message = str(raised.value)
    assert 'same length' in message and 'play' in message


def test_two_different_networks_cannot_share_one_built_net():
    # The question `assert_restorable` never asks, because it runs in a process that builds its
    # network for one policy. An eval wave points one built network at many policies.
    built = an_arch()
    arch_tools.assert_same_network(built, an_arch())
    with pytest.raises(arch_tools.ArchMismatch) as raised:
        arch_tools.assert_same_network(built, an_arch(fc_layer_params=[320, 320]))
    assert 'fc_layer_params' in str(raised.value)
