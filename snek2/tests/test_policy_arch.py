"""Tests for policy_arch.py — the arch.json sidecar that stops a silent mis-restore.

The failure these guard against is the one CLAUDE.md's hall-of-fame section documents: a checkpoint
restores whenever the variable *lengths* line up, and `expect_partial()` says nothing when they do
not, so a network built with the wrong FC shape or against a different observation vector loads
without a word and plays like a beginner. arch.json makes every one of those a loud failure, so the
important tests are the ones that assert a mismatch *raises*.
"""
import os
import tempfile

import policy_arch


def _dir_with_arch(fc=(50, 100, 50), num_actions=3, obs_len=30, obs_era='b09c616'):
    """A fresh temp policy dir carrying one arch.json. Returns the path."""
    path = tempfile.mkdtemp()
    policy_arch.write_arch(path, policy_arch.build_arch(fc, num_actions, obs_len, obs_era))
    return path


# ----------------------------------------------------------- round trip / IO

def test_write_then_read_round_trips_the_fields():
    path = _dir_with_arch(fc=(100, 200, 100), obs_len=30, obs_era='b09c616')
    arch = policy_arch.read_arch(path)
    # fc is stored as a list — JSON has no tuples and every reader iterates it, so the round trip
    # must come back list-shaped, not tuple-shaped.
    assert arch == {'fc_layer_params': [100, 200, 100], 'num_actions': 3,
                    'obs_len': 30, 'obs_era': 'b09c616'}


def test_read_arch_is_none_when_absent():
    empty = tempfile.mkdtemp()
    assert policy_arch.read_arch(empty) is None


def test_write_creates_the_directory():
    # A fresh run writes arch.json before its first checkpoint, so write must not assume the dir.
    base = tempfile.mkdtemp()
    target = os.path.join(base, 'not', 'yet', 'there')
    policy_arch.write_arch(target, policy_arch.build_arch((50,), 3, 30, 'b09c616'))
    assert policy_arch.read_arch(target)['fc_layer_params'] == [50]


# --------------------------------------------------------------- require_arch

def test_require_arch_raises_when_the_sidecar_is_missing():
    empty = tempfile.mkdtemp()
    try:
        policy_arch.require_arch(empty)
    except policy_arch.ArchMismatch:
        return
    raise AssertionError('require_arch must refuse a dir with no arch.json, not return None')


# ------------------------------------------------------------ assert_restorable

def test_assert_restorable_returns_the_arch_when_everything_matches():
    path = _dir_with_arch(fc=(50, 100, 50), num_actions=3, obs_len=30, obs_era='b09c616')
    arch = policy_arch.assert_restorable(path, num_actions=3, obs_len=30, obs_era='b09c616')
    assert arch['fc_layer_params'] == [50, 100, 50]


def test_assert_restorable_raises_on_obs_len_mismatch():
    # A 30-value checkpoint asked to load in a 26-value environment. This is the case a batch-rule
    # backfill would have gotten wrong and a length check catches for free.
    path = _dir_with_arch(obs_len=30)
    try:
        policy_arch.assert_restorable(path, num_actions=3, obs_len=26, obs_era='b09c616')
    except policy_arch.ArchMismatch as exc:
        assert 'obs_len' in str(exc)
        return
    raise AssertionError('a differing observation length must raise, not restore silently')


def test_assert_restorable_raises_on_obs_era_mismatch():
    # Same length, different meaning — the game_over -> board_fill trap at a constant 20 values.
    # Only the era marker separates these, so this is the test that would fail if obs_era stopped
    # being checked.
    path = _dir_with_arch(obs_len=20, obs_era='e4514a8')
    try:
        policy_arch.assert_restorable(path, num_actions=3, obs_len=20, obs_era='b09c616')
    except policy_arch.ArchMismatch as exc:
        assert 'obs_era' in str(exc)
        return
    raise AssertionError('a same-length meaning change must raise')


def test_assert_restorable_raises_on_num_actions_mismatch():
    path = _dir_with_arch(num_actions=3)
    try:
        policy_arch.assert_restorable(path, num_actions=4, obs_len=30, obs_era='b09c616')
    except policy_arch.ArchMismatch:
        return
    raise AssertionError('a differing action count must raise')


def test_assert_restorable_raises_when_the_sidecar_is_missing():
    empty = tempfile.mkdtemp()
    try:
        policy_arch.assert_restorable(empty, num_actions=3, obs_len=30, obs_era='b09c616')
    except policy_arch.ArchMismatch:
        return
    raise AssertionError('assert_restorable must require the sidecar to be present')


# ---------------------------------------------------------- assert_config_matches

def test_assert_config_matches_passes_when_fc_agrees():
    path = _dir_with_arch(fc=(50, 100, 50))
    # tuple vs list must compare equal — the env supplies a tuple, arch stores a list.
    policy_arch.assert_config_matches(path, (50, 100, 50))


def test_assert_config_matches_raises_on_a_changed_fc_shape():
    # The b19 near-revert: resuming under a changed SNEK_FC_LAYERS would build the wrong net and
    # then load weights into it. This must stop before the restore.
    path = _dir_with_arch(fc=(50, 100, 50))
    try:
        policy_arch.assert_config_matches(path, (100, 200, 100))
    except policy_arch.ArchMismatch as exc:
        assert '50' in str(exc) and '100' in str(exc)
        return
    raise AssertionError('a resume under a different fc shape must raise')
