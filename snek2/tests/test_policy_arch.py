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


def _dir_with_arch(fc=(50, 100, 50), num_actions=3, obs_len=30, obs_era='b09c616', **algo_kwargs):
    """A fresh temp policy dir carrying one arch.json. Returns the path."""
    path = tempfile.mkdtemp()
    policy_arch.write_arch(path, policy_arch.build_arch(fc, num_actions, obs_len, obs_era,
                                                        **algo_kwargs))
    return path


def _c51_dir(fc=(200, 100, 100), num_atoms=51, v_min=-5.0, v_max=120.0):
    return _dir_with_arch(fc=fc, algo='c51', num_atoms=num_atoms, v_min=v_min, v_max=v_max)


# ----------------------------------------------------------- round trip / IO

def test_write_then_read_round_trips_the_fields():
    path = _dir_with_arch(fc=(100, 200, 100), obs_len=30, obs_era='b09c616')
    arch = policy_arch.read_arch(path)
    # fc is stored as a list — JSON has no tuples and every reader iterates it, so the round trip
    # must come back list-shaped, not tuple-shaped.
    # An exact dict, not a field-by-field check: a new field that readers must handle should fail
    # here and be added deliberately. `algo` is always present, the categorical fields never are on a
    # scalar arm — a ddqn sidecar carrying three nulls would make "absent means scalar" ambiguous.
    assert arch == {'fc_layer_params': [100, 200, 100], 'num_actions': 3,
                    'obs_len': 30, 'obs_era': 'b09c616', 'algo': 'ddqn'}


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


# ------------------------------------------------------------------- algorithm

def test_a_pre_c51_sidecar_reads_as_ddqn():
    # ~100 policy directories plus every hallOfFame entry have no `algo` field. All of them must keep
    # loading, which is what makes "missing means ddqn" load-bearing rather than cosmetic.
    path = tempfile.mkdtemp()
    policy_arch.write_arch(path, {'fc_layer_params': [50, 100, 50], 'num_actions': 3,
                                  'obs_len': 30, 'obs_era': 'b09c616'})
    arch = policy_arch.read_arch(path)
    assert policy_arch.algo_of(arch) == 'ddqn'
    assert not policy_arch.is_categorical(arch)
    assert policy_arch.support_from_arch(arch) is None
    # And it still restores, which is the property that actually matters.
    policy_arch.assert_restorable(path, num_actions=3, obs_len=30, obs_era='b09c616')


def test_build_arch_records_the_support_for_c51():
    arch = policy_arch.read_arch(_c51_dir())
    assert arch['algo'] == 'c51'
    assert (arch['num_atoms'], arch['v_min'], arch['v_max']) == (51, -5.0, 120.0)
    assert policy_arch.is_categorical(arch)


def test_build_arch_refuses_c51_without_a_support():
    # A c51 sidecar with no support is unusable, and the failure it would produce — a policy whose
    # atoms mean whatever the reader assumed — is silent. So it cannot be written in the first place.
    try:
        policy_arch.build_arch((50,), 3, 30, 'b09c616', algo='c51', num_atoms=51)
    except ValueError:
        return
    raise AssertionError('build_arch must refuse algo=c51 with v_min/v_max missing')


def test_support_from_arch_reconstructs_the_grid():
    support = policy_arch.support_from_arch(policy_arch.read_arch(_c51_dir()))
    assert len(support) == 51
    assert abs(support[0] + 5.0) < 1e-9 and abs(support[-1] - 120.0) < 1e-9
    # 2.5 spacing is the reason these three numbers were chosen together.
    assert abs((support[1] - support[0]) - 2.5) < 1e-9


def test_assert_restorable_raises_when_a_c51_sidecar_has_no_support():
    # Hand-edited or half-written. Caught on the read side too, not only on the write side, because
    # the write side is not what a hallOfFame copy or an rsync goes through.
    path = tempfile.mkdtemp()
    policy_arch.write_arch(path, {'fc_layer_params': [50], 'num_actions': 3, 'obs_len': 30,
                                  'obs_era': 'b09c616', 'algo': 'c51'})
    try:
        policy_arch.assert_restorable(path, num_actions=3, obs_len=30, obs_era='b09c616')
    except policy_arch.ArchMismatch as exc:
        assert 'num_atoms' in str(exc)
        return
    raise AssertionError('a c51 sidecar with no support must not restore')


def test_assert_restorable_raises_when_a_ddqn_sidecar_carries_a_support():
    path = tempfile.mkdtemp()
    policy_arch.write_arch(path, {'fc_layer_params': [50], 'num_actions': 3, 'obs_len': 30,
                                  'obs_era': 'b09c616', 'algo': 'ddqn', 'num_atoms': 51,
                                  'v_min': -5.0, 'v_max': 120.0})
    try:
        policy_arch.assert_restorable(path, num_actions=3, obs_len=30, obs_era='b09c616')
    except policy_arch.ArchMismatch as exc:
        assert 'num_atoms' in str(exc)
        return
    raise AssertionError('a ddqn sidecar carrying atoms means the algorithm was changed by hand')


def test_assert_config_matches_raises_on_a_changed_algo():
    path = _dir_with_arch()
    try:
        policy_arch.assert_config_matches(path, (50, 100, 50), algo='c51', num_atoms=51,
                                          v_min=-5.0, v_max=120.0)
    except policy_arch.ArchMismatch as exc:
        assert 'algo' in str(exc)
        return
    raise AssertionError('resuming a ddqn policy as c51 must raise')


def test_assert_config_matches_raises_on_a_changed_v_max():
    # **The dangerous one.** Every weight restores cleanly under a changed support and the policy is
    # silently different, because the greedy action is argmax_a sum_i z_i p_i(s, a). Nothing about the
    # shapes would object, so this check is the only thing standing there.
    path = _c51_dir(v_max=120.0)
    try:
        policy_arch.assert_config_matches(path, (200, 100, 100), algo='c51', num_atoms=51,
                                          v_min=-5.0, v_max=110.0)
    except policy_arch.ArchMismatch as exc:
        assert 'v_max' in str(exc)
        return
    raise AssertionError('a resume under a different support must raise')


def test_assert_config_matches_raises_on_a_changed_atom_count():
    path = _c51_dir(num_atoms=51)
    try:
        policy_arch.assert_config_matches(path, (200, 100, 100), algo='c51', num_atoms=101,
                                          v_min=-5.0, v_max=120.0)
    except policy_arch.ArchMismatch as exc:
        assert 'num_atoms' in str(exc)
        return
    raise AssertionError('a resume under a different atom count must raise')


def test_assert_config_matches_passes_on_a_matching_c51_resume():
    path = _c51_dir()
    arch = policy_arch.assert_config_matches(path, (200, 100, 100), algo='c51', num_atoms=51,
                                             v_min=-5.0, v_max=120.0)
    assert arch['v_max'] == 120.0


def test_assert_config_matches_ignores_the_support_for_a_ddqn_resume():
    # Every existing arm resumes through this call, and snek2 passes the c51 knobs unconditionally.
    # A ddqn resume must not care what they say.
    path = _dir_with_arch()
    policy_arch.assert_config_matches(path, (50, 100, 50), algo='ddqn', num_atoms=999,
                                      v_min=-1.0, v_max=2.0)


# ------------------------------------------------------------ refuse_categorical

def test_refuse_categorical_raises_for_a_c51_policy():
    try:
        policy_arch.refuse_categorical(_c51_dir(), 'plasticity.py')
    except SystemExit as exc:
        assert 'plasticity.py' in str(exc)
        return
    raise AssertionError('a scalar-head diagnostic must refuse a c51 policy')


def test_refuse_categorical_returns_the_arch_for_a_ddqn_policy():
    arch = policy_arch.refuse_categorical(_dir_with_arch(), 'plasticity.py')
    assert arch['fc_layer_params'] == [50, 100, 50]
