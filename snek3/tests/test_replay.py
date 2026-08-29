"""The prioritised replay buffer and the sum tree under it.

Two things here are easy to get subtly wrong and impossible to notice from a training curve: whether
sampling is really proportional to priority, and whether an unwritten slot can ever be drawn. An
unwritten slot is an all-zero observation, which trains the net on a board that cannot exist.
"""

import numpy as np
import pytest

from dqn.replay import PrioritizedReplay, SumTree, normalize_is_weights


def filled(capacity=8, obs_len=3, count=None, seed=0, **kwargs):
    """A buffer with `count` transitions whose observation is its own index."""
    buffer = PrioritizedReplay(capacity, obs_len, seed=seed, **kwargs)
    for index in range(capacity if count is None else count):
        buffer.add(np.full(obs_len, index, np.float32), index % 3, float(index),
                   np.full(obs_len, index + 1, np.float32), 0.99)
    return buffer


# --- the sum tree ---------------------------------------------------------------------------

def test_the_tree_requires_a_power_of_two_so_every_leaf_is_at_one_depth():
    # The batch descends in lockstep. Mixed depths would silently stop early on the shallow ones.
    SumTree(8)
    with pytest.raises(ValueError):
        SumTree(6)


def test_the_total_is_the_sum_of_the_leaves():
    tree = SumTree(8)
    tree.set([0, 3, 7], [1.5, 2.5, 4.0])
    assert tree.total == pytest.approx(8.0)


def test_setting_a_leaf_twice_replaces_rather_than_accumulates():
    tree = SumTree(8)
    tree.set([2], [5.0])
    tree.set([2], [1.0])
    assert tree.total == pytest.approx(1.0)


def test_find_returns_the_leaf_whose_cumulative_range_contains_the_target():
    tree = SumTree(4)
    tree.set([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])   # ranges [0,1) [1,3) [3,6) [6,10)
    found = tree.find([0.5, 2.0, 4.0, 9.5])
    assert list(found) == [0, 1, 2, 3]


def test_find_reaches_only_the_leaves_that_carry_priority():
    tree = SumTree(4)
    tree.set([0, 1, 2, 3], [0.0, 5.0, 0.0, 0.0])
    assert list(tree.find([0.1, 2.5, 4.999])) == [1, 1, 1]


def test_a_target_of_exactly_zero_lands_on_leaf_zero_whatever_its_priority():
    """The one degenerate case, pinned because the buffer's safety depends on knowing it.

    `find` goes left when `target <= left_sum`, so a target of exactly 0.0 walks to leaf 0 even if
    leaf 0 is empty. That is inherent to a sum tree and not worth a branch in the descent, because
    `PrioritizedReplay` can never be hurt by it: leaf 0 is written as soon as the buffer holds one
    transition, and `sample` returns None before that. Every priority carries `PRIORITY_EPSILON`, so
    a *written* slot's leaf is never zero — which means the only zero leaves reachable are the
    padding above `size`, and those are exactly what the stale guard in `sample` replaces.
    """
    tree = SumTree(4)
    tree.set([0, 1, 2, 3], [0.0, 5.0, 0.0, 0.0])
    assert list(tree.find([0.0])) == [0]


def test_the_tree_repairs_every_level_above_a_changed_leaf():
    # A partial repair leaves the root right and an interior node wrong, which biases sampling
    # toward one half of the buffer without changing the total.
    tree = SumTree(8)
    tree.set(list(range(8)), [1.0] * 8)
    tree.set([5], [9.0])
    assert tree.total == pytest.approx(16.0)
    # Every internal node must equal the sum of its children.
    for node in range(1, tree.size):
        assert tree.nodes[node] == pytest.approx(tree.nodes[2 * node] + tree.nodes[2 * node + 1])


# --- storage --------------------------------------------------------------------------------

def test_a_capacity_that_is_not_a_power_of_two_still_works():
    buffer = filled(capacity=100, count=100)
    assert buffer.size == 100 and buffer.tree.size == 128


def test_size_grows_to_capacity_and_stops():
    buffer = filled(capacity=8, count=20)
    assert buffer.size == 8


def test_the_oldest_transition_is_the_one_evicted():
    buffer = filled(capacity=4, obs_len=1, count=6)
    # Slots hold 4 and 5 where 0 and 1 were; 2 and 3 are untouched.
    assert sorted(buffer.obs[:, 0].tolist()) == [2.0, 3.0, 4.0, 5.0]


def test_a_new_transition_enters_at_the_highest_priority_seen():
    # So everything is trained on at least once before its priority means anything.
    buffer = filled(capacity=8, count=4)
    buffer.update_priorities([0], [10.0])
    slot = buffer.add(np.zeros(3, np.float32), 0, 0.0, np.zeros(3, np.float32), 0.0)
    leaf = buffer.tree.nodes[slot + buffer.tree.size]
    assert leaf == pytest.approx(buffer.max_priority ** buffer.alpha)
    assert buffer.max_priority >= 10.0


# --- sampling -------------------------------------------------------------------------------

def test_sampling_an_empty_buffer_returns_none_rather_than_raising():
    assert PrioritizedReplay(8, 3, seed=0).sample(4, 0) is None


def test_a_sample_never_returns_an_unwritten_slot():
    buffer = filled(capacity=1024, count=5)
    for _ in range(50):
        _, indexes, _ = buffer.sample(64, 0)
        assert indexes.max() < buffer.size, 'an unwritten slot is an impossible board'


def test_sampling_is_proportional_to_priority():
    buffer = filled(capacity=4, obs_len=1, count=4)
    buffer.update_priorities([0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])
    buffer.update_priorities([3], [10.0])
    # alpha=0.6, so slot 3's weight is 10**0.6 = 3.98 against 1.0 for the others.
    expected = 10.0 ** buffer.alpha / (3.0 + 10.0 ** buffer.alpha)
    counts = np.zeros(4)
    for _ in range(200):
        _, indexes, _ = buffer.sample(64, 0)
        counts += np.bincount(indexes, minlength=4)
    observed = counts[3] / counts.sum()
    assert observed == pytest.approx(expected, abs=0.02), (observed, expected)


def test_the_same_seed_samples_the_same_batch():
    first = filled(capacity=64, count=64, seed=7).sample(16, 0)[1]
    second = filled(capacity=64, count=64, seed=7).sample(16, 0)[1]
    assert list(first) == list(second)


def test_different_seeds_sample_differently():
    first = filled(capacity=64, count=64, seed=1).sample(16, 0)[1]
    second = filled(capacity=64, count=64, seed=2).sample(16, 0)[1]
    assert list(first) != list(second)


def test_the_batch_carries_the_fields_the_bellman_target_needs():
    buffer = filled(capacity=16, obs_len=3, count=16)
    batch, _, _ = buffer.sample(4, 0)
    assert set(batch) == {'obs', 'action', 'reward', 'next_obs', 'discount'}
    assert batch['obs'].shape == (4, 3) and batch['next_obs'].shape == (4, 3)
    assert batch['action'].dtype == np.int64


def test_the_batch_rows_belong_together():
    # obs is its own index and next_obs is index+1, so a row that mismatches is a fabricated
    # transition — the exact failure that per-stream windows exist to prevent.
    buffer = filled(capacity=64, obs_len=1, count=64)
    batch, _, _ = buffer.sample(32, 0)
    assert np.allclose(batch['next_obs'][:, 0], batch['obs'][:, 0] + 1)
    assert np.allclose(batch['reward'], batch['obs'][:, 0])


# --- importance sampling --------------------------------------------------------------------

def test_is_weights_are_normalised_to_mean_one():
    # Mean, not max. Dividing by the buffer's largest weight is a blanket 11x-370x cut to the
    # learning rate that worsens as beta anneals.
    buffer = filled(capacity=64, count=64)
    buffer.update_priorities(np.arange(64), np.linspace(0.1, 10.0, 64))
    _, _, weights = buffer.sample(32, 0)
    assert weights.mean() == pytest.approx(1.0)


def test_normalize_is_weights_survives_an_all_zero_input():
    assert list(normalize_is_weights(np.zeros(4))) == [1.0, 1.0, 1.0, 1.0]


def test_a_rarely_sampled_transition_gets_the_larger_weight():
    # This is the bias correction: over-sampled transitions count for less.
    buffer = filled(capacity=4, obs_len=1, count=4)
    buffer.update_priorities([0, 1, 2, 3], [1.0, 1.0, 1.0, 100.0])
    seen = {}
    for _ in range(100):
        _, indexes, weights = buffer.sample(32, 0)
        for index, weight in zip(indexes, weights):
            seen.setdefault(int(index), []).append(float(weight))
    assert np.mean(seen[0]) > np.mean(seen[3])


def test_beta_anneals_linearly_then_holds():
    buffer = PrioritizedReplay(8, 3, initial_beta=0.4, final_beta=1.0, beta_anneal_steps=1000)
    assert buffer.beta_for(0) == pytest.approx(0.4)
    assert buffer.beta_for(500) == pytest.approx(0.7)
    assert buffer.beta_for(1000) == pytest.approx(1.0)
    assert buffer.beta_for(50_000) == pytest.approx(1.0)


def test_beta_one_makes_the_correction_complete_and_weights_still_average_one():
    buffer = filled(capacity=64, count=64, beta_anneal_steps=1)
    _, _, weights = buffer.sample(32, 10)
    assert weights.mean() == pytest.approx(1.0)


# --- priorities -----------------------------------------------------------------------------

def test_a_zero_td_error_stays_reachable():
    # Without the epsilon a slot whose TD error came back exactly zero falls out of the
    # distribution permanently and is never corrected.
    buffer = filled(capacity=4, obs_len=1, count=4)
    buffer.update_priorities([0, 1, 2, 3], [0.0, 0.0, 0.0, 0.0])
    assert buffer.tree.total > 0.0
    _, indexes, _ = buffer.sample(64, 0)
    assert len(set(indexes.tolist())) > 1


def test_priorities_use_the_absolute_td_error():
    buffer = filled(capacity=4, obs_len=1, count=4)
    buffer.update_priorities([0], [-5.0])
    assert buffer.tree.nodes[0 + buffer.tree.size] == pytest.approx((5.0 + 1e-6) ** buffer.alpha)


def test_alpha_zero_makes_sampling_uniform():
    buffer = filled(capacity=4, obs_len=1, count=4, alpha=0.0)
    buffer.update_priorities([0, 1, 2, 3], [1.0, 1.0, 1.0, 1000.0])
    counts = np.zeros(4)
    for _ in range(200):
        _, indexes, _ = buffer.sample(64, 0)
        counts += np.bincount(indexes, minlength=4)
    assert counts.min() / counts.max() > 0.9, counts


# --- persistence ----------------------------------------------------------------------------

def test_a_save_and_load_round_trip_keeps_the_transitions_and_the_priorities(tmp_path):
    buffer = filled(capacity=64, obs_len=3, count=40)
    buffer.update_priorities(np.arange(40), np.linspace(0.5, 5.0, 40))
    buffer.save(str(tmp_path))

    restored = PrioritizedReplay(64, 3, seed=0)
    assert restored.load(str(tmp_path)) is True
    assert restored.size == 40 and restored.write == buffer.write
    assert np.allclose(restored.obs[:40], buffer.obs[:40])
    assert np.allclose(restored.next_obs[:40], buffer.next_obs[:40])
    assert np.allclose(restored.reward[:40], buffer.reward[:40])
    assert restored.tree.total == pytest.approx(buffer.tree.total)


def test_loading_from_a_directory_with_no_buffer_is_false_not_an_error(tmp_path):
    assert PrioritizedReplay(8, 3).load(str(tmp_path)) is False


def test_a_save_leaves_no_partial_file(tmp_path):
    filled(capacity=8, count=8).save(str(tmp_path))
    assert not [name for name in tmp_path.iterdir() if 'partial' in name.name]


def test_loading_a_buffer_larger_than_capacity_says_so(tmp_path):
    # numpy would raise on the assignment anyway, with a shape error naming neither number. The
    # explicit check exists for the message, so the message is what is asserted.
    filled(capacity=64, obs_len=3, count=64).save(str(tmp_path))
    with pytest.raises(ValueError, match='capacity is 16'):
        PrioritizedReplay(16, 3).load(str(tmp_path))


def test_find_can_only_reach_a_written_slot():
    """The property that makes a stale-slot guard in `sample` unnecessary.

    `find` goes right only when `target > left_sum`, and `target <= node_sum` holds inductively from
    `target < total`, so a right turn always has positive weight beneath it. A left turn into an
    empty subtree needs `target <= 0`, and leaf 0 is written as soon as the buffer holds anything.
    Every written slot carries `PRIORITY_EPSILON`, so the padding leaves above `size` are the only
    zero ones — and they are unreachable.

    Swept rather than argued: every partial fill, and targets including 0, the prefix sums exactly,
    and just under the total.
    """
    for count in range(1, 8):
        buffer = filled(capacity=8, obs_len=1, count=count)
        total = buffer.tree.total
        leaves = buffer.tree.nodes[buffer.tree.size:buffer.tree.size + 8]
        prefixes = np.cumsum(leaves)
        targets = np.concatenate([[0.0], prefixes, prefixes - 1e-12,
                                  np.linspace(0.0, total, 97)[:-1]])
        targets = np.clip(targets, 0.0, np.nextafter(total, 0.0))
        found = buffer.tree.find(targets)
        assert found.max() < count, (count, found.max())


def test_the_tree_repair_matches_an_independent_walk_to_the_root():
    """The property that lets `set` skip deduplicating its parents.

    A batch of leaves shares ancestors, so the repair scatters to the same node several times per
    level. Checked against the obvious implementation — one leaf at a time, one walk each — because
    the fast version's correctness rests on those repeated writes being idempotent, and nothing else
    in the suite would notice if a shared ancestor came out holding one child's sum.
    """
    rng = np.random.default_rng(7)
    for size in (8, 64, 1024):
        fast = SumTree(size)
        slow = SumTree(size)
        for _ in range(20):
            leaves = rng.integers(0, size, rng.integers(1, min(size, 40) + 1))
            values = rng.random(leaves.size)
            fast.set(leaves, values)
            for leaf, value in zip(leaves.tolist(), values.tolist()):
                node = leaf + slow.size
                slow.nodes[node] = value
                node //= 2
                while node >= 1:
                    slow.nodes[node] = slow.nodes[2 * node] + slow.nodes[2 * node + 1]
                    node //= 2
            assert np.allclose(fast.nodes, slow.nodes), 'size {0}'.format(size)
        assert fast.total > 0.0


def test_setting_no_leaves_at_all_leaves_the_tree_alone():
    # The repair loop reads `parents[0]`, so an empty batch has to return before it does.
    tree = SumTree(16)
    tree.set([3], [2.0])
    tree.set([], [])
    assert tree.total == pytest.approx(2.0)
