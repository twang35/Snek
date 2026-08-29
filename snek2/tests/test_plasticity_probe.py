"""Fixtures for `hyperparamTuning/perDiagnostics/plasticity_probe.py`.

The probe's first version answered `fit = -283` for every trained checkpoint, because the teacher's
head is `RandomUniform(±0.03)` while a trained network emits Q values of order 1 — it was measuring the
distance between two output scales and calling it plasticity. That failure is the reason for the two
invariants pinned hardest here:

- the target is standardised, so its variance is exactly 1
- the student's head is zeroed, so every network starts at `mse_start == variance` and `fit` is purely
  what the network learned rather than where it happened to start

Both are silent when broken: the run still prints a full table of plausible numbers.

The other class of bug is the probe training the checkpoint it is measuring, which would make every
later row in the same run depend on the earlier ones. `copy_of` exists for that, so it is checked that
the source network is untouched and that a frozen probe really does freeze.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'hyperparamTuning', 'perDiagnostics'))

import numpy as np
import tensorflow as tf

import plasticity_probe as PP
from under_the_hood import build_q_net

FC = (16, 8)
OBS_LEN = 30


def net_of(fc=FC, obs_len=OBS_LEN, seed=5):
    net = build_q_net(3, tuple(fc))
    net(tf.zeros([2, obs_len], tf.float32), step_type=tf.fill([2], 1))
    PP.seeded_reinit(net, seed)
    return net


def boards(rows=200, obs_len=OBS_LEN, seed=1):
    return np.random.default_rng(seed).random((rows, obs_len)).astype(np.float32)


def test_the_target_is_standardised_so_its_variance_is_one():
    obs = boards()
    targets = PP.teacher_targets(FC, obs)
    assert targets.shape == (len(obs), 3)
    # Per output, because `fit` divides by the pooled variance and a single loud output would
    # otherwise dominate the denominator.
    assert np.allclose(targets.mean(axis=0), 0.0, atol=1e-5), targets.mean(axis=0)
    assert np.allclose(targets.std(axis=0), 1.0, atol=1e-5), targets.std(axis=0)
    assert abs(float(np.var(targets)) - 1.0) < 1e-4


def test_different_teacher_seeds_give_different_targets_and_the_same_seed_repeats():
    obs = boards()
    first = PP.teacher_targets(FC, obs, seed=11)
    again = PP.teacher_targets(FC, obs, seed=11)
    other = PP.teacher_targets(FC, obs, seed=12)
    assert np.array_equal(first, again), 'a fixed teacher seed must reproduce its target exactly'
    assert not np.allclose(first, other)


def test_the_copy_zeroes_the_head_and_keeps_every_hidden_weight():
    source = net_of()
    hidden_before = [layer.get_weights()[0].copy() for layer in PP.layer_stack(source)[0]]
    copy = PP.copy_of(source, FC, OBS_LEN)
    hidden_copy, head_copy = PP.layer_stack(copy)
    for before, layer in zip(hidden_before, hidden_copy):
        assert np.array_equal(before, layer.get_weights()[0])
    assert np.count_nonzero(head_copy.get_weights()[0]) == 0
    assert np.count_nonzero(head_copy.get_weights()[1]) == 0


def test_a_zeroed_head_puts_every_network_at_the_same_starting_error():
    """The invariant that makes `fit` comparable across checkpoints, and it is exact.

    With the head at zero the output is identically zero, so the initial MSE against a unit-variance
    target is 1.0 for a fresh net and a heavily trained one alike. A redrawn head instead of a zeroed
    one leaves this depending on the feature scale, which grows ~2.7x over training.
    """
    obs = boards()
    targets = PP.teacher_targets(FC, obs)
    starts = []
    for seed in (1, 2):
        copy = PP.copy_of(net_of(seed=seed), FC, OBS_LEN)
        # Scale one network's hidden weights hard: the start must not move at all.
        if seed == 2:
            for layer in PP.layer_stack(copy)[0]:
                weights = layer.get_weights()
                layer.set_weights([weights[0] * 5.0] + weights[1:])
        result = PP.fit_target(copy, obs, targets, steps=1)
        starts.append(result['mse_start'])
    assert all(abs(start - 1.0) < 1e-4 for start in starts), starts


def test_the_probe_never_moves_the_network_it_was_given():
    """Every row after the first would depend on the rows before it if this failed."""
    source = net_of()
    before = [[v.copy() for v in layer.get_weights()] for layer in source.layers]
    obs = boards()
    PP.fit_target(PP.copy_of(source, FC, OBS_LEN), obs, PP.teacher_targets(FC, obs), steps=5)
    for layer, saved in zip(source.layers, before):
        for old, new in zip(saved, layer.get_weights()):
            assert np.array_equal(old, new)


def test_head_only_freezes_the_hidden_stack_and_still_learns():
    obs = boards()
    targets = PP.teacher_targets(FC, obs)
    copy = PP.copy_of(net_of(), FC, OBS_LEN)
    hidden_before = [layer.get_weights()[0].copy() for layer in PP.layer_stack(copy)[0]]
    result = PP.fit_target(copy, obs, targets, steps=30, head_only=True)
    for before, layer in zip(hidden_before, PP.layer_stack(copy)[0]):
        assert np.array_equal(before, layer.get_weights()[0]), 'hidden layers must not move'
    assert np.count_nonzero(PP.layer_stack(copy)[1].get_weights()[0]) > 0, 'the head must move'
    assert result['mse_end'] < result['mse_start']


def test_the_full_probe_does_move_the_hidden_stack():
    # The complement of the test above: if `head_only=False` also froze the hidden layers, `fit` and
    # `fit_frozen` would be the same measurement under two names.
    obs = boards()
    copy = PP.copy_of(net_of(), FC, OBS_LEN)
    before = [layer.get_weights()[0].copy() for layer in PP.layer_stack(copy)[0]]
    PP.fit_target(copy, obs, PP.teacher_targets(FC, obs), steps=30)
    moved = [not np.array_equal(b, layer.get_weights()[0])
             for b, layer in zip(before, PP.layer_stack(copy)[0])]
    assert any(moved), moved


def test_fit_is_one_minus_the_normalised_residual_and_reduces_the_loss():
    obs = boards()
    targets = PP.teacher_targets(FC, obs)
    result = PP.fit_target(PP.copy_of(net_of(), FC, OBS_LEN), obs, targets, steps=60)
    assert abs(result['fit'] - (1.0 - result['mse_end'] / result['variance'])) < 1e-12
    assert result['mse_end'] < result['mse_start'], 'the probe has to actually train'
    assert 0.0 < result['fit'] < 1.0, result['fit']


def test_two_probes_of_the_same_weights_agree():
    """Otherwise a difference between two checkpoints could be the probe's own sampling noise."""
    obs = boards()
    targets = PP.teacher_targets(FC, obs)
    first = PP.fit_target(PP.copy_of(net_of(), FC, OBS_LEN), obs, targets, steps=40, seed=3)
    second = PP.fit_target(PP.copy_of(net_of(), FC, OBS_LEN), obs, targets, steps=40, seed=3)
    assert abs(first['fit'] - second['fit']) < 1e-9, (first['fit'], second['fit'])
    # A different minibatch order must give a different answer, or `seed` is not reaching the sampler
    # and averaging over repeats would be averaging the same number.
    third = PP.fit_target(PP.copy_of(net_of(), FC, OBS_LEN), obs, targets, steps=40, seed=4)
    assert third['fit'] != first['fit']


def test_a_trained_network_beats_a_fresh_one_on_a_new_target():
    """The direction the whole measurement reports, on a network trained here rather than restored.

    A net pre-fitted on one target should fit a *second* target faster than a random init, since it
    starts from useful features. If the probe reported the opposite for this fixture, the instrument
    would be inverted and every arm would read as having lost plasticity.
    """
    obs = boards(rows=400)
    first_target = PP.teacher_targets(FC, obs, seed=101)
    second_target = PP.teacher_targets(FC, obs, seed=202)
    warmed = PP.copy_of(net_of(seed=9), FC, OBS_LEN)
    PP.fit_target(warmed, obs, first_target, steps=600)
    warm_fit = PP.fit_target(PP.copy_of(warmed, FC, OBS_LEN), obs, second_target, steps=200)
    cold_fit = PP.fit_target(PP.copy_of(net_of(seed=9), FC, OBS_LEN), obs, second_target, steps=200)
    assert warm_fit['fit'] > cold_fit['fit'], (warm_fit['fit'], cold_fit['fit'])
