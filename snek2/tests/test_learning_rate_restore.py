"""`enforce_learning_rate` — the guard on Adam's learning rate surviving a resume.

Adam's `learning_rate` is a `tf.Variable`, so `common.Checkpointer` saves it with the moment
estimates and `initialize_or_restore()` overwrites whatever the constructor was given. Without
`training.enforce_learning_rate`, `SNEK_LEARNING_RATE` is a no-op on every resume and nothing
about the run looks wrong. `test_restore_really_does_overwrite_the_learning_rate` pins the
underlying TF behaviour, so if a future TF version stops restoring it this file says so rather
than the fix quietly becoming dead code.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from training import enforce_learning_rate


def _built_adam(lr):
    """An Adam whose hyper Variables exist — unbuilt, `learning_rate` is not yet checkpointable."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-7)
    optimizer.build([tf.Variable([1.0])])
    return optimizer


def _write_then_restore(saved_lr, resumed_lr):
    """Round-trips a checkpoint written at `saved_lr` into an optimizer built at `resumed_lr`."""
    with tempfile.TemporaryDirectory() as directory:
        path = tf.train.Checkpoint(optimizer=_built_adam(saved_lr)).write(
            os.path.join(directory, 'ckpt'))
        resumed = _built_adam(resumed_lr)
        tf.train.Checkpoint(optimizer=resumed).restore(path).expect_partial()
        return resumed


def test_restore_really_does_overwrite_the_learning_rate():
    """The bug this exists for. If this ever fails, TF changed and the fix can be revisited."""
    resumed = _write_then_restore(saved_lr=1e-5, resumed_lr=1e-6)
    assert np.float32(resumed.learning_rate.numpy()) == np.float32(1e-5), (
        'expected the checkpoint to clobber the configured 1e-6 with 1e-5, got '
        '{0:g}'.format(float(resumed.learning_rate.numpy())))


def test_lowered_learning_rate_is_put_back_after_a_restore():
    resumed = _write_then_restore(saved_lr=1e-5, resumed_lr=1e-6)
    overridden = enforce_learning_rate(resumed, 1e-6)
    assert np.float32(resumed.learning_rate.numpy()) == np.float32(1e-6)
    # Returned so the caller can print it; the *restored* value is what was displaced.
    assert np.float32(overridden) == np.float32(1e-5)


def test_raised_learning_rate_is_put_back_too():
    """Not symmetric by accident — the comparison must not be one-sided."""
    resumed = _write_then_restore(saved_lr=1e-6, resumed_lr=1e-4)
    assert np.float32(enforce_learning_rate(resumed, 1e-4)) == np.float32(1e-6)
    assert np.float32(resumed.learning_rate.numpy()) == np.float32(1e-4)


def test_unchanged_learning_rate_reports_no_override_and_assigns_nothing():
    """The common case: every resume that does not retune the learning rate.

    `None` rather than the value, so `snek2.main` only prints when something was displaced.
    """
    resumed = _write_then_restore(saved_lr=1e-5, resumed_lr=1e-5)
    assert enforce_learning_rate(resumed, 1e-5) is None
    assert np.float32(resumed.learning_rate.numpy()) == np.float32(1e-5)


def test_float32_round_trip_does_not_read_as_a_change():
    """The Variable holds 9.999999747e-06 for a configured 1e-5.

    Comparing the raw floats would call every resume an override and print on all of them, so
    this pins that an untouched optimizer reports `None`.
    """
    optimizer = _built_adam(1e-5)
    stored = float(optimizer.learning_rate.numpy())
    assert stored != 1e-5, 'expected a float32 round-trip, got an exact match'
    assert enforce_learning_rate(optimizer, 1e-5) is None


def test_a_retune_below_the_round6_floor_is_still_seen():
    """Why the comparison is float32 and not `maybe_update_epsilon`'s `round(x, 6)`.

    `round(x, 6)` separates 1e-5 from 1e-6 fine, but floors both 1e-7 and 1e-8 to 0.0 — so that
    idiom would call them equal and silently drop the retune. This test is the mutant check for
    the choice: swapping the float32 comparison for `round(x, 6)` fails here and nowhere else.
    """
    assert round(1e-7, 6) == round(1e-8, 6) == 0.0, 'the premise of this test no longer holds'
    resumed = _write_then_restore(saved_lr=1e-7, resumed_lr=1e-8)
    assert np.float32(enforce_learning_rate(resumed, 1e-8)) == np.float32(1e-7)
    assert np.float32(resumed.learning_rate.numpy()) == np.float32(1e-8)


def test_moments_and_iteration_count_are_left_alone():
    """It puts back the hyperparameter, not the history — Adam's state must survive."""
    with tempfile.TemporaryDirectory() as directory:
        variable = tf.Variable([1.0])
        saved = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-7)
        saved.apply_gradients([(tf.constant([0.5]), variable)])
        path = tf.train.Checkpoint(optimizer=saved).write(os.path.join(directory, 'ckpt'))

        resumed = _built_adam(1e-6)
        tf.train.Checkpoint(optimizer=resumed).restore(path).expect_partial()
        moments_before = [np.array(v.numpy(), copy=True) for v in resumed.variables]

        enforce_learning_rate(resumed, 1e-6)

        assert int(resumed.iterations.numpy()) == 1, 'the step count was reset'
        for before, after in zip(moments_before, resumed.variables):
            assert np.allclose(before, after.numpy()), (
                'enforce_learning_rate touched {0}'.format(after.name))
