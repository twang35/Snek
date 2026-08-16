"""`SNEK_ADAM_EPSILON` — that it reaches both optimizers, and that it is not a no-op.

Batch 32 rests entirely on the claim that Adam's `epsilon` changes which coordinates move, so that
claim gets a fixture rather than a comment. Without one, the knob reads like a divide-by-zero guard
and the obvious "simplification" is to delete it.

The mechanism test is exact rather than empirical, and the exact form is worth having written down
because the obvious guess about it is wrong. Keras applies `epsilon` *outside* the bias correction, so
one step from rest moves

    lr * x / (x + eps),   x = sqrt(1 - beta_2) * |g| = 0.0316 * |g|

which is `lr` when `x >> eps` and `lr * x/eps` when `x << eps`. **The crossover is therefore at
`|g| ~ 31.6 * eps`, not at `|g| ~ eps`** — a factor of 31.6 that decides whether a given epsilon is
doing anything at all. At Dopamine's `3.125e-4` the half-step point is `|g| ~ 1e-2`, so it damps a
much wider band of gradients than the number suggests; a first version of this file assumed the
crossover was at `eps` itself and picked test gradients that all landed on the wrong side of it.

One step is enough to pin the whole regime split, with no dependence on how many iterations a loop
happened to run.
"""
import ast
import os
import sys

import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import snek2

SNEK2_PATH = os.path.join(os.path.dirname(__file__), '..', 'snek2.py')
DOPAMINE_C51_EPSILON = 3.125e-4      # Dopamine's published C51 config, with lr 2.5e-4
RAINBOW_EPSILON = 1.5e-4             # Dopamine's published Rainbow config, with lr 6.25e-5
# sqrt(1 - beta_2) at Keras's default beta_2=0.999. The gradient at which one step from rest is
# exactly half of `lr` is CROSSOVER_SCALE**-1 * eps, i.e. ~31.6 * eps.
CROSSOVER_SCALE = 0.001 ** 0.5


def _source_default(knob):
    """The default literal `snek2.py` actually passes to `tuned(knob, ...)`.

    Parsed out of the source so a test cannot accidentally assert its own argument back at itself,
    which is the way a default-value test ends up asserting nothing at all.
    """
    tree = ast.parse(open(SNEK2_PATH).read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'tuned' and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == knob):
            return ast.literal_eval(node.args[1])
    raise AssertionError('snek2.py has no tuned({0!r}, ...) call'.format(knob))


def _first_step(epsilon, gradient, learning_rate=1e-4):
    """How far one Adam step moves a scalar with a constant gradient."""
    variable = tf.Variable(0.0)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
    optimizer.apply_gradients([(tf.constant(gradient), variable)])
    return abs(float(variable.numpy()))


def test_default_matches_keras_so_earlier_arms_are_unchanged():
    """Unset, the knob must reproduce the optimizer every arm before batch 32 was trained with.

    Asserted against Keras's own attribute rather than the literal 1e-7, so a framework upgrade that
    moved the default would fail here instead of silently redefining what "no override" means for
    every ddqn arm in the project.
    """
    os.environ.pop('SNEK_ADAM_EPSILON', None)
    assert float(tf.keras.optimizers.Adam().epsilon) == 1e-7
    assert snek2.tuned('ADAM_EPSILON', 1e-7) == 1e-7
    # Read out of the source, not passed in from here. A first version of this test supplied its own
    # 1e-7 as the default argument, so raising snek2.py's real default to a C51 value — which would
    # silently retrain every ddqn arm in the project under a different optimizer — passed it.
    assert _source_default('ADAM_EPSILON') == 1e-7, (
        'snek2.py defaults SNEK_ADAM_EPSILON to something other than the Keras default, so an arm '
        'that sets nothing no longer reproduces every arm trained before batch 32')


def test_env_override_is_read_as_a_float():
    os.environ['SNEK_ADAM_EPSILON'] = '3.125e-4'
    try:
        value = snek2.tuned('ADAM_EPSILON', 1e-7)
        assert value == DOPAMINE_C51_EPSILON
        assert isinstance(value, float)
    finally:
        os.environ.pop('SNEK_ADAM_EPSILON', None)


def test_it_reaches_the_optimizer():
    assert float(tf.keras.optimizers.Adam(
        learning_rate=1e-4, epsilon=DOPAMINE_C51_EPSILON).epsilon) == DOPAMINE_C51_EPSILON


def test_every_adam_built_in_snek2_passes_epsilon():
    """The tripwire: a new agent branch that forgets `epsilon` would silently train at 1e-7.

    There are two `Adam(...)` sites, one per algorithm, and they are 16 lines apart — exactly the
    shape of edit where one gets updated and the other does not. Walks the AST rather than grepping
    so a reformatted call cannot slip through on whitespace.
    """
    tree = ast.parse(open(SNEK2_PATH).read())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == 'Adam']
    assert len(calls) == 2, 'expected one Adam per algorithm branch, found {0}'.format(len(calls))
    for call in calls:
        keywords = {kw.arg: kw.value for kw in call.keywords}
        assert 'epsilon' in keywords, (
            'the Adam at snek2.py:{0} does not pass epsilon, so it would train at the Keras '
            'default whatever SNEK_ADAM_EPSILON says'.format(call.lineno))
        assert 'learning_rate' in keywords, 'snek2.py:{0}'.format(call.lineno)
        # It has to be *the knob*, not a literal that happens to equal the default. A wired-but-ignored
        # `epsilon=1e-7` passes a presence check and would make batch 32 a silent null result — every
        # arm training identically while its report claimed otherwise. Same for learning_rate, which
        # has had a live sweep on it for the whole project.
        for knob, expected in (('epsilon', 'adam_epsilon'), ('learning_rate', 'learning_rate')):
            value = keywords[knob]
            assert isinstance(value, ast.Name) and value.id == expected, (
                'snek2.py:{0} passes {1}={2}, not the {3} variable, so the override would be read, '
                'printed and recorded but never applied'.format(
                    call.lineno, knob, ast.dump(value), expected))


def test_a_large_epsilon_barely_changes_a_well_driven_gradient():
    """Well above the crossover both settings take essentially the full `lr` step.

    This is the half that makes the knob safe: it does not scale learning down across the board, so a
    b32 arm is not simply a lower-learning-rate arm in disguise. If this collapsed, the epsilon
    experiment would be confounded with the learning-rate sweep it is meant to be independent of.
    """
    default = _first_step(1e-7, 1.0)
    dopamine = _first_step(DOPAMINE_C51_EPSILON, 1.0)
    assert dopamine / default > 0.98, (default, dopamine)


def test_a_large_epsilon_strongly_damps_a_noise_sized_gradient():
    """Below the crossover the step becomes proportional again — the whole point of the knob.

    At 1e-7 a gradient 100,000x smaller than the one above still buys the majority of a full step,
    which is the behaviour suspected of driving the C51 arms' churn: a coordinate carrying nothing but
    batch noise moves nearly as far as one carrying signal.
    """
    default = _first_step(1e-7, 1e-5)
    dopamine = _first_step(DOPAMINE_C51_EPSILON, 1e-5)
    assert default / _first_step(1e-7, 1.0) > 0.5, 'at 1e-7 a tiny gradient still moves ~half of lr'
    assert default / dopamine > 100, (default, dopamine)


def test_the_crossover_sits_at_31_times_epsilon_not_at_epsilon():
    """Half a step at `|g| = eps / sqrt(1 - beta_2)`, which is the non-obvious part of the form.

    Pins *where* the split happens, not just that it exists. That location is what makes 1.5e-4 and
    3.125e-4 different treatments rather than two ways of spelling the same one, and it is what a
    future reader needs in order to pick a third value on purpose.
    """
    for epsilon in (RAINBOW_EPSILON, DOPAMINE_C51_EPSILON):
        half_step_gradient = epsilon / CROSSOVER_SCALE
        at_crossover = _first_step(epsilon, half_step_gradient)
        assert abs(at_crossover - 1e-4 * 0.5) < 1e-4 * 0.02, (epsilon, at_crossover)
        # And one order of magnitude below it really is in the damped regime, so the crossover is a
        # crossover rather than a point the curve merely passes through.
        assert _first_step(epsilon, half_step_gradient / 10) < 1e-4 * 0.15, epsilon


def test_the_two_batch_32_settings_are_meaningfully_different():
    """1.5e-4 and 3.125e-4 must separate on a real gradient, or b32's two halves test one thing.

    Compared at the gradient sitting between their crossovers, where the difference is largest — a
    check on the experiment's design, not on Keras.
    """
    between = (RAINBOW_EPSILON + DOPAMINE_C51_EPSILON) / 2 / CROSSOVER_SCALE
    rainbow = _first_step(RAINBOW_EPSILON, between)
    dopamine = _first_step(DOPAMINE_C51_EPSILON, between)
    assert rainbow / dopamine > 1.2, (rainbow, dopamine)
