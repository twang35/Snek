"""`SNEK_PERFECT_GAME_REWARD` — the default is untouched, and every guard follows the scale.

Batch 33 turns the win reward down to 10. That is a change of *objective*, and it invalidates three
things that were quietly written against a win of 100:

1. `MEASURED_MAX_RETURN`, a single empirical constant used as a hard clipping bound.
2. `arch.json`, which recorded the support but not what the support was calibrated to — so an arm
   trained at 10 and resumed at 100 restored cleanly and optimised something else.
3. Nothing else, and that is the load-bearing part: perfect games are counted from **score**, so the
   counter that broke the last time a reward term moved is immune to this one by construction.

**The relationship is not linear and that is the whole reason this file exists.** At a win of 100 the
maximum discounted return is just before the win (104.38). At a win of 10 the win stops dominating, so
the maximum moves to the *opening* of an episode, where all 95 meals are still ahead — and it lands at
**32.46**, not at 10.4. Any code that scales the bound by the reward ratio is wrong by 3x.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import categorical_agent
import policy_arch
import snake_constants

CONSTANTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'snake_constants.py')


def _source_default(name):
    """The literal `snake_constants.py` assigns to `name`, read from source.

    Same reason `test_adam_epsilon` parses its default out of the file: a test that passes its own
    expected value in as an argument and gets it back asserts nothing.
    """
    tree = ast.parse(open(CONSTANTS_PATH).read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
            return ast.literal_eval(node.value)
    raise AssertionError('snake_constants.py has no plain assignment to {0}'.format(name))


class _Win(object):
    """Sets `PERFECT_GAME_REWARD` for a block and restores it, since it is module-global.

    Assigning the module attribute is the right lever in a test: the env var is read once at import,
    and every guard under test reads `snake_constants.PERFECT_GAME_REWARD` live.
    """

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.original = snake_constants.PERFECT_GAME_REWARD
        snake_constants.PERFECT_GAME_REWARD = self.value

    def __exit__(self, *exc):
        snake_constants.PERFECT_GAME_REWARD = self.original


def test_the_default_win_is_still_100():
    """Every arm before batch 33 trained at 100, so the unset value must reproduce it exactly."""
    assert _source_default('DEFAULT_PERFECT_GAME_REWARD') == 100
    assert float(snake_constants.DEFAULT_PERFECT_GAME_REWARD) == 100.0


def test_the_knob_reads_from_the_environment():
    """Read in `snake_constants`, not `snek2`, because `Snake.step` pays it in a worker process."""
    source = open(CONSTANTS_PATH).read()
    assert "os.environ.get('SNEK_PERFECT_GAME_REWARD'" in source, (
        'the knob must be read where the env workers import it — a parent-process assignment never '
        'reaches a ParallelPyEnvironment worker, which is why FOOD_DISTANCE_REWARD lives here too')


def test_the_measured_bound_is_per_reward_scale_and_is_not_linear():
    """104.38 at win 100 and 32.46 at win 10 — a 3.1x ratio for a 10x reward change.

    Pinned as an inequality rather than by exact values so the table can gain entries, but any future
    edit that computes the bound by scaling the reward fails here.
    """
    table = categorical_agent.MEASURED_MAX_RETURN_BY_WIN
    assert table[100.0] > table[10.0] * 2, table
    assert table[10.0] > 10.0 * 2, (
        'the win-10 bound must exceed the win value itself: the maximum return moves to the opening '
        'of an episode, where 95 discounted meals dominate a 10-point terminal win')


def test_measured_max_return_follows_the_live_reward():
    with _Win(100.0):
        assert categorical_agent.measured_max_return() == 105.0
    with _Win(10.0):
        assert categorical_agent.measured_max_return() == 33.0


def test_an_unmeasured_scale_warns_rather_than_refusing():
    """`None` is the honest answer, and it must downgrade the hard failure rather than invent a bound.

    Refusing to start on a bound measured under different rewards would block a legitimate experiment;
    silently scaling the bound would be wrong by 3x. Warning and recording it is the third option.
    """
    with _Win(42.0):
        assert categorical_agent.measured_max_return() is None
        warnings = categorical_agent.check_support(-5.0, 20.0, 51)   # would be refused at win 100
        assert any('no measured maximum return' in w for w in warnings), warnings


def test_the_clipping_floor_still_bites_at_a_measured_scale():
    with _Win(10.0):
        try:
            categorical_agent.check_support(-5.0, 30.0, 51)
        except SystemExit as exc:
            assert '33.0' in str(exc) and 'PERFECT_GAME_REWARD=10.0' in str(exc), str(exc)
        else:
            raise AssertionError('v_max=30 is below the measured 33.0 at win 10 and must be refused')
        # And the batch 33 support is accepted, with the below-derived-bound judgement recorded.
        warnings = categorical_agent.check_support(-5.0, 40.0, 51)
        assert any('below the derived maximum return' in w for w in warnings), warnings


def test_win_100_still_refuses_exactly_what_it_refused_before():
    """The reward-scale table must not have loosened the guard for the default configuration."""
    with _Win(100.0):
        try:
            categorical_agent.check_support(-5.0, 104.0, 51)
        except SystemExit:
            pass
        else:
            raise AssertionError('v_max=104 is below the measured 105.0 at win 100 and must be refused')
        assert categorical_agent.check_support(-5.0, 120.0, 51)   # the shipped support: warns, starts


def test_the_arch_records_the_reward_scale_for_both_algorithms():
    scalar = policy_arch.build_arch((50, 100, 50), 3, 30, 'b09c616', perfect_game_reward=10.0)
    assert scalar['perfect_game_reward'] == 10.0
    cat = policy_arch.build_arch((200, 100, 100), 3, 30, 'b09c616', algo='c51',
                                 num_atoms=51, v_min=-5.0, v_max=40.0, perfect_game_reward=10.0)
    assert cat['perfect_game_reward'] == 10.0
    # Omitted means the default, which is what every pre-batch-33 arm trained at.
    assert policy_arch.build_arch((50,), 3, 30, 'b09c616')['perfect_game_reward'] == 100.0


def test_a_sidecar_without_the_field_reads_as_the_default():
    """~100 policy directories and every hallOfFame entry predate the field, and all trained at 100."""
    assert policy_arch.reward_scale_of({'algo': 'ddqn'}) == 100.0
    assert policy_arch.reward_scale_of({'algo': 'c51', 'perfect_game_reward': 10.0}) == 10.0


def test_resuming_under_a_changed_win_is_refused(tmp=None):
    """The trap this field exists for: it restores cleanly and then optimises a different objective.

    Unlike a changed layer width there is no shape error anywhere downstream, and unlike a changed
    support the atoms still mean what they meant — the *environment* is what moved. Only a recorded
    value catches it.
    """
    import json
    import shutil
    import tempfile
    d = tempfile.mkdtemp()
    try:
        arch = policy_arch.build_arch((200, 100, 100), 3, 30, 'b09c616', algo='c51',
                                      num_atoms=51, v_min=-5.0, v_max=40.0, perfect_game_reward=10.0)
        with open(os.path.join(d, policy_arch.ARCH_FILENAME), 'w') as handle:
            json.dump(arch, handle)
        # Same knobs, right reward: fine.
        policy_arch.assert_config_matches(d, (200, 100, 100), algo='c51', num_atoms=51,
                                          v_min=-5.0, v_max=40.0, perfect_game_reward=10.0)
        try:
            policy_arch.assert_config_matches(d, (200, 100, 100), algo='c51', num_atoms=51,
                                              v_min=-5.0, v_max=40.0, perfect_game_reward=100.0)
        except policy_arch.ArchMismatch as exc:
            assert 'perfect_game_reward' in str(exc) and 'different objective' in str(exc), str(exc)
        else:
            raise AssertionError('resuming a win-10 checkpoint at win 100 must raise ArchMismatch')
    finally:
        shutil.rmtree(d)


def test_nothing_counts_a_perfect_game_from_this_reward():
    """The guard that makes the knob safe at all, restated here so it is checked from this side too.

    `tests/test_perfect_game_counting.py` owns the `ast` tripwire. This asserts the positive form: the
    definition is a score predicate, so it is invariant to every reward change.
    """
    import state_helpers
    with _Win(10.0):
        perfect = int(snake_constants.MAX_POSSIBLE_SCORE)
        assert state_helpers.is_perfect_score(perfect)
        assert not state_helpers.is_perfect_score(perfect - 1)
