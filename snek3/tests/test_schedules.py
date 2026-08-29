"""The exploration schedule. Pure functions, so these are cheap and there are a lot of them.

The schedule steers what goes into the replay buffer, so a defect here changes the training and not
only a log line. Two properties carry most of the weight: the phases must hand over without a jump,
and every function must be stateless, because a resume recomputes epsilon from restored history.
"""

import pytest

from dqn import schedules


def test_bootstrap_halves_once_per_threshold_cleared():
    assert schedules.bootstrap_epsilon(0, 0.4) == pytest.approx(0.4)
    assert schedules.bootstrap_epsilon(3, 0.4) == pytest.approx(0.2)
    assert schedules.bootstrap_epsilon(6, 0.4) == pytest.approx(0.1)
    assert schedules.bootstrap_epsilon(12, 0.4) == pytest.approx(0.05)
    assert schedules.bootstrap_epsilon(18, 0.4) == pytest.approx(0.025)


def test_a_reward_exactly_on_a_threshold_stays_on_the_higher_rung():
    # `<=`, not `<`. Which side does not matter much, but it must not change silently.
    assert schedules.bootstrap_epsilon(2, 0.4) == pytest.approx(0.4)
    assert schedules.bootstrap_epsilon(5, 0.4) == pytest.approx(0.2)


def test_bootstrap_stands_down_with_zero_not_with_a_small_epsilon():
    # 0.0 means "this phase has nothing to say". If it returned its last rung instead, the max()
    # in epsilon_for would pin epsilon there forever and refinement would never run.
    assert schedules.bootstrap_epsilon(21, 0.4) == 0.0
    assert schedules.bootstrap_epsilon(1000, 0.4) == 0.0


def test_every_bootstrap_threshold_is_in_the_pre_winning_regime():
    # Nothing here may cut exploration while an arm is still learning to win. A perfect game scores
    # far above 20, so a threshold above that would fire on an arm that is already winning.
    assert max(schedules.BOOTSTRAP_REWARD_THRESHOLDS) <= 20


def test_the_thresholds_ascend():
    thresholds = schedules.BOOTSTRAP_REWARD_THRESHOLDS
    assert list(thresholds) == sorted(thresholds), 'the ladder is walked in order'


def test_refine_starts_at_the_top_and_reaches_the_floor_at_the_target():
    assert schedules.refine_epsilon(0.0, 0.0125, 0.002) == pytest.approx(0.0125)
    assert schedules.refine_epsilon(0.80, 0.0125, 0.002) == pytest.approx(0.002)


def test_refine_is_geometric_not_linear():
    # Halfway to the target must be the geometric mean of top and floor, not the arithmetic one.
    top, floor = 0.0125, 0.002
    midpoint = schedules.refine_epsilon(0.40, top, floor)
    assert midpoint == pytest.approx((top * floor) ** 0.5)
    assert midpoint < (top + floor) / 2.0, 'a linear ramp would sit higher here'


def test_refine_never_goes_below_the_floor_however_good_the_arm_is():
    for rate in (0.85, 0.95, 1.0, 1.5):
        assert schedules.refine_epsilon(rate, 0.0125, 0.002) == pytest.approx(0.002)


def test_refine_never_goes_above_the_top_on_a_negative_rate():
    # A negative rate would give a negative exponent and push epsilon above `top`. Nothing else
    # guards that, which is why the lower clamp on the fraction exists.
    assert schedules.refine_epsilon(-0.5, 0.0125, 0.002) <= 0.0125


def test_refine_degenerates_to_the_floor_when_the_top_is_not_above_it():
    """A reachable misconfiguration, not a hypothetical.

    `top` is `initial_epsilon / 32`, so `SNEK_INITIAL_EPSILON=0.05` puts it at 0.0015625 — *below* a
    default `min_epsilon` of 0.002. Without the guard the ratio `floor/top` exceeds 1, and a perfect
    rate above the target gives an exponent above 1, so **a better arm would explore more**: at rate
    1.6 the un-guarded formula returns 0.004, twice the floor.
    """
    assert schedules.refine_epsilon(0.0, 0.002, 0.002) == pytest.approx(0.002)
    assert schedules.refine_epsilon(0.0, 0.001, 0.002) == pytest.approx(0.002)
    assert schedules.refine_epsilon(1.6, 0.001, 0.002) == pytest.approx(0.002)
    # And the whole realistic range stays pinned at the floor rather than rising with skill.
    inverted = [schedules.refine_epsilon(rate, 0.0015625, 0.002)
                for rate in (0.0, 0.4, 0.8, 1.0)]
    assert inverted == [pytest.approx(0.002)] * 4, inverted


def test_the_phases_hand_over_downward_without_a_jump():
    """Bootstrap's last live rung must sit at or above refinement's ceiling.

    If refinement's top were the higher of the two, the handover would *raise* epsilon at exactly
    the moment the arm started winning.
    """
    initial = 0.4
    last_rung = initial / 2.0 ** (schedules.BOOTSTRAP_RUNGS - 1)
    refine_top = initial / 2.0 ** schedules.BOOTSTRAP_RUNGS
    assert refine_top < last_rung


def test_epsilon_for_follows_bootstrap_while_it_is_live():
    # Perfect rate is 0 through the whole bootstrap band, so refinement would say `top`. Bootstrap
    # is larger, and it is the one that must win.
    assert schedules.epsilon_for(0, 0.0, 0.4, 0.002) == pytest.approx(0.4)
    assert schedules.epsilon_for(6, 0.0, 0.4, 0.002) == pytest.approx(0.1)


def test_epsilon_for_follows_refinement_once_bootstrap_stands_down():
    assert schedules.epsilon_for(30, 0.0, 0.4, 0.002) == pytest.approx(0.0125)
    assert schedules.epsilon_for(30, 0.8, 0.4, 0.002) == pytest.approx(0.002)


def test_epsilon_for_takes_the_max_so_a_strong_arm_early_is_not_cut_off():
    # An arm with a high perfect rate but a low avg_reward is contradictory, and the max() resolves
    # it toward *more* exploration. min() here would drop epsilon to the floor on eval one.
    assert schedules.epsilon_for(0, 0.9, 0.4, 0.002) == pytest.approx(0.4)


def test_epsilon_for_is_stateless_so_a_regression_buys_exploration_back():
    # The property the schedule exists for: no ratchet. A declining arm explores more again.
    strong = schedules.epsilon_for(30, 0.75, 0.4, 0.002)
    collapsed = schedules.epsilon_for(30, 0.05, 0.4, 0.002)
    assert collapsed > strong * 2


def test_epsilon_for_never_returns_exactly_zero():
    assert schedules.epsilon_for(1000, 1.0, 0.4, 0.002) > 0.0
    assert schedules.EPSILON_HARD_FLOOR > 0.0


def test_trailing_mean_divides_by_the_evals_that_exist():
    # Not by `window`. Dividing by the window would read near zero on a fresh run and pin epsilon
    # at the ceiling exactly when it should descend.
    assert schedules.trailing_mean([], 'avg_reward', 10.0, 5) == pytest.approx(10.0)
    rows = [{'avg_reward': 6.0}]
    assert schedules.trailing_mean(rows, 'avg_reward', 10.0, 5) == pytest.approx(8.0)


def test_trailing_mean_counts_the_current_value_and_window_minus_one_rows():
    rows = [{'avg_reward': float(value)} for value in range(1, 11)]
    # window 5 -> the last 4 rows (7, 8, 9, 10) plus the current value.
    assert schedules.trailing_mean(rows, 'avg_reward', 0.0, 5) == pytest.approx(34.0 / 5)


def test_trailing_mean_with_window_one_is_just_the_current_value():
    rows = [{'avg_reward': 99.0}]
    assert schedules.trailing_mean(rows, 'avg_reward', 1.0, 1) == pytest.approx(1.0)


def test_trailing_mean_treats_a_missing_key_as_zero_rather_than_raising():
    # Rows from an older run may not carry every field, and an eval must not crash on one.
    assert schedules.trailing_mean([{}], 'avg_reward', 4.0, 2) == pytest.approx(2.0)


def test_trailing_perfect_rate_converts_stored_percent_to_a_fraction():
    # Getting this wrong scales epsilon by 100, which is the whole reason it is one function.
    rows = [{'perfect_percent': 80.0}]
    assert schedules.trailing_perfect_rate(rows, 0.80, window=2) == pytest.approx(0.80)


def test_trailing_perfect_rate_uses_the_same_window_as_best_perfect30():
    assert schedules.REFINE_TRAILING_WINDOW == 30


def test_trailing_reward_uses_a_short_window_but_not_one():
    # One would let a single noisy eval flap epsilon; the measured case went 0.4 -> 0.2 -> 0.4.
    assert 1 < schedules.BOOTSTRAP_TRAILING_WINDOW <= 10


def test_the_shield_is_off_while_bootstrap_is_live():
    assert schedules.guided_fraction_for(0, 0.4, 0.8) == 0.0
    assert schedules.guided_fraction_for(18, 0.4, 0.8) == 0.0


def test_the_shield_comes_on_when_bootstrap_stands_down():
    assert schedules.guided_fraction_for(30, 0.4, 0.8) == pytest.approx(0.8)


def test_the_shield_switches_off_again_if_an_arm_collapses_back():
    # Stateless, not latching: one rule, "shielded iff refining", rather than two that can disagree.
    assert schedules.guided_fraction_for(30, 0.4, 0.8) > 0.0
    assert schedules.guided_fraction_for(1, 0.4, 0.8) == 0.0


def test_a_configured_fraction_of_zero_keeps_the_shield_off_throughout():
    for reward in (0, 6, 30, 1000):
        assert schedules.guided_fraction_for(reward, 0.4, 0.0) == 0.0
