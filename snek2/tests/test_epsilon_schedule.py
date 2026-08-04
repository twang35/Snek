"""Tests for the two-phase epsilon schedule in training.py.

The schedule this replaced was a one-way ratchet on `avg_reward` that reached 0.001 by step
~15000 of a 3.5M-step run and 0.0 shortly after: 96.8% of every training step across batches
10 and 11 ran fully greedy. These tests pin the three properties that failure turned on —
where the rungs sit in *skill* terms, that epsilon can rise again, and that it never reaches 0.
"""
import training

INITIAL = 0.4
FLOOR = 0.002
# Where the bootstrap phase hands over: INITIAL / 2**3.
TOP = 0.05


def eps(avg_reward, perfect_rate, initial=INITIAL, floor=FLOOR):
    return training.epsilon_for(avg_reward, perfect_rate, initial, floor)


# ------------------------------------------------------- bootstrap phase

def test_bootstrap_halves_once_per_threshold():
    # -5 is the reward of an agent that dies immediately, which is where every run starts.
    assert eps(-5.0, 0.0) == INITIAL
    assert eps(7.0, 0.0) == INITIAL / 2
    assert eps(15.0, 0.0) == INITIAL / 4


def test_bootstrap_hands_over_at_the_refinement_ceiling():
    # Past the last threshold the bootstrap term stands down and refinement takes over at
    # exactly INITIAL/8 — the two phases must meet, not overlap or leave a gap.
    assert eps(25.0, 0.0) == TOP
    assert training.bootstrap_epsilon(25.0, INITIAL) == 0.0


def test_bootstrap_thresholds_are_below_the_first_perfect_game():
    # The whole diagnosis: the old ladder's *last* rung fired at avg_reward > 60, which is
    # "eats 65 of 95 food and never wins". Every threshold that still drops epsilon must sit
    # in the pre-winning regime, so nothing cuts exploration while the arm is learning to win.
    assert max(training.BOOTSTRAP_REWARD_THRESHOLDS) <= 20


def test_a_collapsed_arm_gets_exploration_back():
    # b11b at step 20000: score fell 64.6 -> 8.8 with epsilon pinned at 0.001 by the ratchet.
    # A reward that low is back inside the bootstrap band, so epsilon must rise.
    healthy = eps(90.0, 0.85)
    collapsed = eps(4.0, 0.85)
    assert collapsed > healthy
    assert collapsed == INITIAL


# ------------------------------------------------------ refinement phase

def test_refinement_reaches_the_floor_at_the_target_rate():
    assert abs(eps(90.0, training.REFINE_PERFECT_TARGET) - FLOOR) < 1e-12


def test_refinement_clamps_above_the_target():
    # No arm has sustained a trailing rate near 1.0, but the function must not undershoot.
    assert eps(90.0, 0.95) == FLOOR
    assert eps(90.0, 1.0) == FLOOR


def test_refinement_is_geometric_not_linear():
    # Equal steps in perfect rate must give equal *ratios*, which is the property that keeps
    # epsilon out of the 0.02+ band for most of the run. A linear ramp would give equal
    # differences instead, and would still be at 0.026 by 40% perfect rather than 0.010.
    quarter = eps(90.0, 0.20)
    half = eps(90.0, 0.40)
    three_quarters = eps(90.0, 0.60)
    assert abs((quarter / half) - (half / three_quarters)) < 1e-9
    linear = TOP + (FLOOR - TOP) * 0.5
    assert half < linear


def test_refinement_is_monotone_decreasing_in_skill():
    rates = [i / 20.0 for i in range(21)]
    values = [eps(90.0, r) for r in rates]
    assert all(b <= a for a, b in zip(values, values[1:]))


def test_refinement_never_exceeds_its_ceiling():
    assert eps(90.0, 0.0) == TOP
    assert eps(90.0, -0.1) == TOP  # a nonsense negative rate must not amplify epsilon


# ------------------------------------------------------------ the floor

def test_epsilon_never_reaches_zero():
    # The property the user asked for, across the whole domain.
    for reward in (-5.0, 0.0, 7.0, 15.0, 25.0, 60.0, 100.0, 200.0):
        for rate in (0.0, 0.25, 0.5, 0.8, 1.0):
            assert eps(reward, rate) >= FLOOR > 0.0


def test_a_zero_floor_would_still_not_produce_zero_from_refinement():
    # Defence in depth: snek2.py rejects a floor below EPSILON_HARD_FLOOR, but if that guard
    # were ever bypassed the schedule itself must not hand back a fully greedy policy for a
    # skill level short of the target.
    assert training.EPSILON_HARD_FLOOR > 0.0
    assert eps(90.0, 0.4, floor=training.EPSILON_HARD_FLOOR) > 0.0


def test_floor_at_the_ceiling_degenerates_safely():
    # min_epsilon >= TOP is rejected by snek2.py; the function must still be total.
    assert training.refine_epsilon(0.5, top=0.05, floor=0.05) == 0.05
    assert training.refine_epsilon(0.5, top=0.05, floor=0.2) == 0.2


# --------------------------------------------------- the trailing signal

def rows(*percents):
    return [{'step': 1000 * i, 'perfect_percent': p} for i, p in enumerate(percents)]


def reward_rows(*rewards):
    return [{'step': 1000 * i, 'avg_reward': r} for i, r in enumerate(rewards)]


def test_bootstrap_signal_is_damped_against_flapping():
    # Caught by a smoke run: avg_reward read 7.63 then 4.96 on consecutive evals — noise either
    # side of the first threshold — and the undamped schedule went 0.4 -> 0.2 -> 0.4. The
    # trailing mean of the same two readings stays above 5, so epsilon holds.
    undamped = eps(4.96, 0.0)
    damped = eps(training.trailing_reward(reward_rows(7.63), 4.96), 0.0)
    assert undamped == INITIAL
    assert damped == INITIAL / 2


def test_bootstrap_signal_still_responds_within_its_window():
    # Damped, not frozen: a sustained regression must still raise epsilon, because a one-way
    # bootstrap was half of the original defect.
    sustained = training.trailing_reward(reward_rows(2.0, 2.0, 2.0, 2.0), 1.0)
    assert eps(sustained, 0.0) == INITIAL


def test_bootstrap_window_is_shorter_than_the_refinement_window():
    # The two phases run on different timescales — ~10k steps against ~1M — so a single window
    # cannot serve both. This pins the relationship rather than the values.
    assert 1 < training.BOOTSTRAP_TRAILING_WINDOW < training.REFINE_TRAILING_WINDOW


def test_trailing_mean_handles_a_window_of_one():
    # window=1 must mean "just this eval", not crash on the -(window-1) slice, which would
    # otherwise take eval_rows[0:] and average the entire history.
    assert training.trailing_mean(reward_rows(99.0, 99.0), 'avg_reward', 1.0, window=1) == 1.0


def test_trailing_rate_converts_the_percent_scale():
    # eval_rows store 0-100, the live value arrives as 0-1. Mixing them scales epsilon by 100.
    assert training.trailing_perfect_rate(rows(80, 80, 80), 0.8) == 0.8


def test_trailing_rate_includes_the_current_eval():
    # Window of 2: one stored row at 0% plus a current 100% eval averages to 50%.
    assert training.trailing_perfect_rate(rows(0), 1.0, window=2) == 0.5


def test_trailing_rate_uses_only_the_window():
    # The 30-eval window must not be dragged by ancient history.
    old = rows(*([0] * 100))
    assert training.trailing_perfect_rate(old + rows(*([100] * 29)), 1.0) == 1.0


def test_trailing_rate_averages_a_short_history():
    # A fresh run has no 30 evals yet; it must still produce a usable signal rather than
    # dividing by the window and reading near zero, which would pin epsilon at the ceiling.
    assert training.trailing_perfect_rate([], 0.6) == 0.6
    assert training.trailing_perfect_rate(rows(100), 1.0) == 1.0


def test_trailing_rate_damps_a_single_lucky_eval():
    # The ratchet hazard, which is what pinned epsilon at 0.001 by step ~15000: one 10-episode
    # eval crossing a threshold set the floor permanently. Here a single 100% eval against 29
    # zeroes moves the signal by 1/30, and epsilon by 0.0500 -> 0.0437 — a 13% dip, not a 50x
    # cut, and bounded because the signal is an average.
    flat = training.trailing_perfect_rate(rows(*([0] * 29)), 0.0)
    lucky = training.trailing_perfect_rate(rows(*([0] * 29)), 1.0)
    assert lucky - flat < 0.05
    assert eps(90.0, lucky) > 0.85 * eps(90.0, flat)


def test_a_lucky_eval_does_not_pin_epsilon():
    # And the dip reverses on the next ordinary eval, which is the property the ratchet lacked
    # entirely: nothing here carries the lucky eval forward once it leaves the window.
    lucky = training.trailing_perfect_rate(rows(*([0] * 29)), 1.0)
    back = training.trailing_perfect_rate(rows(*([0] * 29)), 0.0)
    assert eps(90.0, back) == eps(90.0, 0.0) == TOP
    assert eps(90.0, lucky) < TOP


# ------------------------------------------------------- the whole curve

def test_the_schedule_spends_real_time_above_the_old_effective_floor():
    # The headline defect: batches 10-11 ran 99.6% of their steps at epsilon <= 0.001. Under
    # this schedule an arm is still above 0.01 at a 40% trailing perfect rate, which no arm
    # on record reached before ~300k steps.
    assert eps(90.0, 0.40) >= 0.01
    assert eps(90.0, 0.30) > 0.01


def test_a_bigger_initial_epsilon_scales_the_whole_curve():
    # INITIAL_EPSILON stays a meaningful knob: the rungs and the handover are derived from it
    # rather than hardcoded, so doubling it doubles the ceiling the refinement starts from.
    assert eps(25.0, 0.0, initial=0.8) == 0.1
    assert eps(-5.0, 0.0, initial=0.8) == 0.8
