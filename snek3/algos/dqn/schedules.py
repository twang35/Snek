"""The exploration schedule: how much epsilon, and how much of it is shielded.

Pure functions of the eval history, with no state of their own. That is the design and not an
accident of the port: the schedule these replace was a one-way ratchet, so one lucky eval pinned
epsilon permanently and a regression never bought exploration back — a snek2 arm sat at 0.001 while
its score collapsed from 64.6 to 8.8. Stateless also makes a resume exact, because the first eval
after a restart recomputes epsilon from the restored history instead of descending a ladder again.

**This is a feedback loop, not a readout.** `perfect_percent` steers exploration through
`refine_epsilon`, so breaking the measurement changes the training rather than only the log — in
snek2 a shaping term silenced the perfect-game counter and eight arms trained with epsilon pinned at
its ceiling for 300k+ steps while reading 0%. See [`../docs/invariants.md`](../docs/invariants.md)
invariant 2.

Two phases, combined with `max()`:

| phase | driven by | range | at the defaults 0.4 / 0.002 |
|---|---|---|---|
| bootstrap | `avg_reward`, one halving per threshold cleared | `initial` down to `initial/16` | 0.4 -> 0.025, then 0.0 |
| refinement | trailing perfect rate | `initial/32` down to `min_epsilon` | 0.0125 -> 0.002 |

The two do not overlap: bootstrap's last live rung is `initial/16` and refinement's ceiling is
`initial/32`, one halving below it, so the handover is a single step down and never a jump up.
"""

# `avg_reward` thresholds the bootstrap phase halves on. Five rungs, so the phase hands over at
# `initial_epsilon / 32` — 0.0125 for the default 0.4.
#
# **The handover value is the load-bearing number, not the rung count.** A three-rung version handed
# over at 0.05, and 0.05 is too high for this task with or without the shield: arms pinned there for
# up to 942k steps at 0% perfect games, greedy trailing 53-63 where a lower handover gave 84-88. The
# cause is structural rather than a tuning miss — a collect policy that never finishes a board leaves
# no completed trajectories in the buffer to learn the endgame from.
#
# Every threshold still sits in the pre-winning regime (the largest is 20), so nothing here cuts
# exploration while an arm is still learning to win.
BOOTSTRAP_REWARD_THRESHOLDS = (2, 5, 10, 15, 20)
BOOTSTRAP_RUNGS = len(BOOTSTRAP_REWARD_THRESHOLDS)

# Trailing perfect rate at which refinement reaches the floor. 0.80 rather than 1.0 because no arm
# has sustained a trailing rate above ~0.92, and anchoring at 1.0 puts the floor out of reach.
REFINE_PERFECT_TARGET = 0.80

# Evals averaged for the refinement phase's skill signal. 30 x 100 episodes = 3,000, so a single
# lucky eval cannot move epsilon far. Deliberately the same window `run_report` uses for
# `best_perfect30`.
REFINE_TRAILING_WINDOW = 30

# Evals averaged for the bootstrap phase's reward signal. Short, because the phase lasts ~10k steps
# and has to stay responsive — but more than one, because the raw signal flaps: two consecutive evals
# read 7.63 and 4.96 either side of the first threshold and undamped epsilon went 0.4 -> 0.2 -> 0.4.
BOOTSTRAP_TRAILING_WINDOW = 5

# epsilon never reaches exactly 0. A fully greedy collect policy makes the buffer a closed loop on
# its own behaviour, and there is no measured upside: 0.001 and 0.0 were indistinguishable, so
# forbidding the endpoint gives nothing up.
EPSILON_HARD_FLOOR = 1e-4


def trailing_mean(eval_rows, key, current, window, scale=1.0):
    """Mean of `key` over the last `window` evals, counting this one as `current`.

    Divides by however many evals exist when there are fewer than `window`, not by `window`. The
    other way a fresh run reads near zero for its first evals, which pins epsilon at the ceiling
    exactly when it should be descending.

    `scale` divides the *stored* rows only, for the one field whose stored and live units differ.
    """
    history = [row.get(key, 0) / scale for row in eval_rows[-(window - 1):]] if window > 1 else []
    history.append(current)
    return sum(history) / len(history)


def trailing_perfect_rate(eval_rows, current_percent, window=REFINE_TRAILING_WINDOW):
    """The refinement phase's skill signal, as a fraction.

    Rows store `perfect_percent` on 0-100 and the live value arrives as a 0-1 fraction. Reconciled
    here rather than at the call site, because getting it wrong scales epsilon by 100.
    """
    return trailing_mean(eval_rows, 'perfect_percent', current_percent, window, scale=100.0)


def trailing_reward(eval_rows, current_reward, window=BOOTSTRAP_TRAILING_WINDOW):
    """The bootstrap phase's signal: mean `avg_reward` over a short window.

    A damper, not a ratchet — the phase may still raise epsilon when an arm genuinely regresses.
    """
    return trailing_mean(eval_rows, 'avg_reward', current_reward, window)


def bootstrap_epsilon(avg_reward, initial_epsilon):
    """Phase 1: halve epsilon as `avg_reward` clears each threshold, then stand down.

    **Returns 0.0 to mean "this phase has nothing to say", not "epsilon is 0".** `epsilon_for` takes
    the max of the two phases, so standing down hands control to refinement instead of pinning
    epsilon here.

    Driven by `avg_reward` rather than by the perfect rate because score rises 0 to 70 in the first
    ~13k steps and this phase is calibrated to that stretch, where the perfect rate is still flat 0
    and carries no signal at all.
    """
    for index, threshold in enumerate(BOOTSTRAP_REWARD_THRESHOLDS):
        if avg_reward <= threshold:
            return initial_epsilon / (2.0 ** index)
    return 0.0


def refine_epsilon(perfect_rate, top, floor, perfect_target=REFINE_PERFECT_TARGET):
    """Phase 2: geometric interpolation from `top` at 0% perfect to `floor` at `perfect_target`.

    **Geometric, not linear**, because the useful range spans more than an order of magnitude and
    equal *ratios* are what matter to an exploration rate: 0.05 -> 0.025 changes behaviour as much as
    0.004 -> 0.002. A linear ramp sits above 0.02 for more than half its length, where a random move
    every ~40 steps wrecks the endgame, and then crosses the whole low range in its last few percent.
    """
    if top <= floor:
        return floor
    # Only the lower clamp is needed. A rate above `perfect_target` gives a fraction over 1, which
    # undershoots the floor and is caught by the max() — but a *negative* rate would give a negative
    # exponent and push epsilon above `top`, which nothing else guards.
    fraction = max(0.0, perfect_rate / perfect_target)
    return max(floor, top * (floor / top) ** fraction)


def epsilon_for(avg_reward, perfect_rate, initial_epsilon, min_epsilon):
    """The epsilon this eval implies.

    `max()` rather than `min()` because the phases must not fight: the bootstrap term is larger than
    refinement's ceiling while it is live and 0.0 once it is not, so the maximum is whichever phase
    is speaking. `min()` would jump straight to refinement's ceiling on the first eval, before the
    arm has played at all.
    """
    top = initial_epsilon / (2.0 ** BOOTSTRAP_RUNGS)
    return max(bootstrap_epsilon(avg_reward, initial_epsilon),
               refine_epsilon(perfect_rate, top, min_epsilon))


def guided_fraction_for(avg_reward, initial_epsilon, configured_fraction):
    """How much of exploration the shield covers, given where the schedule is.

    Zero while bootstrap is live, the configured value once it stands down. **The shield exists to
    make exploration survivable in the endgame, and during bootstrap there is no endgame to
    protect** — epsilon is 0.1-0.4, the snake is a few segments long, dying is cheap, and the deaths
    are the signal.

    Stateless like `epsilon_for`, so an arm that collapses far enough for bootstrap to re-arm loses
    the shield with it rather than latching. One rule, "shielded iff refining", instead of two that
    can disagree.
    """
    if bootstrap_epsilon(avg_reward, initial_epsilon) > 0.0:
        return 0.0
    return configured_fraction
