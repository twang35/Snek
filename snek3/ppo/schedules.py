"""The entropy coefficient, as a pure function of the arm's progress through its cap.

The same stateless shape as `dqn/schedules.py`, for the same two reasons: a resume recomputes the
value from the restored step instead of descending a ladder again, and a schedule with no state cannot
latch. `dqn/schedules.py` records what latching cost — a one-way ratchet pinned one snek2 arm's
epsilon at 0.001 while its score collapsed from 64.6 to 8.8.

**But it is driven by the step, not by the eval history, and that is the deliberate difference.** DQN's
epsilon is mastery-gated because it has to be: a fixed epsilon either wrecks the endgame or never
explores, and the schedule's whole job is to find the handover. PPO's entropy bonus is a *cost* term
rather than a forced random-action rate — the policy is free to be near-deterministic where it is
confident and stochastic where it is not — so there is no deadlock to gate against, and a mastery-gated
version would add a feedback loop through `perfect_percent`
([`../docs/invariants.md`](../docs/invariants.md) invariant 2) for no measured gain.

So batch b4 runs a **constant** coefficient: `initial == final` is the default and the anneal is off.
The linear ramp exists so b3 can ask whether a fixed coefficient is the binding constraint without a
code change, and an adaptive version is in the plan's backlog for if it is.
"""


def ramped(step, max_steps, initial, final=None):
    """A value at `step`, ramping linearly from `initial` to `final` across the cap.

    The one ramp every PPO schedule uses — the entropy coefficient, and since 2026-09-01 the clip and
    the learning rate (`SNEK_PPO_CLIP_FINAL`, `SNEK_PPO_LEARNING_RATE_FINAL`, the PPO paper's Atari
    recipe). One function rather than three copies so they cannot drift, and so a test of the clamp
    below covers all of them.

    `final is None` means "no anneal", which is not the same as `final == initial` only in intent —
    they compute the same number — but it is what lets `SNEK_PPO_ENTROPY_COEF_FINAL` be absent rather
    than duplicated in every launch.

    Clamped at both ends. **The upper clamp is the one that matters**: `SNEK_MAX_STEPS` is absolute
    and a resumed arm can be *past* its old cap while the new one is higher, so an unclamped fraction
    would run past 1 and push the coefficient beyond `final` — which for a descending ramp means a
    negative entropy bonus, an active push *toward* determinism, and a policy that collapses for a
    reason nothing in the config names.
    """
    initial = float(initial)
    if final is None:
        return initial
    final = float(final)
    span = max(1, int(max_steps))
    fraction = min(1.0, max(0.0, float(step) / span))
    return initial + fraction * (final - initial)


def entropy_coef_for(step, max_steps, initial, final=None):
    """The entropy coefficient at `step`. See `ramped`."""
    return ramped(step, max_steps, initial, final)


def clip_for(step, max_steps, initial, final=None):
    """The PPO clip at `step`. `final` must stay inside (0, 1) — `algo.build_config` refuses otherwise,
    because a clip of 0 admits no update at all and the ramp would silently end training early."""
    return ramped(step, max_steps, initial, final)


def learning_rate_for(step, max_steps, initial, final=None):
    """Adam's step size at `step`. `final` may be 0: the last stretch then takes no gradient steps,
    which is the Atari recipe's intent, and a resumed arm past its cap stays at the floor."""
    return ramped(step, max_steps, initial, final)
