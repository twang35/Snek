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

So batch p1 runs a **constant** coefficient: `initial == final` is the default and the anneal is off.
The linear ramp exists so p0 can ask whether a fixed coefficient is the binding constraint without a
code change, and an adaptive version is in the plan's backlog for if it is.
"""


def entropy_coef_for(step, max_steps, initial, final=None):
    """The coefficient at `step`, ramping linearly from `initial` to `final` across the cap.

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
