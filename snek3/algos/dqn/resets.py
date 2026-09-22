"""Shrink-and-perturb resets (Schwarzer et al. 2023, "Bigger, Better, Faster"; Group D, row D1).

Every `interval` gradient steps the online network is pulled part-way back toward a fresh
initialisation: the **trunk** (every `hidden.*` linear) becomes `alpha * theta + (1 - alpha) *
theta_fresh`, and everything after it -- the Q or distributional **head**, IQN's cosine embedding,
FQF's fraction proposal -- is **re-initialised outright**. The target network is then a copy of the
reset online network and every optimiser's state is cleared, so the moments of the old weights do not
steer the new ones. BBF's argument is that a value net trained for a long time on a replay buffer
loses plasticity and drifts; the reset trades a short re-learning dip for a net that can move again.

Here the reset is a **knob on every value agent**, not an algorithm: `SNEK_RESET_INTERVAL` in
gradient steps (0, the default, is off), `SNEK_RESET_ALPHA` (0.5, the paper's), and
`SNEK_RESET_STOP_AFTER`, the gradient step after which no reset fires (0 never stops; the paper keeps
the last 100k of 200k gradient steps reset-free so the final net has settled). The row that motivates
it is the late drift every Group A cell showed after reaching 88-94%, so the probe is the reset
schedule on A6's best cell at A's cap, judged on hold and drawdowns.

The fresh network is built by the same builder that built the agent's, from a seed derived from the
agent's own and the reset count, so two arms with the same `SNEK_SEED` reset to the same weights.

`partition(net)` is the one place that says what is trunk and what is head, by parameter name: a name
containing `hidden.` is the trunk (`QNet.hidden`, and `qnet.hidden` inside every distributional net);
every other parameter is head-side. Buffers (C51's support, IQN's harmonics) are constants and are
not touched.

**The within-cycle anneal** (`CycleSchedule`, 2026-09-20) is BBF's companion to the reset: after every
reset the n-step horizon and the discount restart at a short, myopic setting and move back to the long
one over `steps` gradient steps -- n from 10 to 3 and gamma from 0.97 to 0.997 in the paper, both on an
exponential schedule (Dopamine's `exponential_decay_scheduler`; gamma interpolated in `log(1 - gamma)`).
A freshly reset head relearns from a target it can fit, then the horizon lengthens again. Off unless
both `SNEK_RESET_ANNEAL_N_STEP` and `SNEK_RESET_ANNEAL_GAMMA` are set, and inert unless resets are on:
with no reset the cycle never restarts, so the schedule sits at its final values from step `steps` on
(the first cycle begins at gradient step 0, as BBF's does).
"""

import math

import torch

TRUNK_MARKER = 'hidden.'


class ResetSchedule(object):
    """When a reset is due, in gradient steps. `interval <= 0` never fires."""

    def __init__(self, interval=0, alpha=0.5, stop_after=0):
        self.interval = int(interval)
        self.alpha = float(alpha)
        self.stop_after = int(stop_after)
        if self.interval < 0:
            raise ValueError('reset_interval must be >= 0 (0 is off), got {0}'.format(interval))
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError('reset_alpha must be in [0, 1], got {0}'.format(alpha))
        if self.stop_after < 0:
            raise ValueError('reset_stop_after must be >= 0 (0 never stops), got {0}'.format(stop_after))

    @property
    def enabled(self):
        return self.interval > 0

    def due(self, train_step):
        """True on the gradient steps that are whole multiples of the interval, and not past the stop."""
        if not self.enabled or train_step <= 0 or train_step % self.interval:
            return False
        return self.stop_after <= 0 or train_step <= self.stop_after

    def describe(self):
        if not self.enabled:
            return ''
        text = 'resets every {0:,} gradient steps at alpha {1}'.format(self.interval, self.alpha)
        if self.stop_after > 0:
            text += ', none after {0:,}'.format(self.stop_after)
        return text


class CycleSchedule(object):
    """n-step and gamma as functions of the gradient steps since the last reset. Both exponential:
    `n(p) = round(n0 * (n1 / n0) ** p)`, `gamma(p) = 1 - exp((1 - p) log(1 - g0) + p log(1 - g1))`, with
    `p = min(1, steps_since_reset / steps)`. `enabled` is False when either pair is None, and then
    `n_step_at` / `gamma_at` return the constants they were given (`n_step`, `gamma`)."""

    def __init__(self, n_steps=None, gammas=None, steps=10000, n_step=1, gamma=0.99):
        self.n_steps = None if n_steps is None else (int(n_steps[0]), int(n_steps[1]))
        self.gammas = None if gammas is None else (float(gammas[0]), float(gammas[1]))
        self.steps = int(steps)
        self.n_step, self.gamma = int(n_step), float(gamma)
        if (self.n_steps is None) != (self.gammas is None):
            raise ValueError('reset_anneal needs both SNEK_RESET_ANNEAL_N_STEP and SNEK_RESET_ANNEAL_GAMMA, or neither')
        if self.enabled:
            if self.steps < 1:
                raise ValueError('reset_anneal_steps must be >= 1, got {0}'.format(steps))
            if min(self.n_steps) < 1:
                raise ValueError('reset_anneal_n_step values must be >= 1, got {0}'.format(n_steps))
            if not all(0.0 < g < 1.0 for g in self.gammas):
                raise ValueError('reset_anneal_gamma values must be in (0, 1), got {0}'.format(gammas))

    @property
    def enabled(self):
        return self.n_steps is not None

    def progress(self, since_reset):
        return min(1.0, max(0.0, float(since_reset) / float(self.steps)))

    def n_step_at(self, since_reset):
        if not self.enabled:
            return self.n_step
        n0, n1 = self.n_steps
        return max(1, int(round(n0 * (n1 / n0) ** self.progress(since_reset))))

    def gamma_at(self, since_reset):
        if not self.enabled:
            return self.gamma
        g0, g1 = self.gammas
        p = self.progress(since_reset)
        return 1.0 - math.exp((1.0 - p) * math.log(1.0 - g0) + p * math.log(1.0 - g1))

    def describe(self):
        if not self.enabled:
            return ''
        return 'n-step {0} -> {1} and gamma {2} -> {3} over {4:,} gradient steps after each reset'.format(
            self.n_steps[0], self.n_steps[1], self.gammas[0], self.gammas[1], self.steps)


def parse_pair(text, cast, knob):
    """`'10,3'` -> `(10, 3)`; empty or None -> None (off). Anything else names the knob."""
    if text is None or str(text).strip() == '':
        return None
    parts = [part.strip() for part in str(text).split(',')]
    if len(parts) != 2:
        raise ValueError('SNEK_{0}={1!r} must be two comma-separated values, start,end'.format(knob, text))
    return cast(parts[0]), cast(parts[1])


def anneal_config(tuned):
    """The three anneal knobs as config keys (each its `SNEK_` name lowercased), read through `tuned`."""
    return {
        'reset_anneal_n_step': str(tuned('RESET_ANNEAL_N_STEP', '', str)).strip(),
        'reset_anneal_gamma': str(tuned('RESET_ANNEAL_GAMMA', '', str)).strip(),
        'reset_anneal_steps': int(tuned('RESET_ANNEAL_STEPS', 10000, int)),
    }


def cycle_from_config(config):
    """A `CycleSchedule` from the config keys `anneal_config` wrote plus the run's `n_step_update` and
    `discount` as the constants. Validates, so a bad value names its knob before anything is built."""
    return CycleSchedule(parse_pair(config.get('reset_anneal_n_step'), int, 'RESET_ANNEAL_N_STEP'),
                         parse_pair(config.get('reset_anneal_gamma'), float, 'RESET_ANNEAL_GAMMA'),
                         steps=config.get('reset_anneal_steps', 10000),
                         n_step=config.get('n_step_update', 1), gamma=config.get('discount', 0.99))


def partition(net):
    """`(trunk_names, head_names)` over `net.named_parameters()`; the trunk is every `hidden.` linear."""
    trunk, head = [], []
    for name, _ in net.named_parameters():
        (trunk if TRUNK_MARKER in name else head).append(name)
    if not trunk or not head:
        raise ValueError('a resettable net needs both a hidden stack and a head; got trunk {0} head {1}'.format(
            trunk, head))
    return trunk, head


def shrink_and_perturb(net, fresh, alpha):
    """In place: the trunk interpolated toward `fresh` by `alpha`, the head replaced by `fresh`'s.

    `alpha = 1` leaves the trunk as it was and still re-initialises the head; `alpha = 0` is a full
    re-initialisation from `fresh`. Returns `(trunk_names, head_names)` for the caller's log.
    """
    trunk, head = partition(net)
    fresh_params = dict(fresh.named_parameters())
    alpha = float(alpha)
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            new = fresh_params[name].to(parameter.device)
            if name in head:
                parameter.copy_(new)
            else:
                parameter.mul_(alpha).add_(new, alpha=1.0 - alpha)
    return trunk, head


def clear_optimizer(optimizer):
    """Drops every moment and step count, so the reset weights start from Adam's (or RMSProp's) zero."""
    optimizer.state.clear()


def reset_seed(seed, count):
    """The seed the `count`-th fresh network is built from; None stays None (torch's global RNG)."""
    if seed is None:
        return None
    return (int(seed) * 1000003 + int(count) * 7919 + 101) % (2 ** 63 - 1)
