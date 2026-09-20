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
"""

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
