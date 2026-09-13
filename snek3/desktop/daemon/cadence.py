"""Wall-clock slots, so the two boxes and the page publish in a fixed order inside each ten minutes.

The laptop's scheduler publishes at minute 8 of each ten (:08, :18, ...), the desktop daemon does its
network half -- fetch the laptop's feeds, publish its status, rebuild the site -- at minute 9, and the
viewer page reloads its manifest at minute 0 (user, 2026-09-13). Before this each side counted seconds
from its own last publish, so the laptop's pictures could land just after the desktop had built the
site and wait a full cycle for the next build, and the page had to be reloaded by hand.

Stdlib only: the daemon runs on base python.
"""


def slot(now, period, offset):
    """The most recent wall-clock instant at or before `now` that is `offset` seconds into a `period`."""
    return now - ((now - offset) % period)


def due(last, now, period, offset=0):
    """Whether a slot boundary has passed since `last` (None: never published, so due now). With
    `period` <= 0 every call is due."""
    if period <= 0 or last is None:
        return True
    return slot(now, period, offset) > last
