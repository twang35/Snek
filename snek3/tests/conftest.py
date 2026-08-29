"""Suite-wide defaults.

One thing, and it is about a side effect rather than a value: a test that runs the training loop
would otherwise open a real chart window on the developer's screen, twice, because a training opens
one without being asked. `SNEK_CHART_WINDOW=0` is the switch the daemon uses for the same reason.
"""

import pytest


@pytest.fixture(autouse=True)
def no_chart_window(monkeypatch):
    monkeypatch.setenv('SNEK_CHART_WINDOW', '0')
