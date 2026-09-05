"""Suite-wide defaults.

One thing, and it is about a side effect rather than a value: a test that runs the scheduler would
otherwise open a real chart window on the developer's screen. `SNEK_CHART_WINDOW=0` is the switch
`tools/window.py` reads, the same one a benchmark and a headless box use.
"""

import pytest


@pytest.fixture(autouse=True)
def no_chart_window(monkeypatch):
    monkeypatch.setenv('SNEK_CHART_WINDOW', '0')
