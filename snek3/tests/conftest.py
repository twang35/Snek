"""Suite-wide defaults. Both are about side effects rather than values.

**Every default path points at a scratch runs directory, never the real one.** Every tool resolves
`runs_dir or constants.RUNS_DIR` at call time, so a test that omits `runs_dir` -- a `Reporter` built
without one, a `main()` called for its argument parsing -- reaches the box's live `runs/.live/`. On
2026-09-05 a pytest run did exactly that while b16's fifth wave trained: `scheduler.main()` read the live
status file, killed the live chart window through `kill_stale`, and a `Reporter.publish(None)` overwrote
the live status with an empty one. The window came back only by hand. `SNEK_RUNS_DIR` cannot do this job
from here, because `env.constants` has read it by the time a fixture runs; patching the attribute can.

**No test opens a real chart window.** `SNEK_CHART_WINDOW=0` is the switch `tools/window.py` reads,
the same one a benchmark and a headless box use.
"""

import pytest

from env import constants


@pytest.fixture(autouse=True)
def isolated_runs_dir(tmp_path_factory, monkeypatch):
    # Its own directory, not under the test's `tmp_path`: tests list that one and mkdir `runs` in it.
    runs = tmp_path_factory.mktemp('isolated-runs')
    monkeypatch.setattr(constants, 'RUNS_DIR', str(runs))
    return str(runs)


@pytest.fixture(autouse=True)
def no_chart_window(monkeypatch):
    monkeypatch.setenv('SNEK_CHART_WINDOW', '0')
