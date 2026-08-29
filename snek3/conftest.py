"""Puts the project root on `sys.path`, so `pytest` needs no `PYTHONPATH`.

pytest inserts the directory holding the rootdir `conftest.py` into `sys.path` under its default
`prepend` import mode, which is what makes `from env import ...` resolve. `tests/` deliberately has
no `__init__.py` for the same reason: without one, pytest also inserts `tests/` itself, so the
modules there can import each other by bare name.
"""

import os

# Set before any test module imports `env.game`, which is the only importer of pygame. The banner is
# a print, not a warning, so nothing else suppresses it.
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
