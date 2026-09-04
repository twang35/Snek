"""`SNEK_RUNS_DIR` (2026-09-03): the one knob that moves every run artifact, read once at import by
`env/constants.py`. The desktop daemon sets it to its gitignored `desktop/runs/`; the laptop leaves it
unset and keeps `runs/`, the archive master tracks."""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _runs_dir(extra_env):
    env = dict(os.environ)
    env.pop('SNEK_RUNS_DIR', None)
    env.update(extra_env)
    env['PYTHONPATH'] = ROOT
    return subprocess.run([sys.executable, '-c', 'from env import constants; print(constants.RUNS_DIR)'],
                          capture_output=True, text=True, check=True, env=env, cwd=ROOT).stdout.strip()


def test_unset_means_the_tracked_runs_directory():
    assert _runs_dir({}) == os.path.join(ROOT, 'runs')


def test_the_knob_moves_every_artifact_root(tmp_path):
    assert _runs_dir({'SNEK_RUNS_DIR': str(tmp_path / 'desktop' / 'runs')}) == str(tmp_path / 'desktop' / 'runs')


def test_an_empty_value_is_unset_not_the_cwd():
    assert _runs_dir({'SNEK_RUNS_DIR': ''}) == os.path.join(ROOT, 'runs')
