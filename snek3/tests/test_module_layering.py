"""The import layering from `CLAUDE.md`'s layout table, asserted rather than documented.

Two rules, each with a cost behind it.

**`vectorized/` must not reach pygame.** `pygame.init()` opens a real CoreAudio stream per process
and `SDL_VIDEODRIVER=dummy` does not affect audio, so in snek2 ten idle headless workers drove
`coreaudiod` to 15% CPU. An eval wave runs sixteen shards, so the fix is for the eval path to have
no pygame in it at all rather than to guard each entry point.

**`vectorized/` must not reach torch.** The engine's only seam is a `policy_fn` of shape
`(m, 30) float32 -> (m,) int64`, which is what lets the whole measurement stack be tested against a
hand-written heuristic. A tensor threaded through the env would also cost more than it bought: the
policy is 8% of a step and the numpy observation build is the rest.

These are import-time facts, so they need a subprocess — by the time this module runs, another test
has already imported `env.game` and pygame is in `sys.modules`.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def loaded_after_importing(*modules):
    """`sys.modules` keys of interest after importing `modules` in a clean interpreter."""
    program = ('import sys\n'
               + ''.join('import {0}\n'.format(name) for name in modules)
               + "print('pygame' in sys.modules, 'torch' in sys.modules)")
    result = subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    pygame, torch = result.stdout.split()
    return pygame == 'True', torch == 'True'


def test_the_vectorised_env_pulls_in_neither_pygame_nor_torch():
    pygame, torch = loaded_after_importing('vectorized.vec_env', 'vectorized.config')
    assert not pygame, 'vectorized/ reached pygame; an eval shard must not open an audio device'
    assert not torch, 'vectorized/ reached torch; the policy seam is a plain callable'


def test_the_measurement_engine_pulls_in_neither_either():
    pygame, torch = loaded_after_importing('vectorized.engine')
    assert not pygame and not torch


def test_the_game_rules_are_importable_without_a_display():
    # `env.constants` is what `vectorized/config.py` imports, so it carries the same rule even
    # though it sits inside `env/`. `env.render` is where the pygame constants live.
    pygame, torch = loaded_after_importing('env.constants', 'env.observations')
    assert not pygame, 'env.constants or env.observations reached pygame'
    assert not torch


def test_the_probe_would_notice_pygame():
    # A fixture whose subject cannot violate it is not a fixture: this proves the three assertions
    # above are testing something, by importing the one module that *does* pull pygame in.
    pygame, _ = loaded_after_importing('env.game')
    assert pygame, 'the probe cannot see pygame, so the assertions above prove nothing'
