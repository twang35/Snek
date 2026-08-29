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

import glob
import os
import subprocess
import sys
import sysconfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = ('env', 'vectorized', 'dqn', 'ppo', 'tools')


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


def test_the_learning_code_never_reaches_pygame():
    """`dqn/` may import torch and may not import pygame.

    Caught a real violation: `dqn/agent.py` read the exploration shield's observation slice from
    `env.scalar_env.block_ranges()`, and `scalar_env` imports `env.game` and therefore pygame. The
    only reason anyone noticed was the SDL banner printing during a smoke run. The layout table moved
    to `env.constants`, which is pygame-free by rule, and this assertion exists so the next one is
    caught by the suite instead of by a banner.
    """
    pygame, torch = loaded_after_importing('dqn.agent', 'dqn.replay', 'dqn.schedules', 'dqn.net')
    assert not pygame, 'dqn/ reached pygame; a trainer must not open an audio device'
    assert torch, 'dqn/ is where torch belongs, so not reaching it means this probe is wrong'


def test_the_measurement_tools_never_reach_pygame():
    # `tools/` may import torch for checkpoint I/O, and several of these run as eval shard
    # subprocesses where a CoreAudio stream per process was a measured 15% of a core.
    pygame, _ = loaded_after_importing('tools.results', 'tools.shard', 'tools.eval_wave',
                                       'tools.stage_b_chart', 'tools.step_selectors')
    assert not pygame, 'tools/ reached pygame'


def test_the_probe_would_notice_pygame():
    # A fixture whose subject cannot violate it is not a fixture: this proves the three assertions
    # above are testing something, by importing the one module that *does* pull pygame in.
    pygame, _ = loaded_after_importing('env.game')
    assert pygame, 'the probe cannot see pygame, so the assertions above prove nothing'


# ------------------------------------------------------------- shadowing the standard library

def project_modules():
    """Every module in the tree, as `(import name, path)`, excluding tests."""
    found = []
    for path in sorted(glob.glob(os.path.join(ROOT, '*.py'))):
        found.append((os.path.basename(path)[:-3], path))
    for package in PACKAGES:
        for path in sorted(glob.glob(os.path.join(ROOT, package, '*.py'))):
            name = os.path.basename(path)[:-3]
            if name != '__init__':
                found.append((name, path))
    return found


def test_no_module_shadows_the_standard_library():
    """A module named after a stdlib one is loaded *instead of* it, and the error names neither.

    This has already cost a debugging session. `tools/selectors.py` shadowed the standard library's
    `selectors`, which `subprocess` imports — so running any script from inside `tools/`, where
    `sys.path[0]` is `tools/`, made `import subprocess` load the project's file, which imports torch,
    which imports `multiprocessing`, which imports `subprocess` again. The traceback reported a
    circular import inside `subprocess.py` and pointed nowhere near the actual file.

    Checked against the real stdlib listing rather than a hand-kept denylist, because the point is
    to catch the name nobody thought of.
    """
    stdlib = {name for name in sys.stdlib_module_names if not name.startswith('_')}
    # `sysconfig`'s stdlib directory catches submodule-level names an import could still resolve to.
    stdlib |= {os.path.basename(path)[:-3]
               for path in glob.glob(os.path.join(sysconfig.get_paths()['stdlib'], '*.py'))}

    offenders = [(name, os.path.relpath(path, ROOT))
                 for name, path in project_modules() if name in stdlib]
    assert not offenders, (
        'these modules shadow standard-library modules of the same name: {0}'.format(offenders))


def test_the_shadowing_check_would_have_caught_the_real_one():
    # A fixture whose subject cannot violate it is not a fixture: `selectors` is the name that
    # actually bit, so the check has to consider it a stdlib name.
    stdlib = {name for name in sys.stdlib_module_names if not name.startswith('_')}
    assert 'selectors' in stdlib and 'subprocess' in stdlib
