"""Set `SNEK_OBS_HISTORY` from a checkpoint's sidecar **before `env.constants` is imported**.

`env/constants.py` reads the observation knobs once, at import, and sizes the vector from them. A
checkpoint trained with move history (`SNEK_OBS_HISTORY=N`) therefore only loads in a process whose
environment says the same N -- and the tools that measure or watch a checkpoint are started by a
scheduler, a close-out or a human who should not have to know. The sidecar knows: its `obs_era` carries
the depth as a suffix (`obs26-20260907-hist4`). So every entry point that takes a policy on its command
line calls `adopt_from_argv(sys.argv)` first thing, and this module sets the variable from the era when
the environment has not set it already. It never overrides an explicit setting: a disagreement is
`tools/arch.py`'s to report, by name.

**stdlib only, imported before anything under `env/`, `vectorized/` or `tools/` that reaches
`env.constants`.** That ordering is the whole point, which is why this is not a function in
`tools/arch.py` (which imports `env.constants`).

Shared stage-A eval workers are the one path this does not cover: a worker serves every arm on the box
from one process, so it is given the wave's depth by whoever starts it (`tools/scheduler.py`
`wave_obs_history`, or a trainer's own environment).
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOB = 'SNEK_OBS_HISTORY'
_SUFFIX = re.compile(r'-hist(\d+)$')


def obs_history_of_era(era):
    """`obs26-20260907-hist4` -> 4; an era with no suffix -> 0."""
    match = _SUFFIX.search(str(era))
    return int(match.group(1)) if match else 0


def sidecar_path(candidate):
    """The `arch.json` a command-line token names, or None: a directory holding one, or a policy
    name under `savedPolicies/` or `hallOfFame/`."""
    for directory in (candidate, os.path.join(ROOT, 'savedPolicies', candidate),
                      os.path.join(ROOT, 'hallOfFame', candidate)):
        path = os.path.join(directory, 'arch.json')
        if os.path.isfile(path):
            return path
    return None


def adopt(policy_dir, environ=None):
    """Sets `SNEK_OBS_HISTORY` from `policy_dir`'s sidecar unless the environment already has a value.
    Returns the depth adopted, or None when nothing was read or the variable was already set."""
    environ = os.environ if environ is None else environ
    path = sidecar_path(policy_dir)
    if path is None or KNOB in environ:
        return None
    with open(path) as handle:
        depth = obs_history_of_era(json.load(handle).get('obs_era', ''))
    environ[KNOB] = str(depth)
    return depth


def adopt_from_argv(argv, environ=None):
    """`adopt` on the first command-line token that names a checkpoint directory. Flags and values
    that name no sidecar are skipped, so this is safe to call with the whole argv."""
    for token in argv[1:]:
        if token.startswith('-'):
            continue
        depth = adopt(token, environ=environ)
        if depth is not None:
            return depth
        if sidecar_path(token) is not None:
            return None                     # a sidecar was found but the environment already decides
    return None
