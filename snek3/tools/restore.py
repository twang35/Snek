"""Policy directory + step -> a `policy_fn` the measurement engine can drive.

The one place that puts `arch.json`, a checkpoint and a network together, so the three checks that
have to happen on a restore happen once rather than in each caller:

1. the sidecar matches the **live environment** — `obs_len`, `obs_era`, `num_actions`;
2. the `state_dict` matches the network built from that sidecar, via torch's `strict=True`;
3. the checkpoint's recorded step matches its filename.

Miss the first and weights load cleanly and play badly, which is snek2's 2026-08-02 bite. Miss the
second and a shape mismatch half-loads. Miss the third and a correct measurement is reported against
the wrong step.
"""

import os

from env import constants
from tools import arch as arch_tools
from tools import checkpoints


def _dqn():
    # Imported here, not at module scope, so `tools/` keeps working for anything that only wants the
    # sidecar or the checkpoint listing. It also means an unknown `algo` fails before torch loads.
    from dqn import net as network
    return network


# `algo` from the sidecar decides both the network and how a greedy action is taken from it. A dict
# rather than an `if`, so adding PPO is one line and an unrecognised value names itself in the error
# instead of falling through to a default — which is how snek2 would have watched a c51 checkpoint
# as a scalar one.
ALGORITHMS = {'dqn': _dqn}


def _module_for(arch):
    algo = arch['algo']
    if algo not in ALGORITHMS:
        raise arch_tools.ArchMismatch('unknown algo {0!r}; this build knows {1}'.format(
            algo, sorted(ALGORITHMS)))
    return ALGORITHMS[algo]()


def build_net(arch, device='cpu'):
    return _module_for(arch).build(arch, device=device)


def policy_fn_for(arch, net, device='cpu'):
    """A greedy `policy_fn` over `net`.

    Separate from `build_net` because a caller that follows a live arm — `watch.py` — reloads weights
    into the *same* net between episodes and must keep the same callable.
    """
    return _module_for(arch).greedy_policy_fn(net, device=device)


def policy_arch(policy_dir):
    """The sidecar, checked against the live env. Cheap, and needs no torch."""
    return arch_tools.assert_restorable(policy_dir, constants.OBS_LEN, constants.OBS_ERA,
                                        constants.NUM_ACTIONS)


def restore(policy_dir, step=None, device='cpu'):
    """`(policy_fn, arch, step)` for one checkpoint. `step=None` takes the newest.

    The newest rather than the best: choosing *which* checkpoint to measure is the eval plan's job,
    and a default that silently picked a good one would make every measurement a selected high.
    """
    arch = policy_arch(policy_dir)
    if step is None:
        step = checkpoints.latest_step(policy_dir)
        if step is None:
            raise checkpoints.CheckpointError('no checkpoints in {0}'.format(policy_dir))
    net = build_net(arch, device=device)
    checkpoints.load(checkpoints.path(policy_dir, step), net, device=device)
    return policy_fn_for(arch, net, device=device), arch, int(step)


def policy_dir(policy):
    """`savedPolicies/<policy>`, or `policy` itself if it is already a directory that has a sidecar.

    Both spellings are used constantly — a bare arm name from a batch spec, and a path to a
    `hallOfFame/` entry that lives outside `savedPolicies/`.
    """
    if os.path.exists(arch_tools.arch_path(policy)):
        return policy
    return os.path.join(constants.POLICY_DIR, policy)
