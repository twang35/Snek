"""Architecture sidecar for a saved policy — ``arch.json``, one per policy directory.

A TF checkpoint stores variable *values*, never the architecture that produced them, and
``restore(...).expect_partial()`` populates whatever layers it can match and stays silent about the
rest. So a checkpoint trained with ``FC_LAYERS=100,200,100`` restored into a ``50,100,50`` net loads
without a word and plays like a beginner — the failure this repo has been bitten by more than once
(see CLAUDE.md's hall-of-fame section and ``eval_agent.py``).

``arch.json`` closes that. It records the few facts that decide whether restored weights *mean* what
the network thinks they mean — the FC layer widths, the action count, and the observation vector's
length and meaning-era — written once when a policy is created and read by everything that restores
it (training resume, ``eval_checkpoints``, ``eval_workers``, ``watch.py``). A restore *requires* the
file and fails loudly on any mismatch, rather than half-loading.

It lives beside the checkpoints (``savedPolicies/<policy>/arch.json``), not inside them, so it is
written once rather than copied into all ~10000 checkpoints, and it travels with the weights when a
checkpoint is copied into ``hallOfFame/`` or rsynced to the desktop.
"""
import json
import os

ARCH_FILENAME = 'arch.json'


class ArchMismatch(Exception):
    """A restore's environment or config disagrees with the recorded architecture.

    Loud on purpose: the whole point of the sidecar is that this can no longer pass silently.
    """


def arch_path(policy_dir):
    return os.path.join(policy_dir, ARCH_FILENAME)


def build_arch(fc_layer_params, num_actions, obs_len, obs_era):
    """The canonical dict.

    ``fc_layer_params`` is stored as a list — JSON has no tuples and every reader iterates it, so
    the round trip compares list-to-list.
    """
    return {
        'fc_layer_params': [int(width) for width in fc_layer_params],
        'num_actions': int(num_actions),
        'obs_len': int(obs_len),
        'obs_era': str(obs_era),
    }


def read_arch(policy_dir):
    """The recorded arch, or ``None`` if the policy has no sidecar (a pre-``arch.json`` dir)."""
    path = arch_path(policy_dir)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def write_arch(policy_dir, arch):
    """Write ``arch.json`` atomically.

    The dir is normally the checkpointer's own ``ckpt_dir``, created before this runs; making it
    here too keeps a fresh run that writes arch before its first checkpoint safe.
    """
    os.makedirs(policy_dir, exist_ok=True)
    path = arch_path(policy_dir)
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(arch, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def require_arch(policy_dir):
    """The recorded arch, or a loud ``ArchMismatch`` if the sidecar is missing."""
    arch = read_arch(policy_dir)
    if arch is None:
        raise ArchMismatch(
            'no {0} in {1}: this policy predates the architecture sidecar, or the sidecar was not '
            'copied with the checkpoint. Run backfill_arch.py (or write one) before restoring.'
            .format(ARCH_FILENAME, policy_dir))
    return arch


def assert_restorable(policy_dir, num_actions, obs_len, obs_era):
    """Require ``arch.json`` and check every field that decides weight shape or meaning against the
    live environment.

    Returns the arch so the caller builds the network from ``arch['fc_layer_params']`` — the
    recorded shape is authoritative, which is what removes the reliance on ``SNEK_FC_LAYERS`` being
    set correctly at eval time.
    """
    arch = require_arch(policy_dir)
    problems = []
    if arch.get('num_actions') != int(num_actions):
        problems.append('num_actions {0} != env {1}'.format(arch.get('num_actions'), num_actions))
    if arch.get('obs_len') != int(obs_len):
        problems.append('obs_len {0} != env {1} (observation vector changed length)'
                        .format(arch.get('obs_len'), obs_len))
    if arch.get('obs_era') != str(obs_era):
        problems.append('obs_era {0!r} != env {1!r} (observation meaning changed at constant '
                        'length)'.format(arch.get('obs_era'), obs_era))
    if problems:
        raise ArchMismatch('{0} in {1} does not match this environment: {2}'
                           .format(ARCH_FILENAME, policy_dir, '; '.join(problems)))
    return arch


def assert_config_matches(policy_dir, fc_layer_params):
    """On a training *resume*, the checkpoint's arch is authoritative and the env's
    ``SNEK_FC_LAYERS`` is the thing that might be wrong.

    Fail loudly if they disagree — a resume under a changed knob is how a near-record arm was almost
    reverted for a loss (CLAUDE.md). Returns the arch on success.
    """
    arch = require_arch(policy_dir)
    recorded = [int(width) for width in arch['fc_layer_params']]
    wanted = [int(width) for width in fc_layer_params]
    if recorded != wanted:
        raise ArchMismatch(
            'resuming {0}: recorded fc_layer_params {1} != SNEK_FC_LAYERS {2}. The checkpoint was '
            'trained with the recorded shape; drop the override or resume a matching run.'
            .format(policy_dir, recorded, wanted))
    return arch
