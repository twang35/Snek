"""`arch.json` — the architecture sidecar, one per policy directory.

A checkpoint stores weight *values* and nothing about what they mean. Torch's `strict=True` catches
a shape mismatch, which is most of the problem, but not the one that actually cost this project a
day: on 2026-08-02 two observation indices were repurposed **at constant length**, so every champion
restored silently, matched every shape, and played like a beginner — 90.3% perfect became scores of
0, 0, 1. `obs_era` is the field that catches that, and it is the reason this file exists rather than
just relying on `load_state_dict`.

The sidecar lives beside the checkpoints (`savedPolicies/<policy>/arch.json`) rather than inside
each one, so it is written once instead of copied into thousands of files, and it travels with the
weights when a checkpoint is copied into `hallOfFame/` or rsynced to the desktop.

**Every field is required.** snek2's equivalent had to default three of them, because arms predating
each knob were still being restored; snek3 writes every sidecar itself, so a missing field is a bug
rather than an old file and is reported as one.
"""

import json
import os

ARCH_FILENAME = 'arch.json'

# The complete set, and the order they are written in. Also exactly what has to match for weights to
# load into an already-built network, which is the question an eval wave asks when it points one
# process at many policies.
FIELDS = ('algo', 'fc_layer_params', 'num_actions', 'obs_len', 'obs_era')


class ArchMismatch(Exception):
    """A restore's environment or config disagrees with the recorded architecture.

    Loud on purpose: the whole point of the sidecar is that this cannot pass silently.
    """


def arch_path(policy_dir):
    return os.path.join(policy_dir, ARCH_FILENAME)


def build_arch(fc_layer_params, num_actions, obs_len, obs_era, algo='dqn'):
    """The canonical dict.

    `fc_layer_params` is stored as a list of ints, because JSON has no tuples and every reader
    iterates it — so the round trip compares list to list rather than list to tuple.
    """
    return {'algo': str(algo),
            'fc_layer_params': [int(width) for width in fc_layer_params],
            'num_actions': int(num_actions),
            'obs_len': int(obs_len),
            'obs_era': str(obs_era)}


def write_arch(policy_dir, arch):
    """Writes the sidecar, refusing to overwrite a *different* one.

    Rewriting an identical arch is fine and happens on every resume. Rewriting a different one would
    mean the directory now holds checkpoints from two architectures, and the second half would load
    into the wrong network with no shape error — so that is an error, not an update.
    """
    missing = [field for field in FIELDS if field not in arch]
    if missing:
        raise ArchMismatch('refusing to write an incomplete {0}: missing {1}'.format(
            ARCH_FILENAME, ', '.join(missing)))
    path = arch_path(policy_dir)
    if os.path.exists(path):
        existing = read_arch(policy_dir)
        if signature(existing) != signature(arch):
            raise ArchMismatch(
                '{0} already records a different architecture: {1} on disk, {2} offered'.format(
                    path, signature(existing), signature(arch)))
        return path
    os.makedirs(policy_dir, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump({field: arch[field] for field in FIELDS}, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return path


def read_arch(policy_dir):
    """The sidecar, or raise. There is no "restore without one" path, by design."""
    path = arch_path(policy_dir)
    if not os.path.exists(path):
        raise ArchMismatch(
            'no {0} in {1}. A checkpoint cannot be restored without one — weights whose meaning is '
            'unknown are worse than no weights'.format(ARCH_FILENAME, policy_dir))
    with open(path) as handle:
        arch = json.load(handle)
    missing = [field for field in FIELDS if field not in arch]
    if missing:
        raise ArchMismatch('{0} is missing {1}'.format(path, ', '.join(missing)))
    return arch


def signature(arch):
    """A hashable form of the fields that decide whether weights can be restored.

    Lets a caller group policies into lanes with a dict rather than comparing pairwise.
    `fc_layer_params` is normalised to a tuple, or two identical arches would hash differently
    depending on which one came off disk.
    """
    return tuple((field, tuple(arch[field]) if isinstance(arch[field], list) else arch[field])
                 for field in FIELDS)


def assert_restorable(policy_dir, obs_len, obs_era, num_actions):
    """Check the sidecar against the live environment, and return it.

    The caller then builds its network from `arch['fc_layer_params']`, which is what makes the
    recorded shape authoritative instead of relying on an environment variable being set right at
    eval time — snek2's exact bite, from a shape that defaulted silently.
    """
    arch = read_arch(policy_dir)
    problems = []
    if arch['num_actions'] != int(num_actions):
        problems.append('num_actions {0} != env {1}'.format(arch['num_actions'], num_actions))
    if arch['obs_len'] != int(obs_len):
        problems.append('obs_len {0} != env {1} (the observation changed length)'.format(
            arch['obs_len'], obs_len))
    if arch['obs_era'] != str(obs_era):
        problems.append(
            'obs_era {0!r} != env {1!r}. The observation is the same length but its indices no '
            'longer mean the same thing, so these weights would load cleanly and play badly'.format(
                arch['obs_era'], obs_era))
    if problems:
        raise ArchMismatch('{0} does not match this environment: {1}'.format(
            arch_path(policy_dir), '; '.join(problems)))
    return arch


def assert_same_network(built, target, built_dir='<built>', target_dir='<target>'):
    """Refuse to load `target`'s weights into a network built for `built`.

    The question `assert_restorable` never has to ask, because it is called by a process that builds
    its network for one policy. An eval wave points one built network at many policies, so it needs
    this as well.
    """
    if signature(built) != signature(target):
        differing = [field for field in FIELDS if built[field] != target[field]]
        raise ArchMismatch(
            'cannot restore {0} into a network built for {1}: {2} differ ({3} vs {4})'.format(
                target_dir, built_dir, ', '.join(differing),
                [built[field] for field in differing], [target[field] for field in differing]))
