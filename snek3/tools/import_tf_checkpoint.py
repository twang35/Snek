"""Converts snek2 TF checkpoints into snek3 torch checkpoints.

    PYTHONPATH=. python tools/import_tf_checkpoint.py <snek2-policy-dir> <snek3-policy-dir> \\
        [--step N | --steps <file> | --all]

    # one champion
    PYTHONPATH=. python tools/import_tf_checkpoint.py \\
        ../snek2/hallOfFame/b44a-lowlr7-b29b-ckpt2739000 savedPolicies/b44a-import

    # a whole arm, for an A/B against snek2's own result file
    PYTHONPATH=. python tools/import_tf_checkpoint.py \\
        ../snek2/savedPolicies/b45a-lowlr8-b29b savedPolicies/b45a-import --all

**Why this is worth its length.** Converting one champion validated the observation vector, the
scalar env, the vectorised env, the policy path and the measurement engine against a number snek2
had already measured — before any snek3 training code existed. Converting a whole arm extends that
to the eval wave, row for row, against a file produced by an independent stack.

**The conversion itself is a transpose.** snek2's network is Keras `Dense` layers, whose kernel is
`(in, out)`; torch's `nn.Linear` holds `(out, in)`. Nothing else changes: the layers are
`relu`-activated with a bare linear head, no normalisation, no dropout, and `dqn/net.py`
reimplements even the initialisers, so the two networks are the same function of the same weights.

**It takes two interpreters.** snek3's env has no TensorFlow and snek2's has no torch, so the read
half runs as a subprocess under the `snek` env (`tools/tf_export.py`) and this half does the rest.
Point `SNEK2_PYTHON` at a different interpreter if that path moves. The export reads every requested
checkpoint in *one* process, because TensorFlow's own import is ~5 s and would otherwise dominate.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from env import constants
from tools import arch as arch_tools
from tools import checkpoints
from tools import restore

SNEK2_PYTHON = os.environ.get('SNEK2_PYTHON', '/opt/miniconda3/envs/snek/bin/python')
EXPORTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tf_export.py')


def tf_steps(policy_dir):
    """Checkpoint steps in a snek2 policy directory, ascending.

    Read from the filenames rather than from `tf.train.latest_checkpoint`, which returns None for a
    `hallOfFame/` entry — those hold a single checkpoint with no state file beside it — and would
    then fail a restore silently.
    """
    if not os.path.isdir(policy_dir):
        raise SystemExit('no such directory: {0}'.format(policy_dir))
    present = sorted(int(name[len('ckpt-'):-len('.index')]) for name in os.listdir(policy_dir)
                     if name.startswith('ckpt-') and name.endswith('.index'))
    if not present:
        raise SystemExit('no ckpt-*.index in {0}'.format(policy_dir))
    return present


def export_weights(policy_dir, steps, destination, step_list_path=None):
    """Runs the exporter under the `snek` env and returns the loaded arrays."""
    if not os.path.exists(SNEK2_PYTHON):
        raise SystemExit('no interpreter at {0}. TensorFlow lives in the `snek` env; set '
                         'SNEK2_PYTHON if it has moved'.format(SNEK2_PYTHON))
    command = [SNEK2_PYTHON, EXPORTER, policy_dir, destination]
    if step_list_path:
        command.append(step_list_path)
    # Streamed rather than captured: a 3,554-checkpoint export runs for minutes and its progress is
    # the only sign it is alive.
    result = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        raise SystemExit('the TensorFlow export failed')
    return np.load(destination)


def state_dict_from(arrays):
    """A torch `state_dict` for `dqn.net.QNet`, and the `fc_layer_params` it implies.

    The last layer is the head; everything before it is hidden. That is the network's own definition,
    not an assumption about widths — snek2's `Sequential` is
    `[dense(w) for w in fc_layer_params] + [q_values_layer]`.
    """
    layers = int(arrays['layers'])
    kernels = [arrays['kernel{0}'.format(i)] for i in range(layers)]
    biases = [arrays['bias{0}'.format(i)] for i in range(layers)]

    for index in range(layers - 1):
        if kernels[index].shape[1] != kernels[index + 1].shape[0]:
            raise SystemExit('layers {0} and {1} do not compose: {2} then {3}'.format(
                index, index + 1, kernels[index].shape, kernels[index + 1].shape))

    state = {}
    for index in range(layers - 1):
        # Transposed: Keras kernels are (in, out), torch Linear weights are (out, in).
        state['hidden.{0}.weight'.format(index)] = torch.from_numpy(kernels[index].T.copy())
        state['hidden.{0}.bias'.format(index)] = torch.from_numpy(biases[index].copy())
    state['head.weight'] = torch.from_numpy(kernels[-1].T.copy())
    state['head.bias'] = torch.from_numpy(biases[-1].copy())

    fc_layer_params = [kernel.shape[1] for kernel in kernels[:-1]]
    return state, fc_layer_params, kernels[0].shape[0], kernels[-1].shape[1]


def slice_for_step(bundle, step):
    """One checkpoint's arrays out of the namespaced bundle `tf_export.py` writes."""
    prefix = '{0}/'.format(step)
    return {key[len(prefix):]: bundle[key] for key in bundle.files if key.startswith(prefix)}


def source_arch(policy_dir):
    """snek2's own `arch.json`. Every policy directory and hall-of-fame entry carries one."""
    path = os.path.join(policy_dir, 'arch.json')
    if not os.path.exists(path):
        raise SystemExit(
            'no arch.json in {0}. obs_era cannot be inferred from weights, and a checkpoint whose '
            'observation era is unknown is not safe to measure'.format(policy_dir))
    with open(path) as handle:
        return json.load(handle)


def check_against(recorded, fc_layer_params, obs_len, num_actions, policy_dir):
    """The sidecar is checked against the weights and against the live env, never trusted.

    `obs_era` is the field that cannot be derived from weights and the one that matters: the length
    and the action count are unchanged when indices are repurposed, so torch would load such a
    checkpoint without a word and it would play like a beginner.
    """
    derived = {'obs_len': obs_len, 'num_actions': num_actions,
               'fc_layer_params': fc_layer_params}
    mismatches = []
    for field, value in sorted(derived.items()):
        was = recorded.get(field)
        if isinstance(value, list):
            was = list(was) if was is not None else None
        if was != value:
            mismatches.append('{0}: sidecar {1!r}, weights {2!r}'.format(field, was, value))
    if mismatches:
        raise SystemExit('{0}/arch.json disagrees with its own weights: {1}'.format(
            policy_dir, '; '.join(mismatches)))
    if obs_len != constants.OBS_LEN or num_actions != constants.NUM_ACTIONS:
        raise SystemExit('the checkpoint is {0}->{1}, this env is {2}->{3}'.format(
            obs_len, num_actions, constants.OBS_LEN, constants.NUM_ACTIONS))
    if recorded['obs_era'] != constants.OBS_ERA:
        raise SystemExit(
            "obs_era {0!r} != this env's {1!r}. The observation is the same length but its indices "
            'no longer mean the same thing, so these weights would load cleanly and play '
            'badly'.format(recorded['obs_era'], constants.OBS_ERA))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('source', help='a snek2 policy or hallOfFame directory')
    parser.add_argument('destination', help='the snek3 policy directory to write')
    parser.add_argument('--step', type=int, default=None, help='one checkpoint')
    parser.add_argument('--steps', default=None, help='a file of steps, one per line')
    parser.add_argument('--all', action='store_true', help='every checkpoint in the source')
    parser.add_argument('--force', action='store_true',
                        help='reconvert steps already present in the destination')
    args = parser.parse_args(argv)

    present = tf_steps(args.source)
    if args.step is not None:
        wanted = [args.step]
    elif args.steps:
        with open(args.steps) as handle:
            wanted = sorted({int(line.split('#')[0].strip()) for line in handle
                             if line.split('#')[0].strip()})
    elif args.all:
        wanted = present
    elif len(present) == 1:
        wanted = present
    else:
        raise SystemExit('{0} holds {1} checkpoints ({2}...{3}); name them with --step, --steps or '
                         '--all'.format(args.source, len(present), present[0], present[-1]))
    unknown = sorted(set(wanted) - set(present))
    if unknown:
        raise SystemExit('not in {0}: {1}'.format(args.source, unknown[:8]))

    already = set() if args.force else set(checkpoints.steps(args.destination))
    todo = [step for step in wanted if step not in already]
    if not todo:
        print('all {0} step(s) already converted in {1}'.format(len(wanted), args.destination))
        return 0
    print('converting {0} of {1} step(s) from {2}'.format(len(todo), len(wanted), args.source))
    # Flushed before the subprocess writes to the same stdout, or the export's progress lines appear
    # above everything printed here.
    sys.stdout.flush()

    recorded = source_arch(args.source)
    started = time.time()
    with tempfile.TemporaryDirectory() as scratch:
        listing = os.path.join(scratch, 'steps.txt')
        with open(listing, 'w') as handle:
            handle.write('\n'.join(str(step) for step in todo) + '\n')
        bundle = export_weights(args.source, todo, os.path.join(scratch, 'weights.npz'), listing)

        arch = None
        for number, step in enumerate(todo, 1):
            state, fc_layer_params, obs_len, num_actions = state_dict_from(
                slice_for_step(bundle, step))
            if arch is None:
                check_against(recorded, fc_layer_params, obs_len, num_actions, args.source)
                arch = arch_tools.build_arch(fc_layer_params, num_actions, obs_len,
                                             recorded['obs_era'], algo='dqn')
                arch_tools.write_arch(args.destination, arch)
                net = restore.build_net(arch)
            else:
                # Every checkpoint of an arm shares one architecture, so a shape change mid-arm is a
                # corrupt export rather than something to accommodate — `strict=True` catches it.
                arch_tools.assert_same_network(
                    arch, arch_tools.build_arch(fc_layer_params, num_actions, obs_len,
                                                recorded['obs_era'], algo='dqn'),
                    args.destination, 'step {0}'.format(step))
            net.load_state_dict(state, strict=True)
            checkpoints.save(args.destination, step, net,
                             extra={'imported_from': os.path.join(args.source,
                                                                  'ckpt-{0}'.format(step))})
            if number == 1 or number % 250 == 0 or number == len(todo):
                print('  wrote {0}/{1} (step {2})'.format(number, len(todo), step))

    print('{0}: {1} checkpoint(s), fc {2}, era {3}, {4:.1f}s'.format(
        args.destination, len(checkpoints.steps(args.destination)), arch['fc_layer_params'],
        arch['obs_era'], time.time() - started))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
