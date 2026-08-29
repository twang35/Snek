"""Converts a snek2 TF checkpoint into a snek3 torch checkpoint.

    python tools/import_tf_checkpoint.py <snek2-policy-dir> <snek3-policy-dir> [--step N]

    python tools/import_tf_checkpoint.py \\
        ../snek2/hallOfFame/b44a-lowlr7-b29b-ckpt2739000 savedPolicies/b44a-import

**Why this is worth its length.** Converting one champion validates the observation vector, the
scalar env, the vectorised env, the policy path and the whole measurement engine against a number
snek2 already measured over 3,000 episodes — before any snek3 training code exists. If the converted
policy scores what it scored in TensorFlow, everything under it is right.

**The conversion itself is a transpose.** snek2's network is Keras `Dense` layers, whose kernel is
`(in, out)`; torch's `nn.Linear` holds `(out, in)`. Nothing else changes: the layers are
`relu`-activated with a bare linear head, no normalisation, no dropout, and `dqn/net.py`
reimplements even the initialisers, so the two networks are the same function of the same weights.

**It takes two interpreters.** snek3's env has no TensorFlow and snek2's has no torch, so the read
half runs as a subprocess under the `snek` env (`tools/tf_export.py`) and this half does the rest.
Point `SNEK2_PYTHON` at a different interpreter if that path moves.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

from dqn import net as network
from env import constants
from tools import arch as arch_tools
from tools import checkpoints

SNEK2_PYTHON = os.environ.get('SNEK2_PYTHON', '/opt/miniconda3/envs/snek/bin/python')
EXPORTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tf_export.py')


def tf_checkpoint_prefix(policy_dir, step=None):
    """The `ckpt-<step>` prefix inside a snek2 policy directory.

    Named explicitly rather than found with `tf.train.latest_checkpoint`, which returns None for a
    `hallOfFame/` entry — those directories hold a single checkpoint with no `checkpoint` state file
    beside it, and the None then fails the restore silently.
    """
    present = sorted(int(name[len('ckpt-'):-len('.index')])
                     for name in os.listdir(policy_dir)
                     if name.startswith('ckpt-') and name.endswith('.index'))
    if not present:
        raise SystemExit('no ckpt-*.index in {0}'.format(policy_dir))
    if step is None:
        if len(present) > 1:
            raise SystemExit('{0} holds {1} checkpoints ({2}...{3}); name one with --step'.format(
                policy_dir, len(present), present[0], present[-1]))
        step = present[0]
    elif step not in present:
        raise SystemExit('no step {0} in {1}; present: {2}'.format(step, policy_dir, present))
    return os.path.join(policy_dir, 'ckpt-{0}'.format(step)), step


def export_weights(prefix, destination):
    """Runs the exporter under the `snek` env and returns the loaded arrays."""
    if not os.path.exists(SNEK2_PYTHON):
        raise SystemExit('no interpreter at {0}. TensorFlow lives in the `snek` env; set '
                         'SNEK2_PYTHON if it has moved'.format(SNEK2_PYTHON))
    result = subprocess.run([SNEK2_PYTHON, EXPORTER, prefix, destination],
                            capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit('the TensorFlow export failed')
    for line in result.stdout.splitlines():
        if line.startswith('layer '):
            print('  ' + line)
    return np.load(destination)


def state_dict_from(arrays):
    """A torch `state_dict` for `dqn.net.QNet`, and the `fc_layer_params` it implies.

    The last layer is the head; everything before it is hidden. That is the network's own
    definition, not an assumption about widths — snek2's `Sequential` is
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


def source_arch(policy_dir):
    """snek2's own `arch.json`, which every hall-of-fame entry carries."""
    path = os.path.join(policy_dir, 'arch.json')
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('source', help='a snek2 policy or hallOfFame directory')
    parser.add_argument('destination', help='the snek3 policy directory to write')
    parser.add_argument('--step', type=int, default=None,
                        help='which checkpoint, if the source holds several')
    args = parser.parse_args(argv)

    prefix, step = tf_checkpoint_prefix(args.source, args.step)
    print('reading {0}'.format(prefix))
    with tempfile.TemporaryDirectory() as scratch:
        arrays = export_weights(prefix, os.path.join(scratch, 'weights.npz'))
        state, fc_layer_params, obs_len, num_actions = state_dict_from(arrays)

    # The recorded arch is checked against the weights rather than trusted, because it is the field
    # that decides whether the observation still *means* what it did and a wrong one converts
    # silently. snek2's own sidecar is the only place `obs_era` can come from.
    recorded = source_arch(args.source)
    if recorded is None:
        raise SystemExit(
            'no arch.json in {0}. obs_era cannot be inferred from weights, and a checkpoint whose '
            'observation era is unknown is not safe to measure'.format(args.source))
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
            args.source, '; '.join(mismatches)))
    if obs_len != constants.OBS_LEN or num_actions != constants.NUM_ACTIONS:
        raise SystemExit('the checkpoint is {0}->{1}, this env is {2}->{3}'.format(
            obs_len, num_actions, constants.OBS_LEN, constants.NUM_ACTIONS))
    if recorded['obs_era'] != constants.OBS_ERA:
        raise SystemExit(
            "obs_era {0!r} != this env's {1!r}. The observation is the same length but its indices "
            'no longer mean the same thing, so these weights would load cleanly and play '
            'badly'.format(recorded['obs_era'], constants.OBS_ERA))

    arch = arch_tools.build_arch(fc_layer_params, num_actions, obs_len, recorded['obs_era'],
                                 algo='dqn')
    net = network.build(arch)
    net.load_state_dict(state, strict=True)

    arch_tools.write_arch(args.destination, arch)
    written = checkpoints.save(args.destination, step, net,
                              extra={'imported_from': prefix})
    print('{0} -> {1}  ({2} params, fc {3})'.format(
        prefix, written, sum(p.numel() for p in net.parameters()), fc_layer_params))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
