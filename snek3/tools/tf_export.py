"""Dumps snek2 TF checkpoints' online Q-networks to an `.npz`. **Runs under the `snek` env.**

    /opt/miniconda3/envs/snek/bin/python tools/tf_export.py <policy-dir> <out.npz> [steps.txt]

The only file in snek3 that touches TensorFlow, and it is never imported — `import_tf_checkpoint.py`
runs it as a subprocess with the `snek` env's python, because snek3's env has no TensorFlow and
snek2's has no torch, so no single interpreter can do both halves of a conversion.

**Every checkpoint in one process, because the import is the cost.** TensorFlow takes ~5 s to import,
so converting b45a's 3,554 checkpoints one subprocess at a time would be five hours of nothing but
imports. Reading them all in one process is a couple of minutes.

Keys are `<step>/kernel<i>` and `<step>/bias<i>` in network order, plus `<step>/layers` holding the
count. Kernels come out in TensorFlow's `(in, out)` orientation and are transposed on the torch side,
not here — this half stays a faithful dump so a disagreement can only be in one of the two places.
"""

import os
import re
import sys

import numpy as np
import tensorflow as tf

# `/_q_network/` selects the *online* net. The substring test is exact: the target net's key reads
# `agent/_target_q_network/...`, in which `_q_network` appears without the leading slash. Getting
# this wrong is silent — the target net holds the same shapes and lags the online one by up to
# `target_update_period` steps, so it would convert cleanly and measure slightly worse.
_KERNEL = re.compile(r'/_q_network/_sequential_layers/(\d+)/kernel/\.ATTRIBUTES/VARIABLE_VALUE$')

_STEM = 'agent/_q_network/_sequential_layers/{0}/{1}/.ATTRIBUTES/VARIABLE_VALUE'


def steps_in(policy_dir):
    """Checkpoint steps present in a snek2 policy directory, ascending."""
    return sorted(int(name[len('ckpt-'):-len('.index')]) for name in os.listdir(policy_dir)
                  if name.startswith('ckpt-') and name.endswith('.index'))


def read_one(prefix):
    """`{'layers': n, 'kernel0': ..., 'bias0': ...}` for one checkpoint prefix."""
    reader = tf.train.load_checkpoint(prefix)
    shapes = reader.get_variable_to_shape_map()

    indices = sorted(int(match.group(1)) for match in
                     (_KERNEL.search(key) for key in shapes) if match)
    if not indices:
        raise SystemExit('no /_q_network/_sequential_layers/*/kernel in {0}'.format(prefix))
    if indices != list(range(len(indices))):
        raise SystemExit('layer indices are not contiguous from 0: {0}'.format(indices))

    arrays = {'layers': np.asarray(len(indices))}
    for index in indices:
        kernel = reader.get_tensor(_STEM.format(index, 'kernel'))
        bias = reader.get_tensor(_STEM.format(index, 'bias'))
        if kernel.ndim != 2 or bias.ndim != 1 or kernel.shape[1] != bias.shape[0]:
            raise SystemExit('layer {0} of {1} is not a dense layer: kernel {2}, bias {3}'.format(
                index, prefix, kernel.shape, bias.shape))
        arrays['kernel{0}'.format(index)] = np.asarray(kernel, dtype=np.float32)
        arrays['bias{0}'.format(index)] = np.asarray(bias, dtype=np.float32)
    return arrays


def export(policy_dir, destination, steps=None):
    """Every requested checkpoint of `policy_dir`, namespaced by step, into one `.npz`."""
    present = steps_in(policy_dir)
    if not present:
        raise SystemExit('no ckpt-*.index in {0}'.format(policy_dir))
    wanted = present if steps is None else [step for step in steps if step in set(present)]
    if steps is not None and len(wanted) != len(set(steps)):
        raise SystemExit('{0} of {1} requested steps are not in {2}'.format(
            len(set(steps)) - len(wanted), len(set(steps)), policy_dir))

    out = {}
    for number, step in enumerate(wanted, 1):
        arrays = read_one(os.path.join(policy_dir, 'ckpt-{0}'.format(step)))
        for key, value in arrays.items():
            out['{0}/{1}'.format(step, key)] = value
        if number == 1 or number % 250 == 0 or number == len(wanted):
            print('read {0}/{1} (step {2})'.format(number, len(wanted), step))
            sys.stdout.flush()

    out['steps'] = np.asarray(wanted, dtype=np.int64)
    np.savez(destination, **out)
    print('wrote {0}: {1} checkpoint(s), {2:.1f} MB'.format(
        destination, len(wanted), os.path.getsize(destination) / 1e6))


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    requested = None
    if len(sys.argv) == 4:
        with open(sys.argv[3]) as handle:
            requested = [int(line.split('#')[0].strip()) for line in handle
                         if line.split('#')[0].strip()]
    export(sys.argv[1], sys.argv[2], requested)
