"""Dumps a snek2 TF checkpoint's online Q-network to an `.npz`. **Runs under the `snek` env.**

The only file in snek3 that touches TensorFlow, and it is never imported — `import_tf_checkpoint.py`
runs it as a subprocess with the `snek` env's python, because snek3's env has no TensorFlow and
snek2's has no torch, so no single interpreter can do both halves of the conversion.

    /opt/miniconda3/envs/snek/bin/python tools/tf_export.py <ckpt-prefix> <out.npz>

`<ckpt-prefix>` is the path *without* the `.index`/`.data-...` suffix, e.g.
`../snek2/hallOfFame/b44a-lowlr7-b29b-ckpt2739000/ckpt-2739000`.

Writes one array per layer parameter, named `kernel0`, `bias0`, `kernel1`, ... in network order,
plus `layers` holding the count. Kernels come out in TensorFlow's `(in, out)` orientation and are
transposed on the torch side, not here — this half stays a faithful dump so that a disagreement can
only be in one of the two places.
"""

import re
import sys

import numpy as np
import tensorflow as tf

# `/_q_network/` selects the *online* net. The substring test is exact: the target net's key reads
# `agent/_target_q_network/...`, in which `_q_network` appears without the leading slash. Getting
# this wrong is silent — the target net holds the same shapes and lags the online one by up to
# `target_update_period` steps, so it would convert cleanly and measure slightly worse.
_KERNEL = re.compile(r'/_q_network/_sequential_layers/(\d+)/kernel/\.ATTRIBUTES/VARIABLE_VALUE$')


def export(prefix, destination):
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
        stem = 'agent/_q_network/_sequential_layers/{0}/'.format(index)
        kernel = reader.get_tensor(stem + 'kernel/.ATTRIBUTES/VARIABLE_VALUE')
        bias = reader.get_tensor(stem + 'bias/.ATTRIBUTES/VARIABLE_VALUE')
        if kernel.ndim != 2 or bias.ndim != 1 or kernel.shape[1] != bias.shape[0]:
            raise SystemExit('layer {0} is not a dense layer: kernel {1}, bias {2}'.format(
                index, kernel.shape, bias.shape))
        arrays['kernel{0}'.format(index)] = np.asarray(kernel, dtype=np.float32)
        arrays['bias{0}'.format(index)] = np.asarray(bias, dtype=np.float32)
        print('layer {0}: kernel {1}, bias {2}'.format(index, kernel.shape, bias.shape))

    np.savez(destination, **arrays)
    print('wrote {0}'.format(destination))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    export(sys.argv[1], sys.argv[2])
