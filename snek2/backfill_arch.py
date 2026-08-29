"""One-off maintenance: write ``arch.json`` into every existing policy directory that lacks one.

Policies created before ``policy_arch.py`` have no architecture sidecar, so a restore of one would
hit the new hard-fail (`ArchMismatch: no arch.json`). This backfills them.

The architecture is read from each checkpoint's **own tensor shapes** — the ground truth, since the
dense kernels are ``[obs_len, w0], [w0, w1], ..., [w_last, num_actions]`` in layer order — rather
than trusting a config file that could have drifted. Where ``runs/<policy>.md`` records
``fc_layer_params``, it is cross-checked and any disagreement is printed loudly. ``obs_era`` is
mapped from the observed ``obs_len`` via the era markers in ``hallOfFame/HOF.md``; that cannot
distinguish two meanings at the *same* length, but every such case here predates the 30-value vector
and does not load on ``master`` regardless, so the length alone is enough to make the current
environment reject it.

Usage (from snek2/):
    PYTHONPATH=. python backfill_arch.py            # write, skipping dirs that already have one
    PYTHONPATH=. python backfill_arch.py --dry-run  # print the table, write nothing
    PYTHONPATH=. python backfill_arch.py --force     # rewrite even where arch.json exists
"""
import glob
import os
import re
import sys

os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import tensorflow as tf

import policy_arch
from snake_constants import POLICY_DIR, RUNS_DIR

HALL_OF_FAME_DIR = os.path.join(os.path.dirname(POLICY_DIR.rstrip('/')), 'hallOfFame')
OBS_LEN_TO_ERA = {20: 'e4514a8', 26: '450e66e', 30: 'b09c616'}


def latest_index_prefix(policy_dir):
    """The checkpoint prefix for the highest step in the dir, or None if there is no checkpoint."""
    indices = glob.glob(os.path.join(policy_dir, 'ckpt-*.index'))
    if not indices:
        return None
    latest = max(indices, key=lambda p: int(re.search(r'ckpt-(\d+)\.index', p).group(1)))
    return latest[:-len('.index')]


def arch_from_checkpoint(prefix):
    """(fc_layer_params, num_actions, obs_len) read from the online q-network's kernel shapes.

    ``/_q_network/`` selects the online net; the target net is ``/_target_q_network/`` and is skipped
    (its ``_q_network`` is not slash-prefixed, so the substring test is exact).
    """
    reader = tf.train.load_checkpoint(prefix)
    shapes = reader.get_variable_to_shape_map()
    layers = {}
    for key, shape in shapes.items():
        if '/_q_network/' in key and key.endswith('kernel/.ATTRIBUTES/VARIABLE_VALUE') and len(shape) == 2:
            index = int(re.search(r'_sequential_layers/(\d+)/kernel', key).group(1))
            layers[index] = tuple(shape)
    ordered = [layers[i] for i in sorted(layers)]
    obs_len = ordered[0][0]
    num_actions = ordered[-1][1]
    fc_layer_params = [rows_cols[1] for rows_cols in ordered[:-1]]
    return fc_layer_params, num_actions, obs_len


def base_policy_name(dir_name):
    """The runs/<name>.md base: hallOfFame entries are '<policy>-ckpt<step>'."""
    return re.sub(r'-ckpt\d+$', '', dir_name)


def recorded_fc(dir_name):
    """fc_layer_params as recorded in runs/<base>.md, or None if that file is absent/silent."""
    md = os.path.join(RUNS_DIR, base_policy_name(dir_name) + '.md')
    if not os.path.exists(md):
        return None
    with open(md) as handle:
        for line in handle:
            match = re.match(r'\|\s*fc_layer_params\s*\|\s*\(([^)]*)\)\s*\|', line)
            if match:
                # A single-layer arm is recorded as the Python tuple repr '(320,)', whose trailing
                # comma leaves an empty token after split — skip the blanks.
                return [int(w) for w in match.group(1).split(',') if w.strip()]
    return None


def target_dirs():
    for base in (POLICY_DIR.rstrip('/'), HALL_OF_FAME_DIR):
        for path in sorted(glob.glob(os.path.join(base, '*'))):
            if os.path.isdir(path):
                yield path


def main(argv):
    dry_run = '--dry-run' in argv
    force = '--force' in argv
    written = skipped = mismatched = 0
    print('{0:<44} {1:>8} {2:>18} {3:>10}  {4}'.format(
        'policy dir', 'obs_len', 'fc_layer_params', 'era', 'action'))
    for policy_dir in target_dirs():
        name = os.path.basename(policy_dir)
        if policy_arch.read_arch(policy_dir) is not None and not force:
            skipped += 1
            continue
        prefix = latest_index_prefix(policy_dir)
        if prefix is None:
            print('{0:<44} {1:>8} {2:>18} {3:>10}  skip (no checkpoint)'.format(
                name[:44], '-', '-', '-'))
            skipped += 1
            continue
        fc_layer_params, num_actions, obs_len = arch_from_checkpoint(prefix)
        era = OBS_LEN_TO_ERA.get(obs_len, 'unknown-{0}v'.format(obs_len))
        note = ''
        from_md = recorded_fc(name)
        if from_md is not None and from_md != fc_layer_params:
            note = '  MISMATCH vs runs.md {0}'.format(tuple(from_md))
            mismatched += 1
        action = 'dry-run' if dry_run else 'write'
        print('{0:<44} {1:>8} {2:>18} {3:>10}  {4}{5}'.format(
            name[:44], obs_len, str(tuple(fc_layer_params)), era, action, note))
        if not dry_run:
            policy_arch.write_arch(policy_dir, policy_arch.build_arch(
                fc_layer_params, num_actions, obs_len, era))
            written += 1
    print('\n{0} written, {1} skipped, {2} fc-vs-runs.md mismatches'.format(
        written, skipped, mismatched))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
