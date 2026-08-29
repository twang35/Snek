"""Measure a saved policy.

    PYTHONPATH=. python -u evaluate.py <policy> [--step N] [--episodes N] [--lanes N]

    PYTHONPATH=. python -u evaluate.py b44a-import --episodes 3000

`<policy>` is either an arm name under `savedPolicies/` or a path to any directory holding an
`arch.json` and a `ckpt-<step>.pt` — a `hallOfFame/` entry, say.

**Phase 1 shape: one checkpoint, one process.** The multiprocess wave over every checkpoint at ≥95
in stage A is phase 2 (`tools/eval_plan.py`, `vectorized/shard.py`, `vectorized/wave.py`); this is
the single-policy path it will keep for one-offs, and it is what the champion transfer is measured
with.
"""

import argparse
import json
import os
import sys
import time

from tools import eval_plan
from tools import restore
from vectorized import config
from vectorized import engine


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policy')
    parser.add_argument('--step', type=int, default=None,
                        help='which checkpoint; the newest by default')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--lanes', type=int, default=None,
                        help='how many games run in lockstep; the engine picks by default')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None,
                        help='write the row as JSON here as well as printing it')
    args = parser.parse_args(argv)

    directory = restore.policy_dir(args.policy)
    policy_fn, arch, step = restore.restore(directory, args.step)

    print('{0}  step {1}  fc {2}  era {3}'.format(
        directory, step, arch['fc_layer_params'], arch['obs_era']))
    print(config.describe())
    print('measuring {0} episodes...'.format(args.episodes))

    started = time.time()
    held = engine.measure(policy_fn, args.episodes, lanes=args.lanes, seed=args.seed)
    row = eval_plan.build_row(step, held)

    print(eval_plan.one_line(row))
    print('{0:.0f}s wall, {1:.1f} episodes/s'.format(
        time.time() - started, args.episodes / max(time.time() - started, 1e-9)))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as handle:
            json.dump({'policy': args.policy, 'arch': arch, 'seed': args.seed, 'rows': [row]},
                      handle, indent=2)
        print('wrote {0}'.format(args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
