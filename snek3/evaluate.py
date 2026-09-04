"""Measure saved policies — one checkpoint, or a whole arm across shard processes.

    PYTHONPATH=. python -u evaluate.py <policy> [selector] [options]

    PYTHONPATH=. python -u evaluate.py b1a-thing                    # stage B: screen:97, 500 eps
    PYTHONPATH=. python -u evaluate.py b1a-thing screen:98 --shards 8
    PYTHONPATH=. python -u evaluate.py b44a-import one --episodes 3000
    PYTHONPATH=. python -u evaluate.py b45a-import steps:runs/ab.txt --episodes 100 --label ab

The selector decides which checkpoints; see [`tools/step_selectors.py`](tools/step_selectors.py).
`one` is the exception: it measures a single checkpoint in this process, which is what a spot check
and a record re-measurement want.

**The default is the protocol.** No selector means `screen:97` at 500 episodes — every checkpoint
whose stage-A eval reached 97/100, measured at full length. There is no second stage behind it:
stage B *is* the hall-of-fame measurement, so a promotion reads its file rather than queueing
another job.
"""

import argparse
import sys
import time

from tools import eval_plan
from tools import restore
from tools import results
from tools import eval_wave
from vectorized import config
from vectorized import engine


def measure_one(policy, step=None, episodes=500, lanes=None, seed=0, out=None):
    """One checkpoint, in this process. Prints the row and optionally writes it."""
    directory = restore.policy_dir(policy)
    policy_fn, arch, resolved = restore.restore(directory, step)

    print('{0}  step {1}  fc {2}  era {3}'.format(
        directory, resolved, arch['fc_layer_params'], arch['obs_era']))
    print(config.describe())
    print('measuring {0} episodes...'.format(episodes))

    started = time.time()
    held = engine.measure(policy_fn, episodes, lanes=lanes, seed=seed)
    row = eval_plan.build_row(resolved, held)
    elapsed = time.time() - started

    print(eval_plan.one_line(row))
    print('{0:.0f}s wall, {1:.1f} episodes/s'.format(elapsed, episodes / max(elapsed, 1e-9)))
    if out:
        results.write(out, {'policy': results.run_name(policy), 'arch': arch, 'seed': seed,
                            'episodes': episodes, 'config': config.describe(), 'rows': [row]})
        print('wrote {0}'.format(out))
    return row


def build_parser():
    """This script's command-line contract, as an object.

    Extracted from `main` so a test can hand it a command line **the desktop daemon built** and find
    out whether it parses. The daemon cannot import this module — it runs on base python, before the
    conda env exists, so it duplicates the spelling of this interface by hand — and that duplication
    silently drifted: it passed several policies and a `--selector` flag against the single positional
    `policy` and positional `selector` below, so every stage-B wave the box dispatched exited 2. Four
    green fixtures asserted the argv the daemon builds and none asked this parser to accept it.

    A test *does* run in the env, so the constraint that forces the duplication does not extend to
    the fixture that guards it. See `tests/test_desktop_runner.py`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policy')
    parser.add_argument('selector', nargs='?', default='screen',
                        help="a step selector, or 'one' for a single checkpoint in this process")
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--label', default=None, help='names the pass, so two do not collide')
    parser.add_argument('--width', type=int, default=None, help='games in lockstep, per process')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--step', type=int, default=None, help="with 'one': which checkpoint")
    parser.add_argument('--out', default=None, help="with 'one': write the row here")
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--no-merge', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.selector == 'one':
        measure_one(args.policy, args.step, args.episodes, args.width, args.seed, args.out)
        return 0
    return eval_wave.run(args.policy, args.selector, args.episodes, args.shards, args.label,
                    args.width, args.seed, not args.no_resume, not args.no_merge)


if __name__ == '__main__':
    sys.exit(main())
