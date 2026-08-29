"""One process measuring one slice of one arm's checkpoints.

    PYTHONPATH=. python -u -m tools.shard <policy> --steps <file> --shard 0 --shards 4

**A shard owns its output file and nothing else writes to it.** That is what makes the wave a set of
independent processes with no central bookkeeping — see `tools/results.py` for the incident that
design avoids.

**It is resumable.** A killed shard's rows are already on disk, and restarting it skips the steps it
already measured. snek2 lost 192 rows and 7,534 episodes in one incident for want of this, and the
per-episode arrays in each row are what make a resumed file identical to an uninterrupted one rather
than merely similar.

**One net per resident checkpoint, pooled.** `engine.measure_stream` keeps several checkpoints live
at once to keep the batch full — that is what `max_live` is — so a single net cannot serve them. A
`30 -> 320 -> 3` net is 45 KB, so the pool costs nothing; it exists to avoid re-allocating one per
checkpoint, which on a 3,000-checkpoint arm is 3,000 allocations of the same shape.
"""

import argparse
import sys
import time

import torch

from tools import checkpoints
from tools import eval_plan
from tools import restore
from tools import results
from tools import step_selectors as selectors
from vectorized import config
from vectorized import engine

class _NetPool:
    """Nets sized for one arch, handed out and returned as checkpoints come and go."""

    def __init__(self, arch, device='cpu'):
        self._arch = arch
        self._device = device
        self._free = []
        self.built = 0

    def take(self):
        if self._free:
            return self._free.pop()
        self.built += 1
        return restore.build_net(self._arch, device=self._device)

    def give_back(self, net):
        self._free.append(net)


def measure_slice(policy_dir, steps, episodes, out_path, policy=None, width=None, seed=0,
                  stage_a=None, on_row=None, resume=True, device='cpu'):
    """Measure `steps` and write rows to `out_path`. Returns every row in the file, by step.

    `stage_a` maps step -> the stage-A percent that selected it, carried into each row so the screen
    and the measurement can be compared on the same weights.
    """
    # One thread. A wave runs 4-16 of these at once, and torch defaults to one thread per core each,
    # so the processes oversubscribe the box and every one of them slows down. The matmul here is
    # (512, 30) x (30, 320), which a single thread saturates anyway.
    torch.set_num_threads(1)

    policy = policy if policy is not None else policy_dir
    arch = restore.policy_arch(policy_dir)
    stage_a = stage_a or {}

    existing = {row['step']: row for row in results.rows_of(results.read(out_path))} if resume else {}
    todo = [step for step in steps if step not in existing]
    header = {'policy': results.run_name(policy), 'arch': arch, 'episodes': episodes,
              'seed': seed, 'config': config.describe()}

    def flush():
        payload = dict(header)
        payload['rows'] = [existing[step] for step in sorted(existing)]
        results.write(out_path, payload)

    if not todo:
        print('{0}: all {1} step(s) already measured'.format(out_path, len(steps)))
        flush()
        return [existing[step] for step in sorted(existing)]

    print('{0}: {1} step(s) to measure, {2} already done, {3} episodes each'.format(
        out_path, len(todo), len(existing), episodes))

    pool = _NetPool(arch, device=device)
    nets = {}
    pending = iter(todo)
    started = time.time()
    completed = [0]

    def next_job():
        step = next(pending, None)
        if step is None:
            return None
        net = pool.take()
        checkpoints.load(checkpoints.path(policy_dir, step), net, device=device)
        nets[step] = net
        return step, restore.policy_fn_for(arch, net, device=device)

    def on_complete(step, held):
        pool.give_back(nets.pop(step))
        row = eval_plan.build_row(step, held, stage_a_percent=stage_a.get(step))
        existing[step] = row
        completed[0] += 1
        elapsed = time.time() - started
        rate = completed[0] / max(elapsed, 1e-9)
        remaining = (len(todo) - completed[0]) / rate if rate else 0.0
        print('[{0:>5}/{1}] {2}  eta {3:.0f}m'.format(
            completed[0], len(todo), eval_plan.one_line(row), remaining / 60.0))
        sys.stdout.flush()
        # Written after **every** row, not on a timer. That makes a `kill -9` cost nothing and lets
        # the wave read exact progress straight off the files, and it is affordable here in a way it
        # was not in snek2: this rewrites one shard's own rows once per completed measurement, so a
        # 400-row shard dumps ~3 MB about every 10 s — tens of milliseconds. snek2 needed a write
        # gate because its single controller rebuilt every lane's rows 125 times *per* measurement,
        # 58 s of bookkeeping against 46 s of lane work. Independent shard files remove the cause
        # rather than throttling the symptom.
        flush()
        # After the flush, so a caller's hook can read the file it is being told about.
        if on_row:
            on_row(row)

    engine.measure_stream(next_job, on_complete, episodes, width=width, seed=seed)
    flush()

    ordered = [existing[step] for step in sorted(existing)]
    print('{0}: {1} row(s), {2} episodes, {3:.1f}m'.format(
        out_path, len(ordered), len(todo) * episodes, (time.time() - started) / 60.0))
    return ordered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policy')
    parser.add_argument('--selector', default='screen', help='see tools/step_selectors.py')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--label', default=None, help='names the pass, so two do not collide')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--width', type=int, default=None, help='games in lockstep')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args(argv)

    directory = restore.policy_dir(args.policy)
    steps, description = selectors.resolve(directory, args.selector, policy=args.policy)
    mine = selectors.slice_for(steps, args.shard, args.shards)
    out_path = results.stage_b_path(args.policy, args.label,
                                    shard=args.shard if args.shards > 1 else None,
                                    shards=args.shards)
    print('shard {0}/{1}: {2} of {3} step(s) — {4}'.format(
        args.shard + 1, args.shards, len(mine), len(steps), description))
    if not mine:
        print('nothing for this shard')
        return 0

    measure_slice(directory, mine, args.episodes, out_path, policy=args.policy,
                  width=args.width, seed=args.seed, resume=not args.no_resume)
    return 0


if __name__ == '__main__':
    sys.exit(main())
