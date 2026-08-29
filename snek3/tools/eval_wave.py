"""Runs a stage-B measurement across several shard processes.

    PYTHONPATH=. python -u -m tools.eval_wave <policy> [--selector screen:95] [--shards 4]

**Shards are independent processes with independent output files, and the wave is only a launcher
and a progress readout.** There is no central row bookkeeping, which is the design's whole point:
snek2's controller banked every lane's episodes itself and re-serialised the result file 125 times
per measurement — 58 s of single-threaded work against the 46 s four lanes needed to produce one — so
it overtook its own workers and folded a 90-minute backlog with 16 of them idle. Here the controller
cannot be the bottleneck because it does no work per episode.

**Progress is read off the shard files, so it is exact and survives the wave being killed.** Each
shard rewrites its own rows after every completed measurement; the wave counts them. Restarting the
wave resumes every shard from where it stopped.

**`--shards` is how parallel the wave is, and nothing else.** snek2's equivalent field meant
"episodes advance in indivisible rounds of this size", and reporting the batch width there inflated
an ETA by 10.24x.

**Named `eval_wave`, not `wave`, because `wave` is a standard-library module** (the WAV file reader).
A `tools/wave.py` is importable as bare `wave` from anything run out of `tools/`, and the failure it
produces names neither file — `tools/selectors.py` shadowed stdlib `selectors`, which `subprocess`
imports, and the traceback reported a circular import inside `subprocess.py`.
`tests/test_module_layering.py` checks the whole tree against the real stdlib listing, and it is what
caught this one.
"""

import argparse
import os
import subprocess
import sys
import time

from env import constants
from tools import restore
from tools import results
from tools import step_selectors as selectors

LOG_DIR = os.path.join(os.path.dirname(constants.RUNS_DIR), 'logs')

# How often the wave counts rows. A 500-episode row takes ~10 s of one shard's time, so anything
# under a few seconds is polling for nothing.
POLL_S = 10.0

# Below this many completed rows an ETA is noise — a wave's first rows are unrepresentative because
# the shards start together and their first measurements overlap.
ETA_MIN_ROWS = 8


def shard_command(policy, selector, episodes, label, shard, shards, width, seed, resume):
    command = [sys.executable, '-u', '-m', 'tools.shard', policy,
               '--selector', selector, '--episodes', str(episodes),
               '--shard', str(shard), '--shards', str(shards), '--seed', str(seed)]
    if label:
        command += ['--label', label]
    if width:
        command += ['--width', str(width)]
    if not resume:
        command.append('--no-resume')
    return command


def rows_done(policy, label, shards):
    """Rows on disk per shard, as a list. A shard that has not written yet contributes 0."""
    counts = []
    for shard in range(shards):
        path = results.stage_b_path(policy, label, shard=shard if shards > 1 else None,
                                    shards=shards)
        counts.append(len(results.rows_of(results.read(path))))
    return counts


def run(policy, selector='screen', episodes=500, shards=4, label=None, width=None, seed=0,
        resume=True, merge=True):
    directory = restore.policy_dir(policy)
    steps, description = selectors.resolve(directory, selector, policy=policy)
    if shards > len(steps):
        # Not an error: a shard with nothing to do exits immediately. Reported because a wave
        # launched with more shards than checkpoints usually means the selector matched almost
        # nothing, which is worth seeing before waiting on it.
        print('note: {0} shards for {1} step(s); the surplus will exit at once'.format(
            shards, len(steps)))

    print('{0}: {1} step(s) — {2}'.format(results.run_name(policy), len(steps), description))
    print('{0} episodes each, {1} shard(s) = {2} episodes total'.format(
        episodes, shards, len(steps) * episodes))

    os.makedirs(LOG_DIR, exist_ok=True)
    processes, logs = [], []
    for shard in range(shards):
        log_path = os.path.join(LOG_DIR, '{0}-s{1}of{2}.log'.format(
            os.path.basename(results.stage_b_path(policy, label))[:-len('.json')],
            shard + 1, shards))
        handle = open(log_path, 'w')
        logs.append((log_path, handle))
        processes.append(subprocess.Popen(
            shard_command(policy, selector, episodes, label, shard, shards, width, seed, resume),
            stdout=handle, stderr=subprocess.STDOUT, cwd=os.path.dirname(constants.RUNS_DIR)))
    print('logs: {0}'.format(os.path.join(LOG_DIR, '')))

    started = time.time()
    baseline = sum(rows_done(policy, label, shards))
    try:
        while any(process.poll() is None for process in processes):
            time.sleep(POLL_S)
            counts = rows_done(policy, label, shards)
            done = sum(counts)
            alive = sum(1 for process in processes if process.poll() is None)
            elapsed = time.time() - started
            fresh = done - baseline
            eta = ''
            if fresh >= ETA_MIN_ROWS:
                rate = fresh / elapsed
                eta = '  eta {0:.0f}m'.format((len(steps) - done) / rate / 60.0)
            print('[{0:>5}/{1}] {2} shard(s) alive, {3:.0f}m elapsed{4}  {5}'.format(
                done, len(steps), alive, elapsed / 60.0, eta, counts))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print('\nstopping shards; their rows are already on disk and a rerun resumes them')
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait()
        raise
    finally:
        for _, handle in logs:
            handle.close()

    failed = [(shard, process.returncode, logs[shard][0])
              for shard, process in enumerate(processes) if process.returncode not in (0, None)]
    counts = rows_done(policy, label, shards)
    print('{0} row(s) in {1:.1f}m  {2}'.format(sum(counts), (time.time() - started) / 60.0, counts))
    for shard, code, log_path in failed:
        print('shard {0} exited {1} — see {2}'.format(shard + 1, code, log_path))

    if failed:
        # Not merged. A short merged file is indistinguishable from a real result, and this is where
        # snek2's waves quietly produced them.
        print('refusing to merge with {0} failed shard(s). Fix and rerun; the surviving rows '
              'resume.'.format(len(failed)))
        return 1

    if sum(counts) != len(steps):
        print('refusing to merge: {0} row(s) on disk for {1} selected step(s)'.format(
            sum(counts), len(steps)))
        return 1

    if merge and shards > 1:
        path, rows = results.merge(policy, label)
        print('merged {0} row(s) into {1}'.format(len(rows), path))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policy')
    parser.add_argument('--selector', default='screen', help='see tools/step_selectors.py')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--label', default=None, help='names the pass, so two do not collide')
    parser.add_argument('--width', type=int, default=None, help='games in lockstep, per shard')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--no-merge', action='store_true')
    args = parser.parse_args(argv)
    return run(args.policy, args.selector, args.episodes, args.shards, args.label,
               args.width, args.seed, not args.no_resume, not args.no_merge)


if __name__ == '__main__':
    sys.exit(main())
