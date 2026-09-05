"""Runs a stage-B measurement across several shard processes.

    PYTHONPATH=. python -u -m tools.eval_wave <policy> [--selector screen:97] [--shards 4]

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

LOG_DIR = os.path.join(constants.ROOT, 'logs')

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


class ArmWave(object):
    """One arm's measurement: its shards, launched **one at a time on request**, and its merge.

    Launching on request rather than all at once is what lets `tools/closeout.py` pool several arms'
    shards under one budget (2026-09-04): a hof30k arm has one or two candidates, so a 12-shard wave
    of it was one busy core and eleven shards exiting at once, arm after arm. The arm's shard count is
    `min(shards, len(steps))`, never more — a surplus shard did nothing but exit — and it is a
    function of the candidates alone, so a rerun computes the same count and every shard resumes
    from its own file.

    `finish()` is the merge, under the same two refusals as always: a failed shard, or fewer rows on
    disk than steps selected, and nothing is merged — a short merged file is indistinguishable from a
    real result, and this is where snek2's waves quietly produced them. An arm with **no candidates**
    still gets its (empty) merged file, so the pass that selects from it reads it and selects nothing.
    """

    def __init__(self, policy, selector='screen', episodes=500, shards=4, label=None, width=None,
                 seed=0, resume=True, merge=True):
        self.policy, self.selector, self.episodes = policy, selector, episodes
        self.label, self.width, self.seed, self.resume, self.merge = label, width, seed, resume, merge
        directory = restore.policy_dir(policy)
        self.steps, self.description = selectors.resolve(directory, selector, policy=policy)
        self.shards = min(int(shards), len(self.steps))
        self.pending = list(range(self.shards))
        self.processes, self.logs = [], []
        self.started = None
        self.baseline = sum(self.counts()) if self.shards else 0

    @property
    def name(self):
        return results.run_name(self.policy)

    def announce(self, requested=None):
        if requested is not None and requested > self.shards:
            print('note: {0} shards for {1} step(s); {2} will run'.format(
                requested, len(self.steps), self.shards))
        print('{0}: {1} step(s) — {2}'.format(self.name, len(self.steps), self.description))
        print('{0} episodes each, {1} shard(s) = {2} episodes total'.format(
            self.episodes, self.shards, len(self.steps) * self.episodes))

    def start_shard(self):
        """Launches the next shard. Returns False when every shard has been launched."""
        if not self.pending:
            return False
        shard = self.pending.pop(0)
        if self.started is None:
            self.started = time.time()
            os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, '{0}-s{1}of{2}.log'.format(
            os.path.basename(results.stage_b_path(self.policy, self.label))[:-len('.json')],
            shard + 1, self.shards))
        handle = open(log_path, 'w')
        self.logs.append((log_path, handle))
        self.processes.append(subprocess.Popen(
            shard_command(self.policy, self.selector, self.episodes, self.label, shard,
                          self.shards, self.width, self.seed, self.resume),
            stdout=handle, stderr=subprocess.STDOUT, cwd=constants.ROOT))
        return True

    def exhausted(self):
        return not self.pending

    def alive(self):
        return sum(1 for process in self.processes if process.poll() is None)

    def finished(self):
        return self.exhausted() and self.alive() == 0

    def counts(self):
        return rows_done(self.policy, self.label, self.shards) if self.shards else []

    def terminate(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            process.wait()

    def finish(self):
        """Closes the logs, checks every shard, merges. Returns 0, or 1 when nothing was merged."""
        for _, handle in self.logs:
            handle.close()
        elapsed = (time.time() - self.started) / 60.0 if self.started else 0.0
        if not self.shards:
            if self.merge:
                path, _ = results.merge(self.policy, self.label)
                print('{0}: no candidates; wrote the empty pass file {1}'.format(self.name, path))
            return 0
        failed = [(shard, process.returncode, self.logs[shard][0])
                  for shard, process in enumerate(self.processes)
                  if process.returncode not in (0, None)]
        counts = self.counts()
        print('{0}: {1} row(s) in {2:.1f}m  {3}'.format(self.name, sum(counts), elapsed, counts))
        for shard, code, log_path in failed:
            print('shard {0} exited {1} — see {2}'.format(shard + 1, code, log_path))
        if failed:
            print('refusing to merge with {0} failed shard(s). Fix and rerun; the surviving rows '
                  'resume.'.format(len(failed)))
            return 1
        if sum(counts) != len(self.steps):
            print('refusing to merge: {0} row(s) on disk for {1} selected step(s)'.format(
                sum(counts), len(self.steps)))
            return 1
        if self.merge and self.shards > 1:
            # `delete_shards=True` because by here every shard has been checked for failure and the
            # row count matches the selection, so the merged file provably covers every row at full
            # length — the two `refusing to merge` guards above are what make that true. Without it
            # the shards stay forever as exact duplicates: measured 2026-09-01, 224 files and 403 MB
            # on the laptop and 1,200 files and 651 MB on the desktop. `stage_b_chart.load` prefers
            # the merged file anyway, and falls back to the shards only while a wave is in flight.
            path, rows = results.merge(self.policy, self.label, delete_shards=True)
            print('merged {0} row(s) into {1} and removed {2} shard file(s)'.format(
                len(rows), path, self.shards))
        return 0


def run(policy, selector='screen', episodes=500, shards=4, label=None, width=None, seed=0,
        resume=True, merge=True):
    """One arm, all of its shards at once — `evaluate.py`'s wave. The pooled form is the close-out."""
    wave = ArmWave(policy, selector, episodes, shards, label, width, seed, resume, merge)
    wave.announce(requested=shards)
    while wave.start_shard():
        pass
    print('logs: {0}'.format(os.path.join(LOG_DIR, '')))
    started = time.time()
    try:
        while wave.alive():
            time.sleep(POLL_S)
            counts = wave.counts()
            done = sum(counts)
            elapsed = time.time() - started
            fresh = done - wave.baseline
            eta = ''
            if fresh >= ETA_MIN_ROWS:
                rate = fresh / elapsed
                eta = '  eta {0:.0f}m'.format((len(wave.steps) - done) / rate / 60.0)
            print('[{0:>5}/{1}] {2} shard(s) alive, {3:.0f}m elapsed{4}  {5}'.format(
                done, len(wave.steps), wave.alive(), elapsed / 60.0, eta, counts))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print('\nstopping shards; their rows are already on disk and a rerun resumes them')
        wave.terminate()
        raise
    return wave.finish()


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
