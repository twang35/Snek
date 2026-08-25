"""Throughput sweep for the vectorised eval: which `VEC_WAVE_PROCS` / width / thread config is fastest.

**Why this exists on a second host.** The laptop's operating point (12 processes x width 1024) was
measured on 14 *physical* cores, and `vec_wave.DEFAULT_PROCS` derives its default from
`os.cpu_count()` -- which counts SMT threads. On the desktop (Ryzen 7 9700X) that is 16 threads over
**8 physical cores**, so the default asks for 14 shards where there are 8 cores to run them, and one
shard saturates about one core because the observation build is single-threaded numpy. A number tuned
on one topology cannot be carried to the other by arithmetic; it has to be measured.

**The protocol, and why each part is the way it is.**

* **One arm, an explicit step list, identical every run.** `top<N>` is a *target, not a quota*
  (`select_top_checkpoints` measures every checkpoint over the mandatory threshold), so two configs
  could silently measure different amounts of work. The set is the last `--checkpoints` steps by
  default -- late checkpoints play long episodes, which is what a real close-out is made of.
* **Everything lands under a `bench-*` name, via a symlink into the real policy dir.** `vec_wave`
  forces its children's chart directory to `evals/`, and the PNG is named after the policy -- so
  measuring `b45a` directly would overwrite `evals/b45a..._eval_progress.png` with bench data. The
  symlink gives the same checkpoints a throwaway identity, and `bench-*` is the project's documented
  disposable-eval prefix.
* **Resume is off and the result file is removed between configs.** `VEC_EVAL_RESUME` defaults on,
  so the second config in a sweep would find every row already banked and measure nothing -- it
  would report an enormous throughput for doing no work.
* **CPU idle, MemAvailable and summed shard RSS are sampled during the run**, not derived. Idle is
  read from `/proc/stat` deltas so there is no `mpstat` dependency; the memory pair is what says
  whether a proc count is even admissible on a 15 GB box.

Usage (from `snek2/`, on the host being measured):

    PYTHONPATH=. python -u hyperparamTuning/perDiagnostics/vec_wave_sweep.py b45a-lowlr8-b29b \\
        --configs 6 8 10 12 14 16 --checkpoints 240 --episodes 100

A config is `procs[:width[:threads]]`, so `12:2048` and `16::1` are both valid. The `threads` field
selects **which pools are pinned**, because there are two independent ones and a bare number sets
both:

| spelling | `TF_NUM_INTRAOP` | `TF_NUM_INTEROP` | `OMP_NUM_THREADS` |
|---|---|---|---|
| omitted | TF's default | TF's default | TF's default |
| `1` | 1 | 1 | 1 |
| `tf1` | 1 | 1 | untouched |
| `intra1` | 1 | untouched | untouched |
| `inter1` | untouched | 1 | untouched |
| `omp1` | untouched | untouched | 1 |

Measured on `the-claw-den`: one process holds **50** threads at the defaults, 20 with TF's two at 1,
35 with `OMP_NUM_THREADS=1` alone and **5** with all three -- so TF's Eigen pools are ~30 of them and
oneDNN's OpenMP pool is ~15, and a 16-shard wave runs ~800 threads on 16 hardware threads unless
pinned. `tf.config.threading.get_intra_op_parallelism_threads()` is **not** a way to check this: it
reads the `ConfigProto` field, which stays 0 whether or not the env var is set.

**Platform split.** CPU idle and `MemAvailable` come from `/proc` and are Linux-only; mean load average
and peak shard RSS are read on both, so darwin gets `episodes/s`, `wall` and `load` -- which is enough
to pick a config, since throughput is the objective and the rest is diagnosis.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SNEK2 = os.path.dirname(os.path.dirname(HERE))
POLICY_DIR = os.path.join(SNEK2, 'savedPolicies')
RUNS_DIR = os.path.join(SNEK2, 'runs')
BENCH_NAME = 'bench-vecsweep'
SUFFIX = '-vecsweep'


def available_steps(policy):
    d = os.path.join(POLICY_DIR, policy)
    steps = sorted(int(f[len('ckpt-'):].split('.')[0])
                   for f in os.listdir(d) if f.startswith('ckpt-') and f.endswith('.index'))
    if not steps:
        raise SystemExit('no checkpoints in {0}'.format(d))
    return steps


def pick_steps(policy, count, spread):
    """`count` checkpoint steps: the last ones by default, evenly strided with `--spread`.

    The last ones are the default because per-checkpoint cost tracks policy quality -- a late
    checkpoint survives longer, so it costs more env steps -- and a close-out is dominated by exactly
    those. `--spread` is there for an arm whose tail is unrepresentative.
    """
    steps = available_steps(policy)
    if count >= len(steps):
        return steps
    if spread:
        stride = len(steps) / float(count)
        return [steps[min(len(steps) - 1, int(i * stride))] for i in range(count)]
    return steps[-count:]


def ensure_bench_link(policy):
    """Point `savedPolicies/bench-vecsweep` at `policy`, and say whether we created it."""
    link = os.path.join(POLICY_DIR, BENCH_NAME)
    if os.path.islink(link):
        if os.readlink(link) == policy:
            return False
        os.unlink(link)
    elif os.path.exists(link):
        raise SystemExit('{0} exists and is not a symlink - refusing to touch it'.format(link))
    os.symlink(policy, link)
    return True


def clear_outputs():
    """Remove the bench result files so nothing resumes and nothing is left behind."""
    for name in sorted(os.listdir(RUNS_DIR)):
        if name.startswith(BENCH_NAME) and name.endswith('.json'):
            os.remove(os.path.join(RUNS_DIR, name))


LINUX = sys.platform.startswith('linux')


def read_stat():
    """`(total, idle)` jiffies from /proc/stat. Linux only; `None` elsewhere."""
    if not LINUX:
        return None
    with open('/proc/stat') as handle:
        parts = handle.readline().split()
    values = [int(x) for x in parts[1:]]
    return sum(values), values[3]


def mem_available_mb():
    """`MemAvailable` in MB, or `None` off Linux.

    Deliberately not emulated on darwin. The honest analogue would be some combination of free,
    inactive, purgeable and compressed pages, and a number assembled from a guess about which of those
    the kernel would actually hand back is worse than no number -- especially here, where the laptop
    has 36 GB and memory is not the constraint being measured.
    """
    if not LINUX:
        return None
    with open('/proc/meminfo') as handle:
        for line in handle:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) // 1024
    return 0


def shard_rss_mb():
    """Summed RSS of every live `vec_eval.py` process.

    On Linux this reads /proc by pid, so the scan cannot match its own command line -- the project's
    repeated `pgrep` trap. On darwin it shells out to `ps` once and filters in Python for the same
    reason: `ps -Ao rss=,command=` does not contain the pattern, so nothing can self-match. Either
    way a process that exits mid-scan is skipped rather than raising.
    """
    total = 0
    if LINUX:
        for pid in os.listdir('/proc'):
            if not pid.isdigit():
                continue
            try:
                with open('/proc/{0}/cmdline'.format(pid), 'rb') as handle:
                    if 'vec_eval.py' not in handle.read().decode('utf-8', 'replace'):
                        continue
                with open('/proc/{0}/statm'.format(pid)) as handle:
                    total += (int(handle.read().split()[1])
                              * os.sysconf('SC_PAGE_SIZE') // (1024 * 1024))
            except (IOError, OSError, IndexError, ValueError):
                continue
        return total
    try:
        out = subprocess.check_output(['ps', '-Ao', 'rss=,command='])
    except (subprocess.CalledProcessError, OSError):
        return 0
    for line in out.decode('utf-8', 'replace').splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and 'vec_eval.py' in parts[1]:
            try:
                total += int(parts[0]) // 1024          # ps reports RSS in KB on both platforms
            except ValueError:
                continue
    return total


class Sampler(threading.Thread):
    """Mean load, CPU idle, min MemAvailable and peak shard RSS across one config's run.

    Load average is sampled rather than read once at the end, because a config's whole point is what
    it does *while* it runs -- and it is the one occupancy signal available on both platforms, so it is
    what a cross-host comparison has to lean on.
    """

    def __init__(self, interval=2.0):
        threading.Thread.__init__(self)
        self.daemon = True
        self.interval = interval
        self.stop_flag = threading.Event()
        self.idle_pct = None
        self.min_avail = mem_available_mb()
        self.peak_rss = 0
        self.load_mean = 0.0

    def run(self):
        start = read_stat()
        loads = []
        while not self.stop_flag.wait(self.interval):
            avail = mem_available_mb()
            if avail is not None:
                self.min_avail = avail if self.min_avail is None else min(self.min_avail, avail)
            self.peak_rss = max(self.peak_rss, shard_rss_mb())
            try:
                loads.append(os.getloadavg()[0])
            except OSError:
                pass
        if loads:
            self.load_mean = sum(loads) / len(loads)
        end = read_stat()
        if start and end:
            span = end[0] - start[0]
            self.idle_pct = 100.0 * (end[1] - start[1]) / span if span else 0.0

    def finish(self):
        self.stop_flag.set()
        self.join(timeout=self.interval * 3)


THREAD_MODES = ('all', 'tf', 'intra', 'inter', 'omp')


def parse_threads(field):
    """`(mode, value)` from a `threads` field: `''`, `'1'`, `'tf1'`, `'intra1'`, `'omp1'`.

    A bare number keeps meaning "all three", so the rows already published in
    `vectorized/README.md` are reproducible by the spelling they were taken with.
    """
    if not field:
        return None, 0
    for mode in ('intra', 'inter', 'omp', 'tf'):   # no mode is a prefix of another, but order the
        if field.startswith(mode):                 # loop by specificity anyway
            return mode, int(field[len(mode):] or 1)
    return 'all', int(field)


def thread_env(mode, value):
    """The env vars one thread mode sets.

    **The default mode sets explicit zeros rather than unsetting**, because `vec_wave.child_env`
    `setdefault`s `TF_NUM_INTEROP_THREADS` to 1 -- so an absent variable would arrive at the shard as
    the pinned config, and this script's own control would silently measure the treatment. 0 is
    TensorFlow's spelling of "size it yourself" (verified: 0 and unset both leave 50 threads in a
    process where 1 leaves 35), so an explicit 0 is a real opt-out rather than a sentinel.
    """
    if not mode:
        return {'TF_NUM_INTRAOP_THREADS': '0', 'TF_NUM_INTEROP_THREADS': '0', 'OMP_NUM_THREADS': '0'}
    env = {}
    if mode in ('all', 'tf', 'intra'):
        env['TF_NUM_INTRAOP_THREADS'] = str(value)
    if mode in ('all', 'tf', 'inter'):
        env['TF_NUM_INTEROP_THREADS'] = str(value)
    if mode in ('all', 'omp'):
        env['OMP_NUM_THREADS'] = str(value)
    return env


def label_threads(mode, value):
    return 'default' if not mode else '{0}{1}'.format('' if mode == 'all' else mode, value)


def parse_config(token):
    parts = (token.split(':') + ['', ''])[:3]
    procs = int(parts[0])
    width = int(parts[1]) if parts[1] else 1024
    mode, value = parse_threads(parts[2])
    if mode and mode not in THREAD_MODES:
        raise SystemExit('unknown thread mode {0!r} in {1!r}'.format(mode, token))
    return procs, width, mode, value


def run_config(steps, episodes, procs, width, mode, value, log_dir):
    """One `vec_wave.py` invocation, timed and sampled. Returns a result dict.

    Every thread var this script knows about is **cleared** before the mode's own are applied, so a
    value inherited from the caller's shell cannot quietly become part of a config that was meant to
    read TensorFlow's default -- which would attribute someone's exported `OMP_NUM_THREADS` to the
    engine.
    """
    clear_outputs()
    env = dict(os.environ)
    env['VEC_WAVE_PROCS'] = str(procs)
    env['VEC_EVAL_WIDTH'] = str(width)
    env['VEC_EVAL_RESUME'] = '0'
    env['EVAL_OUT_SUFFIX'] = SUFFIX
    env['SNEK_CHART_VIEWER'] = '0'
    env['PYTHONPATH'] = SNEK2
    for key in ('TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS'):
        env.pop(key, None)
    env.update(thread_env(mode, value))

    argv = [sys.executable, '-u', os.path.join('vectorized', 'vec_wave.py')]
    argv += [str(step) for step in steps] + [BENCH_NAME]
    tag = '{0}p-{1}w-{2}'.format(procs, width, label_threads(mode, value))
    log_path = os.path.join(log_dir, 'vecsweep-{0}.log'.format(tag))

    sampler = Sampler()
    sampler.start()
    start = time.time()
    with open(log_path, 'w') as log:
        code = subprocess.call(argv, cwd=SNEK2, env=env, stdout=log, stderr=subprocess.STDOUT)
    wall = time.time() - start
    sampler.finish()

    measured = measured_episodes()
    return {
        'procs': procs, 'width': width, 'threads': label_threads(mode, value), 'exit': code,
        'wall': wall, 'episodes': measured,
        'eps_per_s': measured / wall if wall else 0.0,
        'idle_pct': sampler.idle_pct, 'min_avail_mb': sampler.min_avail,
        'peak_rss_mb': sampler.peak_rss, 'load_mean': sampler.load_mean,
        'rss_per_shard_mb': sampler.peak_rss // procs if procs else 0,
        'log': log_path,
    }


def measured_episodes():
    """Episodes actually measured, from the merged bench file -- never assumed from the plan.

    Assuming `checkpoints x episodes` is how a config that resumed, or one whose shard died, reports
    a throughput it did not earn.
    """
    path = os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(BENCH_NAME, SUFFIX))
    if not os.path.exists(path):
        return 0
    with open(path) as handle:
        payload = json.load(handle)
    return sum(row.get('episodes', 0) for row in payload.get('results', []))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policy')
    parser.add_argument('--configs', nargs='+', required=True,
                        help='procs[:width[:threads]] tokens, e.g. 8 12 12:2048 14::1 14::tf1 14::omp1')
    parser.add_argument('--checkpoints', type=int, default=240)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--spread', action='store_true',
                        help='stride the checkpoints over the whole run instead of taking the tail')
    parser.add_argument('--log-dir', default='/tmp')
    parser.add_argument('--out', default='', help='write the result rows here as JSON')
    args = parser.parse_args(argv)

    steps = pick_steps(args.policy, args.checkpoints, args.spread)
    created = ensure_bench_link(args.policy)
    print('sweep: {0} -> {1}{2}'.format(args.policy, BENCH_NAME,
                                        ' (symlink created)' if created else ''))
    print('{0} checkpoints ({1} .. {2}), {3} episodes each = {4} episodes per config'.format(
        len(steps), steps[0], steps[-1], args.episodes, len(steps) * args.episodes))
    avail = mem_available_mb()
    print('{0}, cpu_count {1}{2}\n'.format(
        sys.platform, os.cpu_count(),
        '' if avail is None else ', MemAvailable {0} MB'.format(avail)))

    rows = []
    plan = [parse_config(token) for token in args.configs] * args.repeat
    for index, (procs, width, mode, value) in enumerate(plan, 1):
        print('[{0}/{1}] procs={2} width={3} threads={4} ...'.format(
            index, len(plan), procs, width, label_threads(mode, value)), end=' ')
        sys.stdout.flush()
        row = run_config(steps, args.episodes, procs, width, mode, value, args.log_dir)
        rows.append(row)
        print('{0:.0f}s  {1:.1f} eps/s  {2}load {3:.1f}  peakRSS {4} MB ({5} MB/shard){6}'.format(
            row['wall'], row['eps_per_s'],
            '' if row['idle_pct'] is None else 'idle {0:.1f}%  '.format(row['idle_pct']),
            row['load_mean'], row['peak_rss_mb'], row['rss_per_shard_mb'],
            '' if row['exit'] == 0 and row['episodes'] else '  ** SUSPECT **'))

    print('\n| procs | width | threads | episodes/s | wall | CPU idle | load | peak RSS |')
    print('|---|---|---|---|---|---|---|---|')
    for row in rows:
        print('| {0} | {1} | {2} | {3:.1f} | {4:.0f} s | {5} | {6:.1f} | {7} MB |'.format(
            row['procs'], row['width'], row['threads'], row['eps_per_s'], row['wall'],
            'n/a' if row['idle_pct'] is None else '{0:.1f}%'.format(row['idle_pct']),
            row['load_mean'], row['peak_rss_mb']))
    best = max(rows, key=lambda r: r['eps_per_s'])
    print('\nfastest: procs={0} width={1} threads={2} at {3:.1f} eps/s'.format(
        best['procs'], best['width'], best['threads'], best['eps_per_s']))

    if args.out:
        with open(args.out, 'w') as handle:
            json.dump(rows, handle, indent=2)
        print('rows written to {0}'.format(args.out))
    clear_outputs()
    return 0


if __name__ == '__main__':
    sys.exit(main())
