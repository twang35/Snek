"""Turning the box's `ops` specs into the scheduler's queue, and the scheduler into a subprocess.

**The daemon launches one thing: `tools.scheduler`** (2026-09-05). Before that it launched every
trainer and every close-out itself, which meant a second copy of the wave barrier, the chain, the
launch env and the close-out argv beside the laptop's — and two schedulers drift (`plans/scheduler.md`
§0). Now the daemon writes the specs it reads off `ops` into a local queue directory in the shape the
scheduler reads on the laptop (`<queue>/<batch>/<id>.json`), and the scheduler does the rest: waves,
passes, the chart window, the eval workers.

The scheduler is launched **detached** (`setsid`), so a daemon restart never kills it — a deploy can
restart the daemon mid-batch and the scheduler carries on; the new daemon adopts it by pid. Runtime
knobs reach it as flags and environment at spawn; a change to `runtime.json` applies at the next
spawn, except `paused`/`drain`, which the daemon relays at once through a hold marker the scheduler
checks before every launch (`live_runs.hold_path`).

`SNEK_RUNS_DIR` is one constant directory, `desktop/runs`, gitignored (2026-09-03): the box's checkout
holds nothing under a path master tracks, so the laptop can commit every chart and `deploy`'s
fast-forward cannot refuse. The daemon reads finished work back from the same directory.
"""

import json
import os
import subprocess
import time

# `PYTHONPATH=.` because snek3's entry points are run as `PYTHONPATH=. python -u train.py`, not as
# installed modules — `conftest.py` covers the test path only.
BASE_ENV = {'PYTHONPATH': '.'}

# The two optional `host.env` keys. The daemon is a system service outside the graphical session, so
# the scheduler's chart window only reaches the physical monitor with that session's display and X
# authority; the scheduler inherits both and opens the window itself.
DISPLAY_ENV_KEYS = ('DISPLAY', 'XAUTHORITY')

RUNS_SUBDIR = os.path.join('desktop', 'runs')
# The local queue the scheduler reads: one directory per batch, one spec per file, mirrored from
# `ops` by `daemon.py` every poll. Gitignored, like `desktop/runs`.
QUEUE_SUBDIR = os.path.join('desktop', 'queue-local')

# Type defaults the old `build_command` applied. A smoke scores ~0, so the checkpoint gate would write
# nothing and the run could not resume; a benchmark measures the training loop, so stage A is turned
# off rather than shrunk and no window lands in its numbers.
SMOKE_ENV = {'SNEK_MIN_CHECKPOINT_SCORE': '0', 'SNEK_EVAL_INTERVAL': '500',
             'SNEK_GRAPH_EVAL_EPISODES': '20'}
SMOKE_STEPS = 5000
BENCHMARK_STEPS = 20000


def runs_dir(host):
    """The box's run directory: `<SNEK_DIR>/desktop/runs`, `SNEK_RUNS_DIR` for the scheduler and its jobs."""
    return os.path.join(host['SNEK_DIR'], RUNS_SUBDIR)


def queue_dir(host):
    return os.path.join(host['SNEK_DIR'], QUEUE_SUBDIR)


def materialise(job):
    """A parsed `ops` spec as the dict the scheduler reads, or None for an action.

    Every train-like type becomes a `train` spec with its policy and cap resolved and its type
    defaults folded into `env`, so the scheduler needs no notion of smoke or benchmark; an eval spec
    keeps its selector, depth and extra args, which the scheduler spells into `tools.closeout`
    exactly as the old `eval_command` did. A train spec with no cap raises: the scheduler cannot tell
    when such an arm is finished, and running one forever is not a default to fall back to.
    """
    if job.category == 'action':
        return None
    if job.type == 'eval':
        spec = {'project': job.project, 'id': job.id, 'type': 'eval', 'policies': list(job.policies),
                'label': job.label, 'notes': job.notes, 'eval_args': list(job.eval_args)}
        if job.selector:
            spec['selector'] = job.selector
        if job.episodes:
            spec['episodes'] = job.episodes
        if job.eval_shards:
            spec['eval_shards'] = job.eval_shards
        return spec
    env = dict(job.env)
    policy, max_steps = job.policy, job.max_steps
    if job.type == 'smoke':
        policy = policy or 'smoke'
        max_steps = max_steps or SMOKE_STEPS
        for key, value in SMOKE_ENV.items():
            env.setdefault(key, value)
    elif job.type == 'benchmark':
        policy = policy or 'bench-{0}'.format(job.id)
        max_steps = max_steps or BENCHMARK_STEPS
        env.setdefault('SNEK_MIN_CHECKPOINT_SCORE', '0')
        env.setdefault('SNEK_EVAL_INTERVAL', str(max_steps))
        env.setdefault('SNEK_EVAL_QUEUE', '0')
    if max_steps is None:
        raise ValueError('{0}: a train spec needs max_steps'.format(job.id))
    return {'project': job.project, 'id': job.id, 'type': 'train', 'policy': policy,
            'max_steps': int(max_steps), 'env': env, 'label': job.label, 'notes': job.notes,
            'priority': job.priority}


def scheduler_command(host, runtime):
    """The scheduler's argv: the runtime knobs as flags. Pure, so the spelling is testable."""
    argv = [host['PYTHON_BIN'], '-u', '-m', 'tools.scheduler', '--queue', queue_dir(host),
            '--wave', str(runtime['max_trainers']), '--max-trainers', str(runtime['max_trainers']),
            '--shards', str(runtime['eval_shards']), '--no-status']
    if not runtime.get('auto_stage_b', True):
        argv.append('--no-stage-b')
    return argv


def scheduler_env(host, runtime):
    """The environment overrides the scheduler runs under, inherited by every arm and pass it starts."""
    env = dict(BASE_ENV)
    env['SNEK_RUNS_DIR'] = runs_dir(host)
    if runtime.get('torch_threads', 0) > 0:
        env['SNEK_TORCH_THREADS'] = str(runtime['torch_threads'])
    if runtime.get('omp_num_threads', 0) > 0:
        env['OMP_NUM_THREADS'] = str(runtime['omp_num_threads'])
    for key in DISPLAY_ENV_KEYS:
        if host.get(key):
            env[key] = host[key]
    if not runtime.get('viewer', True):
        env['SNEK_CHART_WINDOW'] = '0'
    return env


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                     # someone else's, but alive
    return True


def spawn_scheduler(host, runtime):
    """Starts the scheduler detached. Returns `(popen, log_path)`."""
    argv = scheduler_command(host, runtime)
    os.makedirs(host['LOG_DIR'], exist_ok=True)
    log_path = os.path.join(host['LOG_DIR'], 'scheduler-{0}.log'.format(
        time.strftime('%Y%m%d-%H%M%S')))
    full_env = dict(os.environ)
    full_env.update(scheduler_env(host, runtime))
    nice = int(runtime.get('nice', 0))

    def preexec():
        os.setsid()          # detach, so a daemon restart does not kill the scheduler
        if nice:
            os.nice(nice)

    log_handle = open(log_path, 'ab')
    process = subprocess.Popen(argv, cwd=host['SNEK_DIR'], env=full_env,
                               stdout=log_handle, stderr=subprocess.STDOUT,
                               preexec_fn=preexec, close_fds=True)
    return process, log_path
