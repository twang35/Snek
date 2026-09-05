"""Turning a Job into a subprocess.

Jobs are launched **detached** (`setsid`), so a daemon restart never kills a running trainer — a
training self-terminates at `SNEK_MAX_STEPS` and the daemon only reaps it. That is also why the
daemon can be restarted for a deploy while four arms are mid-run.

Throughput is read from the arm's own `runs/<policy>_evals.json`, so `status.json` can show live
steps/s without the daemon knowing anything about the trainer.
"""

import json
import os
import subprocess
import time

# `PYTHONPATH=.` because snek3's entry points are run as `PYTHONPATH=. python -u train.py`, not as
# installed modules — `conftest.py` covers the test path only.
BASE_ENV = {'PYTHONPATH': '.'}

# The two optional `host.env` keys, forwarded to every job. The daemon is a system service outside
# the graphical session, so a job only reaches the physical monitor with that session's display and X
# authority. **The daemon does not open the window** — `train.py` does, for its own arm, which is what
# makes one appear on the laptop too and with nobody launching it. All the daemon owes it is these two
# variables and the switch in `build_command`.
DISPLAY_ENV_KEYS = ('DISPLAY', 'XAUTHORITY')

# Where every job on the box writes its run artifacts, and where the daemon reads them back: one constant
# directory under the checkout, gitignored, rather than `runs/`. Before 2026-09-03 the box wrote into
# `snek3/runs/`, the same path master tracks, so every chart the laptop committed collided with the
# box's untracked copy and `git merge --ff-only` refused; `deploy.py` settled those per file and the
# rule tore each time a new directory joined the pattern (37 JSONs and 75 snek2 files on 2026-09-03).
# With the box writing here, its checkout holds nothing under a tracked path and the merge is clean.
RUNS_SUBDIR = os.path.join('desktop', 'runs')


def runs_dir(host):
    """The box's run directory: `<SNEK_DIR>/desktop/runs`, the value of `SNEK_RUNS_DIR` for every job."""
    return os.path.join(host['SNEK_DIR'], RUNS_SUBDIR)


def build_command(job, host, runtime):
    """`(argv, env_overrides, log_name, resolved_policy)`. `env_overrides` merges over `os.environ`.

    ## Threads

    One knob reaches two places. `SNEK_TORCH_THREADS` is read by `train.py` and by the eval shards;
    `OMP_NUM_THREADS` is read by numpy's BLAS underneath both. They default to **1** for a measured
    reason — see `config.py` — and 16 shards each grabbing a thread per core is the specific mistake
    the default prevents.

    ## The eval command

    A batch stays **one job**: one ledger record, one `results` publish, one thing to look at. It is
    also **one process** — `tools/closeout.py`, which takes the whole batch and measures it one arm at
    a time. That is the same file the laptop's close-out runs, which is the point: before 2026-08-30
    this function built a `sh -c` chain of one `evaluate.py` per arm and the laptop ran an agent's
    scratch shell script, and the two sequencers behaved differently under exactly the conditions
    nobody watches — a failed arm, a killed pass, buffered progress.

    Since 2026-09-04 the close-out pools its shards across the arms — `--shards` is the pool, each arm
    gets as many as it has candidates — so a hof30k wave of one-candidate arms no longer runs them
    one at a time on one core each. The daemon passes the pool size and nothing else changes here.

    **No episode count and no gate are set here.** snek2's daemon carried five protocol numbers as a
    second copy of what `eval_plan.py` defines, and they drifted. The rule is inverted instead: the
    daemon passes a selector and an episode count *only if the spec named them*, and otherwise lets
    the close-out's own defaults decide — which are the protocol, `screen:97` at 500 episodes. One
    definition, no import, nothing here to drift. The daemon cannot import the original anyway: it
    runs on base python so it can start before the conda env exists, and `eval_plan` needs numpy.

    **That last constraint is why this interface drifted and went unnoticed.** It duplicated
    `evaluate.py`'s spelling by hand and got it wrong twice — several policies where one is taken, and
    `--selector` where the selector is positional — so every wave the box dispatched exited 2. A test
    runs in the conda env even though the daemon does not, so `tests/test_desktop_runner.py` hands
    what this builds to `closeout.build_parser()`. **Note the one spelling difference it guards:** the
    selector is a flag for the close-out and a positional for `evaluate.py`, because a batch's
    policies take the `nargs='+'`.
    """
    python = host['PYTHON_BIN']
    env = dict(BASE_ENV)
    env['SNEK_RUNS_DIR'] = runs_dir(host)
    if runtime.get('torch_threads', 0) > 0:
        env['SNEK_TORCH_THREADS'] = str(runtime['torch_threads'])
    if runtime.get('omp_num_threads', 0) > 0:
        env['OMP_NUM_THREADS'] = str(runtime['omp_num_threads'])
    for key in DISPLAY_ENV_KEYS:
        if host.get(key):
            env[key] = host[key]
    # `runtime.json`'s `viewer` stays the ops-level switch, but it now reaches the process that opens
    # the window rather than the daemon acting on it. A spec's own `SNEK_CHART_WINDOW` still wins,
    # because `job.env` is applied after this.
    if not runtime.get('viewer', True):
        env['SNEK_CHART_WINDOW'] = '0'
    env.update(job.env)                      # per-job SNEK_* win over the runtime defaults

    if job.type == 'eval':
        return (eval_command(job, host, runtime), env,
                'eval-{0}.log'.format(job.id), job.policy)

    # train / smoke / benchmark all invoke the trainer.
    policy = job.policy
    if job.type == 'smoke':
        policy = policy or 'smoke'
        # A smoke run scores ~0, so the checkpoint gate would write nothing and the run could not
        # resume. Set here rather than in the spec so every smoke gets it.
        env.setdefault('SNEK_MIN_CHECKPOINT_SCORE', '0')
        env.setdefault('SNEK_MAX_STEPS', str(job.max_steps or 5000))
        env.setdefault('SNEK_EVAL_INTERVAL', '500')
        env.setdefault('SNEK_GRAPH_EVAL_EPISODES', '20')
    elif job.type == 'benchmark':
        policy = policy or 'bench-{0}'.format(job.id)
        # A benchmark measures the training loop, and a window costs a couple of percent of a core
        # that would be measured along with it.
        env.setdefault('SNEK_CHART_WINDOW', '0')
        env.setdefault('SNEK_MIN_CHECKPOINT_SCORE', '0')
        env.setdefault('SNEK_MAX_STEPS', str(job.max_steps or 20000))
        # A benchmark measures the training loop, so stage A is turned off rather than shrunk: it is
        # ~90% of an arm's wall clock and would be most of what the benchmark measured.
        env.setdefault('SNEK_EVAL_INTERVAL', str(job.max_steps or 20000))
    if job.max_steps is not None:
        env['SNEK_MAX_STEPS'] = str(job.max_steps)
    argv = [python, '-u', 'train.py', policy]
    return argv, env, '{0}-{1}.log'.format(job.type, job.id), policy


def eval_command(job, host, runtime):
    """The one argv that measures the batch — exactly what someone would type on the laptop.

    That property is the one worth keeping: a spec is a command a human could have run, and now it is
    the *same* command, because the sequencing over arms moved into `tools/closeout.py` where both
    boxes reach it. No `sh -c`, so nothing here quotes anything and a policy name cannot be shell
    syntax.

    Selector and episode count are omitted unless the spec named them, so the close-out's own
    defaults stay the single definition of the protocol.
    """
    shards = job.eval_shards or runtime.get('eval_shards', 0)
    argv = [host['PYTHON_BIN'], '-u', '-m', 'tools.closeout']
    argv += [str(policy) for policy in job.policies]
    if job.selector:
        argv += ['--selector', job.selector]
    argv += list(job.eval_args)
    if job.episodes:
        argv += ['--episodes', str(job.episodes)]
    if shards:
        argv += ['--shards', str(shards)]
    return argv


class RunningJob(object):
    def __init__(self, job, policy, pid, log_path, popen=None, started=None):
        self.job = job
        self.policy = policy
        self.pid = pid
        self.log_path = log_path
        self.popen = popen              # None when re-adopted after a daemon restart
        self.started = started or time.time()
        self.current_step = None
        self.steps_per_sec = None
        self._last_step = None
        self._last_step_time = None

    def is_alive(self):
        if self.popen is not None:
            return self.popen.poll() is None
        return pid_alive(self.pid)

    def returncode(self):
        """None across a restart: the child was reparented to init, so it cannot be waited on.

        The caller infers success from the log tail instead, which is why `_log_has_traceback`
        exists.
        """
        return self.popen.returncode if self.popen is not None else None


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                     # someone else's, but alive
    return True


def spawn(job, host, runtime):
    argv, overrides, log_name, policy = build_command(job, host, runtime)
    os.makedirs(host['LOG_DIR'], exist_ok=True)
    log_path = os.path.join(host['LOG_DIR'], log_name)
    full_env = dict(os.environ)
    full_env.update(overrides)
    nice = int(runtime.get('nice', 0))

    def preexec():
        os.setsid()          # detach, so a daemon restart does not kill the run
        if nice:
            os.nice(nice)

    log_handle = open(log_path, 'ab')
    process = subprocess.Popen(argv, cwd=host['SNEK_DIR'], env=full_env,
                               stdout=log_handle, stderr=subprocess.STDOUT,
                               preexec_fn=preexec, close_fds=True)
    return RunningJob(job, policy, process.pid, log_path, popen=process)


def update_throughput(running_job, host):
    """Refreshes `current_step` and `steps_per_sec` from the arm's `_evals.json` summary.

    Cheap and best-effort — a missing or half-written file is ignored, which matters because the
    trainer writes that file atomically but the daemon reads it on an unrelated clock.

    Reads `running_job.policy`, which for an eval wave is the first of its arms. That is right
    rather than approximate: the field exists to show a *training's* progress, and a wave has no
    single step to report.
    """
    path = os.path.join(runs_dir(host), str(running_job.policy) + '_evals.json')
    try:
        with open(path) as handle:
            step = json.load(handle).get('summary', {}).get('step')
    except (OSError, ValueError):
        return
    if step is None:
        return
    now = time.time()
    if running_job._last_step is not None and now > running_job._last_step_time:
        delta = step - running_job._last_step
        running_job.steps_per_sec = (
            round(delta / (now - running_job._last_step_time), 1) if delta >= 0 else None)
    running_job.current_step = step
    running_job._last_step, running_job._last_step_time = step, now
