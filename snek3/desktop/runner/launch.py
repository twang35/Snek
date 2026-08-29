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


def build_command(job, host, runtime):
    """`(argv, env_overrides, log_name, resolved_policy)`. `env_overrides` merges over `os.environ`.

    ## Threads

    One knob reaches two places. `SNEK_TORCH_THREADS` is read by `train.py` and by the eval shards;
    `OMP_NUM_THREADS` is read by numpy's BLAS underneath both. They default to **1** for a measured
    reason — see `config.py` — and 16 shards each grabbing a thread per core is the specific mistake
    the default prevents.

    ## The eval command

    One process owns the whole wave, so its shards take whichever checkpoint is next regardless of
    which arm it belongs to, instead of a finished arm's share of the box going idle.

    **No episode count and no gate are set here.** snek2's daemon carried five protocol numbers as a
    second copy of what `eval_plan.py` defines, and they drifted. The rule is inverted instead: the
    daemon passes a selector and an episode count *only if the spec named them*, and otherwise lets
    `evaluate.py`'s own defaults decide — which are the protocol, `screen:95` at 500 episodes. One
    definition, no import, nothing here to drift. The daemon cannot import the original anyway: it
    runs on base python so it can start before the conda env exists, and `eval_plan` needs numpy.
    """
    python = host['PYTHON_BIN']
    env = dict(BASE_ENV)
    if runtime.get('torch_threads', 0) > 0:
        env['SNEK_TORCH_THREADS'] = str(runtime['torch_threads'])
    if runtime.get('omp_num_threads', 0) > 0:
        env['OMP_NUM_THREADS'] = str(runtime['omp_num_threads'])
    env.update(job.env)                      # per-job SNEK_* win over the runtime defaults

    if job.type == 'eval':
        # The policies trail the selector — the same spelling typed on the laptop, which is the
        # point: a spec is a command someone could have run by hand.
        argv = [python, '-u', 'evaluate.py']
        argv += list(job.eval_args)
        argv += list(job.policies)
        if job.selector:
            argv += ['--selector', job.selector]
        if job.episodes:
            argv += ['--episodes', str(job.episodes)]
        shards = job.eval_shards or runtime.get('eval_shards', 0)
        if shards:
            argv += ['--shards', str(shards)]
        return argv, env, 'eval-{0}.log'.format(job.id), job.policy

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
        env.setdefault('SNEK_MIN_CHECKPOINT_SCORE', '0')
        env.setdefault('SNEK_MAX_STEPS', str(job.max_steps or 20000))
        # A benchmark measures the training loop, so stage A is turned off rather than shrunk: it is
        # ~90% of an arm's wall clock and would be most of what the benchmark measured.
        env.setdefault('SNEK_EVAL_INTERVAL', str(job.max_steps or 20000))
    if job.max_steps is not None:
        env['SNEK_MAX_STEPS'] = str(job.max_steps)
    argv = [python, '-u', 'train.py', policy]
    return argv, env, '{0}-{1}.log'.format(job.type, job.id), policy


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
    path = os.path.join(host['SNEK_DIR'], 'runs', str(running_job.policy) + '_evals.json')
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
