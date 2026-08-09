"""Turning a Job into a subprocess.

Jobs are launched *detached* (`setsid`) so a daemon restart never kills a running
trainer -- the trainings self-terminate via SNEK_MAX_STEPS, and the daemon only
reaps them. Throughput (steps/second) is read from the policy's evals.json so it
can be surfaced live while you tune capacity.
"""
import json
import os
import subprocess
import time


def build_command(job, host, runtime):
    """Returns (argv, env_overrides, log_name, resolved_policy).

    env_overrides is merged over os.environ. The threading knobs come from the
    live runtime config -- TensorFlow reads TF_NUM_INTRAOP/INTEROP_THREADS and
    oneDNN reads OMP_NUM_THREADS from the environment, so capacity sweeps need no
    code change.
    """
    py = host['PYTHON_BIN']
    env = {}
    if runtime.get('tf_intraop_threads', 0) > 0:
        env['TF_NUM_INTRAOP_THREADS'] = str(runtime['tf_intraop_threads'])
        env['TF_NUM_INTEROP_THREADS'] = str(max(1, runtime['tf_intraop_threads'] // 2))
    if runtime.get('omp_num_threads', 0) > 0:
        env['OMP_NUM_THREADS'] = str(runtime['omp_num_threads'])
    env.update(job.env)  # per-job SNEK_* overrides win

    # Put the job's live chart window on the desktop's session display, if host.env
    # names one. The runner is a systemd service outside the graphical session, so a
    # launched job only reaches the monitor with the session's DISPLAY + X authority.
    # If it can't (display gone), training.py's window is best-effort and the job
    # still runs and writes its PNG.
    if host.get('DISPLAY'):
        env.setdefault('DISPLAY', host['DISPLAY'])
        if host.get('XAUTHORITY'):
            env.setdefault('XAUTHORITY', host['XAUTHORITY'])

    if job.type == 'eval':
        argv = [py, '-u', 'eval_checkpoints.py', job.policy] + list(job.eval_args)
        env['EVAL_WORKERS'] = str(job.eval_workers or runtime.get('eval_workers', 10))
        return argv, env, 'eval-{0}.log'.format(job.id), job.policy

    # train / smoke / benchmark all invoke the trainer.
    policy = job.policy
    if job.type == 'smoke':
        policy = policy or 'smoke'
        env.setdefault('SNEK_MIN_CHECKPOINT_SCORE', '0')  # smokes score ~0
    elif job.type == 'benchmark':
        policy = policy or 'bench-{0}'.format(job.id)
        env.setdefault('SNEK_MIN_CHECKPOINT_SCORE', '0')
        env.setdefault('SNEK_MAX_STEPS', str(job.max_steps or 20000))
    if job.max_steps is not None:
        env['SNEK_MAX_STEPS'] = str(job.max_steps)
    argv = [py, '-u', 'snek2.py', policy]
    return argv, env, '{0}-{1}.log'.format(job.type, job.id), policy


class RunningJob:
    def __init__(self, job, policy, pid, log_path, popen=None, started=None):
        self.job = job
        self.policy = policy
        self.pid = pid
        self.log_path = log_path
        self.popen = popen              # None when re-adopted after a restart
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
        # Unknown across a restart: the child was reparented to init, so we can
        # no longer waitpid() it. The caller infers success from the log tail.
        return self.popen.returncode if self.popen is not None else None


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def spawn(job, host, runtime):
    argv, env_over, log_name, policy = build_command(job, host, runtime)
    os.makedirs(host['LOG_DIR'], exist_ok=True)
    log_path = os.path.join(host['LOG_DIR'], log_name)
    full_env = dict(os.environ)
    full_env.update(env_over)
    nice = int(runtime.get('nice', 0))

    def preexec():
        os.setsid()          # detach so a daemon restart does not kill the run
        if nice:
            os.nice(nice)

    log_fh = open(log_path, 'ab')
    p = subprocess.Popen(argv, cwd=host['SNEK_DIR'], env=full_env,
                         stdout=log_fh, stderr=subprocess.STDOUT,
                         preexec_fn=preexec, close_fds=True)
    return RunningJob(job, policy, p.pid, log_path, popen=p)


def update_throughput(rj, host):
    """Refreshes rj.current_step and rj.steps_per_sec from the policy's evals.json
    summary. Cheap and best-effort -- a missing or half-written file is ignored."""
    path = os.path.join(host['SNEK_DIR'], 'runs', str(rj.policy) + '_evals.json')
    try:
        with open(path) as fh:
            step = json.load(fh).get('summary', {}).get('step')
    except (OSError, ValueError):
        return
    if step is None:
        return
    now = time.time()
    if rj._last_step is not None and now > rj._last_step_time:
        delta = step - rj._last_step
        rj.steps_per_sec = round(delta / (now - rj._last_step_time), 1) if delta >= 0 else None
    rj.current_step = step
    rj._last_step, rj._last_step_time = step, now
