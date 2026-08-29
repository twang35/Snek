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
        # One process owns the whole wave -- not one `eval_checkpoints.py` per arm -- so work moves to
        # whichever arm still has any instead of a finished arm's share of the box sitting idle.
        # `--chain` leads the argv because the selector has to follow it, and the policies trail the
        # selector: the same spelling an agent types on the laptop, which is the point.
        #
        # **The engine defaults to the vectorised one.** `vectorized/vec_wave.py` measures ~40x
        # faster than the TF path and was validated against it at four levels, ending in a
        # 24-checkpoint x 500-episode head-to-head that agreed to -0.058 pp (z = -0.28). Set
        # `SNEK_EVAL_ENGINE=scalar` -- in `runtime.json`'s `eval_engine`, or per job in the spec's
        # `env` -- to force `eval_wave.py`. A c51 batch needs no opt-out either: `vec_wave` measures
        # categorical arms itself since 2026-08-24 (-0.17 pp, z = -0.10 against the scalar path).
        engine = env.get('SNEK_EVAL_ENGINE') or runtime.get('eval_engine', 'vec')
        if engine not in ('vec', 'scalar'):
            raise ValueError('eval_engine={0!r}: expected "vec" or "scalar"'.format(engine))
        argv = [py, '-u', 'vectorized/vec_wave.py' if engine == 'vec' else 'eval_wave.py']
        if job.chain:
            argv.append('--chain')
        argv += list(job.eval_args) + list(job.policies)
        if engine == 'vec':
            # Cores minus two by default (see `vec_wave.DEFAULT_PROCS`); `runtime.json` can say
            # otherwise for a box whose cores are not the binding constraint. `EVAL_WORKERS`/
            # `EVAL_LANES` size *TF worker processes* and this engine has none, so they are not set
            # -- `vec_wave` strips them from what it passes its shards for the same reason.
            procs = job.eval_workers or runtime.get('vec_wave_procs', 0)
            if procs:
                env['VEC_WAVE_PROCS'] = str(procs)
        else:
            env['EVAL_WORKERS'] = str(job.eval_workers or runtime.get('eval_workers', 4))
            env['EVAL_LANES'] = str(job.eval_lanes or runtime.get('eval_lanes', 4))
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
    summary. Cheap and best-effort -- a missing or half-written file is ignored.

    Reads `rj.policy`, which for an eval wave is the first of its arms. That is right rather than
    approximate: the field exists to show a *training's* progress, and a wave has no single step to
    report."""
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
