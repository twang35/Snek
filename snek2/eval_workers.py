"""Independent evaluation workers: each owns its env *and* its own copy of the network.

## Why this exists

`eval_checkpoints.run_round` drives a `ParallelPyEnvironment`: the parent runs `policy_action` over
a batch of `num_workers` observations while every worker sits idle, then the workers step while the
parent sits idle. Measured on `b17b-forkseed2` @1190000 (mean episode ~1500 steps):

| workers | batched ep/s | independent ep/s | speedup |
|---|---|---|---|
| 4 | 3.02 | **4.25** | 1.41x |
| 5 | 2.75 | **5.24** | 1.91x |
| 10 | 4.01 | **9.34** | 2.33x |
| 20 | 4.15 | 8.39 | 2.02x |

Two separate costs were in the batched design, and this module removes both:

1. **The inference/step ping-pong.** All workers idle for 24-30% of wall clock waiting on a batched
   inference call. Batching bought almost nothing to justify it: inference costs **0.25ms at batch 1**
   against **0.22ms at batch 4** — a quarter of the work for 88% of the cost, because the call is
   latency-bound rather than throughput-bound.
2. **The per-episode barrier.** A round ends when the *slowest* worker's episode ends and finished
   workers keep being stepped, wasting 13% of env-step slots at 4 workers and **45% at 40**.

In **one** process the optimum is ~10 workers, with near-linear scaling to there (1.06 / 1.05 / 0.93
ep/s per worker at 4 / 5 / 10). Past it, per-worker inference slows as N independent networks stop
fitting in cache — 0.30ms at 10 workers, 0.41ms at 14, **0.82ms at 20** — and throughput falls.

## ‡ But a close-out runs 4 processes at once, and that changes the answer

**The single-process speedups above do not transfer, and the reason is worth keeping.** Four parallel
processes already overlap each other's inference and stepping phases, so most of the idle this module
removes was gone before it arrived. Measured on the real shape — 4 parallel processes, 800 episodes
per condition, the record checkpoint:

| path | workers each | wall | cpu-s/ep | cores busy |
|---|---|---|---|---|
| batched | 4 | 141.1s | 1.29 | 7.3 |
| batched | **5** | **128.9s** | 1.34 | 8.3 |
| batched | 10 | 132.0s | 1.59 | 9.7 |
| **independent** | **4** | **117.0s** | **1.85** | **12.7** |
| independent | 5 | 118.0s | 1.90 | 12.9 |
| independent | 10 | 134.0s | 2.11 | 12.6 |

**So the real gain is 1.10x, not 2.33x** — best independent (117s) against best batched (128.9s). And
`EVAL_WORKERS=4` is the optimum here rather than 10: 4 processes x 4 workers pins the machine at ~12.7
of 14 cores busy, and every independent condition lands within 117-134s because there is no idle left
to convert. That is why the default moved from 10 to 4.

**The price is CPU: 1.85 against 1.34 cpu-s/episode, +38%.** This is a deliberate trade — see
[`hyperparamTuning.md`](hyperparamTuning/hyperparamTuning.md#-evals-run-hot-on-purpose). No swapping in
any condition (0 swapouts), despite each independent worker carrying a whole TensorFlow where a
batched worker is a bare pygame env.

## The property that must not be broken

`run_round`'s barrier exists to keep the estimate unbiased, and its docstring is worth re-reading
before changing anything here: **truncating in-flight episodes reads high, not low.** Perfect games
average ~1780 steps against ~2200 for non-perfect ones, because a win ends the moment the board
fills while a policy about to fail circles until the starve budget runs out. So "collect N finished
episodes and stop" preferentially drops *failures*.

**This module preserves unbiasedness with a fixed per-worker quota, not a barrier.** Every worker is
told how many episodes to run before it starts, and no episode is ever discarded for being slow. The
only synchronisation is one join at the end.

**Verified against the batched path on 5,120 episodes of one checkpoint**: independent **94.79%**
(1346/1420) against batched **94.03%** (3479/3700), difference +0.76 pp, **p=0.295**. The decisive
paired run was 1000 episodes per path back to back — independent 95.50%, batched 93.90%, p=0.110, sign
*reversed* from an earlier 800-episode scare that read 92.00%. That scare was the minimum of six
samples treated as one hypothesis test; the lesson is that a 2-3 sd surprise found by looking at the
lowest number is a prompt to measure, not evidence.

**Abandonment is the one place this could go wrong.** When `should_abandon` fires, workers stop
between episodes and every *completed* episode is still counted — including ones that land after the
stop flag is set. Discarding those would reintroduce exactly the length-correlated truncation the
quota is designed to avoid, so `collect_results` keeps draining until every worker reports done.
"""
import collections
import multiprocessing
import os
import time

# Message tags on the results queue. One queue, many writers, so every message carries its rank.
TAG_READY = 'ready'
TAG_LOADED = 'loaded'
TAG_EPISODE = 'episode'
TAG_DONE = 'done'
TAG_ERROR = 'error'

Episode = collections.namedtuple('Episode', ['rank', 'score', 'perfect', 'reward', 'steps'])


def split_quota(episodes, num_workers):
    """Per-worker episode counts summing to **exactly** `episodes`.

    The batched path rounds up to whole rounds, which is why `EVAL_EPISODES` had to divide
    `EVAL_WORKERS` or a 100-episode request silently became 108 and stopped matching the rest of the
    arm. Here the remainder is spread one episode at a time, so any combination is exact and the
    divisibility rule stops being a correctness constraint.

    Workers beyond `episodes` get 0 and simply report done, which keeps the collect loop's
    "wait for every worker" invariant true without a special case.
    """
    if num_workers <= 0:
        raise ValueError('need at least one worker, got {0}'.format(num_workers))
    if episodes < 0:
        raise ValueError('episodes cannot be negative, got {0}'.format(episodes))
    base, remainder = divmod(episodes, num_workers)
    return [base + (1 if index < remainder else 0) for index in range(num_workers)]


def collect_results(results, num_workers, stop_flag, episodes_total=None,
                    on_progress=None, should_abandon=None):
    """Drains `results` until every worker has reported `TAG_DONE`.

    Split out from the process machinery so the scheduling logic is testable with a fake queue and a
    fake flag — no TensorFlow, no subprocesses. `results` needs only `get()`; `stop_flag` needs
    `set()` and `is_set()`.

    `should_abandon(perfect_so_far, episodes_so_far)` is consulted after each completed episode,
    which is finer-grained than the batched path's once-per-round. When it first returns True the
    stop flag is set and **collection continues** — workers finish the episode in hand and every
    completed episode is counted. See the module docstring for why dropping them would bias the
    result upward.

    `on_progress(wave, waves_total, perfect_so_far, episodes_so_far, per_wave_perfect)` mirrors the
    batched path's `on_round` signature so the live chart and `write_results` need no changes. A
    "wave" is `num_workers` completed episodes — not a real synchronisation point, just the unit the
    progress display already speaks in.

    **`episodes_total` is what makes `waves_total` a constant, and it is not optional in practice.**
    It is `ceil(episodes_total / num_workers)` — 25 for the usual 100 episodes on 4 workers, 5 for a
    20-episode screen — and it is known before the first episode runs. An earlier version left it
    unset and fell back to the episodes seen so far, so `rounds_total` grew by one every episode and
    the live chart's x axis stretched continuously through a single checkpoint. The x axis must be
    fixed from the first frame: its whole job is showing how much is left.

    Progress is reported **once per completed wave**, not per episode, which matches the batched
    path's cadence. Per-episode reporting also meant `write_results` re-serialising every row of the
    arm up to 100 times a checkpoint instead of 25.

    Returns `(scores, perfect_flags, rewards, steps, abandoned)`.
    """
    scores, perfect_flags, rewards = [], [], []
    steps = 0
    abandoned = False
    finished = 0
    # Fixed before the first episode: ceil(episodes / workers). None only when a caller does not
    # say, in which case there is nothing honest to report as a total.
    waves_total = (-(-episodes_total // num_workers)) if episodes_total else None
    per_wave_perfect = []
    wave_perfect = 0

    while finished < num_workers:
        message = results.get()
        tag = message[0]

        if tag == TAG_ERROR:
            raise RuntimeError('eval worker {0} failed: {1}'.format(message[1], message[2]))

        if tag == TAG_DONE:
            finished += 1
            continue

        if tag != TAG_EPISODE:
            # LOADED/READY belong to a different phase; ignore rather than crash, so a late ack
            # cannot wedge a measurement.
            continue

        episode = message[1]
        scores.append(episode.score)
        perfect_flags.append(bool(episode.perfect))
        rewards.append(episode.reward)
        steps += int(episode.steps)

        wave_perfect += int(bool(episode.perfect))
        if len(scores) % num_workers == 0:
            per_wave_perfect.append(wave_perfect)
            wave_perfect = 0
            if on_progress is not None:
                on_progress(len(per_wave_perfect), waves_total or len(per_wave_perfect),
                            int(sum(perfect_flags)), len(scores), list(per_wave_perfect))

        # Checked every episode, but only ever *sets* the flag — never stops draining.
        if (not abandoned and should_abandon is not None
                and should_abandon(int(sum(perfect_flags)), len(scores))):
            abandoned = True
            stop_flag.set()

    if wave_perfect:
        per_wave_perfect.append(wave_perfect)
    return scores, perfect_flags, rewards, steps, abandoned


def _worker_main(rank, policy_name, ckpt_dir, commands, results, stop_flag):
    """One independent evaluator process: build once, then serve load/run commands.

    The env, network and agent are built **once** and reused for every checkpoint. A close-out
    measures hundreds of checkpoints and TensorFlow import plus graph construction costs ~8s, so
    rebuilding per checkpoint would cost more than the design saves. Restoring different weights
    into the same variables does not retrace the `tf.function`, so only the first checkpoint pays
    for tracing.
    """
    try:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

        import tensorflow as tf
        # One thread each by default: N multi-threaded TF processes on 14 cores measure contention
        # rather than doing work. The parent's batched path wanted many threads; this one wants one.
        #
        # EVAL_WORKER_THREADS=0 leaves TensorFlow's defaults alone. That exists because thread count
        # changes floating-point reduction order in the matmuls, which can flip argmax on a near-tie
        # and so produce a *different* (not wrong) policy realisation. Being able to turn the pinning
        # off is what makes "did pinning move the measured rate?" a testable question rather than a
        # suspicion.
        worker_threads = int(os.environ.get('EVAL_WORKER_THREADS', '1'))
        if worker_threads:
            tf.config.threading.set_intra_op_parallelism_threads(worker_threads)
            tf.config.threading.set_inter_op_parallelism_threads(worker_threads)

        from tf_agents.environments import tf_py_environment
        from tf_agents.utils import common

        from eval_agent import build_eval_agent
        from snake_environment import SnakeEnvironment
        from state_helpers import is_perfect_score

        py_env = SnakeEnvironment(discount=0.99, display=False, policy_name=policy_name)
        tf_env = tf_py_environment.TFPyEnvironment(py_env)
        agent, checkpoint, global_step = build_eval_agent(tf_env, py_env, ckpt_dir)
        policy_action = common.function(agent.policy.action)

        def one_episode():
            steps = 0
            total_reward = 0.0
            time_step = tf_env.reset()
            while True:
                action_step = policy_action(time_step)
                time_step = tf_env.step(action_step.action)
                reward = float(time_step.reward.numpy()[0])
                total_reward += reward
                steps += 1
                if bool(time_step.is_last().numpy()[0]):
                    # Read the score before the next reset overwrites it, exactly as the batched
                    # path does with its `call('get_score')`. The score is also what decides
                    # whether this was a perfect game — never the final reward, which a shaping
                    # term shifts off `PERFECT_GAME_REWARD`. See `state_helpers.is_perfect_score`.
                    score = py_env.get_score()
                    return score, is_perfect_score(score), total_reward, steps

        results.put((TAG_READY, rank))

        while True:
            command = commands.get()
            if command[0] == 'exit':
                return
            if command[0] == 'load':
                step = command[1]
                checkpoint.restore(
                    os.path.join(ckpt_dir, 'ckpt-{0}'.format(step))).expect_partial()
                results.put((TAG_LOADED, rank, int(global_step.numpy())))
                continue
            if command[0] == 'run':
                quota = command[1]
                completed = 0
                while completed < quota and not stop_flag.is_set():
                    score, perfect, reward, steps = one_episode()
                    completed += 1
                    results.put((TAG_EPISODE, Episode(rank, score, perfect, reward, steps)))
                results.put((TAG_DONE, rank, completed))
    except Exception as error:                                  # noqa: BLE001 - reported, not hidden
        import traceback
        results.put((TAG_ERROR, rank, traceback.format_exc()))
        raise


class IndependentWorkerPool:
    """A pool of processes that each evaluate whole episodes on their own.

    Lifecycle mirrors how `eval_checkpoints.measure` already works, so the call site changes by two
    lines: `load(step)` where it restored the parent's checkpoint, and `run(episodes, ...)` where it
    called `evaluate`.

    Not reusable after `close()`.
    """

    def __init__(self, policy_name, ckpt_dir, num_workers, context=None):
        self._policy_name = policy_name
        self._ckpt_dir = ckpt_dir
        self._num_workers = num_workers
        ctx = context or multiprocessing.get_context('spawn')
        self._results = ctx.Queue()
        self._stop_flag = ctx.Event()
        self._commands = [ctx.Queue() for _ in range(num_workers)]
        self._procs = [
            ctx.Process(target=_worker_main,
                        args=(rank, policy_name, ckpt_dir, self._commands[rank],
                              self._results, self._stop_flag),
                        daemon=True)
            for rank in range(num_workers)]
        self.last_global_step = None
        for proc in self._procs:
            proc.start()
        self._await(TAG_READY, num_workers)

    @property
    def num_workers(self):
        return self._num_workers

    def _await(self, tag, count):
        """Waits for `count` messages of `tag`, surfacing worker exceptions instead of hanging."""
        seen, values = 0, []
        while seen < count:
            message = self._results.get()
            if message[0] == TAG_ERROR:
                raise RuntimeError('eval worker {0} failed: {1}'.format(message[1], message[2]))
            if message[0] == tag:
                seen += 1
                values.append(message)
        return values

    def load(self, step):
        """Restores one checkpoint in every worker. Returns the global_step they read back."""
        for queue in self._commands:
            queue.put(('load', step))
        acks = self._await(TAG_LOADED, self._num_workers)
        restored = {ack[2] for ack in acks}
        if len(restored) != 1:
            raise RuntimeError(
                'workers disagree about the restored global_step: {0}'.format(sorted(restored)))
        self.last_global_step = restored.pop()
        return self.last_global_step

    def run(self, episodes, on_progress=None, should_abandon=None):
        """Measures `episodes` episodes. Same return shape as `eval_checkpoints.evaluate`."""
        self._stop_flag.clear()
        quotas = split_quota(episodes, self._num_workers)
        started = time.time()
        for queue, quota in zip(self._commands, quotas):
            queue.put(('run', quota))
        scores, perfect_flags, rewards, steps, abandoned = collect_results(
            self._results, self._num_workers, self._stop_flag, episodes_total=episodes,
            on_progress=on_progress, should_abandon=should_abandon)
        elapsed = time.time() - started
        print('    {0} episodes in {1}s ({2} env steps/s, {3} independent workers)'.format(
            len(scores), round(elapsed, 1),
            round(steps / elapsed) if elapsed else 0, self._num_workers))
        return scores, perfect_flags, rewards, elapsed, abandoned

    def close(self):
        for queue in self._commands:
            try:
                queue.put(('exit',))
            except (OSError, ValueError):
                pass
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
