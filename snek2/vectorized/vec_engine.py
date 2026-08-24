"""Batched greedy measurement: one wide env, many checkpoints, one policy call per checkpoint.

**What this replaces.** `eval_checkpoints.py` measures a checkpoint by spawning `EVAL_WORKERS`
processes, each with its own TensorFlow arena (~230 MB) and its own pygame `Game`, stepping one board
per policy call. That runs at ~2.6 episodes/s per lane, so a 500-episode measurement takes ~183 s.
Almost all of it is per-call overhead: `tf.function` signature machinery dominates the actual ops,
and a batch of one row pays it in full.

Three measurements shaped everything here, all on this laptop with a 98%-perfect checkpoint:

| what | number |
|---|---|
| policy inference, batch 512 | 1.5M rows/s |
| env step **without** the observation, n=512 | 211 us |
| env step **with** it, n=512 | 4323 us |

So **the observation is 95% of a step and the policy is nearly free** — the exact inverse of the
per-worker shape. Two consequences drive the design. The env step gets cheaper per row as the batch
widens (55k steps/s at n=128, 174k at n=2048), so width is the lever; and since the policy is cheap,
one wide env can serve *several checkpoints at once*, each with its own policy call on its own slice.

**Why lanes migrate between checkpoints.** The obvious shape — K fixed blocks of `episodes` lanes —
was measured and it drains: every block starts full and empties as its episodes finish, so mean
utilisation was **39-50%** and the width advantage was half thrown away. Here a lane that finishes an
episode is immediately reassigned to whichever checkpoint still needs episodes started, loading the
next one if the current ones are all fully started. The batch therefore stays full until the *entire
selection* is done rather than until each checkpoint is.

**Episode accounting is the part that is easy to get statistically wrong.** Running N lanes and
stopping at the first N completions is *biased toward short episodes*: a lane that dies early
finishes early, so the ones still playing are the good ones. `eval_workers.split_quota` avoids this
with a fixed per-worker quota. This does the equivalent: a checkpoint has exactly `episodes` episodes
**started** on its behalf, and every started episode is run to completion, so no episode is ever
selected on its length. Which physical lane happens to run one is irrelevant — episodes are
independent given the policy.

**No abandon gate.** `EVAL_MIN_ACHIEVABLE` exists to stop paying 100 episodes for a checkpoint that
cannot reach the threshold. At this throughput that saving is not worth what it costs: an abandoned
row is shorter than a full one, so a file holding both cannot be pooled directly and every reader has
to know which gate it was measured under. Flat equal-effort rows are cheaper to produce here than
they are to reason about.
"""

import time

import numpy as np

from vectorized import config as C
from vectorized.vec_env import VecSnake

# Total lanes in the env. A step gets cheaper per row as the batch widens — 55k steps/s at n=128,
# 88k at 256, 118k at 512, 151k at 1024, 174k at 2048 — but the *run's* tail is paid at full width,
# so the best value is the widest batch a selection can keep busy. 1024 measured best on a 24-
# checkpoint/100-episode run (2.24 s/checkpoint against 2.45 at 2048, where 2400 episodes is barely
# more than one pass and the whole run is tail).
DEFAULT_WIDTH = 1024

# `max_live` is **derived**, not defaulted, because getting it wrong is catastrophic rather than
# suboptimal. A job's quota is consumed the moment its episodes are assigned, and it holds its slot
# until its *last* episode ends — so if `max_live * episodes` merely equals `width`, then after the
# opening assignment no resident job has quota left, no new one may load, and every lane that
# finishes goes idle until some checkpoint completes outright. Measured: width 1200, episodes 100,
# max_live 12 gave **4% utilisation** and ran 10x slower than the same work at max_live 24.
#
# The multiplier is the slack that fixes it: enough jobs to fill the batch (`width / episodes`), times
# four so there is always a fully-started-but-still-finishing cohort *and* a fresh one to draw from.
# The cost of surplus is one policy call per resident job per step, and a job with no live lanes is
# skipped, so overshooting is cheap — 48 was indistinguishable from 24 at width 1024.
MAX_LIVE_SLACK = 4
MIN_MAX_LIVE = 8


def default_max_live(width, episodes):
    """Resident-checkpoint budget for a width and an episode count. See `MAX_LIVE_SLACK`."""
    per_pass = -(-int(width) // max(1, int(episodes)))       # ceil
    return max(MIN_MAX_LIVE, MAX_LIVE_SLACK * per_pass)


class _Job:
    """One checkpoint being measured: its policy, its quota, and its accumulating sample."""

    __slots__ = ('key', 'policy_fn', 'episodes', 'started', 'scores', 'perfect', 'rewards',
                 'started_at', 'live')

    def __init__(self, key, policy_fn, episodes):
        self.key = key
        self.policy_fn = policy_fn
        self.episodes = episodes
        self.started = 0
        self.scores = []
        self.perfect = []
        self.rewards = []
        self.started_at = time.time()
        self.live = 0                    # lanes currently playing an episode for this job

    @property
    def done(self):
        return len(self.scores)

    def wants_more(self):
        return self.started < self.episodes

    def finished(self):
        return self.done >= self.episodes

    def held(self):
        """The sample in the shape `eval_plan.build_row` consumes.

        Raw per-episode lists rather than running totals, because that is what makes a row resumable
        and its median exact — a row rebuilt from summaries carries a quietly wrong median.
        """
        return {'scores': self.scores, 'perfect': self.perfect, 'rewards': self.rewards,
                'seconds': time.time() - self.started_at, 'abandoned': False}


def measure_stream(next_job, on_complete, episodes, width=None, max_live=None, seed=0,
                   shaping_discount=1.0, on_step=None):
    """Measure a stream of checkpoints in one wide env, calling `on_complete(key, held)` for each.

    `next_job` is called whenever a lane needs work and returns `(key, policy_fn)` or None when the
    stream is exhausted. `policy_fn` takes an `(m, OBS_LEN)` float32 array and returns `(m,)` int
    actions. Keeping it a plain callable is deliberate: this module imports no TensorFlow, so it can
    be tested and benchmarked against a hand-written policy without a 3-second import, and the driver
    is the only thing that has to know about agents and checkpoints.

    Returns a stats dict. Nothing is returned per checkpoint — results arrive through `on_complete`
    as they land, so a long run's file is written incrementally rather than at the end.
    """
    width = DEFAULT_WIDTH if width is None else int(width)
    max_live = default_max_live(width, episodes) if max_live is None else int(max_live)
    if width < 1 or max_live < 1:
        raise ValueError('width and max_live must be positive')
    if max_live > 1 and max_live * episodes <= width:
        raise ValueError(
            'max_live * episodes ({0} * {1} = {2}) must exceed width ({3}), or every lane that '
            'finishes an episode goes idle until a whole checkpoint completes — measured at 4% '
            'utilisation. Raise max_live to at least {4}.'.format(
                max_live, episodes, max_live * episodes, width,
                default_max_live(width, episodes)))

    vec = VecSnake(width, seed=seed, shaping_discount=shaping_discount)
    # -1 means the lane is idle. A lane is otherwise an index into `live`, which is compacted only
    # by removal, so the index is stable for a job's whole life.
    owner = np.full(width, -1, dtype=np.int64)
    running = np.zeros(width, dtype=np.float64)     # this episode's undiscounted return, per lane
    live = []                                       # resident jobs, in load order
    exhausted = False

    def take_job():
        """The resident job with quota left, loading a new one if none has any."""
        for job in live:
            if job.wants_more():
                return job
        nonlocal exhausted
        while not exhausted and len(live) < max_live:
            fresh = next_job()
            if fresh is None:
                exhausted = True
                return None
            job = _Job(fresh[0], fresh[1], episodes)
            live.append(job)
            if job.wants_more():
                return job
        return None

    def assign(rows, already_fresh=False):
        """Start an episode on each lane in `rows`, or leave it idle if there is no work left.

        `already_fresh` skips the reset for the opening assignment, where `VecSnake.__init__` has
        just reset every row. Resetting again is not harmless bookkeeping: each reset draws a food
        placement from the env's RNG, so the redundant pass shifted every lane's food stream by one
        and made a run's episodes impossible to reproduce from the seed by hand. Caught by
        `tests/test_vec_engine.py` comparing the engine's returns against a replay.
        """
        for row in rows:
            job = take_job()
            if job is None:
                owner[row] = -1
                continue
            # `take_job` is supposed to be the only place quota is decided, so this is a check on
            # that contract rather than a second implementation of it. It earns its place: a mutant
            # that let `take_job` hand back an exhausted job produced an infinite loop instead of a
            # failing test, because every other guard here is downstream of a job being *retired*.
            if not job.wants_more():
                raise RuntimeError(
                    'checkpoint {0} was assigned a lane past its {1}-episode quota'.format(
                        job.key, job.episodes))
            owner[row] = live.index(job)
            job.started += 1
            job.live += 1
            if not already_fresh:
                vec.reset_rows([row])
            running[row] = 0.0

    stats = {'env_steps': 0, 'batch_steps': 0, 'steps': 0, 'checkpoints': 0,
             'episodes': 0, 'started_at': time.time()}
    # An episode cannot outlast `MAX_STARVE_BUDGET` steps per meal, so while any lane is live some
    # episode must complete inside this many steps. Exceeding it means the loop is not making
    # progress, and raising beats spinning: the first version of this function had a step cap, the
    # rewrite dropped it, and a mutation test then hung for ten minutes instead of failing.
    stall_cap = C.MAX_STARVE_BUDGET * C.PERFECT_SCORE
    since_completion = 0

    assign(range(width), already_fresh=True)
    if owner.max() < 0:
        return stats                                # nothing to measure

    obs = vec.observe()
    while (owner >= 0).any():
        stats['steps'] += 1
        since_completion += 1
        if since_completion > stall_cap:
            raise RuntimeError(
                'no episode completed in {0} steps with {1} lanes live — an episode is bounded by '
                'the starve budget, so this means the loop is not advancing'.format(
                    stall_cap, int((owner >= 0).sum())))
        actions = np.zeros(width, dtype=np.int64)
        active = owner >= 0
        stats['env_steps'] += int(active.sum())
        stats['batch_steps'] += width
        # One policy call per resident job, on that job's own lanes. An idle lane still steps — the
        # env is one array — but its action and its result are discarded, so a zero is fine and keeps
        # each call one contiguous batch instead of a gather and a scatter.
        for index, job in enumerate(live):
            rows = np.flatnonzero(owner == index)
            if rows.size:
                actions[rows] = job.policy_fn(obs[rows])

        # `observe=False` plus one explicit `observe()` below, rather than letting `step` build the
        # observation and rebuilding it after the resets. The observation is 95% of the cost of a
        # step, so computing it twice on every step where a lane finished nearly doubled the bill.
        _, reward, done, _ = vec.step(actions, autoreset=False, observe=False)
        running[active] += reward[active]

        hits = np.flatnonzero(done & active)
        if hits.size:
            since_completion = 0
            for row in hits:
                job = live[owner[row]]
                # The invariant that makes over-counting loud. A lane must never be restarted past
                # its checkpoint's quota, and if it is, every rate in the output file is computed
                # over the wrong denominator — a silent wrong answer, and the exact failure a
                # mutation test produced as an infinite loop rather than a wrong number.
                if len(job.scores) >= job.episodes:
                    raise RuntimeError(
                        'checkpoint {0} banked more than its {1} episodes — a lane was restarted '
                        'past its quota'.format(job.key, job.episodes))
                score = int(vec.score[row])
                job.scores.append(score)
                # Off the score, never the reward. Three counters in this project compared a final
                # reward with PERFECT_GAME_REWARD, and the moment a shaping term shipped they all
                # read 0% while the arms were filling boards.
                job.perfect.append(int(score == C.MAX_POSSIBLE_SCORE))
                job.rewards.append(float(running[row]))
                job.live -= 1
                owner[row] = -1
            stats['episodes'] += int(hits.size)

            # Retire completed jobs before reassigning, so their slots are freed for the next
            # checkpoint in the same pass and `live.index` stays meaningful.
            complete = [job for job in live if job.finished()]
            if complete:
                for job in complete:
                    on_complete(job.key, job.held())
                    stats['checkpoints'] += 1
                keep = [job for job in live if not job.finished()]
                remap = {live.index(job): new for new, job in enumerate(keep)}
                fresh_owner = np.full(width, -1, dtype=np.int64)
                for old, new in remap.items():
                    fresh_owner[owner == old] = new
                owner[:] = fresh_owner
                live[:] = keep

            assign(hits)

        obs = vec.observe()

    stats['seconds'] = time.time() - stats['started_at']
    stats['utilisation'] = (stats['env_steps'] / stats['batch_steps']
                           if stats['batch_steps'] else 0.0)
    return stats


def measure(policy_fn, episodes, lanes=None, seed=0, shaping_discount=1.0):
    """Measure a single checkpoint. Returns the `held` sample.

    Expressed through `measure_stream` rather than as a second loop. Two builders of the same thing
    that drift apart is a failure this project has already paid for more than once, and an engine
    used only by tests would be exactly the copy that drifts.
    """
    jobs = iter([('single', policy_fn)])
    out = {}
    width = lanes if lanes else min(episodes, DEFAULT_WIDTH)
    # One job, so the capacity rule cannot be satisfied by more residents — the single-checkpoint
    # path simply *is* the drain case, and that is fine: it exists for tests and one-offs, not for
    # throughput. `max_live=1` with the guard bypassed is the honest way to say so.
    measure_stream(lambda: next(jobs, None), lambda key, held: out.update(held),
                   episodes, width=min(width, episodes), max_live=1, seed=seed,
                   shaping_discount=shaping_discount)
    if len(out.get('scores', ())) != episodes:
        raise RuntimeError('measured {0} episodes, expected {1}'.format(
            len(out.get('scores', ())), episodes))
    return out
