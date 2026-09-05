"""Train one arm. `python train.py <policy> [--max-steps N]`

    python train.py b1a-baseline
    SNEK_SEED=2 SNEK_FORK_BRANCHES=4 python train.py b1b-fork4

**What lives here is what is not algorithm-specific**: the environment-variable config, seeding, the
`arch.json` sidecar, the checkpoint cadence, the stage-A self-eval and its queue, the progress chart,
the run report and the step cap. **The algorithm sits behind the seam `dqn/algo.py` documents** —
fourteen members, of which `advance()` is the whole loop body — so a second algorithm adds a module
and one entry in `ALGOS` rather than a second copy of this file.

That is not tidiness. An arm's numbers are comparable across algorithms only if the same code screened
the checkpoints, ran the same 100 episodes, wrote the same rows and drew the same chart, so **the one
thing that must not be duplicated for PPO is this file.** See [`plans/ppo.md`](plans/ppo.md).

## The two intervals that are not knobs

The self-eval **is** stage A of the measurement (see [`docs/protocol.md`](docs/protocol.md)): its
100 episodes decide which checkpoints a stage-B wave measures at 500. So the eval interval must equal
the checkpoint interval — otherwise a checkpoint exists that no screen can select — and the episode
count must be 100, because the gate is literally "95 of 100". Both are constants below rather than
`tuned()` calls, and `SNEK_GRAPH_EVAL_EPISODES` exists only to make a smoke test cheap.

**The equality survives an algorithm whose step is larger than the interval**, which is the one
thing the seam had to add here: both are rounded up to a whole number of algorithm steps by
`_whole_steps`, so they stay equal by construction at any width. DQN's granularity is 1, so nothing
about a DQN arm moved.

## Cost, stated plainly

Stage A dominates an arm's wall clock and that is now correct rather than wasteful: a single
checkpoint's 100 episodes drain the measurement engine's lanes with nothing to refill them, so it
runs at ~17 ep/s against ~96 ep/s streamed. Measured on a 3M-step DQN arm, that is **5.3 h of 8.1 h
(66%)**, not the ~90% this note first claimed. `SNEK_EVAL_QUEUE` — on by default — moves it to shared
streamed workers and recovers 3.3-3.4x of it, at a bounded lag on the schedule that
[`tools/eval_queue.py`](tools/eval_queue.py) states exactly.
"""

import argparse
import os
import time

import numpy as np
import torch

from dqn import algo as dqn_algo
from ppo import algo as ppo_algo
# For `trailing_mean` only, which is a plain windowed average over the eval rows and is not about
# epsilon. It lives beside the epsilon schedule because that is its other caller; when `ppo/` lands
# and needs the same trailing score, it moves somewhere both algorithms can reach without one
# importing the other.
from dqn import schedules
from env import constants
from tools import arch as arch_tools
from tools import checkpoints
from tools import eval_queue
from tools import live_runs
from tools import progress_chart
from tools import restore
from tools import run_report
from vectorized import config as reward_config
from vectorized import engine

# **The algorithms this build knows, by their `SNEK_ALGO` value.** A dict rather than an `if`, for the
# reason `tools/restore.py` gives for the same shape: adding PPO is one line, and an unrecognised
# value names itself in the error instead of falling through to a default.
ALGOS = {dqn_algo.NAME: dqn_algo, ppo_algo.NAME: ppo_algo}

# ---------------------------------------------------------------- config

def tuned(name, default, cast=float):
    """Reads a hyperparameter from `SNEK_<NAME>`, printing any override.

    Sweeps run several arms side by side from this one file, so what varies between them comes from
    the environment rather than an edit. Everything read here lands in `run_config`, so
    `runs/<arm>.md` records what the run actually used — and `grep 'hyperparameter override:'` on a
    log confirms an arm received its config, which is why this prints rather than returning quietly.
    """
    raw = os.environ.get('SNEK_' + name)
    if raw is None:
        return default
    value = cast(raw)
    print('hyperparameter override: {0} = {1} (default {2})'.format(name, value, default))
    return value


# ---------------------------------------------------------------- the pinned intervals

# **One knob sets both intervals, and that is the point.** The self-eval is stage A of the
# measurement, so a checkpoint written at a step no eval screens is a checkpoint that can never be
# measured. Expressing the two as one value makes them equal by construction rather than by a comment
# asking the next person not to change one of them. It is settable only so a smoke test need not
# train 1,000 steps to reach its first eval; an arm leaves it at 1,000.
EVAL_INTERVAL = int(tuned('EVAL_INTERVAL', 1000, int))
CHECKPOINT_INTERVAL = EVAL_INTERVAL

# 100 because the stage-B gate is literally "95 of 100"; a different denominator is a different gate.
# Lower it for a smoke test and nothing else.
EVAL_EPISODES = int(tuned('GRAPH_EVAL_EPISODES', 100, int))

# The rolling resume state: net, optimiser, step, and the replay buffer beside it. Ten times the
# checkpoint interval because it is ~20 MB against a checkpoint's ~190 KB and it only warm-starts the
# next run, where a checkpoint is evidence.
RESUME_INTERVAL = 10 * EVAL_INTERVAL
RESUME_FILENAME = 'resume.pt'

# The trailing window the checkpoint gate and the chart read. Evals, not steps.
TRAILING_WINDOW = 30

# One log line per this many evals, plus the first and every new best.
QUIET_EVAL_INTERVAL = 25

# How long a blocked trainer waits for a live worker to publish the front of its queue before
# measuring it itself. A whole round of 32 checkpoints lands inside this, so a slow worker is never
# mistaken for a stuck one; a worker that is alive and not progressing costs one duplicated eval per
# `QUEUE_WAIT_PATIENCE` and nothing more.
QUEUE_WAIT_PATIENCE = 45.0

# The tail drain's patience: how long nothing may land before the trainer stops waiting and measures
# the backlog itself. Generous, because at the cap the queue is at its deepest and a worker is at its
# most efficient — one streamed checkpoint lands in ~1-2 s, so silence this long means a worker that is
# gone rather than one that is busy. Being wrong costs seconds; being impatient cost 3x.
FINAL_DRAIN_PATIENCE = 20.0
FINAL_DRAIN_POLL = 0.25

# How long a blocked trainer waits for a worker that already holds the checkpoint it needs, before
# measuring it anyway. Short, because the wait is only worth taking when it saves a duplicate: one
# streamed checkpoint lands in ~1-2 s, and past that a stalled arm costs more than 100 repeated
# episodes.
RECLAIM_GRACE_SECONDS = 2.0

# The report and the chart are rewritten on their own cadence: both rewrite the whole file from the
# whole series, so doing it per eval would spend a growing fraction of the run on I/O.
REPORT_INTERVAL = 10 * EVAL_INTERVAL


def _whole_steps(target, granularity):
    """`target` rounded up to a whole number of algorithm steps, and never zero.

    DQN's granularity is 1, so all three intervals come out exactly as written and a DQN arm is
    unaffected. PPO's smallest step is a whole rollout — 16,384 transitions at the planned default —
    and an interval that is not a multiple of it would put an eval on a step the loop never reaches,
    which for stage A means a checkpoint no screen can select.
    """
    granularity = max(1, int(granularity))
    whole = -(-int(target) // granularity)
    return granularity * max(1, whole)


def build_config():
    """Every knob, read once. The returned dict is also what the run report prints.

    **Each key is its `SNEK_` variable, lowercased, without exception.** So a `| priority_exponent |
    0.6 |` row in `runs/<arm>.md` tells you the knob is `SNEK_PRIORITY_EXPONENT` with no lookup — and
    a report row that cannot be grepped back to a knob is a config nobody can reproduce.
    `tests/test_train.py` pins the correspondence.
    """
    name = str(tuned('ALGO', dqn_algo.NAME, str))
    if name not in ALGOS:
        raise ValueError('SNEK_ALGO={0!r} is not an algorithm this build knows; it has {1}'.format(
            name, sorted(ALGOS)))
    config = {
        'algo': name,
        'seed': int(tuned('SEED', 1, int)),
        'torch_threads': int(tuned('TORCH_THREADS', 1, int)),
        'max_steps': int(tuned('MAX_STEPS', 10000000, int)),
        # A single hidden layer of 320, which is what every record-holding snek2 arm used — see the
        # `arch.json` beside the imported champions. Phase 3 is a reproduction, so the architecture
        # is not the thing being varied.
        'fc_layers': tuple(int(width) for width in
                           str(tuned('FC_LAYERS', '320', str)).split(',')),
        # Read through `tuned()` like every other knob, so all three print an
        # `hyperparameter override:` line and appear in `runs/<arm>.md` as a row that greps straight
        # back to its variable. They were briefly read inside `tools/eval_queue.py` instead, which
        # broke that contract twice over: the report rows had no matching knob, and a config passed in
        # programmatically was ignored in favour of the environment.
        'eval_queue': bool(int(tuned('EVAL_QUEUE', 1, int))),
        'eval_queue_depth': int(tuned('EVAL_QUEUE_DEPTH', eval_queue.DEFAULT_DEPTH, int)),
        'eval_workers': int(tuned('EVAL_WORKERS', eval_queue.DEFAULT_WORKERS, int)),
        'graph_eval_episodes': EVAL_EPISODES,
        'eval_interval': EVAL_INTERVAL,
        'min_checkpoint_score': constants.MIN_CHECKPOINT_SCORE,
    }
    # **The algorithm's own knobs, and its own validation, come from its module.** One flat dict
    # still, because `runs/<arm>.md` prints it as one table and every key is still its `SNEK_`
    # variable lowercased — what moved is only where the call sits.
    config.update(ALGOS[name].build_config(tuned))
    return config


def reportable(config):
    """`run_config` for the report: whatever the algorithm cannot print as a table row, flattened."""
    return ALGOS[config['algo']].reportable(config)


# ---------------------------------------------------------------- the self-eval

def eval_seed(seed, step):
    """A fresh seed per eval, derived from the arm's seed and the step.

    Derived rather than fixed: a fixed eval seed measures the same 100 boards every time, and a
    policy that improves only on those boards would look like a policy that improved. Derived rather
    than random so that a resumed run re-measuring the step it stopped on gets the same boards.
    """
    return int(np.random.SeedSequence([int(seed), int(step)]).generate_state(1, dtype=np.uint32)[0])


def summarise(held):
    """One measured sample as the five fields a row stores.

    Split from `self_eval` because a queued measurement arrives as a `held` sample from another
    process and must become the same five numbers by the same arithmetic — two summarisers is how the
    in-process and queued paths would quietly disagree about what an `avg_reward` is.

    Every field comes from this one sample, including the extremes. snek2 tracked min and max across
    the whole training interval instead and printed them on the same table line as this eval's mean,
    which describes two different things in one row.
    """
    scores = held['scores']
    return {'avg_score': float(np.mean(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'avg_reward': float(np.mean(held['rewards'])),
            'perfect': float(np.mean(held['perfect']))}


def self_eval(algo, step, seed, episodes):
    """Stage A in this process: `episodes` greedy episodes of the agent's current weights.

    **`episodes` is required, and it used to default to `EVAL_EPISODES`.** A default argument is
    evaluated once at import, so the default froze whatever the module constant was *then* while every
    other reader — `_evaluate`, `_reclaim_oldest` — takes it at call time. Nothing in an arm noticed,
    because both are 100. A test that shrank `train.EVAL_EPISODES` to 4 did: the unqueued path kept
    measuring 100 episodes and the queued path measured 4, so a bit-exactness fixture failed with a
    diff in `avg_reward` that pointed nowhere near the cause.
    """
    return summarise(engine.measure(algo.policy_fn, episodes, seed=eval_seed(seed, step)))


def build_eval_row(step, measured, trailing, steps_per_second, algo_fields=None,
                   transitions=None):
    """One eval as the row the history stores.

    **`transitions` is the game-move count and `step` is not**, which is the units trap
    [`docs/findings.md`](docs/findings.md) records: `collector.step()` advances every lane, so at b2's
    `fork_branches=4` one counted step is four game moves, four buffer rows and four gradient steps.
    Recording both removes the ambiguity at the source rather than leaving it to a doc warning, and it
    is what a PPO row will be compared against — see [`plans/ppo.md`](plans/ppo.md).

    It counts the prefill, because those moves were played and are learned from, so it is not
    `step * width` plus nothing: a fresh arm's first row already carries
    `initial_collect_steps`-worth of them.

    **Absent rather than null when it is unknown.** An arm resumed from a `resume.pt` written before
    this field existed cannot know its true count, and a wrong absolute number on a comparison axis
    is worse than a missing one.

    `algo_fields` is merged verbatim and is **already rounded by the algorithm** — `epsilon` and
    `guided_fraction` for DQN, an entropy coefficient and its diagnostics for PPO. Merged rather than
    named so this function never has to know which algorithm produced the row; a `None` value is
    dropped, which is what keeps a control arm's row free of a `fork` field it never used.
    """
    row = {'step': int(step),
           'avg_score': round(measured['avg_score'], 2),
           'trailing_avg_score': round(trailing, 2),
           'min_score': round(measured['min_score'], 1),
           'max_score': round(measured['max_score'], 1),
           'avg_reward': round(measured['avg_reward'], 3),
           'perfect_percent': round(100.0 * measured['perfect'], 1)}
    # Same rule as `transitions`: absent when unknown. A queued step the trainer had to measure itself
    # with no request file left to read (2026-09-05, b17ai: the worker was between deleting its claim
    # and writing its result) has no speed to report, and a row without one beats no row -- the arm
    # died on this KeyError three relaunches running.
    if steps_per_second is not None:
        row['steps_per_second'] = round(float(steps_per_second), 1)
    if transitions is not None:
        row['transitions'] = int(transitions)
    for key, value in (algo_fields or {}).items():
        if value is not None:
            row[key] = value
    return row


# ---------------------------------------------------------------- the run

class Trainer(object):
    """Owns the loop's state. One instance per process; `run()` returns when the cap is reached."""

    def __init__(self, policy, config, device='cpu'):
        self.policy = policy
        self.config = config
        self.device = device
        self.policy_dir = os.path.join(constants.POLICY_DIR, policy)
        self.history_path = run_report.history_path(constants.RUNS_DIR, policy)
        self.report_path = os.path.join(constants.RUNS_DIR, policy + '.md')
        self.graph_path = progress_chart.chart_path(policy)

        self.arch = self._sidecar()
        self.algo = ALGOS[config['algo']].build(config, self.arch, device=device)

        # **The intervals are rounded up to whole algorithm steps, and that is what makes one
        # `train.py` serve both algorithms.** The self-eval *is* stage A, so a checkpoint written at a
        # step no eval screens can never be measured — and an algorithm whose smallest step is a
        # 16,384-transition rollout cannot land on a 1,000 grid. Rounding up to a multiple of the
        # granularity keeps eval and checkpoint equal by construction at any width. At DQN's
        # granularity of 1 every one of the three is unchanged, which is why the fingerprint of a DQN
        # arm is byte-identical across this refactor.
        self.eval_interval = _whole_steps(EVAL_INTERVAL, self.algo.step_granularity)
        # **The other two round to multiples of the *eval* interval, not of the granularity**, which
        # keeps `RESUME_INTERVAL % EVAL_INTERVAL == 0` true at any width — the property
        # `tests/test_train.py` asserts on the constants. Rounding both to the granularity instead
        # would give 12 and 20 from the test's own values, and 20 is not a multiple of 12: a resume
        # written at a step no eval produced a row for. It is a no-op at DQN's granularity, where the
        # eval interval already divides both.
        self.resume_interval = _whole_steps(RESUME_INTERVAL, self.eval_interval)
        self.report_interval = _whole_steps(REPORT_INTERVAL, self.eval_interval)
        # The report prints `run_config`, so it has to print the interval the arm *ran*, not the one
        # the knob asked for. They differ only when the algorithm's granularity rounded it up, which
        # for DQN is never — but a PPO report that claimed 1,000 while evaluating every 16,384 would
        # be a config row nobody could reproduce from.
        self.config['eval_interval'] = self.eval_interval

        self.step = 0
        # Game moves played, which `self.step` is not — see `build_eval_row`. `None` means "an older
        # resume file did not record it", which every row then omits rather than guessing.
        self.transitions = 0
        self.eval_rows = []
        self.resume_steps = []
        # Steps offered to the queue and not yet merged, oldest first. In-memory because it is only a
        # cursor — every field a row needs lives in the queue's own files, so a resume rebuilds this
        # from the directory rather than from a checkpointed list.
        self.pending_evals = []
        self.queue_depth = int(config['eval_queue_depth'])
        self.eval_workers = []
        self.skipped_checkpoints = 0
        self._resume()

    # ------------------------------------------------------------ setup and resume

    def _sidecar(self):
        """The `arch.json` beside the checkpoints: written on a fresh arm, checked on a resume.

        Checked rather than rewritten, because a resume that silently adopted a new
        `SNEK_FC_LAYER_PARAMS` would load the old weights into a different network — which torch
        would refuse, but only after the run had started and only with a shape error.
        """
        built = arch_tools.build_arch(self.config['fc_layers'], constants.NUM_ACTIONS,
                                      constants.OBS_LEN, constants.OBS_ERA, algo=self.config['algo'])
        if os.path.exists(arch_tools.arch_path(self.policy_dir)):
            existing = arch_tools.read_arch(self.policy_dir)
            arch_tools.assert_same_network(built, existing, '<config>', self.policy_dir)
            return existing
        arch_tools.write_arch(self.policy_dir, built)
        return built

    def _resume_path(self):
        return os.path.join(self.policy_dir, RESUME_FILENAME)

    def _resume(self):
        """Picks up net, optimiser, step, buffer and eval history, or starts fresh.

        The buffer is restored under the same condition as the weights and never on its own: a
        resume that paired old weights with a much newer buffer would train the restored network on
        experience it never generated.
        """
        path = self._resume_path()
        if not os.path.exists(path):
            return
        payload = torch.load(path, map_location=self.device, weights_only=True)
        # **A pre-seam `resume.pt` has no `algo` block** — it holds `agent`, `epsilon` and
        # `guided_fraction` at the top level. Handing the payload itself to the algorithm in that case
        # reads those files correctly rather than stranding b1's and b2's, and it cannot misread one:
        # the key names are identical in both layouts, so a wrong guess would raise on a missing key
        # rather than restore something else.
        self.algo.load_state_dict(payload.get('algo', payload))
        self.step = int(payload['step'])
        self.transitions = payload.get('transitions')
        if self.transitions is not None:
            self.transitions = int(self.transitions)
        loaded = self.algo.load_side_state(self.policy_dir)
        self.eval_rows, resumes = run_report.load_history(self.history_path)
        self.resume_steps = list(resumes) + [self.step]
        print('resumed at step {0:,}: {1} eval row(s), buffer {2}'.format(
            self.step, len(self.eval_rows), 'restored' if loaded else 'empty'), flush=True)
        if self.config['eval_queue']:
            # Whatever the previous run offered and never merged. Adopted rather than discarded so the
            # history has no gap at the resume boundary, which is the worst place for one — the resume
            # step is re-measured and would sit beside missing neighbours.
            measured = {row['step'] for row in self.eval_rows}
            self.pending_evals = [step for step in eval_queue.outstanding(self.policy)
                                  if step not in measured]
            if self.pending_evals:
                print('adopted {0} queued eval(s) from {1:,} to {2:,}'.format(
                    len(self.pending_evals), self.pending_evals[0], self.pending_evals[-1]),
                    flush=True)

    # ------------------------------------------------------------ the loop

    def run(self):
        cap = self.config['max_steps']
        if self.step >= cap:
            print('already at or past the {0:,}-step cap — nothing to train. Raise SNEK_MAX_STEPS '
                  'to continue this arm.'.format(cap), flush=True)
            return
        self._bank_transitions(self.algo.prefill())
        print('{0}: training to {1:,} steps, {2}'.format(
            self.policy, cap, self.algo.describe()), flush=True)
        # The reward and shaping terms, which `hyperparameter override:` does NOT cover: those knobs
        # are read by `env/constants.py` at import, not through `tuned()`, so grepping the log for
        # overrides — which the docs name as the way to confirm an arm got its config — is silent on
        # exactly the settings a shaping experiment is about. b2's `SNEK_CHASE_SAFE_SHAPING` had to be
        # confirmed by reading `/proc/<pid>/environ`, which is not a thing anyone should have to do.
        print('reward config: ' + reward_config.describe(), flush=True)
        # Registered before the first step, so the scheduler -- which reads this registry to adopt an
        # arm a killed predecessor left running, and to count trainers against the cap -- sees this arm
        # at once. **The trainer opens no window** (2026-09-05): the box's one chart window is the
        # scheduler's, and it follows the scheduler's own status file; see `tools/window.py` for why
        # the arms racing for it was the cause of every window incident. Nothing in this loop may ever
        # depend on a window.
        live_runs.register(self.policy)
        if self.config['eval_queue']:
            # Shared across every arm on this box, and idempotent: the slot claim means four arms
            # launched in the same second produce `eval_workers` workers rather than four times as
            # many. Starting none is not a failure — `_reclaim_oldest` measures this arm's own
            # checkpoints, so the arm runs at today's speed instead of not running.
            # `target` from the config, not from the environment. `ensure_workers` defaults to
            # reading `SNEK_EVAL_WORKERS` itself, and letting it do that here meant a caller that set
            # `eval_workers` in the config — every test, and any programmatic launch — was ignored:
            # a fixture asking for zero workers got two, which then measured its checkpoints at their
            # own seed and broke a bit-exactness assertion in a way that pointed at the wrong module.
            self.eval_workers = eval_queue.ensure_workers(self.config['eval_workers'])
            print('stage-A queue on: depth {0}, {1} worker(s) wanted, {2} started here. Rows arrive '
                  "up to {0} evals behind, and the algorithm's schedule reads them at that lag.".format(
                      self.queue_depth, self.config['eval_workers'], len(self.eval_workers)),
                  flush=True)

        window_start, window_step = time.time(), self.step
        while self.step < cap:
            previous = self.step
            steps, transitions = self.algo.advance()
            self.step += steps
            self._bank_transitions(transitions)
            # **"Crossed a multiple of", not "is a multiple of".** An algorithm whose step advances by
            # more than one would step straight over an exact test and never evaluate at all. At DQN's
            # increment of 1 the two are the same expression.
            if self._crossed(previous, self.eval_interval):
                elapsed = max(time.time() - window_start, 1e-9)
                self._evaluate((self.step - window_step) / elapsed)
                window_start, window_step = time.time(), self.step
            if self._crossed(previous, self.resume_interval):
                self._save_resume()
            if self._crossed(previous, self.report_interval):
                self._write_report()
                # A worker this arm started is its child, and a batch outlives several worker
                # generations at five minutes of idle each. `reap` polls; it never blocks on a worker.
                self.eval_workers = eval_queue.reap(self.eval_workers)
        self._save_resume()
        # Before the report, so the close-out reads a complete history. An arm that stopped with its
        # queue full would otherwise be missing rows for its last `depth` intervals — the newest
        # region, which is exactly the part a close-out and a stage-B screen look at.
        if self.config['eval_queue']:
            self._drain(final=True)
            # Anything a worker published after this arm reclaimed the step. Rare — `complete` drops
            # its claim first and refuses to publish without one — and this is what makes rare into
            # gone, since the directory is otherwise never revisited.
            eval_queue.sweep(self.policy, keep=self.pending_evals)
        self._write_report()
        # Not in a `finally`, and not worth one: `live_runs.live` drops an entry whose pid is gone, so
        # a `kill -9`, a crash and a Ctrl-C all clean up on the next read. This call is only so the
        # scheduler sees the arm gone the moment it finishes rather than at the next scan.
        live_runs.unregister(self.policy)
        print('done at step {0:,}'.format(self.step), flush=True)

    def _crossed(self, previous, interval):
        """Whether the step just taken passed a multiple of `interval`."""
        return self.step // interval > previous // interval

    def _bank_transitions(self, transitions):
        """Adds to the game-move count, unless this arm does not know what its count is.

        A `None` stays `None` for the life of the run: a resume from a `resume.pt` that predates the
        field cannot recover the moves the earlier runs played, and counting from there would put a
        confident wrong number on a comparison axis.
        """
        if self.transitions is not None:
            self.transitions += int(transitions)

    # ------------------------------------------------------------ the eval

    def _trainer_fields(self, steps_per_second):
        """The half of an eval row that only the trainer knows, captured at the step it describes.

        `epsilon` and `guided_fraction` are what the arm was **using** while it collected this step.
        With the queue on they are the whole reason the row can still be built correctly several
        intervals later: by then the schedule has moved, and reading them at merge time would label a
        step with an epsilon it never ran under. `steps_per_second` and the fork counters are the same
        kind of thing — a window of training that is over and cannot be reconstructed.
        """
        return {'steps_per_second': float(steps_per_second),
                'transitions': self.transitions,
                'algo': self.algo.fields()}

    def _evaluate(self, steps_per_second):
        """Stage A for the step just reached — measured here, or offered to the queue.

        **The two modes share `_record`, so the arithmetic is the same in both.** What differs is when
        the measurement arrives, and therefore how far behind the schedule's view is. Without the
        queue the row is complete before this returns and the arm stays bit-reproducible from its
        seed; with it, see `tools/eval_queue.py` for the two things that changes.
        """
        fields = self._trainer_fields(steps_per_second)
        if not self.config['eval_queue']:
            self._record(self.step,
                         self_eval(self.algo, self.step, self.config['seed'], EVAL_EPISODES),
                         fields)
            return
        # Written before the offer and unconditionally, because a worker measures a checkpoint by
        # restoring it. That inverts the gate rather than removing it: `_settle_checkpoint` prunes the
        # file when the row lands below the bar, so the set left on disk is the same one.
        checkpoints.save(self.policy_dir, self.step, self.algo.net)
        eval_queue.enqueue(self.policy, self.step, fields, EVAL_EPISODES)
        self.pending_evals.append(self.step)
        self._drain()

    def _record(self, step, measured, fields):
        """Turns one measurement into the arm's next row: schedule, gate, history, log.

        Called at the step's own eval interval without the queue and up to `eval_queue_depth`
        intervals later with it. Every read of `self.eval_rows` here is "the rows that have landed
        before this one", and one `algo.on_eval` serves both modes for that reason.

        **‡ Not "the rows before this one in step order".** That is what this docstring claimed and
        `_merge_landed` retracted: a streamed round does not complete in step order and cannot be made
        to, so `_merge_landed` merges in ascending step order among whatever is *ready* rather than
        waiting for the front. The trailing window is therefore the last N landed rows. See that
        method for why waiting instead cost the queue its whole value.
        """
        # **Always, in both modes: this is what moves the algorithm's schedule on.** What differs by
        # mode is only which of its two spellings reaches the row — see the table below.
        computed = self.algo.on_eval(self.eval_rows, measured)

        trailing = schedules.trailing_mean(self.eval_rows, 'avg_score', measured['avg_score'],
                                           TRAILING_WINDOW)
        keep = self._settle_checkpoint(step, measured['avg_score'], trailing)
        # **The epsilon column shifts by one row between the two modes, and it is measured: every
        # queued row carries the value the unqueued row *before* it carries.** Verified bit-exact at
        # depth 0 over an 8-row arm. The two columns are different quantities and both are true:
        #
        # | mode | `row.epsilon` at step S is |
        # |---|---|
        # | unqueued | what this eval just set, so it governs the interval **starting** at S |
        # | queued | what the arm actually ran under, so it governs the interval **ending** at S |
        #
        # The queued spelling is the one a chart can plot honestly, because the value the schedule
        # computes here does not begin governing until `depth` intervals later — attributing it to S
        # would draw epsilon dropping before the eval that dropped it. The unqueued spelling is kept
        # as it is so b1 and b2 stay reproducible.
        #
        # **Nothing selects on this column**, which is why the shift is acceptable: `screen:97` reads
        # `perfect_percent`, and the measurement half of the row — score, reward, perfect rate and the
        # trailing mean — is bit-identical between the modes. `tests/test_train.py` pins both halves of
        # that sentence.
        #
        # The two spellings are merged rather than chosen between, because only the *schedule's* keys
        # differ: the fork counters are captured at the step in both modes and there is no computed
        # version of them to prefer.
        reported = dict(fields.get('algo') or {})
        if not self.config['eval_queue']:
            reported.update(computed)
        row = build_eval_row(step, measured, trailing, fields.get('steps_per_second'),
                             reported, fields.get('transitions'))
        run_report.merge_eval_row(self.eval_rows, row)
        summary = run_report.save_history(self.history_path, self.eval_rows, self.resume_steps)
        self._log(row, summary, keep)

    def _settle_checkpoint(self, step, avg_score, trailing):
        """Keeps or prunes `ckpt-<step>.pt`. The same final set on disk in either mode.

        Gated on `max(this eval, trailing)` rather than trailing alone: the best checkpoints in this
        project are outliers that spike above their neighbourhood, so a trailing-only test could skip
        exactly the one worth keeping while an arm recovers. Either signal clearing the bar is enough.

        **Only the direction differs by mode.** Without the queue the eval is in hand, so the file is
        written only if it is wanted. With the queue the file had to exist for a worker to restore, so
        this prunes instead — and it must never re-save, because by now `self.algo.net` holds much
        later weights and writing them under an old step number would forge a checkpoint.
        """
        keep = max(avg_score, trailing) >= self.config['min_checkpoint_score']
        if not keep:
            self.skipped_checkpoints += 1
            checkpoints.discard(self.policy_dir, step)
        elif not self.config['eval_queue']:
            checkpoints.save(self.policy_dir, step, self.algo.net)
        return keep

    # ------------------------------------------------------------ the queue

    def _drain(self, final=False):
        """Merges finished measurements and takes work back while the queue is over its bound.

        **The bound counts work still being measured, not rows still to merge**, because
        `_merge_landed` takes everything ready on every pass — so `pending_evals` holds exactly the
        steps no worker has finished. That is the quantity worth bounding: it is the schedule's blind
        spot and the queue's size at once.

        `final` drains to empty regardless of the bound, which is what the end of a run needs: an arm
        that stopped with `depth` measurements in flight would otherwise close out with that many
        missing rows at the top of its history — exactly the region a close-out and a stage-B screen
        read.
        """
        self._merge_landed()
        if final:
            self._drain_patiently()
            return
        waiting_since = time.time()
        while self.pending_evals and len(self.pending_evals) > self.queue_depth:
            if self._merge_landed():
                waiting_since = time.time()
                continue
            # **Do not seize a checkpoint a live worker is already measuring.** A round completes in
            # whatever order its lanes free up, not in step order, so the front of this queue is often
            # among the last of a round to land while later steps pile up behind it as `.done`.
            # Reclaiming then costs 100 duplicated episodes *and* discards the worker's own result for
            # the same step — measured as the difference between no speed-up and the real one. Waiting
            # is the depth bound doing its job: the arm must not run further ahead of its schedule.
            #
            # **The patience is a clock and not a queue length, and the length version deadlocked.**
            # "Wait while fewer than 4x depth are outstanding" cannot fire: this loop blocks before
            # returning to the training step, so the queue it is waiting on can never grow. A clock
            # always advances, so the reclaim below is always eventually reached.
            if self._leave_to_worker(self.pending_evals[0], waiting_since):
                time.sleep(FINAL_DRAIN_POLL)
                continue
            self._reclaim_oldest()
            waiting_since = time.time()

    def _drain_patiently(self):
        """Waits for the backlog rather than seizing it. Only reclaims once nothing is landing.

        **The impatient version cost most of the queue's win and it measured as a 1.13x.** At the cap
        the arm holds its deepest queue of the whole run, which is exactly when a worker is at its most
        efficient — `measure_stream` refills a finished lane from the next checkpoint, so a backlog of
        16 costs ~1.1 s each against ~4.2 s drained. Reclaiming the moment nothing had landed *yet*
        turned that backlog back into serial drained measurements, and the trainer and the worker then
        raced to redo the same work.

        So the rule inverts for the tail: wait while progress is being made, and reclaim only after
        `FINAL_DRAIN_PATIENCE` seconds of silence — which is a dead worker, not a busy one. Forward
        progress is still guaranteed, just no longer instant.
        """
        quiet_since = time.time()
        while self.pending_evals:
            if self._merge_landed():
                quiet_since = time.time()
                continue
            # **Waiting is only worth it if something is coming.** This is the one place the trainer
            # asks whether a worker exists, and it has to: with none alive the patience below is 20
            # seconds of sleep per checkpoint for no reason — it turned a 5-second fixture into 80.
            # The steady-state path still asks nothing, because there reaching the bound is itself the
            # signal.
            if not eval_queue.live_workers():
                self._reclaim_oldest()
                continue
            # No `held_by_live_worker` short-circuit here, and that is deliberate: the patience below
            # already expresses "wait for a live worker", and a `continue` above it would jump the
            # clock and spin forever on a front that never lands. `quiet_since` advances only on a
            # merge, so a stuck round times out.
            if time.time() - quiet_since >= FINAL_DRAIN_PATIENCE:
                self._reclaim_oldest()
                quiet_since = time.time()
                continue
            time.sleep(FINAL_DRAIN_POLL)

    def _leave_to_worker(self, step, waiting_since):
        """Whether to wait for the worker holding `step` rather than measuring it here.

        A predicate rather than an inline condition because it is the rule that decides whether the
        queue pays for itself, and a rule worth a fixture is worth a name. Two clauses, both
        load-bearing:

        - **the holder must still exist** — a dead worker's claim is not a promise, and reading its pid
          out of the claim filename is what makes that answerable rather than merely age-outable;
        - **the wait is a clock** — `QUEUE_WAIT_PATIENCE` from when this arm last made progress. It was
          a queue length first and that version **deadlocked**: "wait while fewer than 4x depth are
          outstanding" can never fire, because the caller blocks before reaching the training step that
          would grow the queue. A clock always advances.
        """
        if time.time() - waiting_since >= QUEUE_WAIT_PATIENCE:
            return False
        return eval_queue.held_by_live_worker(self.policy, step)

    def _merge_landed(self):
        """Merges every measurement that has landed, oldest step first. Returns how many.

        **‡ This held finished rows back until their predecessor landed, and that was wrong.** The
        reasoning was sound and the premise was not: strict step order does make the trailing window
        contain exactly what an unqueued arm's would, but a streamed round does not complete in step
        order and cannot be made to. `engine.measure_stream` keeps every queued checkpoint resident —
        `max_live` derives to 44 at width 1024 — and interleaves their episodes across all 1024 lanes,
        so which one finishes first is set by lane availability and is effectively arbitrary. Waiting
        for the front therefore serialises the whole arm behind the *slowest* member of a round.
        Measured: an arm sat at step 9,000 for 55 s with 2 of 9 landed rows unmerged, and the queue was
        worth nothing.

        So the order constraint is dropped and the trailing window becomes "the last N rows that have
        landed". That is a real difference from an unqueued arm and it is the honest one to accept: the
        schedule is already reading a measurement up to `depth` intervals old, so the window's exact
        membership is the smaller of the two approximations by a wide margin. **What is not
        approximate is any row's own content** — a measurement is of one checkpoint and does not
        depend on when its neighbours arrive — and `merge_eval_row` keeps the file in step order
        regardless. The depth-0 fixture still pins the bit-exact case, because at depth 0 rows arrive
        in order by construction.

        `pending_evals` is ascending (appends are), so iterating it merges oldest-first among whatever
        is ready, which keeps the trailing window's sequence sensible.
        """
        merged = 0
        for step in list(self.pending_evals):
            payload = eval_queue.landed(self.policy, step)
            if payload is None:
                continue
            self.pending_evals.remove(step)
            self._record(step, summarise(payload['held']), payload.get('fields') or {})
            eval_queue.retire(self.policy, step)
            merged += 1
        return merged

    def _reclaim_oldest(self):
        """Measures the oldest outstanding checkpoint in this process. The forward-progress guarantee.

        **This is why no arm can ever wait on a worker that is not coming.** It asks nothing about
        whether workers exist or are keeping up — reaching the bound is itself the signal — so with no
        workers at all the arm simply behaves as it does with the queue off, at that speed. (The one
        place that *does* ask is `_drain_patiently`, and only to decide how long to wait before calling
        this.)

        It measures **the front** of the queue rather than the first unclaimed step, because
        `_merge_landed` merges in step order: a row measured out of turn would sit waiting behind the
        front anyway and the bound would not move.

        `take_back` claims the request when it is still unclaimed, which is the common case and costs
        a worker nothing. When a worker does hold it, this measures it anyway after one short wait:
        a duplicated 100 episodes is a few seconds and a stalled arm is hours. Whichever sample the
        merge sees first wins and `retire` removes the other.
        """
        step = self.pending_evals[0]
        fields = eval_queue.take_back(self.policy, step)
        if fields is None:
            time.sleep(RECLAIM_GRACE_SECONDS)
            if eval_queue.landed(self.policy, step) is not None:
                return
            fields = eval_queue.fields_of(self.policy, step) or {}
            if not fields:
                # No file for the step at all: a worker is between deleting its claim and writing its
                # `.done` (found 2026-09-05 on a resumed arm sitting at its bound while eight workers
                # finished its adopted requests). Measured here anyway -- forward progress -- and if
                # the worker's result lands meanwhile it is preferred below, fields intact.
                print('step {0:,}: no request on disk; measuring it here, its row may carry no '
                      'steps_per_second'.format(step), flush=True)
        # Restored rather than measured from `self.algo.net`: the arm has trained on since this step
        # and stage A is a measurement of the checkpoint, not of the current weights.
        policy_fn, _, _ = restore.restore(self.policy_dir, step=step, device=self.device)
        held = engine.measure(policy_fn, EVAL_EPISODES,
                              seed=eval_seed(self.config['seed'], step))
        landed = eval_queue.landed(self.policy, step)
        if landed is not None and landed.get('fields'):
            held, fields = landed['held'], landed['fields']       # the worker's, complete; ours is the duplicate
        self.pending_evals.pop(0)
        self._record(step, summarise(held), fields)
        eval_queue.retire(self.policy, step)

    def _save_resume(self):
        """Net, optimiser, step, schedules — and the buffer beside it, under the same call."""
        payload = {'step': int(self.step), 'transitions': self.transitions,
                   'algo': self.algo.state_dict()}
        path = self._resume_path()
        os.makedirs(self.policy_dir, exist_ok=True)
        staging = path + '.partial'
        torch.save(payload, staging)
        os.replace(staging, path)
        self.algo.save_side_state(self.policy_dir)

    def _write_report(self):
        progress_chart.render(self.eval_rows, self.graph_path, name=self.policy,
                              resume_steps=self.resume_steps)
        run_report.write_run_report(self.report_path, self.policy, reportable(self.config),
                                    self.eval_rows, os.path.basename(self.graph_path),
                                    self.resume_steps)

    def _log(self, row, summary, keep):
        """One line per `QUIET_EVAL_INTERVAL` evals, plus the first and every new best-30.

        `> 0` on the best-30 value matters: with no perfect games yet `best_perfect30` is 0.0 and its
        step falls back to the current step, which would mark every eval as a new best.
        """
        # The row's step, not `self.step`. With the queue on those differ by up to `depth` intervals,
        # and reading the loop's position here would mark the wrong eval as a new best and quiet the
        # log on the wrong cadence.
        index = row['step'] // self.eval_interval
        is_best = (summary['best_perfect30']['value'] > 0
                   and summary['best_perfect30']['step'] == row['step'])
        if not (index % QUIET_EVAL_INTERVAL == 0 or is_best or index <= 1):
            return
        note = '  <- best so far' if is_best else ('' if keep else '  no ckpt')
        speed = row.get('steps_per_second')
        print('{0:>9,}  score {1:>5.1f}  trail {2:>5.1f}  pf {3:>3.0f}%  best30 {4:>4.1f}%  '
              '{5} {6} st/s{7}'.format(
                  row['step'], row['avg_score'], row['trailing_avg_score'], row['perfect_percent'],
                  summary['best_perfect30']['value'], self.algo.log_note(row),
                  '{0:>5.0f}'.format(speed) if speed is not None else '    ?', note),
              flush=True)
        for line in self.algo.log_extra(row):
            print(line, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policy', help='arm name; the directory is savedPolicies/<policy>')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='overrides SNEK_MAX_STEPS for this launch')
    parser.add_argument('--device', default='cpu', help='torch device (default cpu)')
    args = parser.parse_args(argv)

    config = build_config()
    if args.max_steps is not None:
        config['max_steps'] = int(args.max_steps)
    # **One thread by default, and it is a measured 1.4x rather than a guess.** The net is
    # 30 -> 320 -> 3; every op is far too small to amortise a fork-join, so torch's default of one
    # thread per core spends more time synchronising than computing — 950 gradient steps/s at 10
    # threads against 1,314 at one. It compounds on the laptop, where four arms run side by side.
    torch.set_num_threads(config['torch_threads'])
    torch.manual_seed(config['seed'])
    Trainer(args.policy, config, device=args.device).run()


if __name__ == '__main__':
    main()
