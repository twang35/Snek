"""Batched checkpoint evaluation: the vectorised replacement for `eval_checkpoints.py`.

**What is reused and what is replaced.** Everything about the *result* — the row schema, the payload,
the checkpoint selectors, the atomic write, the chart — comes from `eval_plan` and `eval_progress`
unchanged, so a file written here is readable by every existing tool and directly comparable with a
TF close-out. Only the measurement engine is new: `vectorized.vec_engine` runs one wide numpy env
serving several checkpoints at once instead of `EVAL_WORKERS` processes serving one board each.

Two builders of the same artefact that drift apart is a failure this project has already paid for
(two `build_eval_agent`-shaped constructions, where `expect_partial()` hid the mismatch), which is why
this imports the payload machinery rather than reimplementing a schema that merely looks the same.

**Differences from `eval_checkpoints.py`, all deliberate:**

| | eval_checkpoints | here |
|---|---|---|
| output suffix | `EVAL_OUT_SUFFIX`, default none | `_vec` by default, so a *hand-run* can never overwrite a TF result |
| chart directory | `evals/` | `evals/vec/` by default, so it cannot displace a real close-out's charts (`VEC_EVAL_CHART_DIR`) |
| staging | screen / confirm tiers | **flat**: every checkpoint gets the same episode count |
| abandon gate | `EVAL_MIN_ACHIEVABLE=97` | **none** |
| algorithm | ddqn or c51 | ddqn only, hard-fail on c51 |

The chart directory matters more than it looks. `archive_existing_eval_pngs` moves **every** chart at
the top of `evals/` into the archive before any eval starts, and this has twice blanked a finished
batch's panels. Writing to `evals/vec/` means this driver never calls it at all. `vec_wave.py`, which
*is* the close-out, points `VEC_EVAL_CHART_DIR` at `evals/` and does that archiving once for the whole
wave -- so the rule is unchanged, it just belongs to the wave rather than to each of its shards.

Flat rather than staged is not a simplification for its own sake. Staging exists to avoid paying full
length for a checkpoint that will not place; at ~20x the throughput that saving buys little, and it
costs a lot of interpretive load — `pooled_equal_effort`, the `screen_episodes` field, the rule that
rows of different depths must not be pooled, and the gate recorded in every payload all exist to cope
with rows of unequal effort. Flat rows make all of that vacuously true.

**c51 is refused rather than attempted.** The greedy action for a categorical agent is
`argmax_a sum_i z_i p_i(s, a)`, so a c51 checkpoint restored against the wrong support loads
perfectly and evaluates a *different policy*. Supporting it means reading the support out of
`arch.json` and reducing over atoms here; until that is written and parity-tested, refusing is the
only safe answer.

Usage:

    PYTHONPATH=. python -u vectorized/vec_eval.py <policy_name> [selector]

    selector:  top50            the close-out selector, ranked on the graph eval (default)
               above:98         checkpoints a prior close-out measured at >= 98%  (the HOF pass)
               all              every checkpoint on disk
               1234000,1235000  explicit steps

    VEC_EVAL_EPISODES   episodes per checkpoint (default 100; the HOF pass uses 500)
    VEC_EVAL_WIDTH      total env lanes (default 1024)
    VEC_EVAL_MAX_LIVE   checkpoints resident at once (default: derived from width and
                        episodes by vec_engine.default_max_live — see the note there, a value
                        too low collapses utilisation to a few percent)
    VEC_EVAL_SOURCE     result suffix `above:` selects from (default '', the close-out's own file)
    VEC_EVAL_SEED       food-sampler seed (default 0); the only stochastic input to a measurement
    VEC_EVAL_CHART_DIR  where the progress chart goes (default `evals/vec/`)
    EVAL_OUT_SUFFIX     output suffix (default '_vec')
"""

import collections
import json
import os
import sys
import time

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts

import chart_viewer
import eval_agent
import eval_plan
import eval_progress
import policy_arch
import snake_constants
from snake_constants import EVALS_DIR
from snake_environment import SnakeEnvironment
from vectorized import config as C
from vectorized import vec_engine

POLICY_ROOT = 'savedPolicies'
# `evals/vec/` by default so a hand-run probe can never displace a real close-out's charts, and
# so a vec eval and a TF eval can run side by side during a validation comparison -- which is the
# whole point of such a comparison. `vec_wave.py` overrides it to `evals/`, because a wave *is* the
# close-out and its charts have to land where every viewer and every doc already looks.
CHART_DIR = os.environ.get('VEC_EVAL_CHART_DIR') or os.path.join('evals', 'vec')

# How many recent completions the ETA averages over. Long enough that one slow checkpoint does not
# swing it -- a coiled endgame arm's per-checkpoint time varies by ~2x -- and short enough that the
# estimate tracks a real change in pace, such as a sibling process finishing and handing back cores.
ETA_WINDOW = 60


def _hms(seconds):
    """`None` -> `--`, otherwise a compact `1h23m` / `4m12s`. An ETA nobody can read is not an ETA."""
    if seconds is None:
        return '--'
    seconds = int(max(0, seconds))
    if seconds >= 3600:
        return '{0}h{1:02d}m'.format(seconds // 3600, (seconds % 3600) // 60)
    if seconds >= 60:
        return '{0}m{1:02d}s'.format(seconds // 60, seconds % 60)
    return '{0}s'.format(seconds)
DEFAULT_SUFFIX = '_vec'


def available_steps(policy_dir):
    """Every checkpoint step on disk, ascending, read from the `.index` files.

    Read off the files rather than the `checkpoint` metadata file, because a hall-of-fame directory
    has no metadata file at all — which is how `tf.train.latest_checkpoint` once returned None and a
    champion silently restored nothing, then scored 0.
    """
    steps = []
    for name in os.listdir(policy_dir):
        if name.startswith('ckpt-') and name.endswith('.index'):
            try:
                steps.append(int(name[len('ckpt-'):-len('.index')]))
            except ValueError:
                continue
    return sorted(steps)


def resolve_selection(policy_name, policy_dir, selector, source_suffix):
    """`(steps, selected_by)` for a selector string, using `eval_plan`'s own selectors."""
    steps = available_steps(policy_dir)
    if not steps:
        raise SystemExit('no checkpoints in {0}'.format(policy_dir))
    if selector == 'all':
        return steps, {step: {'selected_by': 'all'} for step in steps}
    if selector.startswith('above:'):
        return eval_plan.select_checkpoints_above(
            policy_name, steps, float(selector.split(':', 1)[1]), source_suffix)
    if selector.startswith('top'):
        count = int(selector[3:]) if selector[3:] else eval_plan.DEFAULT_COUNT
        return eval_plan.select_top_checkpoints(policy_name, steps, count=count)
    explicit = [int(part) for part in selector.replace(',', ' ').split()]
    missing = [step for step in explicit if step not in steps]
    if missing:
        raise SystemExit('no such checkpoint(s): {0}'.format(missing))
    return explicit, {step: {'selected_by': 'explicit'} for step in explicit}


class AgentPool:
    """A fixed set of restorable greedy agents, handed out one per resident checkpoint.

    Built once because building is nearly free (0.03 s) and restoring is cheaper still (0.005 s), so
    the pool costs nothing and keeps the number of live TensorFlow graphs bounded and known.

    Each agent gets its own `tf.function` with an `input_signature` whose batch dimension is `None`.
    That is load-bearing: a job's lane count changes on almost every step as its episodes finish, and
    without a fixed signature every change would retrace the policy — which is precisely the
    per-call overhead this whole exercise exists to escape.
    """

    def __init__(self, size, py_env, tf_env, policy_dir):
        self.policy_dir = policy_dir
        self.free = []
        self.entries = []
        obs_len = int(py_env.observation_spec().shape[0])
        for _ in range(size):
            agent, checkpoint, _ = eval_agent.build_eval_agent(tf_env, py_env, policy_dir)
            policy = agent.policy

            @tf.function(input_signature=[tf.TensorSpec([None, obs_len], tf.float32)])
            def act(observation, policy=policy):
                rows = tf.shape(observation)[0]
                step = ts.TimeStep(
                    step_type=tf.fill([rows], tf.constant(ts.StepType.MID, dtype=tf.int32)),
                    reward=tf.zeros([rows], dtype=tf.float32),
                    discount=tf.ones([rows], dtype=tf.float32),
                    observation=observation)
                return policy.action(step).action

            entry = {'agent': agent, 'checkpoint': checkpoint, 'act': act}
            self.entries.append(entry)
            self.free.append(entry)

    def acquire(self, step):
        """Restore `ckpt-<step>` into a free agent and return `(entry, policy_fn)`."""
        entry = self.free.pop()
        prefix = os.path.join(self.policy_dir, 'ckpt-{0}'.format(step))
        # The explicit prefix, never `latest_checkpoint`. A directory without a `checkpoint`
        # metadata file returns None from that, and `.restore(None).expect_partial()` restores
        # nothing *silently* — measured once as a 90%-perfect champion scoring 0, 0, 1.
        entry['checkpoint'].restore(prefix).expect_partial()
        act = entry['act']

        def policy_fn(observation):
            return act(tf.convert_to_tensor(observation)).numpy()

        return entry, policy_fn

    def release(self, entry):
        self.free.append(entry)


def load_resumable(out_path, episodes):
    """`{step: held}` for rows in `out_path` that can stand as this run's measurement of that step.

    A relaunched job re-measuring from zero is the whole cost this exists to avoid: the desktop marks
    a job `interrupted` after a reboot and dispatches it again, and a HOF pass is thousands of
    checkpoints. `eval_plan.held_from_row` is the inverse of `build_row`'s three `episode_*` fields
    and returns None for anything it cannot rebuild faithfully, so a row that predates per-episode
    storage, or one that was hand-edited, is silently re-measured rather than trusted.

    **Depth has to match exactly.** A row measured at 100 episodes is not this run's 500-episode
    measurement of that step, and pooling the two would produce a file whose rows have different
    weights while claiming to be flat -- the precise property that makes a vec file safe to pool.
    A whole-file depth mismatch is reported and the file ignored, because that means the caller
    changed `VEC_EVAL_EPISODES` and almost certainly wants a fresh measurement.
    """
    if not os.path.exists(out_path):
        return {}
    try:
        with open(out_path) as handle:
            payload = json.load(handle)
    except (ValueError, OSError) as error:
        print('  resume: ignoring {0} ({1})'.format(os.path.basename(out_path), error))
        return {}
    stored = payload.get('episodes_per_checkpoint')
    if stored is not None and stored != episodes:
        print('  resume: ignoring {0} — it holds {1}-episode rows, this run wants {2}'.format(
            os.path.basename(out_path), stored, episodes))
        return {}
    out, skipped = {}, 0
    for row in payload.get('results', []):
        if row.get('episodes') != episodes or row.get('abandoned'):
            skipped += 1
            continue
        held = eval_plan.held_from_row(row)
        if held is None:
            skipped += 1
            continue
        out[row['step']] = held
    if skipped:
        print('  resume: {0} stored row(s) could not be reused and will be re-measured'.format(
            skipped))
    return out


def shard_steps(steps, shard, shards):
    """This shard's slice of `steps`, strided so the shards interleave.

    Every step lands in exactly one shard and their concatenation is a permutation of the input --
    asserted in `tests/test_vec_eval_shard.py`, because a slice that dropped or duplicated a
    checkpoint would produce a merged file that looks complete.
    """
    return list(steps)[shard - 1::shards]


def parse_shard(raw):
    """`"3/12"` -> `(3, 12)`; empty -> `(1, 1)`. Shards are 1-based so a shell `for i in $(seq 12)`
    reads naturally, and `steps[shard - 1::shards]` is the stride that uses it.

    Validated rather than trusted: a typo that silently measured the wrong slice would produce a
    result file that looks complete and covers a third of the arm.
    """
    if not raw:
        return 1, 1
    try:
        shard, shards = (int(part) for part in raw.split('/'))
    except ValueError:
        raise SystemExit('VEC_EVAL_SHARD must look like "3/12", got {0!r}'.format(raw))
    if shards < 1 or not 1 <= shard <= shards:
        raise SystemExit('VEC_EVAL_SHARD out of range: {0!r} (expected 1..n of n)'.format(raw))
    return shard, shards


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    policy_name = argv[0]
    selector = argv[1] if len(argv) > 1 else 'top50'

    episodes = int(os.environ.get('VEC_EVAL_EPISODES', 100))
    width = int(os.environ.get('VEC_EVAL_WIDTH', vec_engine.DEFAULT_WIDTH))
    max_live = int(os.environ.get('VEC_EVAL_MAX_LIVE', 0)) or None
    suffix = os.environ.get('EVAL_OUT_SUFFIX', DEFAULT_SUFFIX)
    source_suffix = os.environ.get('VEC_EVAL_SOURCE', '')
    # Seeds the env's food sampler. Exposed because it is the only stochastic input to a measurement
    # — the policy is greedy and the opening board is fixed — so re-running with a different seed is
    # exactly "draw another independent sample", which is how a difference between two engines is
    # told apart from a difference between two samples.
    seed = int(os.environ.get('VEC_EVAL_SEED', 0))
    # On by default: a relaunch that silently re-measures thousands of checkpoints is the more
    # surprising behaviour, and every row it reuses is depth-checked before it is trusted.
    resume = os.environ.get('VEC_EVAL_RESUME', '1') != '0'
    policy_dir = os.path.join(POLICY_ROOT, policy_name)
    if not os.path.isdir(policy_dir):
        raise SystemExit('no such policy directory: {0}'.format(policy_dir))

    py_env = SnakeEnvironment()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    # `refuse_categorical` rather than a hand-rolled check, so the refusal reads the sidecar through
    # the same loader every other tool does and cannot disagree with it about what c51 looks like.
    policy_arch.refuse_categorical(policy_dir, 'vectorized/vec_eval.py')

    steps, selected_by = resolve_selection(policy_name, policy_dir, selector, source_suffix)
    if not steps:
        print('nothing selected for {0} — exiting cleanly'.format(policy_name))
        return 0

    shard, shards = parse_shard(os.environ.get('VEC_EVAL_SHARD', ''))
    if shards > 1:
        # **Strided, not contiguous.** Per-checkpoint cost tracks policy quality -- a better
        # checkpoint plays longer episodes -- and quality drifts monotonically along a training run,
        # so contiguous blocks would hand one shard every slow checkpoint and another every fast one.
        # A stride interleaves them, which is what makes the shards finish together instead of one
        # process running long after the rest have freed their cores.
        steps = shard_steps(steps, shard, shards)
        # The suffix is extended rather than trusted to the caller: every shard writes
        # `<policy>_checkpoint_evals<suffix>.json`, so a forgotten per-shard suffix would have twelve
        # processes overwriting one file and the loss would look like a short selection rather than a
        # collision. `eval_plan.merge_checkpoint_evals` globs `_checkpoint_evals*`, so it finds these.
        suffix = '{0}-s{1}of{2}'.format(suffix, shard, shards)
        if not steps:
            print('shard {0}/{1} of {2} is empty — exiting cleanly'.format(
                shard, shards, policy_name))
            return 0

    base_suffix = os.environ.get('EVAL_OUT_SUFFIX', DEFAULT_SUFFIX)
    out_path = os.path.join(snake_constants.RUNS_DIR,
                            '{0}_checkpoint_evals{1}.json'.format(policy_name, suffix))
    chart_path = os.path.join(CHART_DIR, '{0}_eval_progress.png'.format(policy_name))
    os.makedirs(CHART_DIR, exist_ok=True)
    # One writer per arm. The chart is drawn from the *arm's* files, and every shard would render the
    # same picture from its own slice, so letting all of them write means N processes racing on one
    # path for a strictly worse image -- each shard sees only its own stride. Shard 1 draws using
    # every shard's suffix, so the picture is the arm's, not the shard's.
    draws_chart = shards == 1 or shard == 1
    chart_suffixes = ((suffix,) if shards == 1 else
                      tuple('{0}-s{1}of{2}'.format(base_suffix, i, shards)
                            for i in range(1, shards + 1)))
    eval_plan.backup_previous_results(out_path)

    print('vectorised eval: {0}'.format(policy_name))
    print('  {0} checkpoints x {1} episodes = {2} episodes, flat, no abandon gate'.format(
        len(steps), episodes, len(steps) * episodes))
    max_live = max_live or vec_engine.default_max_live(width, episodes)
    if shards > 1:
        print('  shard {0} of {1} (strided): {2} of the arm\'s checkpoints'.format(
            shard, shards, len(steps)))
    print('  width {0} lanes, {1} checkpoints resident, seed {2}'.format(
        width, max_live, seed))
    print('  {0}'.format(C.describe()))
    print('  results -> {0}'.format(out_path))
    print('  chart   -> {0}'.format(chart_path))
    # Never `archive_existing_eval_pngs` *here*. At the default `evals/vec/` there is nothing of
    # anyone else's to displace; and when `vec_wave.py` points this at `evals/`, the wave has already
    # archived once with the right `keep_batches`, so a second call per shard -- twelve of them, after
    # the first shards have started writing -- is exactly the mistake that blanked two batches'
    # finished panels.
    # The lock namespace follows the directory, not the tool. Two viewers over one directory would
    # each show the other's panels, so a run writing into `evals/` has to contend for the *same*
    # `-eval` slot a TF eval would -- and a run writing into `evals/vec/` has to contend for its own,
    # or whichever launched first would suppress the other's window entirely.
    chart_viewer.spawn_for_eval(policy_name, watch='vec_eval.py {prefix}',
                                chart_dir=CHART_DIR,
                                slot_suffix='eval' if CHART_DIR == EVALS_DIR else 'veceval')

    spec = eval_plan.PayloadSpec(
        policy_name=policy_name, num_episodes=episodes, all_steps=list(steps),
        # **None, and the batch width must not go here.** In this schema `num_workers` does not mean
        # "how parallel is it" -- it means "episodes advance in indivisible rounds of this size", and
        # `eval_progress.remaining_episodes` multiplies every checkpoint still ahead by
        # `whole_rounds(episodes, num_workers)`. That is right for the batched TF path, which really
        # does run one episode per worker per round. This engine runs an exact quota and never rounds,
        # so reporting width=1024 against a 100-episode request rounded each checkpoint up to 1024
        # and inflated the chart's ETA by **10.24x** -- b45's four arms read 6-8 h against a true
        # ~50 min. `whole_rounds` returns the request unchanged for a falsy count, which is exactly
        # the arithmetic this run wants. The width is in the run header and in `arm_eta_window`.
        num_workers=None,
        screen_episodes=None, confirm_count=None,
        # Both None because there is no gate. `build_payload` writes them as null, which is exactly
        # what a reader needs to know: this file's rows are all full length.
        min_achievable=None, abandon_floor=None,
        measurements_planned=len(steps), episodes_planned=len(steps) * episodes,
        full_planned=len(steps), screen_planned=0, confirm_planned=0)

    pool = AgentPool(max_live, py_env, tf_env, policy_dir)
    resumed = load_resumable(out_path, episodes) if resume else {}
    pending = [step for step in steps if step not in resumed]
    if resumed:
        print('  resuming: {0} of {1} checkpoints already measured, {2} to go'.format(
            len(resumed), len(steps), len(pending)))
    in_use = {}
    # Seeded with the resumed rows so the file this run writes covers the whole selection rather
    # than only the part it measured itself — a resumed run's output has to be indistinguishable
    # from an uninterrupted one, or `complete` means nothing downstream.
    samples = dict(resumed)
    cache = eval_plan.RowCache()
    for step, held in resumed.items():
        cache.put(step, eval_plan.build_row(step, held, selected_by.get(step)))
    gate = eval_plan.WriteGate()
    chart_gate = eval_plan.WriteGate(eval_plan.CHART_MIN_INTERVAL)
    started_at = time.time()
    # `measurements` is the selection's progress and drives the ETA and the `[ n/N ]` line;
    # `session` and `episodes` are this process's own work and drive the pace, which must not be
    # diluted by rows it never ran.
    counters = {'measurements': len(resumed), 'episodes': 0, 'session': 0}
    # Completion timestamps for the ETA, newest last. A trailing window rather than the whole run:
    # the opening seconds include the TF import, the first `max_live` checkpoint restores and a batch
    # that is still filling, so a run-long mean reads slow for the first few minutes and never fully
    # recovers on a long arm.
    completions = collections.deque(maxlen=ETA_WINDOW)

    def eta_seconds():
        """Wall-clock seconds until this arm finishes, from its own measured completion rate.

        Priced in **checkpoints completed per second**, which is the one rate that already accounts
        for this engine's concurrency: up to `max_live` checkpoints are in flight at once, so a row's
        own `seconds` field is concurrent wall clock and summing those over-counts by roughly the
        residency factor (18x on b45a). Completions are the observable that does not need correcting.

        Returned as `arm_eta_seconds`, which `eval_progress.summarize` prefers over its own estimate
        -- the right precedence, because this driver knows its pace and the file cannot express the
        lane migration that produced it.
        """
        if len(completions) < 2:
            return None, 0
        span = completions[-1] - completions[0]
        if span <= 0:
            return None, len(completions)
        rate = (len(completions) - 1) / span
        remaining = max(0, len(steps) - counters['measurements'])
        return remaining / rate, len(completions)

    def rows():
        return cache.rows(samples, {}, lambda step: selected_by.get(step))

    def flush(complete=False, force=False):
        if not (force or gate.due()):
            return
        gate.record()
        done = counters['measurements']
        progress = {'measurements': done, 'session_measurements': counters['session'],
                    'session_episodes': counters['episodes'],
                    'session_seconds': time.time() - started_at,
                    'episodes_saved': 0, 'abandoned': 0,
                    'stage': 'full', 'full_done': done, 'screen_done': 0, 'confirm_done': 0}
        eta, window = eta_seconds()
        progress['arm_eta_seconds'] = eta
        # Doubles as the width's home now that `num_workers` is None: the reader of an ETA wants to
        # know what it was averaged over, and the lane count is what the pace is a property of.
        progress['arm_eta_window'] = window
        progress['wave_lanes'] = width
        # `in_flight` is deliberately left None. The payload's block describes **one** checkpoint —
        # its round, its running rate — and this driver has up to `max_live` in flight at once, so
        # there is no honest single value to put there and naming one of twelve would misreport the
        # other eleven. Nothing is lost: that block exists because a ~5-minute TF measurement is
        # invisible until it lands, and here measurements land every few seconds, so the completed
        # rows *are* the progress display.
        eval_plan.write_payload(out_path, eval_plan.build_payload(
            spec, progress, samples, rows(), complete, in_flight=None))

    def redraw(force=False):
        if not (draws_chart and (force or chart_gate.due())):
            return
        chart_gate.record()
        try:
            eval_progress.live_frame(policy_name, out_path=chart_path, suffixes=chart_suffixes)
        except Exception as error:
            # A chart is never worth a measurement, the same rule the eval viewer follows.
            print('chart refresh failed ({0}: {1}) — measurement continues'.format(
                type(error).__name__, error))

    def next_job():
        if not pending:
            return None
        step = pending.pop(0)
        entry, policy_fn = pool.acquire(step)
        in_use[step] = entry
        return step, policy_fn

    def on_complete(step, held):
        pool.release(in_use.pop(step))
        samples[step] = held
        cache.put(step, eval_plan.build_row(step, held, selected_by.get(step)))
        counters['measurements'] += 1
        counters['session'] += 1
        counters['episodes'] += len(held['scores'])
        row = cache.rows({step: held}, {}, lambda s: selected_by.get(s))[0]
        completions.append(time.time())
        elapsed = time.time() - started_at
        eta, _ = eta_seconds()
        print('[ {0:>4}/{1} ] {2:>9}  {3:>5.1f}% perfect  avg score {4:>5.2f}  '
              '{5:.1f}s  ({6:.1f} ckpt/min, eta {7})'.format(
                  counters['measurements'], len(steps), step, row['perfect_percent'],
                  row['avg_score'], held['seconds'], 60.0 * counters['session'] / elapsed,
                  _hms(eta)))
        flush()
        redraw()

    if not pending:
        print('  every selected checkpoint is already measured — writing the payload and exiting')
        flush(complete=True, force=True)
        redraw(force=True)
        return 0

    stats = vec_engine.measure_stream(
        next_job, on_complete, episodes, width=width, max_live=max_live, seed=seed)
    flush(complete=True, force=True)
    redraw(force=True)

    elapsed = stats.get('seconds', time.time() - started_at) or 1e-9
    # Totals and session work are reported separately, because a resumed run's rates are properties
    # of what it actually ran: dividing this session's seconds by the whole selection would flatter
    # a resume by exactly the fraction it inherited.
    print('\ndone: {0} of {1} checkpoints ({2} measured here, {3} episodes) in {4:.1f}s'.format(
        counters['measurements'], len(steps), counters['session'], counters['episodes'], elapsed))
    print('  {0:.2f} s/checkpoint, {1:.1f} episodes/s, {2:.0f} env-steps/s, '
          'utilisation {3:.0f}%'.format(
              elapsed / max(1, counters['session']), counters['episodes'] / elapsed,
              stats['env_steps'] / elapsed, 100.0 * stats.get('utilisation', 0.0)))
    best = eval_plan.best_full_length_row(rows(), episodes)
    if best:
        print('  best checkpoint: {0} at {1:.1f}% ({2} episodes)'.format(
            best['step'], best['perfect_percent'], best['episodes']))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
