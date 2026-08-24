"""Measures a saved checkpoint's true perfect-game rate over many episodes.

The graph in runs/<policy>.png plots one 10-episode eval per point, so its value moves in
10-point jumps and is far too coarse to quote. This script reloads specific checkpoints and
evaluates each over hundreds of episodes. The graph point is still a good *selector* even
though it is a bad measurement — see select_top_checkpoints.

Protocol rationale, measured costs and cross-batch comparability rules live in
hyperparamTuning/hyperparamTuning.md. This docstring covers how to run it and the traps.

Usage:

    cd snek2
    PYTHONPATH=. python -u eval_checkpoints.py <policy_name> <step> [<step> ...]

    # the normal close-out
    EVAL_WORKERS=10 PYTHONPATH=. python -u eval_checkpoints.py <policy_name> top50

    # flat one-pass protocol (every arm before batch 10 was measured this way)
    EVAL_SCREEN_EPISODES=0 ... top50

    # continue an interrupted close-out
    EVAL_RESUME=1 ... top50

Selection (`top50`, `top`, `top:50`) ranks on the single graph eval, breaking ties
on the surrounding rate. Selection (`above:98`, `above`) instead reads a *prior close-out's*
100-episode measurement and takes every checkpoint whose `perfect_percent` is at or above the
threshold — the HOF re-measure path, which reconfirms already-excellent checkpoints rather than
re-discovering them from the noisy graph (see select_checkpoints_above):

- every checkpoint at **>=95%** is measured, even past N
- remaining slots go to the best of the rest down to **>=90%**
- nothing below 90% is measured

N is a target, not a quota — a weak arm may run 1 or 0, and a *strong* arm blows past N, which is
what makes a continuation close-out expensive. Because a graph point is **20 episodes** (since
2026-08-19; it was 10 before, and the thresholds were 90/60 to match), the mandatory tier is
exactly {95, 100} and the fill band is exactly {90}. Adjacent steps are
allowed through on purpose: 1000 train steps can change the perfect rate by tens of points, so
neighbours are separate policies rather than repeat samples.

Three stages when screening is on (the default):

1. every checkpoint whose graph point is **>=95%** (19/20 or 20/20 of a 20-episode graph eval),
   plus any explicitly named step, gets the full EVAL_EPISODES immediately. Uncapped — but bounded
   by EVAL_MIN_ACHIEVABLE, which abandons a row once the gate is unreachable, and that is what
   makes an uncapped tier affordable. See ALWAYS_FULL_SINGLE.
2. everything else selected gets EVAL_SCREEN_EPISODES (20)
3. the best EVAL_CONFIRM_COUNT **of those screened** get the remaining 80

A promoted checkpoint ends with exactly EVAL_EPISODES, so its number is comparable with arms
measured flat. The 100% tier is excluded from confirmation slots — it already has the
measurement a slot would buy. Since 2026-08-19 the full tier *is* the mandatory tier, so stage 2
holds only the fill band (>=90% and <95%) and stage 3 ranks within it.

Early abandonment (EVAL_MIN_ACHIEVABLE, default 97) stops a checkpoint once its ceiling — its
rate if every remaining episode were perfect — falls below the gate. At 100 episodes and a 97%
gate that is "stop once more than 3 have failed". The rule is arithmetic, not predictive, which
is what makes it safe: a checkpoint that would reach the gate is never stopped, and an abandoned
row's own rate is always below the gate, so it can never outrank a kept one.

What the gate costs, and it is not nothing:

- **best-checkpoint degrades on an arm that never clears the gate**, which at 95 is most arms.
  Such an arm has no full-length row at all; best_full_length_row falls back to half-depth rows
  and the printed line is marked `[truncated]`.
- **abandoned rows are shorter, noisier and not comparable with full-length rows.** They carry
  `abandoned: true`; the payload records `min_achievable` so files measured under different
  gates can be told apart. Do not pool raw rows across them.
- **pooled_equal_effort is exact at any gate** — it truncates to screen depth, and the abandon
  floor is never below that. It is the arm-level figure to use.

Take the arm-level rate from the equal-effort figure the run prints, never by pooling the rows
in the output file: those have different depths and the deep ones are by construction the arm's
best, so pooling weights the winners 5x.

Results are written after every checkpoint, so an interrupted run keeps what it measured;
`complete` stays false until the last one lands. EVAL_RESUME=1 relaunches the identical command
and skips whatever the file already holds at full length. Part-measured checkpoints are redone
from scratch, so no run pools two summaries of one checkpoint.

Environment:
    EVAL_EPISODES     episodes per checkpoint, rounded up to whole rounds (default 100)
    EVAL_WORKERS      parallel envs inside this process (default 10)
    EVAL_OUT_SUFFIX   appended to the output filename
    EVAL_SCREEN_EPISODES  screen depth before promotion (default 20; 0 turns screening off)
    EVAL_CONFIRM_COUNT    how many screened checkpoints get promoted (default 100)
    EVAL_MIN_ACHIEVABLE   abandon once this rate is unreachable (default 97; 0 disables)
    EVAL_ABANDON_FLOOR    never abandon before this many episodes (default 20, raised to
                      EVAL_SCREEN_EPISODES if larger)
    EVAL_RESUME       1 to skip checkpoints already measured at full length, or a
                      comma-separated list of suffixes to take them from
    EVAL_PERFECT_WAIT_MS  window pause on a perfect game (default 400)
    EVAL_RENDER       1 to show a game in a window (default 0, all workers headless)

Traps, all of them learned the hard way:

**Use a throwaway EVAL_OUT_SUFFIX for anything exploratory.** The first write happens seconds
in and unconditionally overwrites whatever is at that path, so reusing the suffix of a complete
measurement destroys it immediately, killed early or not. backup_previous_results() keeps one
rolling `.previous` copy, but a distinct suffix is what actually prevents this.

**Prefer a worker count that divides EVAL_EPISODES.** Episodes round up to whole rounds, so 12
workers turn a 100-episode request into 108 and the rows stop matching the rest of the arm.
More workers is close to free; fewer is actively slower *and* costs more CPU per episode. XLA
(jit_compile=True) measured *worse* here, 0.38 s/episode against 0.32, and is not used.

**Rendering is the most expensive thing in an eval** — 163us per game step headless against
6050us in a window — and because ParallelPyEnvironment waits for the slowest worker, one
rendering worker paces all of them. Hence headless by default. EVAL_RENDER=1 puts a window on
worker 0 for watching by hand; it cannot affect the numbers.

Two things about that window look like bugs and are not, both cosmetic because only each
worker's *first* episode of a round is counted:

1. **It stops mid-game and closes.** Workers that finish early keep being stepped and
   auto-reset into uncounted episodes, so the visible worker is usually part-way through a
   throwaway game when the round ends.
2. **It used to freeze for seconds.** snake_constants.PERFECT_GAME_WAIT_MS blocks the whole
   round via pygame.time.wait(); EVAL_PERFECT_WAIT_MS overrides it.

**evals/ holds only the latest work.** Before writing anything this script moves whatever is
in evals/ into a timestamped evals/archive/ folder. Nothing is lost, but **any** eval launched
for any reason displaces every chart there, including a one-checkpoint verification run — and
EVAL_OUT_SUFFIX does not protect it, since the chart path has no suffix in it. Simultaneous
processes do not archive each other (the archive step runs before any of them writes a chart);
starting one *more* while another runs archives the live one's chart, which reappears within a
round.

A live progress window always opens and cannot be suppressed; it disables itself if no window
can be created, so unattended runs need no configuration. It pools every result file for the
policy, so several processes on one arm each show the same consolidated view. To attach a
window to an eval already running, use
`EVAL_PROGRESS_WINDOW_MODE=1 EVAL_PROGRESS_WATCH=20 python -u eval_progress.py <policy>`.

Results go to runs/<policy_name>_checkpoint_evals<suffix>.json. Two levels of parallelism
compose: EVAL_WORKERS spreads one checkpoint's episodes, and several copies of this script can
run on different checkpoints. Give each copy its own EVAL_OUT_SUFFIX or they overwrite each
other; merge afterwards with merge_checkpoint_evals().
"""
import json
import os
import sys
import time

os.environ['SDL_VIDEODRIVER'] = 'dummy'  # must precede any pygame import
# Belt-and-braces against audio. Snake.Game inits only display+font, but a bare
# pygame.init() anywhere would open a CoreAudio stream per worker and spin coreaudiod
# (measured 15% CPU for 10 idle workers). Unlike the video driver, this is never
# unset for the visible worker — nothing in this project plays sound.
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.system import system_multiprocessing
from tf_agents.utils import common

import snake_constants
from eval_agent import build_eval_agent
from eval_workers import IndependentWorkerPool
from snake_constants import EVALS_DIR, POLICY_DIR, RUNS_DIR
from snake_environment import SnakeEnvironment
from state_helpers import is_perfect_score

# The planning and record-keeping half of this file now lives in `eval_plan.py`, which is
# deliberately TensorFlow-free so `eval_wave.py` can plan a wave without paying for an arena. This
# re-export is the compatibility surface: every name below used to be defined here, so callers and
# the 90 fixtures in `tests/test_eval_checkpoints.py` keep resolving `eval_checkpoints.build_row`
# and friends. Do not collapse it into `import eval_plan` — that would break them all.
from eval_plan import (  # noqa: F401 - re-exported for callers and tests
    ALWAYS_EVAL_SINGLE,
    ALWAYS_FULL_SINGLE,
    CHART_MIN_INTERVAL,
    DEFAULT_ABANDON_FLOOR,
    DEFAULT_ABOVE_THRESHOLD,
    DEFAULT_CONFIRM_COUNT,
    DEFAULT_COUNT,
    DEFAULT_MIN_ACHIEVABLE,
    DEFAULT_SCREEN_EPISODES,
    HOF_EPISODES,
    HOF_GATE,
    HOF_SUFFIX,
    MIN_EVAL_SINGLE,
    PayloadSpec,
    RowCache,
    WRITE_MIN_INTERVAL,
    WriteGate,
    achievable_percent,
    backup_previous_results,
    best_full_length_row,
    build_payload,
    build_row,
    equal_effort_pooled,
    held_from_row,
    hof_settings,
    load_finished_results,
    make_abandon_test,
    merge_checkpoint_evals,
    pick_finalists,
    plan_stages,
    protocol_from_sources,
    resolve_screen_episodes,
    resume_suffixes,
    select_checkpoints_above,
    select_top_checkpoints,
    skips_screening,
    wilson_interval,
    write_payload,
)


def run_round(parallel_env, policy_action, worker_envs):
    """One episode per worker, every worker run to completion.

    Deliberately *not* "collect N finished episodes and stop". Truncating mid-flight throws away
    the episodes still running, and which ones those are is correlated with the outcome, so the
    surviving sample is not a random one. Running whole rounds costs some idle time on the fast
    workers and keeps the estimate unbiased.

    Which way the bias runs is worth knowing if you reason about a variant of this loop:
    **truncation would read high, not low.** Perfect games average ~1780 steps against ~2200 for
    non-perfect ones — a win ends the moment the board fills, while a policy about to fail circles
    until the starve budget runs out — so truncating drops *failures* preferentially.

    The idle time is real: across 20-40 workers, 20-35% of the env steps in a round belong to
    workers that already finished. Giving each worker a fixed back-to-back quota instead of a
    barrier every episode would recover ~1.2x and stay unbiased. Not implemented.
    """
    num_workers = len(worker_envs)
    scores = np.zeros(num_workers, dtype=np.float64)
    rewards = np.zeros(num_workers, dtype=np.float64)
    perfect = np.zeros(num_workers, dtype=bool)
    done = np.zeros(num_workers, dtype=bool)
    steps = 0

    time_step = parallel_env.reset()
    while not np.all(done):
        action_step = policy_action(time_step)
        time_step = parallel_env.step(action_step.action)
        step_rewards = time_step.reward.numpy()
        is_last = time_step.is_last().numpy()

        active = ~done
        rewards[active] += step_rewards[active]
        steps += int(np.sum(active))

        # A finished worker auto-resets on its next step, so read get_score now,
        # before that reset overwrites current_score.
        newly_done = active & is_last
        if np.any(newly_done):
            indices = np.flatnonzero(newly_done)
            promises = [worker_envs[i].call('get_score') for i in indices]
            for i, promise in zip(indices, promises):
                scores[i] = promise()
                # From the score, not `step_rewards[i]`: a shaping term moves the winning step's
                # reward off `PERFECT_GAME_REWARD`. See `state_helpers.is_perfect_score`.
                perfect[i] = is_perfect_score(scores[i])
        done |= is_last

    return scores.tolist(), perfect.tolist(), rewards.tolist(), steps


def evaluate(parallel_env, policy_action, num_episodes, on_round=None, should_abandon=None):
    """Collects at least num_episodes, in whole rounds of one episode per worker.

    `on_round(round_index, rounds_total, perfect_so_far, episodes_so_far, per_round_perfect)`
    is called after each round if given. It exists so the running perfect rate can be
    persisted while a checkpoint is still in flight — a checkpoint takes ~5 minutes and
    without this the only observable state is "started" until it finishes.

    `should_abandon(perfect_so_far, episodes_so_far)` is checked after each round, and stops this
    pass early when it returns True. The counts passed to it are **this pass's**, so a caller
    topping up a screened checkpoint has to fold in the episodes already held — see `measure`. The
    predicate is the caller's business rather than this function's because only the caller knows
    the checkpoint's target length; a round is the smallest unit that can be stopped at, since a
    round runs until every worker's episode ends.

    Returns `(scores, perfect_flags, rewards, elapsed, abandoned)`.
    """
    worker_envs = parallel_env.pyenv.envs
    num_workers = len(worker_envs)
    rounds = -(-num_episodes // num_workers)  # ceil

    scores, perfect_flags, rewards = [], [], []
    per_round_perfect = []
    steps = 0
    abandoned = False
    start = time.time()
    for index in range(rounds):
        round_scores, round_perfect, round_rewards, round_steps = run_round(
            parallel_env, policy_action, worker_envs)
        scores.extend(round_scores)
        perfect_flags.extend(round_perfect)
        rewards.extend(round_rewards)
        steps += round_steps
        per_round_perfect.append(int(sum(round_perfect)))
        print('    round {0}/{1}: {2} episodes, {3} perfect'.format(
            index + 1, rounds, len(round_scores), sum(round_perfect)))
        if on_round is not None:
            on_round(index + 1, rounds, int(sum(perfect_flags)), len(scores), list(per_round_perfect))
        if should_abandon is not None and should_abandon(int(sum(perfect_flags)), len(scores)):
            abandoned = True
            print('    abandoned after {0} episodes: cannot reach the threshold from here'.format(
                len(scores)))
            break

    elapsed = time.time() - start
    print('    {0} episodes in {1}s ({2} env steps/s)'.format(
        len(scores), round(elapsed, 1), round(steps / elapsed)))
    return scores, perfect_flags, rewards, elapsed, abandoned


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    policy_name = argv[1]
    # Live chart window on the laptop only. `viewer_enabled()` is darwin-gated, so this is a no-op
    # on the desktop, where the runner daemon owns the viewer (`desktop/runner/runner.py`) — two
    # owners would open two windows per wave. Best-effort: a chart is never worth an eval.
    if sys.platform == 'darwin':
        # HiDPI: chart_viewer only magnifies the PNG, and 110 dpi looks soft blown up on a Retina
        # panel while the 200-dpi training chart stays crisp. 220 gives the source enough pixels to
        # match, at the same window size. setdefault so an explicit SNEK_EVAL_CHART_DPI still wins.
        os.environ.setdefault('SNEK_EVAL_CHART_DPI', '220')
        try:
            import chart_viewer
            chart_viewer.spawn_for_eval(policy_name)
        except Exception as error:
            print('chart viewer skipped ({0}: {1})'.format(type(error).__name__, error))
    num_episodes = int(os.environ.get('EVAL_EPISODES', 100))
    # 4, lowered from 10 on 2026-08-08. Measured on the real close-out shape (4 parallel eval
    # processes, 800 episodes each): with independent workers 4 gives 117s against 118s at 5 and
    # 134s at 10, because 4 processes x 4 workers already saturates 14 cores at ~12.7 busy. More
    # workers past that add cost, not throughput. See eval_workers.py for the full table.
    num_workers = int(os.environ.get('EVAL_WORKERS', 4))
    screen_requested = os.environ.get('EVAL_SCREEN_EPISODES')
    screen_episodes, screen_note = resolve_screen_episodes(screen_requested, num_episodes)
    confirm_count = int(os.environ.get('EVAL_CONFIRM_COUNT', DEFAULT_CONFIRM_COUNT))
    if screen_note:
        print(screen_note)
    min_achievable = float(os.environ.get('EVAL_MIN_ACHIEVABLE', DEFAULT_MIN_ACHIEVABLE))
    if min_achievable and not 0 < min_achievable <= 100:
        raise SystemExit('EVAL_MIN_ACHIEVABLE={0} must be a percentage in (0, 100], or 0 to '
                         'disable early abandonment.'.format(min_achievable))
    # Never below the screen depth: equal_effort_pooled truncates to it and drops shorter rows, so
    # a lower floor would quietly delete checkpoints from the one arm-level figure meant to be
    # comparable across arms. Raised silently rather than rejected, because the two knobs are
    # independent and a screen deeper than the floor is a reasonable thing to ask for.
    abandon_floor = max(int(os.environ.get('EVAL_ABANDON_FLOOR', DEFAULT_ABANDON_FLOOR)),
                        screen_episodes)
    if min_achievable:
        print('abandoning any checkpoint that can no longer reach {0}%, once it has run {1}+ '
              'episodes (EVAL_MIN_ACHIEVABLE=0 to disable)'.format(min_achievable, abandon_floor))
    perfect_wait_ms = int(os.environ.get('EVAL_PERFECT_WAIT_MS', 400))
    # Off by default: 163us per game step headless against 6050us in a window, and because
    # ParallelPyEnvironment waits for the slowest worker, one rendering worker paces all of them.
    render_worker = os.environ.get('EVAL_RENDER', '0') not in ('0', '', 'false', 'False')
    # On by default from 2026-08-08. Each worker owns its env and its own network copy, so there is
    # no batched inference to idle through and no per-episode barrier: 1.41x at 4 workers, 1.91x at
    # 5, 2.33x at 10. EVAL_INDEPENDENT=0 restores the ParallelPyEnvironment path, which is what
    # every measurement before this date used and which EVAL_RENDER still needs.
    independent_workers = os.environ.get('EVAL_INDEPENDENT', '1') not in ('0', '', 'false', 'False')

    ckpt_dir = POLICY_DIR + policy_name
    available = {int(f[len('ckpt-'):].split('.')[0])
                 for f in os.listdir(ckpt_dir) if f.startswith('ckpt-') and f.endswith('.index')}

    # Spelled `top` rather than `--top`: tf_agents' handle_main routes argv through absl,
    # which rejects any unregistered `--flag` before main() is reached.
    if argv[2].startswith('top'):
        rest = argv[2][len('top'):].lstrip(':=')
        count = int(rest) if rest else (int(argv[3]) if len(argv) > 3 else DEFAULT_COUNT)
        requested_steps, selected_by = select_top_checkpoints(policy_name, available, count)
    elif argv[2].startswith('above'):
        rest = argv[2][len('above'):].lstrip(':=')
        threshold = float(rest) if rest else (
            float(argv[3]) if len(argv) > 3 else DEFAULT_ABOVE_THRESHOLD)
        requested_steps, selected_by = select_checkpoints_above(policy_name, available, threshold)
        # No qualifying checkpoint is the common case, not a failure: exit clean so the job is
        # marked done rather than failed (and never re-tried on the desktop).
        if not requested_steps:
            print('no close-out checkpoint reached {0:g}% — nothing to re-measure'.format(threshold))
            return 0
    else:
        requested_steps = [int(a) for a in argv[2:]]
        selected_by = {step: {'selected_by': 'explicit'} for step in requested_steps}
        missing = [s for s in requested_steps if s not in available]
        if missing:
            raise SystemExit('no checkpoint for step(s) {0} in {1}'.format(missing, ckpt_dir))

    # Resume before the expensive setup, so a run that has nothing left to do says so in
    # seconds rather than after building an agent and 20 worker processes.
    suffix = os.environ.get('EVAL_OUT_SUFFIX', '')
    resumed_rows, resumed_steps, source_screens, resumed_partial = load_finished_results(
        policy_name, resume_suffixes(os.environ.get('EVAL_RESUME'), suffix), num_episodes)
    if resumed_steps:
        skipped = [s for s in requested_steps if s in resumed_steps]
        requested_steps = [s for s in requested_steps if s not in resumed_steps]
        print('resuming: {0} of the selected checkpoints are already measured at >={1} '
              'episodes, {2} left to do'.format(len(skipped), num_episodes, len(requested_steps)))
        if not requested_steps:
            print('nothing left to measure')
            return 0
        if screen_episodes and not screen_requested:
            # Resume means continue: an arm that started under the flat protocol finishes under it,
            # because a mix of 20- and 100-episode rows is not comparable with itself, let alone with
            # the arm it is meant to be compared against.
            #
            # Which protocol it started under is READ from the source files, never inferred from the
            # depth of the resumed rows — see protocol_from_sources for the batch-18 failure that
            # rule replaced.
            keep_screening, recorded_depth = protocol_from_sources(source_screens)
            if keep_screening:
                if recorded_depth != screen_episodes:
                    print('    continuing at the recorded screen depth of {0} rather than {1}, so '
                          'this arm keeps one protocol throughout'.format(
                              recorded_depth, screen_episodes))
                    screen_episodes = recorded_depth
                    abandon_floor = max(abandon_floor, screen_episodes)
                if len(source_screens) > 1:
                    print('    warning: the source files disagree about the protocol ({0}); '
                          'continuing screened at {1}'.format(
                              sorted(str(v) for v in source_screens), screen_episodes))
                print('    resuming a screened arm: keeping screening on at {0} episodes '
                      '({1} full-length rows carried over)'.format(screen_episodes, len(skipped)))
            elif keep_screening is False:
                print('    screening off: the source files record a flat run, so the rest of the '
                      'arm will be too (EVAL_SCREEN_EPISODES={0} to override)'.format(
                          screen_episodes))
                screen_episodes = 0
            else:
                # Pre-dates `screen_episodes` in the payload. Keep the old behaviour rather than
                # guess, but say which branch was taken and why.
                print('    screening off: these {0} resumed rows come from a file that does not '
                      'record its protocol, so it is assumed flat '
                      '(EVAL_SCREEN_EPISODES={1} to override)'.format(
                          len(skipped), screen_episodes))
                screen_episodes = 0

    # Screens carried over from a killed run. Restricted to this run's candidate set, because a
    # partial sample for a checkpoint nobody selected is not work this run owes.
    resumed_partial = {step: held for step, held in resumed_partial.items()
                       if step in set(requested_steps)}
    if resumed_partial:
        carried = sum(len(held['scores']) for held in resumed_partial.values())
        print('    reusing {0} completed screen{1} ({2} episodes) from the earlier run rather than '
              're-measuring them'.format(len(resumed_partial),
                                         '' if len(resumed_partial) == 1 else 's', carried))

    print('policy {0}: evaluating {1} checkpoints x {2} episodes on {3} workers'.format(
        policy_name, len(requested_steps), num_episodes, num_workers))

    # Same env, network and agent as training, so the restored weights line up.
    spec_env = SnakeEnvironment(discount=0.99, display=False, policy_name=policy_name)
    spec_env.reset()
    spec_tf_env = tf_py_environment.TFPyEnvironment(spec_env)

    # One definition, shared with every independent worker in eval_workers.py. A second copy of
    # this construction is the failure mode this project has hit twice: expect_partial() hides a
    # mismatch, so two builders that drift produce a policy that loads silently and plays badly.
    agent, checkpoint, global_step = build_eval_agent(spec_tf_env, spec_env, ckpt_dir)

    def make_headless_worker():
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        return SnakeEnvironment(discount=0.99, display=False, policy_name=policy_name)

    def make_visible_worker():
        # Each worker is its own process, so clearing the dummy driver here gives
        # exactly one real window per eval process — several checkpoints evaluated
        # in parallel each get their own window to watch.
        os.environ.pop('SDL_VIDEODRIVER', None)
        # The game's 5000ms perfect-game celebration is a *blocking* pygame.time.wait(), which
        # stalls every worker in the round and leaves the window unresponsive. Override it.
        snake_constants.PERFECT_GAME_WAIT_MS = perfect_wait_ms
        return SnakeEnvironment(discount=0.99, display=True, policy_name=policy_name)

    # Independent workers (the default) never build a ParallelPyEnvironment at all: each worker
    # owns its env *and* its own network copy, so there is no batched inference to share and no
    # per-episode barrier. Measured 1.41x at 4 workers, 1.91x at 5 and 2.33x at 10 — see
    # eval_workers.py for the full table and for why ~10 workers is the optimum.
    #
    # EVAL_RENDER forces the batched path: rendering needs one visible worker among headless ones,
    # which is a property of the shared ParallelPyEnvironment. Watching a game is interactive and
    # already ~37x off the critical path, so it does not need the faster collector.
    pool = None
    parallel_env = None
    policy_action = None
    if independent_workers and not render_worker:
        print('    collecting with {0} independent workers '
              '(EVAL_INDEPENDENT=0 for the batched path)'.format(num_workers))
        pool = IndependentWorkerPool(policy_name, ckpt_dir, num_workers)
    else:
        if independent_workers and render_worker:
            print('    EVAL_RENDER=1 forces the batched path, which is the one that can show a '
                  'window')
        # All headless unless EVAL_RENDER=1, in which case worker 0 renders. Episodes are i.i.d.
        # across workers, so which worker produced one carries no information either way — the
        # window is purely for watching, and it costs ~37x on the critical path (see EVAL_RENDER
        # above), so a close-out nobody is watching should not pay for it.
        if render_worker:
            constructors = [make_visible_worker] + [make_headless_worker] * (num_workers - 1)
        else:
            constructors = [make_headless_worker] * num_workers
        parallel_env = tf_py_environment.TFPyEnvironment(
            parallel_py_environment.ParallelPyEnvironment(constructors))

        # Inference in a tf.function rather than eager: 208us per call against 1421us — 6.8x — for
        # byte-identical actions. Traced once and reused, since restoring weights writes into the
        # same variables and does not force a retrace.
        policy_action = common.function(agent.policy.action)

    out_path = os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy_name, suffix))
    backup_previous_results(out_path)

    # Everything this arm is accountable for, resumed work included, so the output file and the
    # progress chart describe the whole job rather than only what this process happened to run.
    all_steps = sorted(resumed_steps.union(requested_steps))

    # Planned work, in the two units a reader cares about: how many checkpoint measurements will
    # happen, and how many episodes they add up to. A screening protocol measures some
    # checkpoints twice, so "checkpoints" alone no longer tracks progress and the episode count
    # is what gives eval_progress.py an honest ETA across stages of different lengths.
    # The screening split: a graph point at ALWAYS_FULL_SINGLE (or an explicitly named step) is
    # measured at full length straight away, everything else is screened first and only the best
    # `confirm_count` of *those* are taken the rest of the way.
    plan = plan_stages(requested_steps, selected_by, screen_episodes, confirm_count,
                       num_episodes, num_workers, resumed=len(resumed_steps))
    full_steps, screen_steps = plan['full'], plan['screened']
    measurements_planned = plan['measurements_planned']
    episodes_planned = plan['episodes_planned']
    # One confirmation's episode cost, rounded to whole rounds the way plan_stages does it, for the
    # correction applied after the screens run.
    whole_confirm_rounds = (-(-(num_episodes - screen_episodes) // num_workers) * num_workers
                            if screen_episodes else 0)
    if screen_episodes:
        print('plan: {0} at full length ({1:.0f}% graph point or explicit), {2} screened at {3}, '
              'of which the best {4} confirmed — {5} episodes against {6} for a flat pass '
              '({7:.2f}x)'.format(
                  len(full_steps), ALWAYS_FULL_SINGLE, len(screen_steps), screen_episodes,
                  plan['confirmed'], episodes_planned, plan['flat_episodes'],
                  plan['flat_episodes'] / episodes_planned if episodes_planned else 0))

    # Resumed rows arrive already measured, so they start the count rather than being re-earned.
    # `session_*` covers only what this process has done, which is what an ETA has to be built
    # from: resumed rows carry the pace of whatever settings the earlier run used, and a
    # close-out relaunched at a different EVAL_WORKERS can differ by 3x. Averaging over every
    # row put b10b's ETA out by nearly 3x — 7h10m against a real ~2h.
    #
    # `stage` and the per-stage counts exist so the progress chart can say which of the three
    # passes is running and how far through it is. Without them a screening close-out shows one
    # undifferentiated bar that stalls for an hour on the full tier and then races, which reads
    # as a hung run.
    progress = {'measurements': len(resumed_steps),
                'session_measurements': 0, 'session_episodes': 0, 'session_seconds': 0.0,
                'stage': 'full' if screen_episodes else 'flat',
                'full_done': 0, 'screen_done': 0, 'confirm_done': 0}

    # Everything the payload needs that does not change once measuring starts. Paired with
    # `progress` above, which is the half that does.
    payload_spec = PayloadSpec(
        policy_name=policy_name, num_episodes=num_episodes, all_steps=all_steps,
        num_workers=num_workers, screen_episodes=screen_episodes, confirm_count=confirm_count,
        min_achievable=min_achievable, abandon_floor=abandon_floor,
        measurements_planned=measurements_planned, episodes_planned=episodes_planned,
        full_planned=len(full_steps), screen_planned=len(screen_steps),
        confirm_planned=plan['confirmed'])

    # The `WRITE_MIN_INTERVAL` gate on `on_round`'s progress writes. Recorded by every write,
    # including the unconditional per-checkpoint one, or the round after it would write again at once.
    write_gate = WriteGate()

    def write_results(results, complete, in_flight=None):
        """Rewrites the whole file after every checkpoint, and on a wall-clock tick within one.

        A long run is expensive — 20 checkpoints x 100 episodes is well over an hour, and a
        63-checkpoint run took four — so writing only at the end means any interruption
        throws all of it away. The numbers would still be in the log, but nothing
        machine-readable would survive. Rewriting is cheap next to a checkpoint's runtime.

        The payload itself is built by `eval_plan.build_payload`, which is the single definition
        shared with `eval_wave.py` — two builders that drift is a failure class this project has
        already paid for. `write_payload` does the `.partial` + `os.replace` so a reader never sees
        a half-written file even if the process dies mid-write.
        """
        write_gate.record()
        write_payload(out_path, build_payload(
            payload_spec, progress, samples, results, complete, in_flight))
        update_chart()

    # Live progress window, same mechanism as training's graph. Refreshed from write_results()
    # so it advances every round; a frame is ~0.1s against a ~4s round.
    #
    # Always on, deliberately — update_chart() disables itself if no window can be opened, so an
    # unattended run needs no configuration. The chart is scoped to this job's suffixes — its own
    # EVAL_OUT_SUFFIX plus any it resumes — so parallel EVAL_OUT_SUFFIX processes still show the
    # whole job, but a *different* eval that left a result file on this arm is not merged in. That
    # last case is why the scoping is explicit: load_runs' mtime-window guess pulled a close-out
    # finished <1h earlier into an HOF re-measurement's live chart, reading ~1700 checkpoints.
    chart_suffixes = {suffix} | set(resume_suffixes(os.environ.get('EVAL_RESUME'), suffix))
    chart_path = os.path.join(EVALS_DIR, '{0}_eval_progress.png'.format(policy_name))
    # 'off' disables everything after a real error. 'window_off' disables only the live
    # window (SNEK_CHART_WINDOW=0, the default) while live_frame() keeps writing the PNG on
    # every update -- conflating the two once meant a headless eval wrote the chart a single
    # time and it looked frozen (the decoupled chart_viewer.py then showed a stale snapshot).
    chart = {'screen': None, 'last': 0.0, 'off': False, 'window_off': False}

    def update_chart(force=False):
        if chart['off']:
            return
        now = time.time()
        if not force and now - chart['last'] < CHART_MIN_INTERVAL:
            return
        chart['last'] = now
        try:
            import eval_progress
            frame = eval_progress.live_frame(policy_name, chart_path, suffixes=chart_suffixes)
            if frame is None:
                return
            if chart['window_off']:
                return  # the PNG was written by live_frame() above; only the window is off
            if chart['screen'] is None:
                # Window off by default (the decoupled chart_viewer.py is the way to
                # watch); live_frame() above still writes the PNG every update. Set
                # SNEK_CHART_WINDOW=1 to opt back in. See training.py for why an
                # in-process live window is a fatal-XIO liability under memory pressure.
                if os.environ.get('SNEK_CHART_WINDOW', '0') in ('0', '', 'false', 'False'):
                    chart['window_off'] = True
                    return
                import pyformulas
                chart['screen'] = pyformulas.screen(
                    np.zeros(frame.shape[:2], dtype=np.uint8),
                    '{0} eval progress'.format(policy_name))
            # cv2 reads a three-channel array as BGR and matplotlib produces RGB, so the window
            # needs the channels reversed. The PNG keeps RGB and is written by live_frame().
            chart['screen'].update(frame[:, :, ::-1])
        except Exception as error:
            # A chart is never worth losing an eval over — no display, no cv2, a closed window.
            chart['off'] = True
            print('    progress chart off ({0}: {1})'.format(type(error).__name__, error))

    # Episodes measured in this process, kept raw per step so a screening pass can be topped up
    # to full length later without pooling summary statistics. 660 checkpoints x 100 floats is
    # nothing next to the agent itself.
    # Seeded with any screens carried over from a killed run, so `measure` tops them up to full
    # length exactly the way it tops up a screen measured in this session — same code path, and the
    # median and extremes are recomputed from the pooled raw episodes rather than approximated.
    samples = {step: dict(held) for step, held in resumed_partial.items()}
    resumed_by_step = {row['step']: row for row in resumed_rows}
    # Rows are memoised per step, so **`samples[step]` may be mutated only by `measure`**, which
    # pairs the mutation with a `row_cache.put`. See `eval_plan.RowCache` for the cost this bounds
    # (a 1,300-checkpoint close-out rebuilt every row 125 times per measurement) and for what a
    # broken invariant looks like.
    row_cache = RowCache()

    def current_results():
        """Result rows for every checkpoint with episodes banked.

        The checkpoint in flight is skipped until its episodes land: `on_round` writes progress
        while `evaluate` is still running, so its sample is still empty at that point and there
        is no rate to report. Its running state travels in the payload's `in_flight` block
        instead, which is what eval_progress.py draws it from.
        """
        return row_cache.rows(samples, resumed_by_step, selected_by.get)

    def measure(step, episodes, label, stage=None):
        """Restores one checkpoint and adds `episodes` more episodes to its sample."""
        if stage:
            progress['stage'] = stage
        print('\ncheckpoint {0} ({1})'.format(step, label))
        # With independent workers the parent holds no weights that matter — each worker restores
        # into its own network and reports the global_step it read, which is a stronger check than
        # the parent's own restore was: it confirms all N agree.
        if pool is not None:
            restored = pool.load(step)
        else:
            checkpoint.restore(os.path.join(ckpt_dir, 'ckpt-{0}'.format(step))).expect_partial()
            restored = int(global_step.numpy())
        if restored != step:
            print('    warning: global_step reads {0}, expected {1}'.format(restored, step))

        held = samples.setdefault(step, {'scores': [], 'perfect': [], 'rewards': [], 'seconds': 0.0})
        already = len(held['scores'])
        started_at = time.time()

        def on_round(round_index, rounds_total, perfect_so_far, episodes_so_far, per_round):
            # One progress write per `WRITE_MIN_INTERVAL`, not one per round: the write is O(banked
            # rows) and rounds arrive O(episodes), so the two multiplied. Nothing durable is at stake
            # here — this block is progress only, and the measurement's own write is unconditional.
            if not write_gate.due():
                return
            # Report the checkpoint's whole sample, screening episodes included, so a topped-up
            # finalist shows its true running rate rather than restarting from zero.
            total_perfect = sum(held['perfect']) + perfect_so_far
            total_episodes = already + episodes_so_far
            write_results(current_results(), complete=False, in_flight={
                'step': step,
                'round': round_index,
                'rounds_total': rounds_total,
                'perfect_so_far': total_perfect,
                'episodes_so_far': total_episodes,
                # Just this pass, where `episodes_so_far` is the checkpoint's whole sample. The
                # two differ by `already` whenever a screened checkpoint is being topped up, and
                # only this one shares a denominator with `round`.
                'episodes_this_pass': episodes_so_far,
                'running_percent': round(100.0 * total_perfect / total_episodes, 1),
                'per_round_perfect': per_round,
                'started_at': started_at,
            })

        # The stopping rule reasons about the checkpoint's *whole* sample against its target
        # length, but `evaluate` only counts this pass — the two differ by `already` whenever a
        # screened checkpoint is being topped up. Fold the held tally in here, where it is known.
        held_perfect = int(sum(held['perfect']))
        abandon_test = make_abandon_test(min_achievable, num_episodes, abandon_floor)
        should_abandon = (None if abandon_test is None else
                         lambda perfect, done: abandon_test(held_perfect + perfect, already + done))

        if pool is not None:
            scores, perfect_flags, rewards, elapsed, abandoned = pool.run(
                episodes, on_progress=on_round, should_abandon=should_abandon)
        else:
            scores, perfect_flags, rewards, elapsed, abandoned = evaluate(
                parallel_env, policy_action, episodes, on_round=on_round,
                should_abandon=should_abandon)
        if abandoned:
            held['abandoned'] = True
            progress['abandoned'] = progress.get('abandoned', 0) + 1
            progress['episodes_saved'] = (progress.get('episodes_saved', 0)
                                          + (episodes - len(scores)))
        held['scores'].extend(scores)
        held['perfect'].extend(perfect_flags)
        held['rewards'].extend(rewards)
        held['seconds'] += elapsed
        progress['measurements'] += 1
        progress[progress['stage'] + '_done'] = progress.get(progress['stage'] + '_done', 0) + 1
        progress['session_measurements'] += 1
        progress['session_episodes'] += len(scores)
        progress['session_seconds'] += elapsed

        row = build_row(step, held, selected_by.get(step))
        # Built from the sample just extended above, so it is exactly what `current_results` would
        # produce for this step. This is the `put` the cache's invariant requires: `measure` is the
        # only place a sample is mutated, so it is the only place the cache can go stale.
        row_cache.put(step, row)
        write_results(current_results(), complete=False)
        print('    perfect {0}/{1} = {2}%  (95% CI {3}-{4}%)'.format(
            row['perfect_games'], row['episodes'], row['perfect_percent'],
            row['perfect_ci95'][0], row['perfect_ci95'][1]))
        print('    score mean {0}  median {1}  min {2}  max {3}'.format(
            row['avg_score'], row['median_score'], row['min_score'], row['max_score']))

    if screen_episodes:
        print('\nstage 1: {0} checkpoints at the full {1} episodes (graph point '
              '{2:.0f}%, or explicitly named)'.format(len(full_steps), num_episodes,
                                                      ALWAYS_FULL_SINGLE))
        for index, step in enumerate(full_steps, 1):
            # Same top-up logic as stage 2: a checkpoint the gate stopped at 60 of 100, or one a kill
            # interrupted, needs only the remainder. Restarting it would throw away real episodes for
            # nothing, which is what the old resume did to every partial row it found.
            have = len(samples.get(step, {}).get('scores', []))
            if have >= num_episodes:
                continue
            label = ('full {0} of {1}'.format(index, len(full_steps)) if not have
                     else 'full {0} of {1}, topping up {2} to {3}'.format(
                         index, len(full_steps), have, num_episodes))
            measure(step, num_episodes - have, label, stage='full')

        # A carried screen already has its episodes, so it is not screened again. One that was cut
        # short of the screen depth — the abandonment gate, or a kill mid-checkpoint — is topped up by
        # the difference rather than restarted, which is the whole point of storing raw episodes.
        to_screen = []
        for step in screen_steps:
            have = len(samples.get(step, {}).get('scores', []))
            if have >= screen_episodes:
                continue
            to_screen.append((step, screen_episodes - have, have))
        reused = len(screen_steps) - len(to_screen)
        topped = sum(1 for _, _, have in to_screen if have)
        note = ''
        if reused or topped:
            note = ' ({0} already screened, {1} topped up)'.format(reused, topped)
        print('\nstage 2: screening {0} of {1} checkpoints at {2} episodes each{3}'.format(
            len(to_screen), len(screen_steps), screen_episodes, note))
        for index, (step, needed, have) in enumerate(to_screen, 1):
            label = ('screen {0} of {1}'.format(index, len(to_screen)) if not have
                     else 'screen {0} of {1}, topping up {2} to {3}'.format(
                         index, len(to_screen), have, screen_episodes))
            measure(step, needed, label, stage='screen')

        # Ranked among the screened only. The full tier is excluded because it already has the
        # measurement a confirmation slot would buy — spending one there would spend it on
        # finished work, and the whole point of the split is that these slots go to checkpoints
        # the graph did *not* already flag.
        finalists = pick_finalists(current_results(), confirm_count,
                                   already_full=set(resumed_steps) | set(full_steps))
        # Mandatory 20/20 screens can take this past the planned confirm count, so the totals are
        # corrected here rather than left to report >100% done. plan_stages could not know the
        # number: it depends on how the screens actually came out.
        overshoot = max(0, len(finalists) - plan['confirmed'])
        if overshoot:
            plan['confirmed'] = len(finalists)
            measurements_planned += overshoot
            episodes_planned += overshoot * whole_confirm_rounds
            print('    plan raised by {0} confirmation{1} for the perfect screens'.format(
                overshoot, '' if overshoot == 1 else 's'))
        print('\nstage 3: confirming the best {0} of {1} screened checkpoints at {2} episodes '
              '({3} more each)'.format(len(finalists), len(screen_steps), num_episodes,
                                       num_episodes - screen_episodes))
        for entry in finalists:
            print('    {0:>8}  screen {1:>5.1f}%   graph {2:>5.0f}%   surrounding {3}'.format(
                entry['step'], entry['perfect_percent'],
                entry.get('graph_single_eval') or 0,
                '-' if entry.get('graph_surrounding') is None
                else '{0:.1f}%'.format(entry['graph_surrounding'])))
        for index, entry in enumerate(finalists, 1):
            measure(entry['step'], num_episodes - screen_episodes,
                    'confirm {0} of {1}'.format(index, len(finalists)), stage='confirm')
    else:
        for index, step in enumerate(requested_steps, 1):
            measure(step, num_episodes,
                    '{0} of {1}'.format(len(resumed_by_step) + index, len(all_steps)),
                    stage='flat')

    results = current_results()
    write_results(results, complete=True)
    update_chart(force=True)
    print('\nwrote {0}'.format(out_path))

    print('\n{0:>9}  {1:>11}  {2:>11}  {3:>8}  {4:>16}  {5:>9}'.format(
        'step', 'graph eval', 'surrounding', 'perfect', '95% CI', 'avg score'))
    for row in results:
        graph = row.get('graph_single_eval')
        near = row.get('graph_surrounding')
        print('{0:>9}  {1:>11}  {2:>11}  {3:>7}%  {4:>7}-{5:<7}  {6:>9}{7}'.format(
            row['step'],
            '-' if graph is None else '{0:.0f}%'.format(graph),
            '-' if near is None else '{0:.1f}%'.format(near),
            row['perfect_percent'],
            row['perfect_ci95'][0], row['perfect_ci95'][1], row['avg_score'],
            # Flagged because the rate on this row is over fewer episodes than its neighbours', so
            # it reads as "provably below the gate" rather than as a comparable measurement.
            '  abandoned at {0}'.format(row['episodes']) if row.get('abandoned') else ''))

    if min_achievable and progress.get('abandoned'):
        planned = progress['session_episodes'] + progress['episodes_saved']
        print('\nabandoned {0} checkpoints that could no longer reach {1}%, saving {2} of {3} '
              'episodes ({4:.0f}% of this session\'s planned work)'.format(
                  progress['abandoned'], min_achievable, progress['episodes_saved'], planned,
                  100.0 * progress['episodes_saved'] / max(1, planned)))
        print('    None of them could have reached the gate arithmetically, so no ranking changed.')

    if screen_episodes:
        perfect, episodes, count = equal_effort_pooled(samples, screen_episodes)
        low, high = wilson_interval(perfect, episodes)
        print('\npooled (first {0} episodes of every checkpoint, equal effort): {1}/{2} = {3:.1f}%'
              '  (95% CI {4:.1f}-{5:.1f}%)  over {6} checkpoints'.format(
                  screen_episodes, perfect, episodes, 100.0 * perfect / episodes,
                  100.0 * low, 100.0 * high, count))
        print('    this is the arm-level rate. Do not pool the rows in the output file instead:'
              '\n    the full-length rows hold {0}x the episodes of the screened ones and are the'
              '\n    arm\'s best by construction, so that figure reads high however good the '
              'policy is.'.format(num_episodes // max(1, screen_episodes)))
    else:
        perfect = sum(r['perfect_games'] for r in results)
        episodes = sum(r['episodes'] for r in results)
        low, high = wilson_interval(perfect, episodes)
        print('\npooled: {0}/{1} = {2:.1f}%  (95% CI {3:.1f}-{4:.1f}%)  over {5} checkpoints'.format(
            perfect, episodes, 100.0 * perfect / episodes,
            100.0 * low, 100.0 * high, len(results)))

    best = best_full_length_row(results, num_episodes)
    print('best checkpoint: {0} at {1}% (95% CI {2}-{3}%) over {4} episodes{5}'.format(
        best['step'], best['perfect_percent'], best['perfect_ci95'][0],
        best['perfect_ci95'][1], best['episodes'],
        '  [truncated — no checkpoint reached the abandonment gate]'
        if best['episodes'] < num_episodes else ''))
    print('\nPooled rates only compare across arms when the selection rule matches.')
    if pool is not None:
        pool.close()
    return 0


if __name__ == '__main__':
    system_multiprocessing.handle_main(lambda argv: sys.exit(main(argv)))
