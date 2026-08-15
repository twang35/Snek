from __future__ import absolute_import, division, print_function

import snake_constants
from snake_constants import *
from state_helpers import is_perfect_score

import imageio
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import os
import random
import time

import tensorflow as tf
from tf_agents.networks import sequential


# Share of the perfect-game score an arm has to reach before display_progress draws the
# perfect-score guide. Named rather than inline because it is a display judgement, not a rule about
# the game: at 0.8 the guide appears while there is still visible headroom to read, and it stays off
# for the whole early run where it would only compress the score trace.
PERFECT_SCORE_GUIDE_FRACTION = 0.8


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def derive_seed(seed, stream):
    """A distinct seed per (base seed, stream), or None when seeding is off.

    **Every RNG consumer needs its own stream, and getting this wrong is silent and severe.**
    Food placement in Snake.Food uses the *global* `random` module, and
    `ParallelPyEnvironment` builds its workers from one constructor called once per worker
    process. Seed them all identically and all ten workers deal the same food in the same order:
    a 10-episode eval becomes one episode measured ten times, every confidence interval it
    produces is a fiction, and nothing anywhere raises. So the worker index goes in here.

    The multiplier is a large prime so adjacent base seeds do not produce overlapping streams —
    seed 1 stream 2 must not collide with seed 2 stream 1. Masked into positive int32 because
    `numpy.random.seed` rejects anything wider.
    """
    if seed is None:
        return None
    return (seed * 1000003 + stream) % (2 ** 31 - 1)


def seed_process(seed, stream):
    """Seeds this process's `random`, numpy and TensorFlow RNGs. No-op when seed is None.

    Call once per process, and in a `ParallelPyEnvironment` worker that means inside the env
    constructor — the constructor is what runs in the child, so it is the only place a
    per-worker stream can be applied.

    **This buys reduced variance, not reproducibility, and the reason is `cpprb`.** Measured
    2026-08-07: two arms on the same seed and config have identical network initialisation and an
    identical environment, but their weights diverge inside the first 1000 steps. The cause is that
    prioritized sampling happens in cpprb's C++ RNG, which nothing here seeds — its `seed=` kwarg is
    accepted and silently ignored. So each training step draws a different batch, and 200 gradient
    steps on a *fixed* batch are bit-identical, which locates the nondeterminism in the data rather
    than the math.

    `TF_DETERMINISTIC_OPS=1`, single-threaded TF and `PYTHONHASHSEED=0` were all tried and only
    delay the divergence. Treat a seed as making a run *describable*, and do not report a re-run as
    confirmation of an exact number.
    """
    derived = derive_seed(seed, stream)
    if derived is None:
        return None
    random.seed(derived)
    np.random.seed(derived)
    tf.random.set_seed(derived)
    return derived


def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))


def build_q_net(num_actions, fc_layer_params):
    """The Q-network training builds, so restored weights line up.

    Shared by snek2.main, eval_agent.py and watch.py — one architecture, built in one place, since a
    second copy is how they quietly drift apart. ``fc_layer_params`` is required: a fresh run reads
    it from SNEK_FC_LAYERS, and anything restoring a checkpoint reads it from that checkpoint's
    arch.json (see policy_arch.py), so the shape can no longer default silently to (50, 100, 50) and
    mis-restore. That silent default, driven off an unset SNEK_FC_LAYERS, was the exact bite.
    """
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(0.0))
    return sequential.Sequential(dense_layers + [q_values_layer])


def compute_avg_return(parallel_environment, policy, metrics, eval_only, num_episodes=10):
    """Runs num_episodes greedy episodes, one per worker, and folds them into metrics.

    Every episode runs in parallel. This used to play the first episode by itself on a
    separate single environment in the main process, because that was the only episode that
    could be drawn in a window — pygame allows one display per process and the workers are
    separate processes. Watching moved to watch.py, so the reason went away and with it the
    whole serial path, the extra environment and the display switch.

    Flattening it is worth about 24% of an eval: at champion skill the split shape measured
    5.95s (1.67s serial + 4.28s round) against 4.55s for a single round of 10. Adding the
    tenth worker costs only ~0.27s, because a round ends with its slowest episode and the
    slowest of ten is barely worse than the slowest of nine.

    Statistically unchanged: still num_episodes independent greedy episodes, and
    run_parallel_eval_episodes already counts only each worker's first episode.
    """
    start_time = time.time()

    rewards, scores, total_steps = run_parallel_eval_episodes(
        parallel_environment, policy, num_episodes)
    episode_rewards = rewards.tolist()
    episode_scores = scores.tolist()

    for episode_reward in episode_rewards:
        if metrics.min_reward > episode_reward:
            metrics.min_reward = episode_reward
        if metrics.max_reward < episode_reward:
            metrics.max_reward = episode_reward

    for episode_score in episode_scores:
        if metrics.min_score > episode_score:
            metrics.min_score = episode_score
        if metrics.max_score < episode_score:
            metrics.max_score = episode_score

    # Counted off the score, never off `last_rewards`. See `state_helpers.is_perfect_score`: the
    # reward test this replaces read 0% perfect for every shaped arm, which also pinned epsilon.
    perfect_games = sum(1 for score in episode_scores if is_perfect_score(score))

    if snake_constants.DEBUG_LOGGING:
        print('eval steps/second: ', round(total_steps / (time.time() - start_time), 2))

    metrics.last_eval_perfect_percent = perfect_games / num_episodes
    if eval_only:
        metrics.append_perfect_percent(metrics.last_eval_perfect_percent)

    avg_reward = sum(episode_rewards) / num_episodes
    avg_score = sum(episode_scores) / num_episodes
    return avg_reward, avg_score


def run_parallel_eval_episodes(parallel_environment, policy, num_parallel):
    """One greedy episode per worker. Returns per-episode totals, scores and the step count.

    It used to return each episode's *final* reward as a fourth array, for the sole purpose of
    identifying perfect games by `== PERFECT_GAME_REWARD`. Removed with that test rather than left
    unused: the array is exactly the wrong signal to leave lying around, since any new reward term
    shifts it while looking harmless. `state_helpers.is_perfect_score` has the history.
    """
    worker_envs = parallel_environment.pyenv.envs

    time_step = parallel_environment.reset()
    episode_rewards = np.zeros(num_parallel, dtype=np.float32)
    episode_scores = np.zeros(num_parallel, dtype=np.float32)
    done = np.zeros(num_parallel, dtype=bool)
    total_steps = 0

    while not np.all(done):
        action_step = policy.action(time_step)
        time_step = parallel_environment.step(action_step.action)
        rewards = time_step.reward.numpy()
        is_last = time_step.is_last().numpy()

        active = ~done
        episode_rewards[active] += rewards[active]
        total_steps += int(np.sum(active))

        # an already-finished worker auto-resets into a new episode on its
        # next step; `active` keeps that from being double-counted. Scores
        # must be fetched now, before that reset overwrites current_score.
        newly_done = active & is_last
        if np.any(newly_done):
            finished_indices = np.flatnonzero(newly_done)
            score_promises = [worker_envs[i].call('get_score') for i in finished_indices]
            for i, promise in zip(finished_indices, score_promises):
                episode_scores[i] = promise()
        done = done | newly_done

    return episode_rewards, episode_scores, total_steps


def trailing_average(values, window):
    """Trailing moving average: element i is the mean of the last `window` values up to and
    including i, with a shorter window at the very start where fewer values exist yet.

    Trailing rather than centred on purpose -- the chart is read live, so the newest point must
    reflect only evals already seen; a centred average would pull the latest point toward values
    that do not exist yet. Used to overlay a readable perfect-rate trend on the thin, noisy raw
    trace in display_progress.
    """
    averaged = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1):i + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged


def display_progress(eval_rows, resume_steps, screen, graph_path=None):
    """Draws the whole history of a policy, across however many runs made it.

    Takes explicit (step, score, percent) rows rather than assuming evenly spaced
    evals from a starting step, because a resumed policy's history has gaps
    wherever it was stopped and restarted.
    """
    # SNEK_CHART_SCALE sets the PNG render dpi (dpi = 100 * scale). The decoupled
    # chart_viewer.py only *magnifies* the PNG (~1.5-2x), so a low dpi looks blurry blown
    # up. Default 2.0 keeps the on-screen chart crisp on both hosts without anyone setting
    # an env var; scaling dpi rather than figsize keeps text and line weights proportional.
    # Cosmetic and safe to raise now that the per-eval matplotlib leak below is fixed -- a
    # high value used to amplify it. Override with the env var if a sharper/lighter PNG is
    # wanted. (env baked at launch, so a change only affects new runs, not ones already up.)
    chart_scale = float(os.environ.get('SNEK_CHART_SCALE', '2.0'))
    # Build the figure through the OO API (Figure + FigureCanvasAgg) rather than
    # plt.subplots(). pyplot registers every figure in a process-global manager (Gcf)
    # and a callback registry, and in this matplotlib version plt.close() does not fully
    # release them -- so a new figure per eval leaked Line2D/Text/Transform artists and
    # grew the trainer ~1 MB/eval (worse at high SNEK_CHART_SCALE, where it OOM'd the
    # desktop). A bare Figure() is never registered, so it is freed when it goes out of
    # scope and there is nothing to close. Do NOT switch this back to plt.subplots.
    fig = Figure(figsize=(3.65, 2.25), dpi=100 * chart_scale)
    FigureCanvasAgg(fig)
    score_axis = fig.add_subplot(1, 1, 1)
    steps = [row['step'] for row in eval_rows]
    scores = [row['avg_score'] for row in eval_rows]
    perfect_percents = [row['perfect_percent'] for row in eval_rows]

    label_size = 6
    tick_size = 5

    score_color = 'tab:blue'
    # Thin on purpose: a long run has thousands of points packed into ~300px of plot width,
    # so the default 1.5pt stroke overlaps itself into a near-solid band with almost no
    # visible texture. 0.3 keeps individual excursions readable instead of just filling the
    # whole band one color; chosen by comparing 0.3/0.5/0.8/1.2 side by side on a real arm.
    score_axis.plot(steps, scores, color=score_color, linewidth=0.3)
    score_axis.set_ylabel('Average Score', color=score_color, fontsize=label_size)
    # Put the latest step in the x label so how far the run has progressed reads at a glance --
    # late in training the traces fill the plot and the axis's own "1e6" offset is coarse, so
    # otherwise there is no number on the chart saying where it is now. Comma-grouped thousands:
    # 2685000 -> "Iterations (2,685k steps)".
    xlabel = 'Iterations'
    if steps:
        xlabel = 'Iterations ({:,}k steps)'.format(steps[-1] // 1000)
    score_axis.set_xlabel(xlabel, fontsize=label_size)
    score_axis.tick_params(axis='y', labelcolor=score_color, labelsize=tick_size)
    score_axis.tick_params(axis='x', labelsize=tick_size)
    # score_axis.set_ylim(top=250)

    percent_color = 'tab:red'
    percent_axis = score_axis.twinx()
    percent_axis.plot(steps, perfect_percents, color=percent_color, linewidth=0.3)
    percent_axis.set_ylabel('Perfect Game %', color=percent_color, fontsize=label_size)
    percent_axis.tick_params(axis='y', labelcolor=percent_color, labelsize=tick_size)
    percent_axis.set_ylim(bottom=0, top=100)

    # Horizontal guides at 20/40/60/80 on the perfect-game axis. Without them the red
    # trace can't be read off the right-hand axis by eye — the left axis ticks are on a
    # different scale and give no help. Faint, dashed and in the percent axis's colour so
    # it is unambiguous which axis they belong to, and thin enough not to compete with
    # either trace. zorder keeps them behind the red line they annotate.
    # zorder 1 sits above the axes patch (0) and below both traces (2), so the guides
    # never hide data. Alpha below ~0.4 is invisible at this figure size (3.65x2.25in).
    percent_axis.set_yticks([0, 20, 40, 60, 80, 100])
    for level in (20, 40, 60, 80):
        percent_axis.axhline(level, color=percent_color, linestyle=(0, (4, 3)),
                             linewidth=0.5, alpha=0.55, zorder=1)

    # A trailing 10-eval moving average of the perfect-game %, drawn over the thin raw trace.
    # Late in a long run the 0.3pt red line packs thousands of evals into ~300px and smears into a
    # band with no readable level; this answers "what has the perfect rate been lately" at a glance.
    # Same axis and same red family as the trace it summarises -- it is the smoothed version of that
    # signal, not a new quantity -- just darker so it reads as the level through the noise. Width
    # 0.4 is only a hair over the raw trace's 0.3 on purpose: a 10-eval average still wiggles at
    # this density, and a bold line just reprints the noise heavier (tried 1.1, too heavy). darkred
    # over tab:red gives the contrast instead. zorder 3 keeps it above both traces (2) and the
    # dashed guides (1). Window 10 chosen against 5 on a real ~3000-eval arm: 5 barely smoothed.
    if perfect_percents:
        percent_trend = trailing_average(perfect_percents, 10)
        percent_axis.plot(steps, percent_trend, color='darkred', linewidth=0.4,
                          alpha=0.9, zorder=3)

    # The perfect-game score (MAX_POSSIBLE_SCORE, 95 on a 10x10 board — a perfect game triggers at
    # 95 food eaten, since the snake starts with START_SEGMENTS + 1 cells already on the board).
    # Answers the question the score trace raises once an arm is good: how much is actually left.
    #
    # Only drawn once the arm has been within 80% of it. Before that the line sits far above every
    # point and its only effect is to stretch the y axis and squash the score trace into the bottom
    # of the plot, which costs more than the reference is worth. Gated on `max(scores)` rather than
    # the latest point so it does not flicker on and off between renders as the score oscillates.
    #
    # On the *score* axis, so it reads against the left-hand ticks rather than the percent ones, and
    # at zorder 1 — above the axes patch (0), below both traces (2) — so it cannot hide data. Thin
    # and dashed for the same reason as the percent guides above: at 3.65x2.25in anything heavier
    # competes with the traces it is meant to annotate.
    perfect_score = snake_constants.MAX_POSSIBLE_SCORE
    if scores and max(scores) > PERFECT_SCORE_GUIDE_FRACTION * perfect_score:
        score_axis.axhline(perfect_score, color='tab:green', linestyle=(0, (4, 3)),
                           linewidth=0.5, alpha=0.65, zorder=1)

    # One dashed line per point where training was picked back up, so a dip or
    # jump can be tied to a restart rather than to the policy itself.
    for resume_step in resume_steps:
        score_axis.axvline(resume_step, color='gray', linestyle='--', linewidth=0.6)

    fig.tight_layout()
    fig.canvas.draw()

    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

    # pyformulas is a thin cv2.imshow wrapper, and cv2 reads a three-channel array as BGR while
    # matplotlib produces RGB — so passing `image` straight through swapped red and blue in the
    # window, making the blue score trace look red and vice versa. The saved PNG was always
    # correct, which is why this went unnoticed. Reverse the channels for the window only.
    if screen is not None:  # None on a headless box; the PNG below is the durable chart
        screen.update(image[:, :, ::-1])
    if graph_path is not None:
        # Same array that goes to the window, so the file always matches what's
        # on screen. Written beside the target and renamed so anything reading it
        # mid-eval never sees a half-written PNG.
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        partial = graph_path + '.partial.png'
        imageio.imwrite(partial, image)
        os.replace(partial, graph_path)
    # No plt.close(): this Figure was never registered with pyplot, so it is collected
    # normally when the function returns. That is the whole point of the OO API above.


def create_policy_eval_video(eval_py_env, eval_env, policy, filename, num_episodes=5, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())
