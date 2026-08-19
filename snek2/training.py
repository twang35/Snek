import os
import time
from under_the_hood import *
from run_report import history_path, load_history, merge_eval_row, save_history, write_run_report

from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common
import pyformulas as pf

trailing_avg_window = 5
log_interval = 200
num_eval_episodes = 10
eval_interval = 1000
display_progress_interval = eval_interval
buffer_save_interval = 10 * eval_interval

# Quiet mode prints one compact line per this many evals instead of ~5 lines per eval plus a
# loss line every 200 steps. At 1000 steps per eval a 2M-step run goes from ~20000 lines to
# ~200. Every number dropped from the console is still in <policy>_evals.json, so nothing is
# lost — the file is the durable record and the console was only ever a live feed.
quiet_eval_log_interval = 10

# --- epsilon schedule -------------------------------------------------------------------
#
# `avg_reward` thresholds that drive the bootstrap phase, and the window used for the
# refinement phase's skill signal. See `epsilon_for` for the whole design.
BOOTSTRAP_REWARD_THRESHOLDS = (2, 5, 10, 15, 20)
# Rungs halve, so the phase hands over at initial_epsilon / 2**len(thresholds) — 0.0125 for
# the default 0.4. Named because the refinement phase starts from exactly that value.
#
# **Five rungs, not the original three.** Three handed over at 0.05, and batch 12 measured what
# sitting at 0.05 costs: four arms pinned there for up to 942k steps, 0% perfect games, greedy
# trailing 53-63 against batch 11's 84-88. A smoke run with the exploration shield on removed the
# *decay* but still plateaued at trailing ~83 with 0.3% perfect, improving at 4.7 points per 100k
# against `b11a`'s 11.1 — so 0.05 is too high for this task with or without the shield, because
# a collect policy that never finishes a board leaves no completed trajectories in the buffer.
#
# The two new rungs go *below* the existing ones rather than above, so the property that made
# the rewrite worth doing survives: every threshold that drops epsilon still sits in the
# pre-winning regime (max is still 20), and nothing cuts exploration while an arm is learning to
# win. What changed is only how far the ladder descends before refinement takes over.
BOOTSTRAP_RUNGS = len(BOOTSTRAP_REWARD_THRESHOLDS)
# Trailing perfect rate at which refinement reaches the floor. 0.80 rather than 1.0 because
# no arm has ever sustained a trailing rate above ~0.92, so anchoring at 1.0 would mean the
# floor is unreachable in practice.
REFINE_PERFECT_TARGET = 0.80
# Evals averaged for the skill signal. 30 x 10 episodes = 300, giving ~2.6pp standard error
# at p=0.8 — enough that a single lucky 10-episode eval cannot move epsilon far. The same
# window `run_report.build_summary` uses for `best_perfect30`, deliberately.
REFINE_TRAILING_WINDOW = 30
# Evals averaged for the bootstrap phase's reward signal. Much shorter than the refinement
# window because the phase only lasts ~10k steps, but > 1 so it does not flap on noise.
BOOTSTRAP_TRAILING_WINDOW = 5
# epsilon is never allowed to reach exactly 0. A fully greedy collect policy makes the replay
# buffer a closed loop on the policy's own behaviour, and it is a degenerate case with no
# measured upside: batch 3 found 0.001 and 0.0 indistinguishable, so nothing is given up by
# forbidding the endpoint.
EPSILON_HARD_FLOOR = 1e-4


def random_play(time_step_spec, action_spec, train_py_env, rb_observer, initial_collect_steps):
    if snake_constants.DEBUG_LOGGING:
        print('Random play to populate replay buffer')

    random_policy = random_tf_policy.RandomTFPolicy(time_step_spec, action_spec)

    py_driver.PyDriver(
        train_py_env,
        py_tf_eager_policy.PyTFEagerPolicy(
            random_policy, use_tf_function=True),
        [rb_observer],
        max_steps=initial_collect_steps).run(train_py_env.reset())


def train(max_steps, eval_parallel_env, train_py_env, agent, collect_driver, batch_size, replay_buffer,
          train_checkpointer, replay_buffer_dir, global_step, epsilon, initial_epsilon, min_epsilon,
          guided_fraction, configured_guided_fraction,
          eval_only, policy_name, run_config, priority_signal='td_error', use_is_weights=True,
          forking_collector=None):
    # (Optional) Optimize by wrapping some code in a graph using TF function.
    agent.train = common.function(agent.train)
    step = global_step.numpy()
    initial_step = np.copy(step)

    # Reset the train step.
    # agent.train_step_counter.assign(0)

    # Live results window. **Off by default now** -- the decoupled chart_viewer.py
    # is the way to watch charts (it only reads the PNGs written every eval, so it
    # cannot affect training). The in-process cv2 window is the old way and is unsafe
    # on an X11 box under memory pressure: if the X session breaks (e.g. an OOM kill
    # disrupts it) it raises a *fatal* XIO error that calls exit() below Python and
    # cannot be caught -- which killed all four desktop arms at once on 2026-08-09.
    # Set SNEK_CHART_WINDOW=1 to opt back in; opening it is still best-effort.
    if os.environ.get('SNEK_CHART_WINDOW', '0') in ('0', '', 'false', 'False'):
        screen = None
    else:
        try:
            screen = pf.screen(np.zeros((480, 560)), '{0} results'.format(policy_name))
        except Exception as _e:
            print('live results window unavailable ({0}: {1}); charts still saved to disk'.format(
                type(_e).__name__, _e))
            screen = None
    # Every eval refreshes the window and these three files, so a run always leaves
    # behind its own graph and write-up without anyone having to screenshot it.
    graph_path = os.path.join(snake_constants.RUNS_DIR, '{0}.png'.format(policy_name))
    report_path = os.path.join(snake_constants.RUNS_DIR, '{0}.md'.format(policy_name))
    graph_history_path = history_path(snake_constants.RUNS_DIR, policy_name)

    # Evaluate the agent's policy once before training
    training_metrics = TrainingMetrics(agent.train_step_counter)

    # Carry the graph over from earlier runs of this policy so resuming continues
    # the same curve instead of starting again at the current iteration.
    training_metrics.eval_rows, training_metrics.resume_steps = load_history(graph_history_path)
    if training_metrics.eval_rows:
        print('resuming graph from {0} earlier evals (through step {1})'.format(
            len(training_metrics.eval_rows), training_metrics.eval_rows[-1]['step']))
        if initial_step not in training_metrics.resume_steps:
            training_metrics.resume_steps.append(int(initial_step))
            training_metrics.resume_steps.sort()

    avg_reward, avg_score = compute_avg_return(eval_parallel_env, agent.policy, training_metrics,
                                               eval_only, num_eval_episodes)
    merge_eval_row(training_metrics.eval_rows,
                   build_eval_row(int(initial_step), avg_score, avg_score, avg_reward, training_metrics, epsilon))
    if snake_constants.DEBUG_LOGGING:
        print('before training score: ', round(avg_score, 2))

    remaining = steps_remaining(step, max_steps)
    print('Begin training: {0}  (stopping at step {1}, {2} to go)'.format(
        time.strftime("%d/%m %H:%M:%S", time.localtime()), max_steps, remaining))
    if remaining == 0:
        print('already at or past the {0}-step cap — nothing to train. Raise SNEK_MAX_STEPS to '
              'continue this arm.'.format(max_steps))

    # Reset the environment.
    time_step = train_py_env.reset()

    for _ in range(remaining):
        # Collect a few steps and save to replay buffer.
        # To view q_values, breakpoint at line 160 in tf_agents/policies/q_policy.py
        #
        # The forking collector is a drop-in for the driver that advances one of several branches
        # of the same game instead of only the main line, still one counted step per iteration —
        # see forking_collector.py. `None` only when SNEK_FORK_BRANCHES is 1, which since the
        # 2026-08-14 default raise means someone asked for it explicitly; the PyDriver branch is
        # kept because that is the plain single-line collect every arm before batch 17 ran.
        if forking_collector is None:
            time_step, _ = collect_driver.run(time_step)
        else:
            time_step = forking_collector.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        loss_info = 0
        if not eval_only:
            experience, indexes, is_weights = replay_buffer.sample(batch_size, step)
            # is_weights undo the bias that sampling by priority introduces.
            # theSchlong applied none at all, so use_is_weights=False reproduces it.
            loss_info = agent.train(experience, weights=is_weights if use_is_weights else None)
            # Transitions the network is worst at get sampled more often next time.
            # td_error by default, not td_loss: td_loss is Huber, which is quadratic
            # below |e|=1 and so shrinks small errors, making its spread wider than
            # the raw error's. Feeding it in gives an effective exponent near
            # |e|^1.6 instead of |e|^0.6 and measured worse at 30k steps -- but
            # theSchlong used td_loss, so it is selectable and under test.
            extra = loss_info.extra
            signal = extra.td_error if priority_signal == 'td_error' else extra.td_loss
            replay_buffer.update_priorities(indexes, signal.numpy())

        step += 1
        log_messages_and_eval(training_metrics, loss_info, eval_parallel_env, agent, train_py_env, screen,
                              graph_path, report_path, graph_history_path, train_checkpointer, replay_buffer,
                              replay_buffer_dir, global_step, epsilon, initial_epsilon, min_epsilon,
                              guided_fraction, configured_guided_fraction, step,
                              eval_only, initial_step, policy_name, run_config, forking_collector)


class TrainingMetrics:
    def __init__(self, step_counter):
        self.starting_step = step_counter.numpy()
        self.step_counter = step_counter
        self.trailing_avg_scores = []
        # Plotted history for this policy across every run of it, plus the steps
        # at which training was resumed. Both are loaded from disk in train().
        self.eval_rows = []
        self.resume_steps = []
        self.steps_start_time = time.time()
        self.training_start_time = time.time()
        self.eval_start_time = time.time()
        self.min_reward = 1000
        self.max_reward = 0
        self.min_score = 1000
        self.max_score = 0
        self.last_eval_perfect_percent = 0.0
        self.perfect_percentage = 0.0
        self.num_of_percents = 0
        self.recent_steps_per_second = 0.0
        # Evals whose checkpoint was skipped for scoring below MIN_CHECKPOINT_SCORE. Counted
        # so a progress check can tell "this arm is not saving" from "this arm is not running".
        self.skipped_checkpoints = 0

    def reset(self):
        self.steps_start_time = time.time()
        self.training_start_time = time.time()
        self.min_reward = 1000
        self.max_reward = -1000
        self.min_score = 1000
        self.max_score = -1000

    def append_perfect_percent(self, percentage):
        self.perfect_percentage = (self.perfect_percentage * self.num_of_percents + percentage) \
                                  / (self.num_of_percents + 1)
        self.num_of_percents += 1


def build_eval_row(step, avg_score, trailing_avg_score, avg_reward, metrics, epsilon,
                   forking_collector=None):
    """One row of the run report, and one point on the graph.

    A forking arm gets one extra key, `fork`, holding the collector's counters. It rides into
    `runs/<policy>_evals.json` durably and is ignored everywhere else: `write_run_report` iterates
    the fixed `EVAL_COLUMNS` list rather than the row's keys, and `build_summary` reads named keys.
    Absent entirely on a non-forking arm, so nothing has to distinguish "0 forks" from "no forking".
    """
    row = {
        'step': int(step),
        'avg_score': round(avg_score, 2),
        'trailing_avg_score': round(trailing_avg_score, 2),
        'min_score': int(round(metrics.min_score)),
        'max_score': '{0}/{1}'.format(int(round(metrics.max_score)),
                                      int(snake_constants.MAX_POSSIBLE_SCORE)),
        'avg_reward': round(avg_reward, 3),
        'perfect_percent': round(metrics.last_eval_perfect_percent * 100),
        'epsilon': round(float(epsilon.numpy()), 4),
    }
    if forking_collector is not None:
        row['fork'] = forking_collector.counters()
    return row


def log_messages_and_eval(metrics, loss_info, eval_parallel_env, agent, train_py_env, screen, graph_path,
                          report_path, graph_history_path, train_checkpointer, replay_buffer, replay_buffer_dir,
                          global_step, epsilon, initial_epsilon, min_epsilon, guided_fraction,
                          configured_guided_fraction, step, eval_only, initial_step,
                          policy_name, run_config, forking_collector=None):
    debug = snake_constants.DEBUG_LOGGING

    if step % log_interval == 0:
        steps_per_second = log_interval / (time.time() - metrics.steps_start_time)
        metrics.recent_steps_per_second = steps_per_second

        if debug:
            if eval_only:
                print('step = {0}: steps/second = {1}'.format(step, round(steps_per_second, 2)))
            else:
                print('step = {0}: loss = {1}, steps/second = {2}'.format(step,
                                                                         str(round(loss_info.loss.numpy(), 4)),
                                                                         round(steps_per_second, 2)))
        metrics.steps_start_time = time.time()

    if step % eval_interval == 0:
        if debug:
            print('training time: ', get_time(metrics.training_start_time))
            print('train_py_env high score: ', train_py_env.high_score)
        metrics.eval_start_time = time.time()
        avg_reward, avg_score = compute_avg_return(eval_parallel_env, agent.policy, metrics, eval_only,
                                                   num_eval_episodes)
        if debug:
            print('eval time: ', get_time(metrics.eval_start_time))

        # Skill signal for the refinement phase, computed before this eval's row is merged so
        # the window is the last 30 evals *including* this one and no row is counted twice.
        perfect_rate = trailing_perfect_rate(metrics.eval_rows, metrics.last_eval_perfect_percent)
        reward_signal = trailing_reward(metrics.eval_rows, avg_reward)
        maybe_update_epsilon(reward_signal, perfect_rate, epsilon, initial_epsilon, min_epsilon)
        # Same reward signal, so the shield switches on at exactly the eval the bootstrap phase
        # hands over on — the two cannot drift apart by a window's worth of evals.
        maybe_update_guided_fraction(reward_signal, initial_epsilon, configured_guided_fraction,
                                     guided_fraction)

        metrics.trailing_avg_scores.append(avg_score)
        if len(metrics.trailing_avg_scores) > trailing_avg_window:
            metrics.trailing_avg_scores.pop(0)
        trailing_avg_score = sum(metrics.trailing_avg_scores) / len(metrics.trailing_avg_scores)

        # Skip checkpointing a policy that is not worth keeping. Two reasons, and the
        # second is the one that actually cost this project evidence:
        #
        # 1. Disk. A checkpoint is 188 KB and one is written every 1000 steps.
        # 2. `max_to_keep` is a *rolling* window, so a dead arm that keeps training keeps
        #    writing worthless checkpoints and **evicts the good ones behind them**.
        #    `b8d-disc995clip` ran to 11.64M steps with its last 4.5M at trailing ~1, hit
        #    the 10000 cap, and deleted everything before step 1.64M. Its 80% checkpoint at
        #    2538k survived by luck; a few million more steps would have taken it.
        #
        # Gate on `max(this eval, trailing)` rather than trailing alone. The best
        # checkpoints in this project are *outliers* that spike well above their
        # neighbourhood, so a trailing-only test could skip exactly the checkpoint worth
        # keeping while an arm is recovering. Either signal clearing the bar is enough.
        #
        # The bar is deliberately below any useful policy: anything capable of a perfect game
        # scores ~85-95 out of 95. Checked against every checkpoint this project has measured
        # at 100 episodes — of the 232 that reached 30% perfect games, the lowest
        # max(avg_score, trailing) was 49.8 — so the default of 40 discards only arms that are
        # dead or have not started learning.
        keep_checkpoint = max(avg_score, trailing_avg_score) >= snake_constants.MIN_CHECKPOINT_SCORE

        if not eval_only:
            if keep_checkpoint:
                if debug:
                    print('saving checkpoint')
                train_checkpointer.save(global_step)
                # The buffer is ~20 MB and only warm-starts the next run, so it saves
                # far less often than the agent to keep disk churn down. It is gated on the
                # same condition so the two never desync — a resume that paired an old
                # policy with a much newer buffer would train the restored weights on
                # experience they never generated.
                if step % buffer_save_interval == 0:
                    replay_buffer.save(replay_buffer_dir)
            else:
                metrics.skipped_checkpoints += 1
                if debug:
                    print('skipping checkpoint: score {0:.1f} / trailing {1:.1f} below {2}'.format(
                        avg_score, trailing_avg_score, snake_constants.MIN_CHECKPOINT_SCORE))

        eval_str = 'step = {0}: avg_score = {1}, trailing_avg_score = {2}, min_score = {3}, ' \
                   'max_score = {4}/{5}, avg_reward = {6}, min_reward = {7}, max_reward = {8}, ' \
                   'perfect_percent = {9}, epsilon = {10}'\
            .format(step,
                    round(avg_score, 2),
                    round(trailing_avg_score, 2),
                    int(round(metrics.min_score)),
                    int(round(metrics.max_score)),
                    int(snake_constants.MAX_POSSIBLE_SCORE),
                    round(avg_reward, 3),
                    round(metrics.min_reward, 2),
                    round(metrics.max_reward, 2),
                    '{0}%'.format(round(metrics.last_eval_perfect_percent * 100)),
                    round(float(epsilon.numpy()), 4))
        if eval_only:
            eval_str += ', cumulative_perfect_percent = {0}, initial_step = {1}'\
                .format('{0}%'.format(round(metrics.perfect_percentage * 100)), initial_step)
        if debug:
            print(eval_str)

        # Built before reset() clears the min/max trackers below.
        merge_eval_row(metrics.eval_rows,
                       build_eval_row(step, avg_score, trailing_avg_score, avg_reward, metrics,
                                      epsilon, forking_collector))
        summary = save_history(graph_history_path, metrics.eval_rows, metrics.resume_steps)

        if not debug:
            # One line per quiet_eval_log_interval evals, plus the first eval of the run and
            # any eval that sets a new best 30-eval perfect rate — so the log still shows
            # when an arm is improving without a line for every point.
            eval_index = step // eval_interval
            # `> 0` matters: with no perfect games yet, best_perfect30 is 0.0 and its step
            # field falls back to the current step, which would mark every single eval as a
            # new best.
            is_new_best = (summary['best_perfect30']['value'] > 0
                           and summary['best_perfect30']['step'] == step)
            if eval_index % quiet_eval_log_interval == 0 or is_new_best or eval_index <= 1:
                print('{0:>8}  score {1:>5.1f}  trail {2:>5.1f}  pf {3:>3.0f}%  '
                      'best30 {4:>4.1f}%  eps {5:<6}{6}'.format(
                          step,
                          round(avg_score, 1),
                          round(trailing_avg_score, 1),
                          metrics.last_eval_perfect_percent * 100,
                          summary['best_perfect30']['value'],
                          round(float(epsilon.numpy()), 4),
                          '  <- best so far' if is_new_best else
                          ('  no ckpt' if not keep_checkpoint else '')),
                      flush=True)
                if forking_collector is not None:
                    # On the same cadence as the line above, so a forking arm's log shows whether
                    # branches are actually happening. `skipped` counts decision points that wanted
                    # a branch and found no free slot — a large number means the cap, not the
                    # probability, is deciding which points get explored.
                    fork = forking_collector.counters()
                    print('          forks {0}  live {1}  branch share {2:.0%}  '
                          'ended {3}/trunc {4}  skipped {5}'.format(
                              fork['forks'], fork['live_now'], fork['branch_share'],
                              fork['terminated'], fork['truncated'], fork['skipped_full']),
                          flush=True)
        # restart time because compute_avg_return() takes a while and messes up the timing
        metrics.reset()

    if step % display_progress_interval == 0:
        display_progress(metrics.eval_rows, metrics.resume_steps, screen, graph_path)
        write_run_report(report_path, policy_name, run_config, metrics.eval_rows, os.path.basename(graph_path),
                         metrics.resume_steps)


def get_time(start_time):
    total_time = time.time() - start_time
    if total_time > 60:
        return str(round(total_time / 60.0, 2)) + ' min'
    return str(round(total_time, 1)) + 's'


def steps_remaining(current_step, max_steps):
    """How many more training steps this run should take, given where a resume left off.

    `max_steps` is **absolute**, not "this run's steps": the loop increments `global_step`, which
    a resume restores, so counting relatively would let an arm resumed at 4M run to 9M. Returns 0
    rather than a negative when an arm is already past its cap, so `range()` is a no-op and the
    run exits after its opening eval instead of raising.
    """
    return max(0, int(max_steps) - int(current_step))


def trailing_mean(eval_rows, key, current, window, scale=1.0):
    """Mean of `key` over the last `window` evals, including this one as `current`.

    Averages over however many evals exist when there are fewer than `window`, rather than
    dividing by `window` — a fresh run would otherwise read near zero for its first evals,
    which would pin epsilon at the ceiling exactly when it needs to descend.

    `scale` divides the stored rows only, for the one field whose stored and live units differ.
    """
    history = [row.get(key, 0) / scale for row in eval_rows[-(window - 1):]] if window > 1 else []
    history.append(current)
    return sum(history) / len(history)


def trailing_perfect_rate(eval_rows, current_percent, window=REFINE_TRAILING_WINDOW):
    """Skill signal for the refinement phase: mean perfect rate as a fraction.

    `eval_rows` store `perfect_percent` on a 0-100 scale and `current_percent` arrives as a
    0-1 fraction from `metrics.last_eval_perfect_percent`; the two are reconciled here rather
    than at the call site, because getting that wrong scales epsilon by 100.
    """
    return trailing_mean(eval_rows, 'perfect_percent', current_percent, window, scale=100.0)


def trailing_reward(eval_rows, current_reward, window=BOOTSTRAP_TRAILING_WINDOW):
    """Signal for the bootstrap phase: mean `avg_reward` over a short window.

    Short because the phase it drives lasts ~10k steps and has to stay responsive, but not
    one, because the raw signal is 10 episodes and flaps. A smoke run caught this: avg_reward
    read 7.63 then 4.96 across two consecutive evals — noise either side of the first
    threshold — and undamped epsilon went 0.4 -> 0.2 -> 0.4. Harmless where it happened, since
    exploration is nearly free at score ~5, but the collected distribution should not oscillate
    for no reason. Five evals is the same window `trailing_avg_score` already uses.

    Note this is a *damper*, not a ratchet: the bootstrap phase is still allowed to raise
    epsilon when an arm genuinely regresses, which is the behaviour the old ladder lacked.
    """
    return trailing_mean(eval_rows, 'avg_reward', current_reward, window)


def bootstrap_epsilon(avg_reward, initial_epsilon):
    """Phase 1: halve epsilon as `avg_reward` clears each threshold, then stand down.

    Returns 0.0 once the arm is past the last threshold, which means "this phase has nothing
    to say" rather than "epsilon is 0" — `epsilon_for` takes the max of the two phases, so
    standing down hands control to the refinement term instead of pinning epsilon here.

    Kept on `avg_reward` deliberately. Score rises 0 -> 70 in the first ~13k steps and this
    phase is calibrated to that stretch, where it demonstrably works: arms reach avg_score
    55-75 by step 13k. The defect being fixed was never the early descent, it was the three
    further rungs that dropped epsilon to 0.001 while the arm was still at 0% perfect games.
    """
    for index, threshold in enumerate(BOOTSTRAP_REWARD_THRESHOLDS):
        if avg_reward <= threshold:
            return initial_epsilon / (2.0 ** index)
    return 0.0


def refine_epsilon(perfect_rate, top, floor, perfect_target=REFINE_PERFECT_TARGET):
    """Phase 2: geometric interpolation from `top` at 0% perfect to `floor` at `perfect_target`.

    Geometric rather than linear because the useful range spans more than an order of
    magnitude (0.05 to 0.002), and equal *ratios* are what matter to an exploration rate:
    0.05 -> 0.025 changes behaviour as much as 0.004 -> 0.002. A linear ramp would sit above
    0.02 for more than half its length, where a random move every ~40 steps wrecks the
    endgame, and then cross the entire low range in its last few percent.

    A pure function of the current skill estimate, with no memory. That is the point: the old
    ladder was a one-way ratchet, so a single lucky eval pinned epsilon permanently and a
    regression never bought exploration back — `b11b` sat at 0.001 while its score collapsed
    from 64.6 to 8.8. Here a declining arm automatically explores more.
    """
    if top <= floor:
        return floor
    # Only the lower clamp is needed. A rate above `perfect_target` gives a fraction over 1,
    # which undershoots the floor and is caught by the max() below — but a *negative* rate
    # would give a negative exponent and push epsilon above `top`, which nothing else guards.
    fraction = max(0.0, perfect_rate / perfect_target)
    return max(floor, top * (floor / top) ** fraction)


def epsilon_for(avg_reward, perfect_rate, initial_epsilon, min_epsilon):
    """The epsilon this eval implies. Two phases, combined with max().

    | phase | driven by | range |
    |---|---|---|
    | bootstrap | `avg_reward`, one halving per threshold | initial_epsilon -> initial/4 |
    | refinement | trailing perfect rate | initial/8 -> min_epsilon |

    `max()` rather than `min()` because the phases must not fight: the bootstrap term is
    larger than the refinement term's ceiling while it is active and returns 0.0 once it is
    not, so the maximum is whichever phase is live. `min()` would jump straight to the
    refinement ceiling on the first eval, before the arm can play at all.

    Stateless, which also makes a resume safe: the first eval after restarting recomputes the
    right epsilon from the restored history rather than having to descend a ladder again.
    """
    top = initial_epsilon / (2.0 ** BOOTSTRAP_RUNGS)
    return max(bootstrap_epsilon(avg_reward, initial_epsilon),
               refine_epsilon(perfect_rate, top, min_epsilon))


def guided_fraction_for(avg_reward, initial_epsilon, configured_fraction):
    """Fraction of episodes the exploration shield covers, given where the schedule is.

    Zero while the bootstrap phase is live, the configured value once it stands down. The
    shield exists to make exploration survivable in the *endgame*, and during bootstrap there
    is no endgame to protect — epsilon is 0.1-0.4 and the snake is a few segments long, so
    dying is cheap and the deaths are the signal. Turning it on only at the handover also keeps
    the early curve identical to batch 11's, which is the part that already works.

    Stateless, like `epsilon_for`: if an arm collapses far enough for bootstrap to re-arm, the
    shield switches back off with it rather than latching. That keeps one rule — "shielded iff
    refining" — instead of two that can disagree, and makes a resume recompute the right state
    from restored history.
    """
    if bootstrap_epsilon(avg_reward, initial_epsilon) > 0.0:
        return 0.0
    return configured_fraction


def maybe_update_guided_fraction(avg_reward, initial_epsilon, configured_fraction, guided_fraction):
    """Assigns the scheduled guided fraction, if it differs from what the Variable holds.

    Rounded comparison for the same float32 round-trip reason as `maybe_update_epsilon`.
    """
    target = guided_fraction_for(avg_reward, initial_epsilon, configured_fraction)
    if round(float(guided_fraction.numpy()), 6) != round(target, 6):
        guided_fraction.assign(target)
        # One line per transition, and transitions are rare — the shield switches on once at the
        # handover and only switches off again if the arm collapses back into the bootstrap
        # band. Without it there is no way to tell from a log whether an arm ever got shielded.
        if configured_fraction > 0.0:
            print('exploration shield {0} (guided fraction {1})'.format(
                'ON' if target > 0.0 else 'OFF', target), flush=True)


def enforce_learning_rate(optimizer, configured_lr):
    """Re-asserts the configured learning rate over whatever the checkpoint restored.

    Adam's `learning_rate` is a `tf.Variable`, not a plain attribute, so `common.Checkpointer`
    saves it alongside the moment estimates and `initialize_or_restore()` silently overwrites the
    value the constructor was given. Measured on tf 2.15.1 / keras 2.15.0: an optimizer built at
    1e-6 reads 1e-5 back after restoring a checkpoint written at 1e-5. That makes
    `SNEK_LEARNING_RATE` a **no-op on every resume** — the knob works on a fresh arm and is
    discarded by exactly the runs that most want it, which is the same silent-config-override shape
    as the `v_max` and observation-era traps `policy_arch.py` exists to catch, and it is quieter
    because nothing about the run looks wrong afterwards.

    `epsilon` needs no equivalent: it is a plain Python float on the optimizer, so nothing restores
    it. `iterations` and the `m`/`v` moments are genuine training state and are deliberately left
    alone — this puts back the hyperparameter, not the history.

    Comparison is on the float32 round-trip because that is what the Variable holds: a configured
    1e-5 reads back as 9.999999747e-06, so comparing the raw floats would call every resume an
    override. `round(x, 6)` — the idiom `maybe_update_epsilon` uses — does separate 1e-5 from 1e-6,
    but it floors anything at or below 5e-7 to 0.0, so it would call 1e-7 and 1e-8 equal and
    silently drop a retune in that range. float32 equality needs no threshold and is what the
    Variable can actually represent.

    Returns the restored value when it differed from the configured one and was replaced, else
    None, so a caller can report the override.
    """
    restored = float(optimizer.learning_rate.numpy())
    if np.float32(restored) == np.float32(configured_lr):
        return None
    optimizer.learning_rate.assign(configured_lr)
    return restored


def maybe_update_epsilon(avg_reward, perfect_rate, epsilon, initial_epsilon, min_epsilon):
    """Assigns the scheduled epsilon, if it differs from what the Variable already holds.

    Round-trip through float32 makes the stored value slightly larger than the literal (0.2
    comes back as 0.20000000298), so the comparison is on the rounded value — the original
    ladder pinned itself at 0.2 forever without this.
    """
    target = epsilon_for(avg_reward, perfect_rate, initial_epsilon, min_epsilon)
    if round(float(epsilon.numpy()), 6) != round(target, 6):
        epsilon.assign(target)
