import os
import time
from under_the_hood import *
from run_report import history_path, load_history, merge_eval_row, save_history, write_run_report
from schmid_policy import TheSchmidPolicy

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


def initial_populate_replay_buffer(use_theschmid_bot,
                                   time_step_spec,
                                   action_spec,
                                   train_py_env,
                                   schmid_py_env,
                                   rb_observer,
                                   initial_collect_steps):
    if not use_theschmid_bot:
        random_play(time_step_spec, action_spec, train_py_env, rb_observer, initial_collect_steps)
    else:
        schmid_play(time_step_spec, action_spec, schmid_py_env, rb_observer, initial_collect_steps)


def random_play(time_step_spec, action_spec, train_py_env, rb_observer, initial_collect_steps):
    print('Random play to populate replay buffer')

    random_policy = random_tf_policy.RandomTFPolicy(time_step_spec, action_spec)

    py_driver.PyDriver(
        train_py_env,
        py_tf_eager_policy.PyTFEagerPolicy(
            random_policy, use_tf_function=True),
        [rb_observer],
        max_steps=initial_collect_steps).run(train_py_env.reset())


def schmid_play(time_step_spec, action_spec, train_py_env, rb_observer, initial_collect_steps):
    print('theSchmid play to populate replay buffer')

    schmid_policy = TheSchmidPolicy(time_step_spec, action_spec)

    py_driver.PyDriver(
        train_py_env,
        py_tf_eager_policy.PyTFEagerPolicy(
            schmid_policy, use_tf_function=True),
        [rb_observer],
        max_steps=initial_collect_steps).run(train_py_env.reset())


def train(num_iterations, eval_env, eval_parallel_env, train_py_env, agent, collect_driver, batch_size, replay_buffer,
          train_checkpointer, replay_buffer_dir, global_step, epsilon, eval_only, policy_name, run_config):
    # (Optional) Optimize by wrapping some code in a graph using TF function.
    agent.train = common.function(agent.train)
    step = global_step.numpy()
    initial_step = np.copy(step)

    # Reset the train step.
    # agent.train_step_counter.assign(0)

    screen = pf.screen(np.zeros((480, 560)), '{0} results'.format(policy_name))
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

    avg_reward, avg_score = compute_avg_return(eval_env, eval_parallel_env, agent.policy, training_metrics,
                                               eval_only, num_eval_episodes)
    merge_eval_row(training_metrics.eval_rows,
                   build_eval_row(int(initial_step), avg_score, avg_score, avg_reward, training_metrics, epsilon))
    print('before training score: ', round(avg_score, 2))

    print('Begin training: ', time.strftime("%d/%m %H:%M:%S", time.localtime()))

    # Reset the environment.
    time_step = train_py_env.reset()

    for _ in range(num_iterations):
        # Collect a few steps and save to replay buffer.
        # To view q_values, breakpoint at line 160 in tf_agents/policies/q_policy.py
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        loss_info = 0
        if not eval_only:
            experience, indexes, is_weights = replay_buffer.sample(batch_size, step)
            # is_weights undo the bias that sampling by priority introduces.
            loss_info = agent.train(experience, weights=is_weights)
            # Transitions the network is worst at get sampled more often next time.
            # td_error, not td_loss: td_loss is Huber, which is quadratic below
            # |e|=1 and so shrinks small errors, making its spread wider than the
            # raw error's. Feeding it in gave an effective exponent near |e|^1.6
            # instead of |e|^0.6 and measurably hurt learning.
            replay_buffer.update_priorities(indexes, loss_info.extra.td_error.numpy())

        step += 1
        log_messages_and_eval(training_metrics, loss_info, eval_env, eval_parallel_env, agent, train_py_env, screen,
                              graph_path, report_path, graph_history_path, train_checkpointer, replay_buffer,
                              replay_buffer_dir, global_step, epsilon, step, eval_only, initial_step, policy_name,
                              run_config)


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


def build_eval_row(step, avg_score, trailing_avg_score, avg_reward, metrics, epsilon):
    """One row of the run report, and one point on the graph."""
    return {
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


def log_messages_and_eval(metrics, loss_info, eval_env, eval_parallel_env, agent, train_py_env, screen, graph_path,
                          report_path, graph_history_path, train_checkpointer, replay_buffer, replay_buffer_dir,
                          global_step, epsilon, step, eval_only, initial_step, policy_name, run_config):
    if step % log_interval == 0:
        steps_per_second = log_interval / (time.time() - metrics.steps_start_time)

        if eval_only:
            print('step = {0}: steps/second = {1}'.format(step, round(steps_per_second, 2)))
        else:
            print('step = {0}: loss = {1}, steps/second = {2}'.format(step,
                                                                      str(round(loss_info.loss.numpy(), 4)),
                                                                      round(steps_per_second, 2)))
        metrics.steps_start_time = time.time()

    if step % eval_interval == 0:
        print('training time: ', get_time(metrics.training_start_time))
        print('train_py_env high score: ', train_py_env.high_score)
        metrics.eval_start_time = time.time()
        avg_reward, avg_score = compute_avg_return(eval_env, eval_parallel_env, agent.policy, metrics, eval_only,
                                                   num_eval_episodes)
        print('eval time: ', get_time(metrics.eval_start_time))

        maybe_update_epsilon(avg_reward, epsilon)

        if not eval_only:
            print('saving checkpoint')
            train_checkpointer.save(global_step)
            # The buffer is ~20 MB and only warm-starts the next run, so it saves
            # far less often than the agent to keep disk churn down.
            if step % buffer_save_interval == 0:
                replay_buffer.save(replay_buffer_dir)

        metrics.trailing_avg_scores.append(avg_score)
        if len(metrics.trailing_avg_scores) > trailing_avg_window:
            metrics.trailing_avg_scores.pop(0)
        trailing_avg_score = sum(metrics.trailing_avg_scores) / len(metrics.trailing_avg_scores)

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
        print(eval_str)

        # Built before reset() clears the min/max trackers below.
        merge_eval_row(metrics.eval_rows,
                       build_eval_row(step, avg_score, trailing_avg_score, avg_reward, metrics, epsilon))
        save_history(graph_history_path, metrics.eval_rows, metrics.resume_steps)
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


def maybe_update_epsilon(avg_reward, epsilon):
    # For grid length 15
    # if train_py_env.epsilon > 0.2 and avg_return > 40:
    #     train_py_env.epsilon = 0.2
    # elif train_py_env.epsilon > 0.1 and avg_return > 60:
    #     train_py_env.epsilon = 0.1
    # elif train_py_env.epsilon > 0.05 and avg_return > 80:
    #     train_py_env.epsilon = 0.05
    # elif train_py_env.epsilon > 0.01 and avg_return > 100:
    #     train_py_env.epsilon = 0.01
    # elif avg_return > 140:
    #     train_py_env.epsilon = 0.001
    # For grid length 9
    # Round-trip through float32 makes the stored value slightly larger than the
    # literal (0.2 comes back as 0.20000000298), so comparing it directly would
    # re-match the first branch forever and pin epsilon at 0.2. Rounding restores
    # the exact comparisons the ladder relies on to step down one level per eval.
    current = round(float(epsilon.numpy()), 6)
    if current > 0.2 and avg_reward > 5:
        epsilon.assign(0.2)
    elif current > 0.1 and avg_reward > 10:
        epsilon.assign(0.1)
    elif current > 0.05 and avg_reward > 20:
        epsilon.assign(0.05)
    elif current > 0.01 and avg_reward > 40:
        epsilon.assign(0.01)
    elif current > 0.001 and avg_reward > 60:
        epsilon.assign(0.001)
    elif avg_reward > 100:
        epsilon.assign(0.0)
