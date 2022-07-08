import time
from under_the_hood import *
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


def train(num_iterations, eval_env, train_py_env, agent, collect_driver, iterator, replay_buffer):
    # (Optional) Optimize by wrapping some code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training
    training_metrics = TrainingMetrics(agent.train_step_counter)
    avg_return = compute_avg_return(eval_env, agent.policy, training_metrics, num_eval_episodes)
    training_metrics.returns.append(avg_return)
    print('before training return: ', training_metrics.returns)

    screen = pf.screen(np.zeros((480, 560)), 'Training results')

    print('Begin training: ', time.strftime("%H:%M:%S", time.localtime()))

    # Reset the environment.
    time_step = train_py_env.reset()

    for _ in range(num_iterations):
        # Collect a few steps and save to replay buffer.
        # To view q_values, breakpoint at line 160 in tf_agents/policies/q_policy.py
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, sample_info = next(iterator)
        loss_info = agent.train(experience)

        # check that loss_info.extra.td_loss is different for each key
        replay_buffer.update_priorities([element[0] for element in sample_info.key],
                                        tf.cast(loss_info.extra.td_loss, tf.float64))

        log_messages_and_eval(training_metrics,
                              loss_info,
                              eval_env,
                              agent,
                              train_py_env,
                              screen)


class TrainingMetrics:
    def __init__(self, step_counter):
        self.step_counter = step_counter
        self.returns = []
        self.trailing_avg = []
        self.steps_start_time = time.time()
        self.training_start_time = time.time()
        self.eval_start_time = time.time()
        self.min_score = 1000
        self.max_score = 0

    def reset(self):
        self.steps_start_time = time.time()
        self.training_start_time = time.time()
        self.min_score = 1000
        self.max_score = -1000


def log_messages_and_eval(metrics, loss_info, eval_env, agent, train_py_env, screen):
    step = metrics.step_counter.numpy()
    if step % log_interval == 0:
        steps_per_second = log_interval / (time.time() - metrics.steps_start_time)
        print('step = {0}: loss = {1}, steps/second = {2}'.format(step,
                                                                  str(round(loss_info.loss.numpy(), 4)),
                                                                  round(steps_per_second, 2)))
        metrics.steps_start_time = time.time()

    if step % eval_interval == 0:
        print('training time: ', get_time(metrics.training_start_time))
        print('train_py_env high score: ', train_py_env.high_score)
        metrics.eval_start_time = time.time()
        avg_return = compute_avg_return(eval_env, agent.policy, metrics, num_eval_episodes)
        print('eval time: ', get_time(metrics.eval_start_time))

        maybe_update_epsilon(avg_return, train_py_env)

        metrics.trailing_avg.append(avg_return)
        if len(metrics.trailing_avg) > trailing_avg_window:
            metrics.trailing_avg.pop(0)

        print('step = {0}: avg_return = {1}, trailing_avg = {2}, min_score = {3}, max_score = {4}'
              .format(step,
                      str(round(avg_return, 3)),
                      str(round(compute_trailing_avg_return(metrics.trailing_avg), 3)),
                      metrics.min_score,
                      metrics.max_score))
        metrics.returns.append(avg_return)
        # restart time because compute_avg_return() takes a while and messes up the timing
        metrics.reset()

    if step % display_progress_interval == 0:
        display_progress(step + 1, eval_interval, metrics.returns, screen)


def get_time(start_time):
    total_time = time.time() - start_time
    if total_time > 60:
        return str(round(total_time / 60.0, 2)) + ' min'
    return str(round(total_time, 1)) + 's'


def maybe_update_epsilon(avg_return, train_py_env):
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
    if train_py_env.epsilon > 0.2 and avg_return > 5:
        train_py_env.epsilon = 0.2
    elif train_py_env.epsilon > 0.1 and avg_return > 10:
        train_py_env.epsilon = 0.1
    elif train_py_env.epsilon > 0.05 and avg_return > 20:
        train_py_env.epsilon = 0.05
    elif train_py_env.epsilon > 0.01 and avg_return > 30:
        train_py_env.epsilon = 0.01
    elif train_py_env.epsilon > 0.001 and avg_return > 40:
        train_py_env.epsilon = 0.001
    elif avg_return > 100:
        train_py_env.epsilon = 0.0
