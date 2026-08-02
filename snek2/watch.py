"""Watches a policy play, in a window, without costing the training run anything.

    cd snek2
    PYTHONPATH=. python -u watch.py <policy_name>          # follow a live arm
    PYTHONPATH=. python -u watch.py <policy_name> <step>    # pin one checkpoint

Training used to draw one eval episode itself, which is expensive for a reason no amount of
tuning the drawing code fixes: the game flips the display once per game step, and a flip is a
round trip to the window server. Measured on this machine, a flip costs ~5.2ms against ~2us
for everything else render() does per frame, so a 2000-step episode spends ~11.6s in the
window server against 1.7s headless. That landed on every eval, i.e. every 1000 training
steps, and grew as the policy improved and played longer episodes.

Rendering here instead separates the two concerns completely. Training runs headless and never
flips; this process flips as much as it likes, on a core that would otherwise be idle, and can
be started and stopped whenever you want to look. Nothing it does can affect a training run or
an eval — it only reads checkpoint files.

**It follows the arm.** With no step argument it loads the newest checkpoint under
savedPolicies/<policy_name>/ and re-checks between episodes, so leaving it running shows the
policy getting better in near-real-time. Training writes a checkpoint every 1000 steps (when
the score clears SNEK_MIN_CHECKPOINT_SCORE), so what you see is at most one eval behind.

A dead or very young arm may have no checkpoint at all, since training skips writing them
below the score gate. That is reported rather than treated as an error.

Environment:
    WATCH_FPS         frame rate cap (default 90; 0 for uncapped)
    WATCH_EPISODES    episodes to play, 0 for forever (default 0)
    WATCH_PERFECT_WAIT_MS  pause on a win (default 2000, long enough to see it)

One frame is one game step, so 90 fps moves the snake 90 tiles a second and a full board
(~3000 steps) takes about 35 seconds. That is close to the ceiling: a display flip costs ~5.2ms,
so uncapped tops out near 180 fps and the cap is doing real work at 90. Drop WATCH_FPS to 20 or
30 to actually follow the moves.
"""
import os
import sys
import time

# No SDL_VIDEODRIVER here: this is the one script that wants a real window. Audio is still
# silenced, for the same reason everything else does it — a bare pygame.init() anywhere would
# open a CoreAudio stream and spin coreaudiod.
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

import snake_constants
from snake_constants import POLICY_DIR
from snake_environment import SnakeEnvironment
from snek2 import build_q_net


def available_steps(ckpt_dir):
    """Checkpoint steps present in a policy directory, ascending."""
    if not os.path.isdir(ckpt_dir):
        return []
    steps = set()
    for name in os.listdir(ckpt_dir):
        if name.startswith('ckpt-') and '.index' in name:
            try:
                steps.add(int(name[len('ckpt-'):].split('.')[0]))
            except ValueError:
                continue
    return sorted(steps)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1

    policy_name = argv[1]
    pinned_step = int(argv[2]) if len(argv) > 2 else None
    fps = int(os.environ.get('WATCH_FPS', 90))
    max_episodes = int(os.environ.get('WATCH_EPISODES', 0))
    snake_constants.PERFECT_GAME_WAIT_MS = int(os.environ.get('WATCH_PERFECT_WAIT_MS', 2000))

    ckpt_dir = os.path.join(POLICY_DIR, policy_name)
    steps = available_steps(ckpt_dir)
    if not steps:
        print('no checkpoints in {0}'.format(ckpt_dir))
        print('A young or dead arm may have none: training skips writing a checkpoint when '
              'the score is below SNEK_MIN_CHECKPOINT_SCORE (currently {0:.0f}).'
              .format(snake_constants.MIN_CHECKPOINT_SCORE))
        return 1
    if pinned_step is not None and pinned_step not in steps:
        print('no checkpoint for step {0} in {1}'.format(pinned_step, ckpt_dir))
        return 1

    # limit_fps caps the frame rate inside Game.render() via its own clock, which is the right
    # place for it: the flip is the expensive part, so throttling frames throttles the cost.
    env = SnakeEnvironment(discount=0.99, display=True, limit_fps=fps > 0,
                           policy_name=policy_name)
    if fps > 0:
        snake_constants.FPS_LIMIT = fps
    tf_env = tf_py_environment.TFPyEnvironment(env)

    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    global_step = tf.compat.v1.train.get_or_create_global_step()
    agent = dqn_agent.DdqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=build_q_net(num_actions),
        epsilon_greedy=0.0,  # watching the greedy policy, same as an eval
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        td_errors_loss_fn=common.element_wise_huber_loss,
        target_update_period=8,
        train_step_counter=global_step)
    agent.initialize()
    # Mirrors the keys common.Checkpointer writes in snek2.py, so one ckpt-<step> can be
    # restored by name rather than only the latest.
    checkpoint = tf.train.Checkpoint(agent=agent, policy=agent.policy, global_step=global_step)
    policy_action = common.function(agent.policy.action)

    loaded_step = None
    episode = 0
    print('watching {0} at {1} fps — ctrl-c to stop'.format(
        policy_name, fps if fps > 0 else 'uncapped'))

    while max_episodes == 0 or episode < max_episodes:
        # Re-read between episodes so a live arm's progress shows up without a restart.
        wanted = pinned_step
        if wanted is None:
            current = available_steps(ckpt_dir)
            wanted = current[-1] if current else loaded_step
        if wanted != loaded_step:
            checkpoint.restore(os.path.join(ckpt_dir, 'ckpt-{0}'.format(wanted))).expect_partial()
            loaded_step = wanted
            print('  loaded checkpoint {0}'.format(loaded_step))

        time_step = tf_env.reset()
        started = time.time()
        while not time_step.is_last():
            time_step = tf_env.step(policy_action(time_step).action)

        episode += 1
        game = env._game
        outcome = 'PERFECT' if game.perfect_game else ('starved' if game.starved else 'died')
        print('  episode {0}: score {1}, {2} steps, {3}, {4:.1f}s'.format(
            episode, env.get_score(), game.current_step, outcome, time.time() - started))

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        print('\nstopped')
