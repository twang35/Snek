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

The window is titled `<policy_name> — ckpt <step>`, and sized 1.5x the game's native 100x100 so
macOS has room to show some of that title. Both matter when several of these are open at once,
which is the normal case with a batch of four arms running.

Environment:
    WATCH_FPS         frame rate cap (default 60; 0 for uncapped)
    WATCH_EPISODES    episodes to play, 0 for forever (default 0)
    WATCH_PERFECT_WAIT_MS  pause on a win (default 2000, long enough to see it)
    SNEK_TILE_PIXELS  pixels per tile, so 10x this is the window edge (default here 15)

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
# A 150x150 window instead of the 100x100 the game defaults to. That is about the floor for
# this: macOS needs roughly this much width to show any of the title, and 12pt 'Policy: <arm>'
# only just fits across it. setdefault, and before snake_constants is imported, because every
# pixel constant is derived from this at import time. Training keeps the small default — it
# never draws, but it still allocates a surface per sprite.
os.environ.setdefault('SNEK_TILE_PIXELS', '15')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common

import snake_constants
from eval_agent import build_eval_agent
from snake_constants import POLICY_DIR
from snake_environment import SnakeEnvironment


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
    fps = int(os.environ.get('WATCH_FPS', 60))
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

    # Through build_eval_agent, which is where the greedy agent for a saved policy is built: it reads
    # the layer widths, the algorithm and (for c51) the atom support out of arch.json and checks the
    # sidecar against this environment first. Watching used to build its own DdqnAgent, which meant a
    # c51 checkpoint would have been watched as a scalar one — the third copy of a construction this
    # repo has already been bitten by twice.
    agent, checkpoint, global_step = build_eval_agent(tf_env, env, ckpt_dir)
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
            # Game.reset() applies this on every episode, so it follows a live arm forward
            # rather than being clobbered by the next reset.
            env._game.caption = '{0} — ckpt {1}'.format(policy_name, loaded_step)

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
