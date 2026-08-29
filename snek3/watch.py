"""Watch a policy play, in a window, without costing a training run anything.

    PYTHONPATH=. python -u watch.py <policy>            # follow a live arm
    PYTHONPATH=. python -u watch.py <policy> <step>     # pin one checkpoint

**Training never draws, and this is why.** The game flips the display once per game step, and a flip
is a round trip to the window server: ~5.2 ms against ~2 us for everything else `render()` does. A
2,000-step episode costs ~11.6 s on screen against ~1.7 s headless, and snek2 paid that on every
eval, i.e. every 1,000 training steps, growing as the policy improved and played longer.

So this is a separate process that only *reads* checkpoint files. Nothing it does can affect a
trainer or an eval, and it can be started and stopped whenever you want to look.

**It follows the arm.** With no step argument it loads the newest checkpoint in the policy directory
and re-checks between episodes, so leaving it running shows the policy improving in near-real time.

Environment:
    WATCH_FPS               frame rate cap, 0 for uncapped (default 60)
    WATCH_EPISODES          episodes to play, 0 for forever (default 0)
    WATCH_PERFECT_WAIT_MS   pause on a win, long enough to see it (default 2000)
    SNEK_TILE_PIXELS        pixels per tile, so 10x this is the window edge (default 15 here)

One frame is one game step, so 60 fps moves the snake 60 tiles a second and a full board (~3,000
steps) takes about a minute. Uncapped tops out near 180 fps because of the flip, so the cap is doing
real work; drop it to 20 or 30 to actually follow the moves.
"""

import os
import sys
import time

# No SDL_VIDEODRIVER here — this is the one script that wants a real window. Audio is still silenced
# for the usual reason, and before any pygame import.
os.environ['SDL_AUDIODRIVER'] = 'dummy'
# A 150x150 window rather than the game's 100x100 default. About the floor for this: macOS needs
# roughly that much width to show any of the title bar, which matters when four of these are open.
# Before `env.render` is imported, because every pixel constant is derived from it at import.
os.environ.setdefault('SNEK_TILE_PIXELS', '15')

from env import render as R                                            # noqa: E402
from env.scalar_env import SnakeEnv                                    # noqa: E402
from tools import checkpoints, restore                                 # noqa: E402


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1

    policy = argv[1]
    pinned_step = int(argv[2]) if len(argv) > 2 else None
    fps = int(os.environ.get('WATCH_FPS', 60))
    max_episodes = int(os.environ.get('WATCH_EPISODES', 0))
    R.PERFECT_GAME_WAIT_MS = int(os.environ.get('WATCH_PERFECT_WAIT_MS', 2000))
    if fps > 0:
        # Assigned on the module, and `env.game` reads it as `R.FPS_LIMIT` rather than through its
        # star import, which is what makes this reach the clock at all.
        R.FPS_LIMIT = fps

    directory = restore.policy_dir(policy)
    present = checkpoints.steps(directory)
    if not present:
        print('no checkpoints in {0}'.format(directory))
        print('A young or dead arm may have none: training skips writing one below '
              'SNEK_MIN_CHECKPOINT_SCORE.')
        return 1
    if pinned_step is not None and pinned_step not in present:
        print('no checkpoint for step {0} in {1}; present {2}..{3}'.format(
            pinned_step, directory, present[0], present[-1]))
        return 1

    arch = restore.policy_arch(directory)
    net = restore.build_net(arch)
    policy_fn = restore.policy_fn_for(arch, net)

    # limit_fps caps the rate inside `Game.render()` via its own clock, which is the right place for
    # it: the flip is the expensive part, so throttling frames throttles the cost.
    env = SnakeEnv(discount=0.99, display=True, limit_fps=fps > 0, policy_name=policy)

    loaded_step = None
    episode = 0
    print('watching {0} at {1} fps — ctrl-c to stop'.format(
        policy, fps if fps > 0 else 'uncapped'))

    while max_episodes == 0 or episode < max_episodes:
        # Re-read between episodes, so a live arm's progress shows without a restart.
        wanted = pinned_step
        if wanted is None:
            wanted = checkpoints.latest_step(directory) or loaded_step
        if wanted != loaded_step:
            checkpoints.load(checkpoints.path(directory, wanted), net)
            loaded_step = wanted
            print('  loaded checkpoint {0}'.format(loaded_step))
            # `Game.reset()` applies this every episode, so it follows a live arm forward rather
            # than being clobbered by the next reset.
            env.game.caption = '{0} — ckpt {1}'.format(policy, loaded_step)

        observation = env.reset()
        started = time.time()
        done = False
        while not done:
            action = int(policy_fn(observation.reshape(1, -1))[0])
            observation, _, done, info = env.step(action)

        episode += 1
        outcome = 'PERFECT' if info['perfect'] else ('starved' if info['starved'] else 'died')
        print('  episode {0}: score {1}, {2} steps, {3}, {4:.1f}s'.format(
            episode, info['score'], info['steps'], outcome, time.time() - started))

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        print('\nstopped')
