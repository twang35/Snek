"""A snake that never thinks: one Hamiltonian cycle over the 10x10 board, followed forever.

    PYTHONPATH=. python -m tools.fixed_path                         # the cycle alone, 2000 games
    PYTHONPATH=. python -m tools.fixed_path --hof --episodes 1000   # and every hallOfFame/ entry
    PYTHONPATH=. python -m tools.fixed_path --policy hallOfFame/<entry> --episodes 500

**Why it exists.** A snake that walks a closed tour of every cell can never hit itself and can never
starve (the food is always ahead of it on the tour, at most 99 cells away, under the 100-step floor of
the starve budget), so every game it plays is perfect. What it cannot do is take a shortcut, so it
pays the *whole* tour for every meal on average. That makes it the natural floor for the
"how long does a perfect game take" question the eval tables do not answer: a learned policy that
finishes 100 games in fewer steps than this is routing, and one that takes more is circling for safety
(`b10ck`, the undiscounted policy, ran 2,774-3,727 steps a game in its gifs).

**The tour.** Column 0 is the spine, walked top to bottom; the other nine columns are swept in
boustrophedon rows, bottom to top, so row 3 runs left to right -- which is the direction the opening
body already lies in (`START_TILE` (5, 3) with the body trailing to (1, 3)). `cycle()` asserts all of
that rather than trusting the drawing.

**Steps per game has a closed form.** With the snake at length L the free cells are the 100 - L cells
ahead of the head on the tour, the food is uniform over them, so the meal costs (101 - L) / 2 steps on
average; summed over L = 5 .. 99 that is 2,327.5. `expected_steps()` computes it and the CLI prints it
beside the measurement, which is how the driver below is checked against the game it drives.

**Every checkpoint measures in its own process.** `SNEK_OBS_HISTORY` is read once at import and the
hall of fame mixes history depths, so `--hof` runs one subprocess per entry (with the sidecar adopted,
as every other entry point does) and gathers their JSON. The fixed path itself needs no observation
at all, and `play` skips building one for it, which is most of the cost of a step.
"""

import argparse
import json
import os
import subprocess
import sys

from tools import sidecar_env  # noqa: E402  -- must precede anything that imports env.constants
sidecar_env.adopt_from_argv(sys.argv)

import numpy as np

from env import constants
from vectorized import config as C
from vectorized import vec_env
from vectorized.vec_env import VecSnake

PLAY = C.PLAY                     # 10 cells a side


def cycle():
    """The tour, as `PLAY * PLAY` (x, y) cells in walking order, starting at (0, 0).

    Down column 0, then rows PLAY-1 .. 0 over columns 1 .. PLAY-1, alternating direction so each row
    ends beside the start of the next: the bottom row runs right, the one above it left, and so on up
    to row 0, which ends at (1, 0) beside the start. Needs PLAY even, which 10 is.
    """
    if PLAY % 2:
        raise ValueError('the boustrophedon tour closes only on an even side, not {0}'.format(PLAY))
    cells = [(0, y) for y in range(PLAY)]
    for y in range(PLAY - 1, -1, -1):
        xs = range(1, PLAY) if (PLAY - 1 - y) % 2 == 0 else range(PLAY - 1, 0, -1)
        cells.extend((x, y) for x in xs)
    _check_tour(cells)
    return cells


def _check_tour(cells):
    """Every cell once, each step adjacent, closed, and the opening body lies along it head-forward."""
    if len(cells) != PLAY * PLAY or len(set(cells)) != len(cells):
        raise AssertionError('the tour does not visit every cell exactly once')
    for a, b in zip(cells, cells[1:] + cells[:1]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            raise AssertionError('tour cells {0} and {1} are not adjacent'.format(a, b))
    # The snake starts at START_TILE facing right with START_SEGMENTS cells trailing to its left, so
    # the tour must pass tail -> head in that order or the very first step is a reversal.
    at = {cell: i for i, cell in enumerate(cells)}
    x0, y0 = C.START_TILE
    body = [(x0 - k, y0) for k in range(C.START_SEGMENTS, -1, -1)]      # tail first
    for behind, ahead in zip(body, body[1:]):
        if at[ahead] != (at[behind] + 1) % len(cells):
            raise AssertionError('the opening body {0} does not lie along the tour'.format(body))


def next_cell_table():
    """`next_flat[cell] -> next cell on the tour`, on the padded flat index `VecSnake` uses.

    -1 off the tour (the wall ring), so a finished lane -- whose head may sit in the ring -- indexes
    to a harmless value rather than raising.
    """
    table = np.full(C.PAD, -1, dtype=np.int64)
    tour = cycle()
    for a, b in zip(tour, tour[1:] + tour[:1]):
        table[vec_env.flat(*a)] = vec_env.flat(*b)
    return table


class FixedPath:
    """`actions(vec) -> (n,)` relative actions that keep every lane on the tour."""

    name = 'fixed-path'

    def __init__(self):
        self.next_flat = next_cell_table()

    def actions(self, vec):
        head = vec.heads()
        target = self.next_flat[head]
        on_tour = target >= 0
        # `DIRCODE` is keyed by the flat delta plus GRID; an off-tour head gets `forward`, which is
        # only ever read on a lane whose result is discarded.
        delta = np.where(on_tour, target - head, vec_env.DELTA[vec.head_dir])
        new_dir = vec_env.DIRCODE[delta + vec_env.GRID]
        return vec_env.REL[vec.head_dir, new_dir]


class Checkpoint:
    """A restored checkpoint as the same interface: greedy over the observation."""

    def __init__(self, policy_dir, step=None):
        from tools import restore
        self.policy_fn, self.arch, self.step = restore.restore(policy_dir, step)
        self.name = '{0} @{1}'.format(os.path.basename(policy_dir.rstrip('/')), self.step)

    def actions(self, vec, obs=None):
        return np.asarray(self.policy_fn(obs if obs is not None else vec.observe()))


def expected_steps():
    """Mean steps a perfect game costs the tour: sum over lengths of the mean distance to the food.

    At length L the food is uniform over the 100 - L free cells, which lie 1 .. 100 - L ahead on the
    tour, so a meal costs (101 - L) / 2 steps; the game runs L from the opening length to the last
    meal at PERFECT_SCORE - 1.
    """
    cells = PLAY * PLAY
    first = C.START_SEGMENTS + 1
    return sum((cells + 1 - length) / 2.0 for length in range(first, cells))


def play(actor, episodes, seed=0, width=None, needs_obs=True):
    """`episodes` games under `actor`, returning per-episode arrays: steps, score, perfect, starved, died.

    Lanes run in lockstep; a lane whose quota is spent is reset and plays on with its result discarded,
    the way `vectorized/engine.py` treats an idle lane, so the array stays one contiguous batch.
    """
    width = int(width or min(episodes, 256))
    vec = VecSnake(width, seed=seed)
    quota_left = episodes - width          # games still to start, beyond the ones already on the board
    counted = np.ones(width, dtype=bool)   # lanes whose current game is one of the `episodes`
    out = {key: [] for key in ('steps', 'score', 'perfect', 'starved', 'died')}
    obs = vec.observe() if needs_obs else None
    banked = 0
    while banked < episodes:
        actions = actor.actions(vec, obs) if needs_obs else actor.actions(vec)
        _, _, done, info = vec.step(actions, autoreset=False, observe=False)
        hits = np.flatnonzero(done)
        for row in hits:
            if counted[row]:
                out['steps'].append(int(vec.step_count[row]))
                out['score'].append(int(vec.score[row]))
                out['perfect'].append(bool(info['perfect'][row]))
                out['starved'].append(bool(info['starved'][row]))
                out['died'].append(bool(info['died'][row]))
                banked += 1
                if quota_left > 0:
                    quota_left -= 1
                else:
                    counted[row] = False
        if hits.size:
            vec.reset_rows(hits)
        if needs_obs:
            obs = vec.observe()
    return {key: np.asarray(value) for key, value in out.items()}


def summarise(sample, name):
    """The row: the perfect-game rate and the step distribution of the perfect games."""
    perfect = sample['perfect']
    steps = sample['steps']
    won = steps[perfect]
    row = {'name': name, 'episodes': int(len(steps)),
           'perfect_games': int(perfect.sum()),
           'perfect_percent': round(100.0 * perfect.mean(), 2) if len(steps) else None,
           'starved': int(sample['starved'].sum()), 'died': int(sample['died'].sum())}
    if won.size:
        row.update({
            'steps_mean': round(float(won.mean()), 1),
            'steps_median': float(np.median(won)),
            'steps_std': round(float(won.std()), 1),
            'steps_min': int(won.min()), 'steps_max': int(won.max()),
            'steps_p90': float(np.percentile(won, 90)),
            # Steps per meal over a perfect game, which is the number the tour's closed form is in.
            'steps_per_food': round(float(won.mean()) / C.MAX_POSSIBLE_SCORE, 2),
        })
    return row


def table(rows):
    """Markdown, sorted by mean steps of a perfect game, fastest first."""
    head = ('| policy | perfect | steps: mean | median | min | max | p90 | per food |\n'
            '|---|---|---|---|---|---|---|---|')
    lines = [head]
    for row in sorted(rows, key=lambda r: r.get('steps_mean', float('inf'))):
        if 'steps_mean' not in row:
            lines.append('| {name} | {perfect_games}/{episodes} | -- | -- | -- | -- | -- | -- |'
                         .format(**row))
            continue
        lines.append('| {name} | {perfect_games}/{episodes} ({perfect_percent}%) | {steps_mean} | '
                     '{steps_median:.0f} | {steps_min} | {steps_max} | {steps_p90:.0f} | '
                     '{steps_per_food} |'.format(**row))
    return '\n'.join(lines)


def hof_entries():
    """Every directory under `hallOfFame/` holding an `arch.json`, sorted."""
    root = constants.HOF_DIR
    return sorted(os.path.join(root, name) for name in os.listdir(root)
                  if os.path.isfile(os.path.join(root, name, 'arch.json')))


def loadable_here(policy_dir):
    """Whether the entry's observation era is this checkout's, at any history depth.

    An entry from before `obs26-20260907` (the 30-value layout) refuses to restore under this env and
    can only be measured from a checkout of its era, so `--hof` reports it as skipped rather than
    dying on the first old entry. The depth suffix is the subprocess's business (`sidecar_env`).
    """
    with open(os.path.join(policy_dir, 'arch.json')) as handle:
        era = str(json.load(handle).get('obs_era', ''))
    return era.split('-hist')[0] == constants.BASE_OBS_ERA


def measure_in_subprocess(policy_dir, episodes, seed):
    """One checkpoint, in a fresh interpreter, so its sidecar decides `SNEK_OBS_HISTORY`."""
    env = dict(os.environ)
    env.pop(sidecar_env.KNOB, None)
    env['PYTHONPATH'] = constants.ROOT + os.pathsep + env.get('PYTHONPATH', '')
    argv = [sys.executable, '-m', 'tools.fixed_path', '--policy', policy_dir,
            '--episodes', str(episodes), '--seed', str(seed), '--json']
    done = subprocess.run(argv, cwd=constants.ROOT, env=env, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError('{0} failed:\n{1}'.format(policy_dir, done.stderr))
    return json.loads(done.stdout.strip().splitlines()[-1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--policy', default=None,
                        help='measure this checkpoint directory instead of the fixed path')
    parser.add_argument('--hof', action='store_true',
                        help='also measure every hallOfFame/ entry, one subprocess each')
    parser.add_argument('--json', action='store_true', help='print one JSON row and nothing else')
    parser.add_argument('--out', default=None, help='write every row as JSON here')
    args = parser.parse_args(argv)

    rows = []
    if args.policy:
        actor = Checkpoint(args.policy)
        sample = play(actor, args.episodes, seed=args.seed, width=args.width, needs_obs=True)
        rows.append(summarise(sample, actor.name))
        if args.json:
            print(json.dumps(rows[0]))
            return 0
    else:
        actor = FixedPath()
        sample = play(actor, args.episodes, seed=args.seed, width=args.width, needs_obs=False)
        row = summarise(sample, actor.name)
        row['steps_expected'] = expected_steps()
        rows.append(row)
        if args.json:
            print(json.dumps(row))
            return 0
        print('fixed path: {0} games, {1} perfect, mean {2} steps against a closed-form {3:.1f}'.format(
            row['episodes'], row['perfect_games'], row.get('steps_mean'), row['steps_expected']))
        if args.hof:
            for entry in hof_entries():
                if not loadable_here(entry):
                    print('skipping {0}: an earlier observation era, not loadable here'.format(
                        os.path.basename(entry)), file=sys.stderr)
                    continue
                print('measuring {0}...'.format(os.path.basename(entry)), file=sys.stderr)
                rows.append(measure_in_subprocess(entry, args.episodes, args.seed))

    print(table(rows))
    if args.out:
        with open(args.out, 'w') as handle:
            json.dump({'episodes': args.episodes, 'seed': args.seed, 'rows': rows}, handle, indent=1)
        print('wrote {0}'.format(args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
