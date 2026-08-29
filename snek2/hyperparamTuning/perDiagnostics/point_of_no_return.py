"""For each lost episode, the last snake length at which the food was still reachable.

This measures the **credit-assignment distance** for the endgame failure: how many moves before
death the game was already decided. That number is what decides whether any short-horizon change
(n-step returns, a deeper target, a one-step shield) can address it. If a lost game becomes
unrecoverable two moves before the collision, a short horizon is well matched; if it happens
thirty moves earlier, nothing with a short horizon reaches.

**What "still winnable" means here, exactly.** A full "does a winning continuation exist" is not
well posed, because food spawns at a uniformly random free cell, so whether the board can be
filled depends on placements that have not happened yet. Three criteria are used instead, all
exactly decidable from the current state and none assuming anything about future food:

- **`geom`** — some legal move sequence eats the current food, **ignoring the starve clock**. This
  is the pure board-geometry question: is the food reachable at all?
- **`reach`** — the same, but the sequence must fit inside the remaining starve budget.
- **`safe`** — `reach`, and immediately after eating the head can still reach its own tail through
  free space, so the snake is not sealed into a pocket.

**`geom` is the one to read for "when did the board become hopeless".** `reach` looks similar but is
confounded: the starve budget shrinks every move the snake fails to eat, so `reach` turns false
partly because the clock ran down rather than because the geometry changed — and on a starvation
death that is close to tautological near the end. The pair separates the two causes.

All three are **one food of lookahead**, so each gives an *upper bound* on the true point of no
return: a state can pass and still be doomed two foods later. Read the reported distances as
**lower bounds on how early the mistake was**.

All are searched over real game states rather than over currently-free cells, because following
your own vacating tail is a legitimate and necessary endgame manoeuvre -- restricting to free
cells would call winnable positions lost.

The step function is a 15-line reimplementation of `Snake.step()`'s movement rule, and it is
**checked against the live game on every step of every episode**. `mismatches` in the output must
be 0; anything else means the game changed and the search is measuring a different game. That is
the same guard `hyperparamTuning/diagnostics/diag*.py` uses, and the reason those scripts caught a
stale recomputation rather than reporting nonsense.

Usage, from `snek2/`:

    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/point_of_no_return.py \
        <policy-or-ckpt-path> <episodes> <seed> <out.json>

`<policy-or-ckpt-path>` is either a policy name under `savedPolicies/` (latest checkpoint) or a
full `ckpt-<step>` prefix, which is how a `hallOfFame/` entry is passed. Greedy play, so a given
seed reproduces the same games and two checkpoints can be compared on identical food sequences.
Shard it across seeds the way the `diagnostics/README.md` does; these are eval-style processes, so
the four-trainer rule does not apply, but each takes a core.
"""
import json
import os
import sys
import collections

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('SNEK_FC_LAYERS', '50,100,50')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment

from tf_agents.utils import common

from eval_agent import build_eval_agent
from snake_environment import SnakeEnvironment
from snake_constants import (SCREENTILES, MOVE_VECTORS, CURRENT_DIRECTION_MAPS,
                             TF_ACTION_TO_ACTIONS)
from state_helpers import (count_groups, get_adjacent_groups, starve_budget)
from under_the_hood import seed_process

# How far back from the death to look before giving up. The measurement is the whole point, so
# this is generous; a walk-back that reaches it is reported as censored rather than as a number.
MAX_WALKBACK = 150
# Search node cap per state evaluated. A nearly-full board has few legal moves so the real
# searches finish in hundreds of nodes; the cap only fires where free space is large, and those
# are reported as 'unknown' rather than as 'unwinnable'.
NODE_CAP = 60000
# Depth cap for the clock-free `geom` search. Well above the longest tail-following detour a
# nearly-full board admits, so it stands in for "unbounded" without letting the search run away.
GEOM_DEPTH = 200


def sim_step(body, food, move_dir, action):
    """One move. Returns (body, food, move_dir) or None if fatal.

    Mirrors `Snake.step()`: the tail vacates as the head advances, except on a step that eats,
    where `add_segment()` refills the tile it came from. Copied rather than imported because
    `diagnostics/diag.py` does not import on master, and checked against the live game by
    `verify_step()` on every step of every episode.
    """
    # `action` is a TF action index; the direction map is keyed by the relative-turn *name*.
    # TF_ACTION_TO_ACTIONS is the environment's own mapping, so this cannot drift from it.
    new_dir = CURRENT_DIRECTION_MAPS[move_dir][TF_ACTION_TO_ACTIONS[action]]
    vector = MOVE_VECTORS[new_dir]
    head = (body[0][0] + vector[0], body[0][1] + vector[1])
    if not (0 <= head[0] <= SCREENTILES[0] and 0 <= head[1] <= SCREENTILES[1]):
        return None
    eats = food is not None and head == food
    occupied = body if eats else body[:-1]
    if head in occupied:
        return None
    return ((head,) + tuple(occupied if eats else body[:-1]),
            None if eats else food, new_dir)


def build_grid(body, food):
    """The padded grid `Snake._rebuild_grid()` builds, from a body tuple and food cell."""
    grid = np.zeros((SCREENTILES[1] + 3, SCREENTILES[0] + 3))
    grid[[0, -1], :] = 4
    grid[:, [0, -1]] = 4
    if food is not None:
        grid[food[1] + 1, food[0] + 1] = 1
    for index, cell in enumerate(body):
        grid[cell[1] + 1, cell[0] + 1] = 2 if index == 0 else 3
    return grid


def tail_reachable(body, food):
    """Can the head reach its own tail through open space? The 'not sealed in' test."""
    grid = build_grid(body, food)
    regions, _ = count_groups(grid)
    cols = grid.shape[1]
    head_regions = get_adjacent_groups(regions, cols, body[0])
    tail_regions = get_adjacent_groups(regions, cols, body[-1])
    return bool(head_regions & tail_regions)


def food_still_gettable(body, food, move_dir, steps_left, require_tail=False):
    """Is there a legal move sequence that eats `food` within `steps_left` moves?

    Breadth-first over real game states, so tail-following is allowed. Returns
    (True, moves) / (False, None) / (None, None) when the node cap is hit, which is 'unknown'
    and must not be read as 'unwinnable'.
    """
    if food is None or steps_left <= 0:
        return False, None
    start = (tuple(body), move_dir)
    seen = {start}
    frontier = collections.deque([(tuple(body), move_dir, 0)])
    nodes = 0
    while frontier:
        b, d, depth = frontier.popleft()
        if depth >= steps_left:
            continue
        for action in range(3):
            nodes += 1
            if nodes > NODE_CAP:
                return None, None
            nxt = sim_step(b, food, d, action)
            if nxt is None:
                continue
            nb, nfood, nd = nxt
            if nfood is None:                      # this move ate the food
                if not require_tail or tail_reachable(nb, None):
                    return True, depth + 1
                continue                           # ate but sealed in; not a success
            key = (nb, nd)
            if key in seen:
                continue
            seen.add(key)
            frontier.append((nb, nd, depth + 1))
    return False, None


def verify_step(before, action, after):
    """Assert `sim_step` reproduces what the live game did. Returns True on a mismatch."""
    predicted = sim_step(tuple(before.body), before.food, before.head_move_dir, action)
    if after.finished:
        # The game ended. Either the move was fatal (predicted None) or it filled the board,
        # in which case the body is the whole grid and the snapshot is of a finished game.
        return not (predicted is None or after.perfect_game or before.starved or after.starved)
    if predicted is None:
        return True
    body, food, move_dir = predicted
    return not (list(body) == list(after.body) and move_dir == after.head_move_dir
                and (food == after.food or after.food is not None))


def load_agent(target):
    # The checkpoint is resolved before the agent is built, because `build_eval_agent` needs the
    # directory holding `arch.json` to rebuild the recorded network. A bare policy name means its
    # newest checkpoint; a `ckpt-<step>` path pins one, and its directory is the policy's.
    path = target
    if not os.path.basename(target).startswith('ckpt-'):
        path = tf.train.latest_checkpoint(os.path.join('savedPolicies', target))
        if path is None:
            raise SystemExit('no checkpoint for ' + target)
    py_env = SnakeEnvironment(discount=0.9975, display=False, policy_name='smoke')
    py_env.reset()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    agent, checkpoint, global_step = build_eval_agent(tf_env, py_env, os.path.dirname(path))
    checkpoint.restore(path).expect_partial()
    return py_env, tf_env, agent, path, int(global_step.numpy())


def main():
    target, episodes, seed, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    seed_process(seed, stream=0)
    py_env, tf_env, agent, path, step = load_agent(target)
    # common.function matches eval_workers' inference path; the greedy policy makes a seed
    # reproduce the same games, which is what lets two checkpoints be compared on identical food.
    policy_action = common.function(agent.policy.action)
    print('%s -> %s (global_step %d), %d episodes, seed %d'
          % (target, os.path.basename(path), step, episodes, seed), flush=True)

    results = []
    mismatches = 0
    outcomes = collections.Counter()

    for episode in range(episodes):
        time_step = tf_env.reset()
        trail = []                          # (snapshot, action) per step
        while True:
            snap = py_env.snapshot()
            action_step = policy_action(time_step)
            action = int(action_step.action.numpy()[0])
            time_step = tf_env.step(action_step.action)
            after = py_env.snapshot()
            if verify_step(snap, action, after):
                mismatches += 1
            trail.append((snap, action))
            if bool(time_step.is_last().numpy()[0]):
                break

        final = py_env.snapshot()
        if final.perfect_game:
            outcomes['perfect'] += 1
            continue
        outcomes['starved' if final.starved else 'collision'] += 1

        row = {'episode': episode, 'outcome': 'starved' if final.starved else 'collision',
               'final_length': len(trail[-1][0].body), 'steps': len(trail)}
        for name, require_tail, use_clock in (('geom', False, False),
                                              ('reach', False, True),
                                              ('safe', True, True)):
            found = None
            for back, (snap, _) in enumerate(reversed(trail[:len(trail)])):
                if back >= MAX_WALKBACK:
                    break
                if use_clock:
                    left = starve_budget(len(snap.body)) - (snap.current_step - snap.last_food_step)
                else:
                    left = GEOM_DEPTH
                ok, moves = food_still_gettable(snap.body, snap.food, snap.head_move_dir,
                                               left, require_tail)
                if ok is None:
                    found = {'status': 'unknown', 'moves_before_death': back}
                    break
                if ok:
                    found = {'status': 'winnable', 'length': len(snap.body),
                             'moves_before_death': back, 'moves_to_food': moves}
                    break
            row[name] = found or {'status': 'censored'}
        results.append(row)
        if (episode + 1) % 20 == 0:
            print('  %d episodes, %s, %d losses analysed'
                  % (episode + 1, dict(outcomes), len(results)), flush=True)

    payload = {'target': target, 'checkpoint': os.path.basename(path), 'global_step': step,
               'seed': seed, 'episodes': episodes, 'mismatches': mismatches,
               'outcomes': dict(outcomes), 'losses': results}
    with open(out_path, 'w') as handle:
        json.dump(payload, handle, indent=1)
    report(payload)


def report(payload):
    print('\n%s @%d, seed %d: %s, sim mismatches %d'
          % (payload['target'], payload['global_step'], payload['seed'],
             payload['outcomes'], payload['mismatches']))
    losses = payload['losses']
    if not losses:
        print('  no losses to analyse')
        return
    for name in ('geom', 'reach', 'safe'):
        rows = [r for r in losses if r[name].get('status') == 'winnable']
        other = collections.Counter(r[name]['status'] for r in losses)
        if not rows:
            print('  %-6s no resolved rows (%s)' % (name, dict(other)))
            continue
        back = np.array([r[name]['moves_before_death'] for r in rows])
        length = np.array([r[name]['length'] for r in rows])
        final = np.array([r['final_length'] for r in rows])
        print('  %-6s n=%d of %d  %s' % (name, len(rows), len(losses), dict(other)))
        print('         last still-%s length: median %.0f  mean %.1f  range %d-%d'
              % (name, np.median(length), length.mean(), length.min(), length.max()))
        print('         moves before death: median %.0f  mean %.1f  p90 %.0f  max %d'
              % (np.median(back), back.mean(), np.percentile(back, 90), back.max()))
        print('         length lost between then and death: median %.0f'
              % np.median(final - length))


if __name__ == '__main__':
    main()
