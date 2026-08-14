"""How well is the body packed in the endgame, and does the food land somewhere safely edible?

[`eat_and_survive.py`](eat_and_survive.py) established that the losses are trapped positions: eating
the reachable food leaves the head no legal move in 54% of them. That is a property of the board the
snake had already built, not of the move it chose — at length 98 there are two free cells, and whether
one of them can be entered and left depends on how the body was arranged many meals earlier.

This measures that arrangement directly, on **won and lost episodes alike**, so it describes the
policy rather than its failures. Two families:

**Packing, at every state from `MIN_LENGTH` up — pure geometry, no search.**

| column | meaning |
|---|---|
| `regions` | connected regions of free space, food counted as free. **1 is packed**; more means the snake has cut its remaining space into pieces |
| `largest` | cells in the biggest region — with `regions`, says whether the split is even or a shaving |
| `food_nbrs` | open orthogonal neighbours of the food cell. 0 means arriving there fills a pocket |

At length 98 there are exactly two free cells, so `regions == 1` is precisely "the two free cells are
adjacent" and the measure becomes a yes/no about the final meal.

**Edibility, at each food spawn from `MIN_ASSESS_LENGTH` up — the outcome that packing buys.**

- `at_spawn` — the moment the food appeared, was there a route that eats it and survives 100 moves?
- `ever` — was there such a route at **any** state before the next meal? A spawn can be unsafe and
  become safe as the tail vacates, so `at_spawn` alone understates what the snake had available.
  Waiting is legitimate play; `ever` is the honest denominator, and `ever` false is a board that
  offered no safe meal at all.

Spawn states are found by watching the snake's length increase, which is exactly when a meal happened
and a new food was placed. `ever` stops at the first survivable state, so a well-packed policy is
cheap to measure and a dithering one pays the `WINDOW_CAP`.

Usage, from `snek2/`:

    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/endgame_packing.py \
        <policy-or-ckpt-path> <episodes> <seed> <out.json>

Same sharding, same live-game guard, `mismatches` must be 0.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from eat_and_survive import eat_routes, free_neighbours, survives_for
from point_of_no_return import build_grid, load_agent, sim_step, verify_step

from tf_agents.utils import common

from state_helpers import count_groups
from snake_constants import PERFECT_SCORE
from under_the_hood import seed_process

# Packing is recorded from here up. 80 rather than 90 so the run shows where the regions start to
# fragment, not only the state they are in at the end.
MIN_LENGTH = 80
# Edibility search is the expensive half, so it runs only in the band the losses live in.
MIN_ASSESS_LENGTH = 90
# States assessed per meal window when looking for `ever`. A window that never becomes survivable
# pays this in full, which is why it is not generous; windows longer than this report `capped`.
WINDOW_CAP = 15
# Caps for `safe_meal_available`. Deliberately tighter than `eat_and_survive.py`'s, which runs once
# per loss and can afford to be exhaustive; this runs thousands of times. Every cap makes the answer
# **conservatively false** — a missed route or an abandoned search reports "no safe meal" — so the
# safe-meal shares below are lower bounds, and the comparison between checkpoints is still fair
# because both pay the same caps.
SAFE_ROUTE_CAP = 40
SAFE_DEPTH = 100
SAFE_NODES = 30000
# Route enumeration bounds. `eat_and_survive.py` searches to depth 200 with a 60k node cap because it
# runs once per loss; this runs once per step, and at length 90 there are ten free cells, which is
# enough branching to make the unbounded version allocate **9.2 GB in 90 seconds**. 40 moves is far
# longer than any route a board with ≤10 free cells admits, so the depth bound is slack in practice
# while the node cap is what actually holds memory down.
SAFE_ROUTE_DEPTH = 40
SAFE_ROUTE_NODES = 20000

Packing = collections.namedtuple('Packing', 'length regions largest food_nbrs free_cells')

# Length bands for the report. 98 and 99 stand alone because the board holds two and one free cells
# there, so `regions == 1` stops being a summary and becomes a yes/no about the final meals.
BANDS = ((range(90, 95), '90-94'), (range(95, 98), '95-97'),
         (range(98, 99), '98'), (range(99, 100), '99'))


def safe_meal_available(snap):
    """(food reachable, some route eats it and survives `SAFE_DEPTH` moves, answer is exact).

    The third value is the one that keeps the comparison honest. A `False` verdict is only exact if
    the route enumeration ran to completion; the caps fire on boards with *more* free space, so
    treating a truncated `False` as a real one would penalise exactly the policy that fragments its
    space — the hypothesis under test. Report the safe-meal rates over exact rows only.

    The boolean `eat_and_survive.assess_state` computes as part of a full tier report. Written out
    separately because that function walks a descending depth ladder — 100, 50, 20, 5, 1 — and
    re-searches every route at every rung, which is the right shape for one state per loss and far
    too slow for one per step. Two orders of magnitude of the cost is avoided by the legal-move
    pre-filter: a sealed head is three `sim_step` calls, not a search.
    """
    if snap.food is None:
        return False, False, True
    stats = {}
    reachable = False
    abandoned = False
    for post_body, post_dir, _ in eat_routes(snap.body, snap.food, snap.head_move_dir,
                                             SAFE_ROUTE_DEPTH, max_routes=SAFE_ROUTE_CAP,
                                             node_cap=SAFE_ROUTE_NODES, stats=stats):
        reachable = True
        if len(post_body) == PERFECT_SCORE:
            return True, True, True                 # eating filled the board: the game is won
        if not any(sim_step(post_body, None, post_dir, action) is not None
                   for action in range(3)):
            continue                                # sealed in, so no search is needed
        survived = survives_for(post_body, post_dir, SAFE_DEPTH, SAFE_NODES)
        if survived is True:
            return True, True, True                 # a positive needs no completeness caveat
        if survived is None:
            abandoned = True                        # the survival search ran out of nodes
    return reachable, False, not (stats.get('truncated') or abandoned)


def packing(snap):
    """Region structure of the free space in one state. No search, one bitwise flood fill."""
    grid = build_grid(snap.body, snap.food)
    regions, count = count_groups(grid)
    largest = max((bin(region).count('1') for region in regions), default=0)
    return Packing(length=len(snap.body), regions=count, largest=largest,
                   food_nbrs=free_neighbours(snap.body, snap.food, snap.food)
                   if snap.food is not None else -1,
                   free_cells=PERFECT_SCORE - len(snap.body))


def meal_windows(trail):
    """(spawn_index, end_index) per meal, detected by the snake's length increasing.

    The state at `spawn_index` is the first one carrying the newly placed food; the window runs to
    the state before the next meal, which is the last chance to eat that food.
    """
    windows = []
    previous = len(trail[0][0].body)
    start = 0
    for index, (snap, _) in enumerate(trail):
        length = len(snap.body)
        if length > previous:
            windows.append((start, index))
            start = index
            previous = length
    windows.append((start, len(trail)))
    # The first window opens on the starting board rather than on a meal, so drop it.
    return windows[1:]


def main():
    target, episodes, seed, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    seed_process(seed, stream=0)
    py_env, tf_env, agent, path, step = load_agent(target)
    policy_action = common.function(agent.policy.action)
    print('%s -> %s (global_step %d), %d episodes, seed %d'
          % (target, os.path.basename(path), step, episodes, seed), flush=True)

    states = []
    meals = []
    mismatches = 0
    outcomes = collections.Counter()

    for episode in range(episodes):
        time_step = tf_env.reset()
        trail = []
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
        outcome = ('perfect' if final.perfect_game else
                   'starved' if final.starved else 'collision')
        outcomes[outcome] += 1

        for snap, _ in trail:
            if len(snap.body) >= MIN_LENGTH and snap.food is not None:
                states.append(packing(snap)._asdict())

        for start, end in meal_windows(trail):
            spawn = trail[start][0]
            if len(spawn.body) < MIN_ASSESS_LENGTH or spawn.food is None:
                continue
            reachable, safe, exact = safe_meal_available(spawn)
            spawn_packing = packing(spawn)
            row = {'episode': episode, 'outcome': outcome, 'length': len(spawn.body),
                   'window': end - start, 'at_spawn': safe, 'spawn_reachable': reachable,
                   'spawn_exact': exact, 'food_nbrs': spawn_packing.food_nbrs,
                   'regions': spawn_packing.regions, 'largest': spawn_packing.largest}
            ever = safe
            ever_exact = exact
            assessed = 1
            for snap, _ in trail[start + 1:end]:
                if ever or assessed >= WINDOW_CAP:
                    break
                if snap.food is None:
                    continue
                assessed += 1
                _, ever, state_exact = safe_meal_available(snap)
                ever_exact = ever_exact and state_exact
            row['ever'] = ever
            # `ever` is exact only if it found a safe state, or if it scanned the whole window with
            # every state resolved exactly. A window cut short by WINDOW_CAP is not evidence.
            row['ever_exact'] = ever or (ever_exact and assessed < WINDOW_CAP)
            meals.append(row)

        print('  ep %d %s: %d endgame states, %d meals assessed'
              % (episode, outcome, len(states), len(meals)), flush=True)

    payload = {'target': target, 'checkpoint': os.path.basename(path), 'global_step': step,
               'seed': seed, 'episodes': episodes, 'mismatches': mismatches,
               'min_length': MIN_LENGTH, 'min_assess_length': MIN_ASSESS_LENGTH,
               'window_cap': WINDOW_CAP, 'outcomes': dict(outcomes),
               'states': states, 'meals': meals}
    with open(out_path, 'w') as handle:
        json.dump(payload, handle)
    report(payload)


def report(payload):
    print('\n%s @%d, seed %d: %s, sim mismatches %d'
          % (payload['target'], payload['global_step'], payload['seed'],
             payload['outcomes'], payload['mismatches']))
    states = payload['states']
    if states:
        # Time-weighted: a policy that dithers 900 steps at length 98 contributes 900 rows here
        # against a policy that passes through in 30. Read the per-meal table below for the
        # comparison between checkpoints; this one describes where a policy spends its endgame.
        print('  packing over states (time-weighted), by snake length')
        print('    %-6s %7s %8s %8s %9s' % ('length', 'states', 'regions', 'one-piece', 'food nbr 0'))
        for band, label in ((range(80, 90), '80-89'), (range(90, 95), '90-94'),
                            (range(95, 98), '95-97'), (range(98, 99), '98'),
                            (range(99, 100), '99')):
            rows = [s for s in states if s['length'] in band]
            if not rows:
                continue
            regions = np.array([s['regions'] for s in rows])
            zero = np.array([s['food_nbrs'] == 0 for s in rows])
            print('    %-6s %7d %8.2f %7.0f%% %8.0f%%'
                  % (label, len(rows), regions.mean(), 100 * (regions == 1).mean(),
                     100 * zero.mean()))
    meals = payload['meals']
    if meals:
        # One row per meal, so every checkpoint contributes the same number of samples per game
        # reached. This is the table to compare policies on.
        print('  per meal (one sample each), by snake length at the spawn — packing, exact')
        print('    %-6s %6s %8s %10s %11s %9s'
              % ('length', 'meals', 'regions', 'one-piece', 'food nbr 0', 'largest'))
        for band, label in BANDS:
            rows = [m for m in meals if m['length'] in band]
            if not rows:
                continue
            regions = np.array([m['regions'] for m in rows])
            print('    %-6s %6d %8.2f %9.0f%% %10.0f%% %9.1f'
                  % (label, len(rows), regions.mean(), 100 * (regions == 1).mean(),
                     100 * np.mean([m['food_nbrs'] == 0 for m in rows]),
                     np.mean([m['largest'] for m in rows])))
        # A safe verdict is always exact — the search returns the moment it finds a surviving route —
        # so restricting to exact rows would keep every positive and drop only negatives, inflating
        # the rate. The rates below are over **all** rows and are therefore **lower bounds**, and
        # the `neg exact` column says how tight: at 100% every negative was searched to completion
        # and the bound is the answer.
        print('  per meal, safe-meal availability — lower bounds, see `neg exact`')
        print('    %-6s %6s %10s %10s %11s %9s'
              % ('length', 'meals', 'safe@spawn', 'neg exact', 'ever safe', 'capped'))
        for band, label in BANDS:
            rows = [m for m in meals if m['length'] in band]
            if not rows:
                continue
            negatives = [m for m in rows if not m['at_spawn']]
            capped = sum(1 for m in rows if not m['ever_exact'])
            print('    %-6s %6d %9.0f%% %9.0f%% %10.0f%% %9d'
                  % (label, len(rows),
                     100 * np.mean([m['at_spawn'] for m in rows]),
                     100 * np.mean([m['spawn_exact'] for m in negatives]) if negatives else 100,
                     100 * np.mean([m['ever'] for m in rows]), capped))


if __name__ == '__main__':
    main()
