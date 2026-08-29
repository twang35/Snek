"""Was eating the last reachable food actually survivable, or does `geom` count eat-and-die?

[`point_of_no_return.py`](point_of_no_return.py) established that the food stays **geometrically
reachable** until the last 0-2 moves in every loss, and `findings.md` reads that as "the board is
never a dead end". That reading has a hole, and this script measures it: `geom` returns success the
moment *any* move sequence reaches the food, and asks nothing about the position it leaves behind.

The hole is a real mechanism, not a hypothetical. Eating does not vacate the tail —
`Snake.add_segment()` refills the tile the tail came from — so the move that eats is the one move
in the game that shrinks free space. If the food sits in a pocket, arriving there fills the pocket's
last cell with a head that has nowhere to go. `geom` would call that state winnable and the snake
would be dead two moves later.

So this asks the stronger question, at the same state `geom` resolves on:

| tier | question |
|---|---|
| `geom` | some route eats the food (reproduces `point_of_no_return.py`, as a cross-check) |
| `legal` | some route eats it and the head has **at least one legal move** afterwards |
| `tail` | some route eats it and the head can then reach its own tail through free space |
| `surv<N>` | some route eats it and the snake can then survive **N more moves**, by exact search |

**Every tier searches over all eating routes, not the shortest one.** Different routes to the same
food leave different bodies behind, so "eating was survivable" is a question about the best route,
and a search that stops at the first successful eat cannot answer it. That is the specific respect
in which this is not a re-run of `point_of_no_return.py`.

**The survival search assumes no food on the board**, which is deliberate and makes every tier an
*upper* bound on survivability: the next food spawns on a random free cell, and a snake that must
eventually eat is more constrained than one that never grows again. Ignoring the starve clock after
the meal is exact rather than optimistic, though — eating resets it, and `MIN_STARVE_BUDGET` is 100,
so any `surv100` claim fits inside the budget the meal buys.

A post-eat body covering the whole board is scored as **`perfect`**, not as "no legal move": at
length 99 the food is the last free cell and eating it wins the game.

The step function, the grid builder and the live-game guard are imported from
`point_of_no_return.py` rather than copied, so the two scripts cannot drift into measuring
different games. `mismatches` must be 0 in the output.

Usage, from `snek2/`:

    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/eat_and_survive.py \
        <policy-or-ckpt-path> <episodes> <seed> <out.json>

Shard across seeds like `point_of_no_return.py`; eval-style processes, so the four-trainer rule
does not apply, but each takes a core.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Imported for the movement rule, the padded grid, the tail test, the live-game guard and the
# checkpoint loader. Importing rather than copying is the whole reason the two scripts stay
# comparable; it also pulls in that module's SDL/TF environment setup, which must precede the
# tensorflow import below.
from point_of_no_return import (build_grid, food_still_gettable, load_agent, sim_step,
                                tail_reachable, verify_step, GEOM_DEPTH)

from tf_agents.utils import common

from snake_constants import PERFECT_SCORE, SCREENTILES
from under_the_hood import seed_process

# How far back from the death to walk when looking for the latest state at which each tier still
# held. 150 in `point_of_no_return.py`; smaller here because every state costs an eat-route
# enumeration plus a survival search, and the numbers being extended (`geom` median 0, max 2) sit
# at the very end of the episode. A walk-back that reaches this is reported as censored.
MAX_WALKBACK = 40
# Distinct post-eat states examined per state. A route enumeration on a nearly-full board finds a
# handful; the cap only fires with lots of free space, and those states report 'unknown'.
MAX_ROUTES = 200
# States expanded by one route enumeration, matching `point_of_no_return.NODE_CAP`. **This one is
# load-bearing for memory, not just time.** The breadth-first frontier keys on the whole body, so a
# key is several KB at endgame lengths, and an *unbounded* enumeration over a board with 10+ free
# cells reached **9.2 GB of RSS in 90 seconds**. Hitting it truncates the route list, which makes the
# answer conservatively negative — never a false "survivable".
ROUTE_NODE_CAP = 60000
# Node budget for one survival search. Branching is 1-2 on a packed board, so a real search
# finishes in thousands; the cap turns a runaway into 'unknown' rather than into 'doomed'.
SURVIVAL_NODES = 400000
# Entries kept in one search's memo before it is dropped. **This is a memory bound, not a
# correctness knob** — the memo is a pure cache, so clearing it costs repeated work and changes no
# answer. It needs a bound because a key is the whole body: a ~98-segment tuple of pairs is several
# KB, so an unbounded memo reached **3.7 GB of RSS per process** when `endgame_packing.py` called
# this once per step instead of once per loss, and four of those nearly took the laptop out.
MEMO_LIMIT = 5000
# Survival depths reported, descending. 100 is the headline — it is far longer than the 0-2 moves
# `geom` buys and comfortably inside the starve budget a meal resets. The shorter rungs exist to
# tell "trapped instantly" apart from "trapped in a pocket a few moves later", which is exactly
# the failure this script was written to look for.
SURVIVAL_DEPTHS = (100, 50, 20, 5, 1)


def free_neighbours(body, food, cell):
    """Open orthogonal neighbours of `cell` on the board `body`/`food` describe.

    The direct read of the mechanism in question: the food's own free-neighbour count is what
    decides whether arriving on it leaves the head anywhere to go. Counted on the *pre-eat* board,
    where the food cell itself is open and every body cell including the tail is occupied.
    """
    grid = build_grid(body, food)
    count = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        # build_grid pads by one, so a neighbour off the board reads as the wall value 4.
        if grid[cell[1] + 1 + dy, cell[0] + 1 + dx] == 0:
            count += 1
    return count


def eat_routes(body, food, move_dir, steps_left, max_routes=MAX_ROUTES,
               node_cap=ROUTE_NODE_CAP, stats=None):
    """Every distinct state reachable *immediately after* eating `food`, breadth-first.

    The generalisation of `point_of_no_return.food_still_gettable`, which returns as soon as one
    route eats and therefore describes one post-eat board out of many. Yields
    (post_eat_body, post_eat_dir, moves_to_food) in nondecreasing route length, deduped on the
    post-eat state so two routes arriving the same way are searched once.

    Yields nothing if the food is unreachable. Pass a dict as `stats` to learn whether the answer is
    complete: it gets `truncated=True` if either cap fired, which means "there may be routes this
    did not look at" and therefore that a negative verdict from the caller is not exact. Callers
    that compare two policies **must** record it — the caps fire on boards with more free space, so
    an unrecorded truncation silently penalises whichever policy fragments its space more.
    """
    if stats is not None:
        stats['truncated'] = False
    if food is None or steps_left <= 0:
        return
    seen_states = {(tuple(body), move_dir)}
    seen_post_eat = set()
    frontier = collections.deque([(tuple(body), move_dir, 0)])
    expanded = 0
    while frontier:
        current_body, current_dir, depth = frontier.popleft()
        if depth >= steps_left:
            continue
        expanded += 1
        if expanded > node_cap:
            if stats is not None:
                stats['truncated'] = True
            return
        for action in range(3):
            nxt = sim_step(current_body, food, current_dir, action)
            if nxt is None:
                continue
            new_body, new_food, new_dir = nxt
            if new_food is None:
                # This move ate the food. The post-eat body keeps the old tail, which is what
                # makes an eat able to seal the head in.
                key = (new_body, new_dir)
                if key in seen_post_eat:
                    continue
                seen_post_eat.add(key)
                yield new_body, new_dir, depth + 1
                if len(seen_post_eat) >= max_routes:
                    if stats is not None:
                        stats['truncated'] = True
                    return
                continue
            key = (new_body, new_dir)
            if key in seen_states:
                continue
            seen_states.add(key)
            frontier.append((new_body, new_dir, depth + 1))


def survives_for(body, move_dir, depth, budget=SURVIVAL_NODES):
    """Can some move sequence survive `depth` more moves? Exact, over real game states.

    No food on the board, so the tail vacates on every move and the snake never grows — see the
    module docstring on why that makes this an upper bound. A body covering the board is a won
    game, not a trapped head.

    Returns True / False / None, where None means the node budget ran out and the state is
    'unknown'. It must never be read as 'doomed'.
    """
    memo = {}
    nodes = [0]

    def search(current_body, current_dir, remaining):
        if remaining == 0 or len(current_body) == PERFECT_SCORE:
            return True
        key = (current_body, current_dir, remaining)
        cached = memo.get(key)
        if cached is not None:
            return cached
        if len(memo) >= MEMO_LIMIT:
            memo.clear()
        for action in range(3):
            nodes[0] += 1
            if nodes[0] > budget:
                return None
            nxt = sim_step(current_body, None, current_dir, action)
            if nxt is None:
                continue
            got = search(nxt[0], nxt[2], remaining - 1)
            if got is None:
                return None
            if got:
                memo[key] = True
                return True
        memo[key] = False
        return False

    return search(tuple(body), move_dir, depth)


def assess_state(snap):
    """The full tier report for one state: could this food be eaten, and then survived?

    Returns None when the food is unreachable at all, which is the `geom`-false case the caller
    walks further back on.
    """
    body, food, move_dir = snap.body, snap.food, snap.head_move_dir
    routes = list(eat_routes(body, food, move_dir, GEOM_DEPTH))
    if not routes:
        return None

    report = {'routes': len(routes),
              'routes_truncated': len(routes) >= MAX_ROUTES,
              'length': len(body),
              'food_free_neighbours': free_neighbours(body, food, food),
              'moves_to_food': min(depth for _, _, depth in routes),
              'perfect': False,
              'legal': False,
              'tail': False,
              'survival_depth': 0,
              'survival_unknown': False}

    best_legal = 0
    for post_body, post_dir, _ in routes:
        if len(post_body) == PERFECT_SCORE:
            # Eating filled the board. This route wins the game outright, so every tier holds and
            # there is nothing left to search.
            report.update(perfect=True, legal=True, tail=True,
                          survival_depth=max(SURVIVAL_DEPTHS))
            return report
        moves = sum(1 for action in range(3)
                    if sim_step(post_body, None, post_dir, action) is not None)
        best_legal = max(best_legal, moves)
        if moves and tail_reachable(post_body, None):
            report['tail'] = True
    report['legal'] = best_legal > 0

    # Deepest survival any route achieves. Descending, so the first hit is the answer and a
    # doomed state falls through every rung cheaply — a trapped head fails depth 1 in three nodes.
    for depth in SURVIVAL_DEPTHS:
        if report['survival_depth'] >= depth:
            break
        for post_body, post_dir, _ in routes:
            got = survives_for(post_body, post_dir, depth)
            if got is None:
                report['survival_unknown'] = True
                continue
            if got:
                report['survival_depth'] = depth
                break
        if report['survival_depth'] >= depth:
            break
    return report


def analyse_loss(trail):
    """Walk back from the death and resolve every tier on the states before it."""
    row = {}
    # The state `geom` resolves on: the latest one from which the food can still be eaten. This
    # reproduces point_of_no_return.py's search, and it is the state the headline question is
    # about.
    at_geom = None
    for back, (snap, _) in enumerate(reversed(trail)):
        if back >= MAX_WALKBACK:
            break
        ok, _ = food_still_gettable(snap.body, snap.food, snap.head_move_dir, GEOM_DEPTH, False)
        if ok is None:
            at_geom = {'status': 'unknown', 'moves_before_death': back}
            break
        if ok:
            assessment = assess_state(snap)
            at_geom = {'status': 'resolved', 'moves_before_death': back}
            at_geom.update(assessment or {})
            break
    row['at_last_geom'] = at_geom or {'status': 'censored'}

    # And the latest state from which eating leaves a snake that survives the headline depth. The
    # `safe` column of point_of_no_return.py asked a version of this with the static-tail test that
    # `diagnostics/README.md` measures at 22.1%; this uses the exact search instead.
    strong = None
    for back, (snap, _) in enumerate(reversed(trail)):
        if back >= MAX_WALKBACK:
            break
        assessment = assess_state(snap)
        if assessment is None:
            continue
        if assessment['survival_depth'] >= max(SURVIVAL_DEPTHS):
            strong = {'status': 'resolved', 'moves_before_death': back,
                      'length': assessment['length'],
                      'moves_to_food': assessment['moves_to_food']}
            break
    row['at_last_eat_and_survive'] = strong or {'status': 'censored'}
    return row


def main():
    target, episodes, seed, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    seed_process(seed, stream=0)
    py_env, tf_env, agent, path, step = load_agent(target)
    policy_action = common.function(agent.policy.action)
    print('%s -> %s (global_step %d), %d episodes, seed %d'
          % (target, os.path.basename(path), step, episodes, seed), flush=True)

    results = []
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
        if final.perfect_game:
            outcomes['perfect'] += 1
            continue
        outcomes['starved' if final.starved else 'collision'] += 1

        row = {'episode': episode, 'outcome': 'starved' if final.starved else 'collision',
               'final_length': len(trail[-1][0].body), 'steps': len(trail)}
        row.update(analyse_loss(trail))
        results.append(row)
        print('  ep %d %s len %d: %s' % (episode, row['outcome'], row['final_length'],
                                         summarise(row)), flush=True)

    payload = {'target': target, 'checkpoint': os.path.basename(path), 'global_step': step,
               'seed': seed, 'episodes': episodes, 'mismatches': mismatches,
               'board': '%dx%d' % (SCREENTILES[0] + 1, SCREENTILES[1] + 1),
               'survival_depths': list(SURVIVAL_DEPTHS),
               'outcomes': dict(outcomes), 'losses': results}
    with open(out_path, 'w') as handle:
        json.dump(payload, handle, indent=1)
    report(payload)


def summarise(row):
    at = row['at_last_geom']
    if at.get('status') != 'resolved':
        return 'geom %s' % at.get('status')
    return ('geom -%d moves, food nbrs %d, %d routes, legal %s, tail %s, survives %d'
            % (at['moves_before_death'], at['food_free_neighbours'], at['routes'],
               at['legal'], at['tail'], at['survival_depth']))


def report(payload):
    print('\n%s @%d, seed %d: %s, sim mismatches %d'
          % (payload['target'], payload['global_step'], payload['seed'],
             payload['outcomes'], payload['mismatches']))
    losses = payload['losses']
    resolved = [r['at_last_geom'] for r in losses
                if r['at_last_geom'].get('status') == 'resolved']
    if not resolved:
        print('  no resolved losses')
        return
    print('  at the last state where the food was still reachable, n=%d of %d losses'
          % (len(resolved), len(losses)))
    depths = np.array([r['survival_depth'] for r in resolved])
    print('    head has a legal move after eating: %d' % sum(r['legal'] for r in resolved))
    print('    head can reach its tail after eating: %d' % sum(r['tail'] for r in resolved))
    print('    eating wins the game outright:       %d' % sum(r['perfect'] for r in resolved))
    for depth in sorted(SURVIVAL_DEPTHS):
        print('    survives >= %3d moves after eating: %d' % (depth, int((depths >= depth).sum())))
    print('    survival depth: median %.0f  mean %.1f' % (np.median(depths), depths.mean()))
    nbrs = collections.Counter(r['food_free_neighbours'] for r in resolved)
    print('    food free-neighbour count: %s' % dict(sorted(nbrs.items())))
    unknown = sum(r['survival_unknown'] for r in resolved)
    truncated = sum(r['routes_truncated'] for r in resolved)
    print('    node budget hit: %d, route cap hit: %d' % (unknown, truncated))
    strong = [r['at_last_eat_and_survive'] for r in losses
              if r['at_last_eat_and_survive'].get('status') == 'resolved']
    if strong:
        back = np.array([r['moves_before_death'] for r in strong])
        print('  latest eat-and-survive-%d state, n=%d of %d: moves before death median %.0f, '
              'p90 %.0f, max %d' % (max(SURVIVAL_DEPTHS), len(strong), len(losses),
                                    np.median(back), np.percentile(back, 90), back.max()))
    else:
        print('  no loss had an eat-and-survive-%d state within %d moves of death'
              % (max(SURVIVAL_DEPTHS), MAX_WALKBACK))


if __name__ == '__main__':
    main()
