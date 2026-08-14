"""Fixtures for `hyperparamTuning/perDiagnostics/eat_and_survive.py`.

The claim that script exists to test is "`geom` counts eating routes without asking whether eating
was survivable", so the fixture that matters is a board where **the food is reachable and eating it
kills the snake**. If that case ever stopped being distinguishable from a survivable one, the script
would silently agree with the measurement it was written to check.

Three things are pinned here, all of which a plausible simplification would break:

- **The eat does not vacate the tail.** `Snake.add_segment()` refills the tile the tail came from, so
  the eating move is the one move that shrinks free space. Drop that and a sealed pocket looks open.
- **Every eating route is searched, not the shortest.** Different routes to the same food leave
  different bodies behind, and `test_shortest_route_can_die_while_a_longer_one_lives` is a real board
  where the first route found is fatal and a later one is not.
- **A post-eat body covering the board is a won game**, not a trapped head. At length 99 the food is
  the last free cell.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'hyperparamTuning', 'perDiagnostics'))

import collections

import eat_and_survive as eas
from point_of_no_return import food_still_gettable, sim_step, GEOM_DEPTH
from snake_constants import PERFECT_SCORE

Snap = collections.namedtuple('Snap', 'body food head_move_dir')

DIRECTIONS = {(-1, 0): 'left', (1, 0): 'right', (0, -1): 'up', (0, 1): 'down'}


def serpentine_body(length, flip=False):
    """A snake of `length` laid along a boustrophedon path over the board, head last-laid.

    The packed-board shape the endgame actually reaches, and the only practical way to hand-build a
    body of 44 or 99 segments. Returns (body, head_move_dir); body[0] is the head.
    """
    path = []
    for y in range(10):
        columns = list(range(10)) if (y % 2 == 0) != flip else list(range(9, -1, -1))
        for x in columns:
            path.append((x, y))
    body = tuple(reversed(path[:length]))
    delta = (body[0][0] - body[1][0], body[0][1] - body[1][1])
    return body, DIRECTIONS[delta], path


def legal_moves(body, move_dir):
    return sum(1 for action in range(3) if sim_step(body, None, move_dir, action) is not None)


def test_open_board_survives_the_headline_depth():
    assert eas.survives_for(((5, 5), (5, 6), (5, 7), (5, 8)), 'up', 100) is True


def test_boxed_in_head_survives_nothing():
    # Head at (0,0) facing right: forward hits (1,0), the relative-left turn leaves the board, and
    # the relative-right turn hits (0,1). Both blockers are body, not the tail, so nothing vacates.
    boxed = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 2))
    assert legal_moves(boxed, 'right') == 0
    assert eas.survives_for(boxed, 'right', 1) is False


def test_reachable_food_can_still_be_fatal_to_eat():
    """The headline case: `geom` says winnable, and eating it seals the head in.

    The food sits in the (0,0) corner with (0,1) held by a body segment. The only route arrives from
    (1,0) heading left, so after eating, forward and the relative-right turn leave the board and the
    relative-left turn hits (0,1) — which is body, and is not the tail, so it does not vacate.
    """
    body = ((2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (2, 2))
    food = (0, 0)
    reachable, moves = food_still_gettable(body, food, 'left', GEOM_DEPTH, False)
    assert reachable is True and moves == 2, 'fixture must be geom-winnable'

    report = eas.assess_state(Snap(body, food, 'left'))
    assert report['legal'] is False, 'eating this food leaves no legal move'
    assert report['tail'] is False
    assert report['survival_depth'] == 0
    assert report['perfect'] is False


def test_shortest_route_can_die_while_a_longer_one_lives():
    """Why every route is enumerated. Measured board, not a constructed one.

    Found by sweeping boustrophedon bodies: at length 44 with the food at (8,9), the first route the
    breadth-first search returns cannot survive 20 further moves and a later route of the same length
    can. A version of this script that stopped at the first successful eat would score this state as
    doomed.
    """
    body, move_dir, _ = serpentine_body(44)
    food = (8, 9)
    routes = list(eas.eat_routes(body, food, move_dir, GEOM_DEPTH, max_routes=40))
    assert len(routes) > 1
    assert eas.survives_for(routes[0][0], routes[0][1], 20) is False
    assert any(eas.survives_for(b, d, 20) is True for b, d, _ in routes[1:])
    # And the assessment reports the best route, which is the whole point.
    assert eas.assess_state(Snap(body, food, move_dir))['survival_depth'] >= 20


def test_eating_the_last_free_cell_is_a_win_not_a_trap():
    body, move_dir, path = serpentine_body(PERFECT_SCORE - 1)
    food = path[PERFECT_SCORE - 1]
    report = eas.assess_state(Snap(body, food, move_dir))
    assert report['perfect'] is True
    # A head on a full board has no legal move; scoring that as trapped would report the win as a
    # death, which is the specific misreading this branch exists to prevent.
    assert report['legal'] is True
    assert report['survival_depth'] == max(eas.SURVIVAL_DEPTHS)


def test_post_eat_body_keeps_the_tail_and_grows_by_one():
    body = ((2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (2, 2))
    routes = list(eas.eat_routes(body, (0, 0), 'left', GEOM_DEPTH))
    assert len(routes) == 1
    post_body, post_dir, moves = routes[0]
    # Two moves. The first adds a head and vacates the tail, so the length holds at 7; the eating
    # move adds a head and keeps the tail, so the length goes to 8. That asymmetry is the mechanism
    # under test — a snake that vacated on the eat would end at 7 and never seal itself in.
    assert moves == 2
    assert len(post_body) == len(body) + 1
    assert post_body[0] == (0, 0)
    assert post_body[-1] == (1, 2), 'the eating move must not vacate the tail'
    assert post_dir == 'left'


def test_memo_bound_is_a_cache_and_changes_no_answer():
    """`MEMO_LIMIT` exists to bound RSS, so dropping the cache must not move a single verdict.

    A memo key is the whole body, several KB at endgame lengths, and an unbounded memo reached 3.7 GB
    per process once this search ran per step rather than per loss. Forcing a clear on every lookup
    is the strongest version of that pressure.
    """
    boards = [(((5, 5), (5, 6), (5, 7), (5, 8)), 'up', 100),
              (((0, 0), (1, 0), (1, 1), (0, 1), (0, 2)), 'right', 1),
              (serpentine_body(90)[0], serpentine_body(90)[1], 20)]
    original = eas.MEMO_LIMIT
    try:
        expected = [eas.survives_for(body, move_dir, depth) for body, move_dir, depth in boards]
        eas.MEMO_LIMIT = 1
        assert [eas.survives_for(b, d, k) for b, d, k in boards] == expected
    finally:
        eas.MEMO_LIMIT = original


def test_food_free_neighbours_counts_open_cells_only():
    body = ((2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (2, 2))
    # (0,0): (1,0) is open, (0,1) is body, and two sides are off the board.
    assert eas.free_neighbours(body, (0, 0), (0, 0)) == 1
    # (5,5) is in open space, so all four neighbours are free.
    assert eas.free_neighbours(body, (5, 5), (5, 5)) == 4
    # A corner of the board with nothing near it still only has two neighbours on the board.
    assert eas.free_neighbours(body, (9, 9), (9, 9)) == 2


def test_unreachable_food_yields_no_routes():
    # A closed ring of body around the food: every neighbour of (1,1) is occupied by a segment that
    # is not the tail, so no route reaches it and `assess_state` declines the state.
    ring = ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    assert list(eas.eat_routes(ring, (1, 1), 'up', GEOM_DEPTH)) == []
    assert eas.assess_state(Snap(ring, (1, 1), 'up')) is None
    assert list(eas.eat_routes(ring, None, 'up', GEOM_DEPTH)) == []
