"""Tests for Game.snapshot() / Game.restore_snapshot(), which is how a forked branch is created.

The whole feature rests on one claim: a restored game is indistinguishable from the game it was
copied from. Nothing raises if it is not — a snake whose segment directions are wrong simply walks
apart over the next few steps, which looks like a policy playing badly rather than like corrupt
state, and this project has already spent a run on that class of bug.

So the load-bearing test here is `test_a_restored_game_replays_a_fixed_action_sequence_identically`,
which compares every observation byte-for-byte across a real replay. The narrower tests exist to say
*which* field broke when it fails.

`diagnostics/diag.py`'s pygame-free simulator is used as an independent oracle for the grid, since it
was validated against the real game over 16,173 steps. It is deliberately never used in production —
a second implementation of the rules would have to re-derive rewards, terminal states, food sampling
and starvation.
"""
import importlib
import os
import random
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import numpy as np

import Snake
from snake_constants import (ACTIONS, CURRENT_DIRECTION_MAPS, MOVE_VECTORS, SCREENTILES,
                             TILE_SIZE)
from state_helpers import distance_to_food, get_observations


def snake_walking_up_a_column(length=8):
    """A hand-built snapshot: a snake heading up the middle of the board, food off to the side.

    Hand-built rather than played, so a test can assert on known cells. Head at (4, 2) facing up
    with the body trailing downward, which makes every segment's move_dir 'up'.
    """
    body = tuple((4, 2 + offset) for offset in range(length))
    return Snake.GameSnapshot(body=body, head_move_dir='up', tail_last_move_dir='up',
                              food=(8, 8), current_score=length - 5, current_step=120,
                              last_food_step=100, finished=False, starved=False,
                              perfect_game=False)


def coiled_snapshot():
    """A snapshot whose body turns corners, so the derived directions are not all the same.

    A straight snake cannot catch a direction bug: 'up' for every segment is also what a buggy
    derivation that ignored position would produce if it happened to pick 'up'.
    """
    body = ((3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (6, 3))
    # tail_last_move_dir is deliberately 'right' while the tail's *derived* direction is 'left',
    # which is the shape a just-added segment really has (add_segment sets move_dir but leaves
    # last_move_dir at the SnakeSegment default). With the two equal, a restore that dropped the
    # field entirely would still round-trip and the round-trip test would assert nothing.
    return Snake.GameSnapshot(body=body, head_move_dir='up', tail_last_move_dir='right',
                              food=(0, 9), current_score=3, current_step=60, last_food_step=55,
                              finished=False, starved=False, perfect_game=False)


def boustrophedon_cells():
    """Every cell of the board in one adjacent chain, sweeping each row alternately."""
    cells = []
    for y in range(SCREENTILES[1] + 1):
        columns = range(SCREENTILES[0] + 1)
        cells.extend((x, y) for x in (columns if y % 2 == 0 else reversed(columns)))
    return cells


def long_snapshot(length=85):
    """A length-85 snake filling most of the board, which is the regime forking targets.

    Laid out along a boustrophedon and then **reversed**, so the head sits at the open end of the
    chain with somewhere to go. Head-first along the sweep instead would box it in immediately —
    zero safe moves, which is a dead state and would make a replay test assert nothing.
    """
    cells = boustrophedon_cells()
    body = tuple(cells[length - 1::-1])
    return Snake.GameSnapshot(body=body, head_move_dir=Snake.direction_between(body[1], body[0]),
                              tail_last_move_dir='right', food=cells[95],
                              current_score=length - 5, current_step=1500, last_food_step=1480,
                              finished=False, starved=False, perfect_game=False)


def dead_snake_snapshot():
    """A body from a real self-collision death: the head sits on top of a body cell.

    Verbatim from what a forking smoke run snapshotted when it forked on the boundary step. A live
    snake never overlaps itself — measured zero overlaps in 2743 live steps against 149 self-collision
    deaths over 300 random episodes — so a repeated cell is a reliable signal that the game is over,
    which is what makes rejecting it useful rather than pedantic.
    """
    body = ((1, 8), (2, 8), (2, 9), (1, 9), (1, 8), (1, 7))
    return Snake.GameSnapshot(body=body, head_move_dir='down', tail_last_move_dir='down',
                              food=(7, 7), current_score=1, current_step=20, last_food_step=20,
                              finished=True, starved=False, perfect_game=False)


def restored(snapshot):
    game = Snake.Game(display=False)
    game.restore_snapshot(snapshot)
    return game


def played_snapshot(seed, grow_steps):
    """A snapshot of a game the code itself played, with the RNG state at that moment.

    Kept short deliberately: a food-chasing walk with no lookahead traps itself at around length
    20 and then has no safe move at all, so growing further produces a stuck game rather than a
    longer one. `long_snapshot` covers the long-snake case instead.
    """
    game = Snake.Game(display=False)
    random.seed(seed)
    game.reset()
    walk_toward_food(game, grow_steps)
    assert not game.finished, 'the snapshot must be of a live game'
    assert len(game.snake_group) > 8, 'the walk must grow the snake or this asserts nothing'
    return game, game.snapshot(), random.getstate()


def walk_toward_food(game, steps, prefer_food=True):
    """Plays a non-fatal move each step, so a game actually grows. Returns the actions taken.

    The safety filter is not decoration: a walk that ignores it dies within ~30 steps, which leaves
    the head off the board and makes the snapshot unrestorable — so these tests would be asserting
    against a corpse. Safety comes from observation indices 6-8, the same slice the exploration
    shield masks on, rather than a fourth copy of the collision rules.

    `prefer_food=False` just takes the first safe move. That is the better walker on a nearly-full
    board, where chasing food with no lookahead seals a region within a handful of moves — measured
    at 5 steps from the length-85 fixture against 10+ for first-safe.

    Stops before dying, so the caller always holds a live game.
    """
    taken = []
    for _ in range(steps):
        if game.finished or game.current_food == 'no food':
            break
        observation = game.get_observation()
        safe = [action for index, action in enumerate(ACTIONS) if observation[6 + index] > 0.5]
        if not safe:
            break
        action = safe[0]
        if prefer_food:
            head, facing = game.head.tile_pos, game.snake.move_dir
            target = game.current_food.position
            action = next((a for a in safe if _closes_the_gap(head, facing, a, target)), safe[0])
        taken.append(action)
        if game.step(action)[0]:
            break
    return taken


def _closes_the_gap(head, facing, action, target):
    vector = MOVE_VECTORS[CURRENT_DIRECTION_MAPS[facing][action]]
    moved = (head[0] + vector[0], head[1] + vector[1])
    if not (0 <= moved[0] <= SCREENTILES[0] and 0 <= moved[1] <= SCREENTILES[1]):
        return False
    return distance_to_food(moved, target) < distance_to_food(head, target)


# ------------------------------------------------------------------ snapshot round trip

def test_snapshot_round_trips_every_field():
    # Catches any field silently dropped from the namedtuple or the restore, including
    # tail_last_move_dir — which no behavioural test can cover, because it is dead state today.
    for snapshot in (snake_walking_up_a_column(), coiled_snapshot()):
        assert restored(snapshot).snapshot() == snapshot


def test_snapshot_of_a_played_game_round_trips():
    """The same claim against a body the game itself produced, rather than a hand-built one."""
    _, snapshot, _ = played_snapshot(5, 120)
    assert restored(snapshot).snapshot() == snapshot


# ------------------------------------------------------------------ derived state

def test_every_segment_direction_points_at_the_segment_ahead():
    """The invariant `_build_snake` derives, checked on a body that turns corners.

    Observations never read a segment's move_dir, so `get_observation` cannot catch this — only
    stepping can, which is why the replay test below is the real guard. This one names the field.
    """
    snapshot = coiled_snapshot()
    game = restored(snapshot)
    segment = game.head
    for cell, ahead in zip(snapshot.body[1:], snapshot.body):
        segment = segment.behind_segment
        assert segment.tile_pos == cell
        assert segment.move_dir == Snake.direction_between(cell, ahead), (cell, ahead)


def test_the_segment_chain_is_linked_both_ways_and_ends_at_the_tail():
    snapshot = coiled_snapshot()
    game = restored(snapshot)
    assert game.snake.get_positions() == [tuple(cell) for cell in snapshot.body]
    assert game.head.front_segment is None
    assert game.tail.behind_segment is None
    assert game.tail.tile_pos == snapshot.body[-1]
    segment = game.tail
    while segment.front_segment is not None:
        assert segment.front_segment.behind_segment is segment
        segment = segment.front_segment
    assert segment is game.head


def test_every_segment_rect_matches_its_tile_pos():
    """Rects are game logic here, not just drawing.

    Both collision checks go through `pygame.sprite.groupcollide` and the out-of-bounds test reads
    `self.snake.rect.topleft`, so a body with correct tile_pos and stale rects would collide with
    itself in the wrong places.
    """
    game = restored(coiled_snapshot())
    for segment in game.snake_group:
        expected = (segment.tile_pos[0] * TILE_SIZE[0], segment.tile_pos[1] * TILE_SIZE[1])
        assert segment.rect.topleft == expected, (segment.tile_pos, segment.rect.topleft)
    assert game.current_food.rect.topleft == (0 * TILE_SIZE[0], 9 * TILE_SIZE[1])


def test_group_membership_after_restore():
    """`taken_up_group` must hold the food as well as the body, or the next Food() lands on it.

    That failure is silent and rare: food spawning under the snake's own body is legal as far as
    every other check is concerned, and it would show up as an occasional impossible episode.
    """
    snapshot = coiled_snapshot()
    game = restored(snapshot)
    assert len(game.snake_group) == len(snapshot.body)
    assert list(game.snake_head_group) == [game.head]
    assert list(game.food_group) == [game.current_food]
    assert set(game.taken_up_group) == set(game.snake_group) | {game.current_food}
    assert set(game.all) == set(game.snake_group) | {game.current_food}


def test_restored_grid_matches_the_independent_simulator():
    """Checked against `diagnostics/diag.py`'s build_grid, which was validated against the game."""
    diagnostics = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'hyperparamTuning', 'diagnostics')
    if diagnostics not in sys.path:
        sys.path.insert(0, diagnostics)
    diag = importlib.import_module('diag')

    for snapshot in (snake_walking_up_a_column(), coiled_snapshot()):
        game = restored(snapshot)
        assert np.array_equal(game.grid, diag.build_grid(list(snapshot.body), snapshot.food)), \
            snapshot.body


def test_restored_observation_matches_get_observations_on_plain_data():
    snapshot = coiled_snapshot()
    game = restored(snapshot)

    class FakeFood:
        position = snapshot.food

    expected = get_observations(game.grid, snapshot.body[0], snapshot.body[-1], snapshot.body[-2],
                                snapshot.head_move_dir, FakeFood(), snapshot.current_step,
                                snapshot.last_food_step, len(snapshot.body))
    assert list(game.get_observation()) == list(expected)


def test_a_game_with_no_food_restores_the_sentinel():
    # 'no food' is a string sentinel checked by identity-ish comparison all over the code, so a
    # restore that used None instead would raise deep inside _rebuild_grid.
    snapshot = coiled_snapshot()._replace(food=None)
    game = restored(snapshot)
    assert game.current_food == 'no food'
    assert len(game.food_group) == 0
    assert game.snapshot().food is None


# ------------------------------------------------------------------ the real guard

def test_a_restored_game_matches_the_original_step_for_step():
    """Grow a game, snapshot it, restore a copy, and replay both against the same actions.

    Copy against *original*, which is the comparison that matters — copy against copy would only
    catch non-determinism, not a restore that is consistently wrong.

    The RNG is bracketed rather than reseeded, because food placement draws from the module-global
    `random`: the copy has to start its replay from the state the original held at the snapshot, or
    the two disagree the moment either eats, which would be a property of the test rather than of
    the restore. The actions come from a throwaway scout copy so that the original is still sitting
    at the snapshot state when it is replayed.

    Verified to fail at the first step when `_build_snake` leaves segment directions at the
    SnakeSegment default.
    """
    cases = [('played seed 11', ) + played_snapshot(11, 60) + (40, True),
             ('played seed 7', ) + played_snapshot(7, 120) + (40, True)]
    # The long case has no "original" to compare against — it was never played — so the copy is
    # measured against a second independent restore of the same snapshot. That still catches a
    # wrong derivation, because a wrong one would take both copies somewhere the snapshot does not
    # describe, and the field-level tests above pin the derivation itself.
    long_snap = long_snapshot()
    cases.append(('hand-built length 85', restored(long_snap), long_snap, random.getstate(),
                  8, False))

    for label, original, snapshot, rng_state, min_actions, prefer_food in cases:
        random.setstate(rng_state)
        actions = walk_toward_food(restored(snapshot), 200, prefer_food=prefer_food)
        assert len(actions) >= min_actions, '{0}: only {1} actions'.format(label, len(actions))

        original_trace = _replay(original, actions, rng_state)
        copy_trace = _replay(restored(snapshot), actions, rng_state)
        assert len(original_trace) >= min_actions
        if original_trace != copy_trace:
            step = next(i for i, (a, b) in enumerate(zip(original_trace, copy_trace)) if a != b)
            raise AssertionError('{0} diverged at step {1}'.format(label, step))


def _replay(game, actions, rng_state):
    random.setstate(rng_state)
    trace = []
    for action in actions:
        finished, reward = game.step(action)
        trace.append((finished, reward, game.current_score, game.perfect_game, game.starved,
                      game.get_observation().tobytes()))
        if finished:
            break
    return trace


def test_the_two_games_do_not_share_state():
    """Stepping the copy must not move the original. Guards against reusing sprites or groups."""
    snapshot = coiled_snapshot()
    first, second = restored(snapshot), restored(snapshot)
    before = first.snapshot()
    second.step('forward')
    assert first.snapshot() == before
    assert second.snapshot() != before
    assert not (set(first.snake_group) & set(second.snake_group))


# ------------------------------------------------------------------ Food(position=)

def test_food_with_an_explicit_position_consumes_no_rng():
    game = restored(coiled_snapshot())
    random.seed(1)
    state = random.getstate()
    food = Snake.Food(game.taken_up_group, position=(2, 7))
    assert food.position == (2, 7)
    assert food.rect.topleft == (2 * TILE_SIZE[0], 7 * TILE_SIZE[1])
    assert random.getstate() == state, 'a restore must not perturb the food stream'


def test_food_without_a_position_still_samples():
    # The other half: a mutation that skipped the sampling loop unconditionally would pass the
    # test above and break every real episode.
    game = restored(coiled_snapshot())
    random.seed(1)
    state = random.getstate()
    food = Snake.Food(game.taken_up_group)
    assert random.getstate() != state
    assert food.position not in set(coiled_snapshot().body)


# ------------------------------------------------------------------ validation

def test_restore_rejects_a_malformed_snapshot():
    base = coiled_snapshot()
    cases = {
        'empty body': base._replace(body=()),
        # Doubling back: every consecutive pair is adjacent, so only the duplicate check can
        # reject this. A body with two identical *consecutive* cells is caught by the adjacency
        # test instead, which is why that was the wrong fixture to rely on.
        'body doubling back onto itself': base._replace(body=((3, 3), (3, 4), (3, 5), (3, 4))),
        'off the board': base._replace(body=((0, 0), (-1, 0), (-2, 0), (-3, 0))),
        'non-adjacent cells': base._replace(body=((3, 3), (3, 4), (7, 7), (7, 8))),
        'food on the body': base._replace(food=(3, 4)),
        'unknown head direction': base._replace(head_move_dir='sideways'),
    }
    # Each case is judged independently, and a non-ValueError counts as a failure rather than
    # aborting the loop. Without that, deleting the validate_snapshot() call made the first case
    # raise IndexError out of _build_snake and the remaining five never ran — the suite reported a
    # failure, but for the wrong reason and with five blind spots behind it.
    problems = []
    for label, snapshot in cases.items():
        try:
            restored(snapshot)
        except ValueError:
            continue
        except Exception as error:
            problems.append('{0}: crashed with {1} instead of ValueError'.format(
                label, type(error).__name__))
            continue
        problems.append('{0}: accepted'.format(label))
    assert not problems, problems


def test_a_dead_snakes_body_is_rejected():
    """The real body a smoke run tried to fork from, once the episode had already ended.

    Kept as its own test rather than folded into the malformed-snapshot cases because this one is
    not hypothetical — it is the state that actually reached `restore_snapshot` in a live run, and
    it is the reason the collector now refuses to fork on a boundary step.
    """
    try:
        restored(dead_snake_snapshot())
    except ValueError as error:
        assert 'already ended' in str(error), str(error)
        return
    raise AssertionError("a dead snake's overlapping body was accepted")


def test_an_off_board_death_is_rejected_too():
    # The other half of the deaths: 151 of 300 leave the head outside the board.
    snapshot = dead_snake_snapshot()._replace(body=((6, -1), (6, 0), (6, 1), (6, 2), (5, 2)))
    try:
        restored(snapshot)
    except ValueError:
        return
    raise AssertionError('a body with the head off the board was accepted')


def test_a_rejected_snapshot_leaves_the_game_untouched():
    """Validation runs before anything is mutated, so a bad restore is atomic.

    This is what the adjacency pre-check earns its place with: `_build_snake` would raise on the
    same body anyway, but only after `_new_groups()` had already thrown away the game's sprites,
    leaving a half-restored husk that no exception handler could repair. A pooled env that failed
    a restore would then be quietly broken for every later branch.
    """
    game = restored(coiled_snapshot())
    before = game.snapshot()
    try:
        game.restore_snapshot(coiled_snapshot()._replace(body=((3, 3), (9, 9), (9, 8))))
    except ValueError:
        pass
    assert game.snapshot() == before
    assert len(game.snake_group) == len(before.body)


def test_a_restored_game_can_be_drawn_and_cleared():
    """A pooled Game restored without ever being reset() still needs a background surface.

    `reset()` is what normally creates `self.bg`, and `all.clear(screen, bg)` blits through it.
    Under the dummy driver with rendering off the sprites are never drawn, so the blit is skipped
    and a missing bg goes unnoticed — until something renders. Asserted through the draw/clear
    pair rather than `bg is not None`, so it fails the way a caller would.
    """
    game = Snake.Game(display=False)
    assert game.bg is None, 'a fresh Game has no background, which is the situation under test'
    game.restore_snapshot(coiled_snapshot())
    game.all.draw(game.screen)
    game.all.clear(game.screen, game.bg)


def test_direction_between_rejects_non_adjacent_cells():
    assert Snake.direction_between((3, 3), (3, 2)) == 'up'
    assert Snake.direction_between((3, 3), (4, 3)) == 'right'
    try:
        Snake.direction_between((0, 0), (5, 5))
    except ValueError:
        return
    raise AssertionError('direction_between accepted two cells that are not adjacent')
