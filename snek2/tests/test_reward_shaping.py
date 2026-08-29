"""Tests for the food-distance shaping term and its `SNEK_FOOD_DISTANCE_REWARD` knob.

The rule is one `if` inside `Snake.step`, and its two exclusions — the step that ate, and the step
that ended the episode — were both bugs before they were exclusions. Nothing raises if either comes
back: the reward changes by 0.001 on some steps and every metric this project reads still looks
sane, which is exactly the class of defect that took a measurement to find the first time.

Runs the real `Game` because the rule reads `old_moves_to_food` from before the move and
`current_food` from after it, and a hand-built fixture that fakes that ordering would be testing
the fixture. Headless via the dummy SDL drivers, the same way the env workers run.

One mutant is deliberately not covered: `>` to `>=` is untestable, not untested. A unit move
changes exactly one coordinate by one, so Manhattan distance moves by exactly ±1 and can never be
held — the only way to end a step at the same distance is to have eaten, which the rule already
excludes. The two operators are the same function here.
"""
import importlib
import os
import random

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')


def build_game(shaping):
    """A fresh `Game` whose `FOOD_DISTANCE_REWARD` is `shaping`.

    Reloaded rather than assigned, because `Snake.py` does `from state_helpers import *` and
    `state_helpers` does `from snake_constants import *`, so both bind their own copy of the
    constant at import. Setting `snake_constants.FOOD_DISTANCE_REWARD` would leave `Snake`'s copy
    untouched — the same trap that makes the knob an environment variable rather than a `tuned()`
    call in the parent process.
    """
    previous = os.environ.get('SNEK_FOOD_DISTANCE_REWARD')
    os.environ['SNEK_FOOD_DISTANCE_REWARD'] = repr(shaping)
    try:
        import snake_constants
        import state_helpers
        import Snake
        importlib.reload(snake_constants)
        importlib.reload(state_helpers)
        importlib.reload(Snake)
        assert Snake.FOOD_DISTANCE_REWARD == shaping, Snake.FOOD_DISTANCE_REWARD
        return Snake
    finally:
        if previous is None:
            os.environ.pop('SNEK_FOOD_DISTANCE_REWARD', None)
        else:
            os.environ['SNEK_FOOD_DISTANCE_REWARD'] = previous


def restore():
    """Puts the three modules back on the default shaping, for tests that run after these."""
    import snake_constants
    import state_helpers
    import Snake
    importlib.reload(snake_constants)
    importlib.reload(state_helpers)
    importlib.reload(Snake)


def play(shaping, actions, seed=7):
    """Plays a fixed action sequence and records what each step did.

    Seeded immediately before `reset()` so both shaping settings get the same food stream —
    placement draws from the global `random` module, so the sequence is reproducible only if
    nothing else draws in between.
    """
    module = build_game(shaping)
    try:
        game = module.Game(display=False)
        random.seed(seed)
        game.reset()
        steps = []
        for action in actions:
            before = game.head.tile_pos
            food_before = game.current_food.position if game.current_food != 'no food' else None
            score_before = game.current_score
            finished, reward = game.step(action)
            steps.append({
                'reward': reward,
                'ate': game.current_score > score_before,
                'finished': finished,
                'moved_away': (food_before is not None
                               and module.distance_to_food(game.head.tile_pos, food_before)
                               > module.distance_to_food(before, food_before)),
                'head': game.head.tile_pos,
                'food': game.current_food.position if game.current_food != 'no food' else None,
            })
            if finished:
                break
        return steps
    finally:
        restore()


# ------------------------------------------------------------------------ the knob

def test_the_shaping_term_defaults_to_the_historical_value():
    # Every arm on record trained with 0.001, so the default is what keeps a rerun comparable.
    import snake_constants
    importlib.reload(snake_constants)
    assert snake_constants.FOOD_DISTANCE_REWARD == 0.001


def test_the_knob_reaches_the_module_that_consumes_it():
    # Snake.py holds its own copy from `import *`, which is the failure this asserts against.
    module = build_game(0.0)
    try:
        assert module.FOOD_DISTANCE_REWARD == 0.0
    finally:
        restore()
    module = build_game(0.05)
    try:
        assert module.FOOD_DISTANCE_REWARD == 0.05
    finally:
        restore()


# ------------------------------------------------------------------- the rule itself

def ordinary_steps(shaping, seeds=(1, 3, 7, 11, 13)):
    """Ordinary moves — neither eating nor ending the episode — pooled over several food rolls.

    Pooled because which moves increase the distance depends on where the food landed, and a
    single seed can produce a run that is all one kind. The pool keeps the assertions about the
    *rule* rather than about one placement.
    """
    steps = []
    for seed in seeds:
        for actions in (['left'] * 12, ['forward', 'forward', 'right', 'forward', 'left'] * 3):
            steps.extend(s for s in play(shaping, actions, seed=seed)
                         if not s['ate'] and not s['finished'])
    return steps


def test_a_move_away_from_food_costs_exactly_the_shaping_term():
    away = [s for s in ordinary_steps(0.001) if s['moved_away']]
    assert away, 'the fixture must contain at least one move away from food'
    for step in away:
        assert step['reward'] == -0.001, step


def test_a_move_toward_food_is_not_rewarded():
    # Deliberately a penalty for moving away rather than a bonus for approaching. If it were
    # flipped to a bonus, a policy could farm reward by oscillating next to the food. Note a move
    # that holds the distance is also unpenalised — the test is `>`, not `>=`.
    toward = [s for s in ordinary_steps(0.001) if not s['moved_away']]
    assert toward, 'the fixture must contain at least one move toward food'
    for step in toward:
        assert step['reward'] == 0.0, step


def test_turning_the_knob_off_removes_the_penalty_and_nothing_else():
    """The ablation, end to end: same game, same food, only the reward differs.

    This is the whole design of the experiment. If the two runs diverged in position or food
    placement, the comparison would be between two environments rather than two reward functions.
    """
    actions = ['left', 'forward', 'left', 'forward', 'right', 'forward', 'left', 'forward']
    on = play(0.001, actions)
    off = play(0.0, actions)

    assert len(on) == len(off)
    assert [s['head'] for s in on] == [s['head'] for s in off], 'the trajectory must be identical'
    assert [s['food'] for s in on] == [s['food'] for s in off], 'the food stream must be identical'
    assert [s['moved_away'] for s in on] == [s['moved_away'] for s in off]

    penalised = [s['moved_away'] and not s['ate'] and not s['finished'] for s in on]
    assert any(penalised), 'the fixture must have something to ablate'
    for was_penalised, with_shaping, without in zip(penalised, on, off):
        expected = 0.001 if was_penalised else 0.0
        assert round(without['reward'] - with_shaping['reward'], 9) == expected, (with_shaping,
                                                                                 without)
    assert not any(s['reward'] == -0.001 for s in off), off


def test_the_step_that_eats_is_not_penalised():
    """Eating must pay exactly FOOD_REWARD, with no shaping deducted.

    `old_moves_to_food` measures the food that was just consumed and `current_food` is already its
    replacement, so the comparison is meaningless on this step — and since the head had to be
    adjacent to eat, the old distance is always 1 and the replacement is almost never that close.
    Before the exclusion the penalty fired on 96.8% of food-eating steps, quietly making every
    FOOD_REWARD 0.999.
    """
    module = build_game(0.001)
    try:
        game = module.Game(display=False)
        random.seed(3)
        game.reset()
        # Walk the head onto the food by shortest path, re-planned each step.
        ate = False
        for _ in range(60):
            target = game.current_food.position
            head = game.head.tile_pos
            facing = game.snake.move_dir
            action = next(a for a in ('forward', 'left', 'right')
                          if _closes_the_gap(module, head, facing, a, target))
            score_before = game.current_score
            finished, reward = game.step(action)
            if game.current_score > score_before:
                assert reward == module.FOOD_REWARD, reward
                ate = True
                break
            if finished:
                break
        assert ate, 'the walk must reach the food for this test to assert anything'
    finally:
        restore()


def _closes_the_gap(module, head, facing, action, target):
    direction = module.CURRENT_DIRECTION_MAPS[facing][action]
    vector = module.MOVE_VECTORS[direction]
    moved = (head[0] + vector[0], head[1] + vector[1])
    if not (0 <= moved[0] <= module.SCREENTILES[0] and 0 <= moved[1] <= module.SCREENTILES[1]):
        return False
    return module.distance_to_food(moved, target) < module.distance_to_food(head, target)


def test_the_step_that_ends_the_episode_is_not_penalised():
    """A death pays exactly DEATH_REWARD. Off the board the distance comparison is noise.

    The seeds are pooled and the assertion needs a death that *moved away* from the food: on a
    death toward the food the exclusion makes no difference, so a single seed can pass while the
    exclusion is gone. `left` then `forward` drives straight up from the start tile into the wall.
    """
    deaths = []
    for seed in (1, 3, 7, 11, 13, 17, 19):
        steps = play(0.001, ['left'] + ['forward'] * 8, seed=seed)
        deaths.extend(s for s in steps if s['finished'])
    away_deaths = [s for s in deaths if s['moved_away']]
    assert away_deaths, 'need a death that increased the distance to food, or this asserts nothing'
    for step in away_deaths:
        assert step['reward'] == -5.0, step


# ============================================================ chase-safe potential shaping
#
# `F = c * (gamma * Phi(s') - Phi(s))`, with Phi = 1 when the head, the food and the tail share one
# open region, gated to snake lengths at or above CHASE_SAFE_GATE.
#
# **The fixtures below are coiled on purpose.** On an open board every variant of the region test
# agrees — the head, the tail and the food are all in the one region whatever the rule is — so an
# open-board fixture asserts nothing about this potential. That is the exact trap `../CLAUDE.md`
# records for `group_obs`, where 24 tests passed across a signature change because every board was
# open. `SEALED_POCKET` and `TAIL_UNSEALS` both have a one-cell region that only the intersection
# test gets right.

GAMMA = 0.9975

# Food at (9,9), whose only two neighbours (9,8) and (8,9) are both body, so it is a sealed one-cell
# region. The sealing triple sits in the *middle* of the snake — the only route between (9,8) and
# (8,9) that avoids the food is through (8,8) — so neither end touches the pocket, and the cell the
# tail vacates is in the main region. Phi = 0, and it stays 0 across a move.
SEALED_POCKET = ((9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (8, 8), (8, 9), (7, 9), (6, 9), (5, 9))
SEALED_FOOD = (9, 9)
SEALED_FACING = 'up'          # (9,4) was entered from (9,5), so forward is (9,3)

# The same pocket, but now (9,8) is the *tail*. It is adjacent to the pocket, the head is not, so
# the intersection excludes the pocket and Phi = 0 — and one forward move vacates (9,8), which
# joins the pocket to the main region and flips Phi to 1. The 0->1 case with no hand-set state.
TAIL_UNSEALS = ((5, 9), (6, 9), (7, 9), (8, 9), (8, 8), (9, 8))
TAIL_UNSEALS_FACING = 'left'  # (5,9) was entered from (6,9), so forward is (4,9)

OPEN_BOARD = ((5, 5), (4, 5), (3, 5))
OPEN_FOOD = (8, 8)
OPEN_FACING = 'right'

# **The fixture that discriminates the intersection from the head's regions alone**, which the two
# above do not: a full row of body seals row 0 off from the rest, and the head at (0,1) sits on the
# junction so it neighbours *both* regions while the tail at (9,5) neighbours only the lower one.
# With food in row 0 the correct answer is 0 — the head can reach that food and then has no route
# back to its tail, which is the entire point of the flag. Testing the food against the head's
# regions alone returns 1 here.
#
# Added after a mutation pass: swapping `head_groups & tail_groups` for `head_groups` alone was
# caught by **no test** against the sealed-pocket fixtures, because in those the head does not
# neighbour the pocket either and both rules agree.
HEAD_ONLY_TRAP = ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1),
                  (9, 2), (9, 3), (9, 4), (9, 5))
HEAD_ONLY_TRAP_FOOD = (5, 0)      # in the sealed top row: reachable by the head, no way back
HEAD_ONLY_TRAP_SAFE_FOOD = (5, 5)  # in the lower region, which both head and tail touch
HEAD_ONLY_TRAP_FACING = 'left'

# A snake coiled clockwise all the way around a single food cell, head and tail meeting at the corner.
# Every neighbour of the food is body, so the food is a sealed one-cell region the head sits *directly*
# on: the head can eat it, but doing so walls the head in with its own body and it dies the next step
# (this is not the winning move — the board is far from full). The tail is enclosed in the corner too,
# so it neighbours no open region: `tail_groups` is empty and the head/tail intersection is empty, and
# Phi = 0. Picture, y increasing downward, each arrow pointing at the next body segment toward the head:
#
#     T H <
#     D F U
#     > > U
#
# Head (1,0), tail (0,0), food (1,1). The head is adjacent to both the food and the tail (the tail is
# only diagonal to the food), so it is the "everything is touching yet eating is fatal" case. Testing
# the food against the head's regions alone returns 1 here, so it also pins the intersection rule.
COILED_AROUND_FOOD = ((1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1), (0, 0))
COILED_AROUND_FOOD_FOOD = (1, 1)
COILED_AROUND_FOOD_FACING = 'left'  # the neck (2,0) is right of the head, so it entered moving left


def build_chase_game(c, gate=0, distance_shaping=0.0, free_space=0.0, free_space_gate=0):
    """A fresh `Snake` module whose chase-safe knobs are `c` and `gate`.

    Reloaded rather than assigned, for the reason `build_game` documents: `Snake.py` does
    `from state_helpers import *` and `state_helpers` does `from snake_constants import *`, so both
    bind their own copy of each constant at import.

    The distance shaping is forced to 0.0 as well, so a reward is the chase-safe term alone. That
    matches every arm from batch 17 on, and leaving it at its 0.001 default would put an unrelated
    penalty on some of the moves these tests assert exact values for.

    `free_space` / `free_space_gate` drive the second PBRS term (`FREE_SPACE_SHAPING`), both 0/off by
    default so existing callers are unchanged and a reward stays the chase-safe term alone unless a
    test asks otherwise.
    """
    previous = {name: os.environ.get(name) for name in
                ('SNEK_CHASE_SAFE_SHAPING', 'SNEK_CHASE_SAFE_GATE', 'SNEK_FOOD_DISTANCE_REWARD',
                 'SNEK_FREE_SPACE_SHAPING', 'SNEK_FREE_SPACE_GATE')}
    os.environ['SNEK_CHASE_SAFE_SHAPING'] = repr(c)
    os.environ['SNEK_CHASE_SAFE_GATE'] = str(gate)
    os.environ['SNEK_FOOD_DISTANCE_REWARD'] = repr(distance_shaping)
    os.environ['SNEK_FREE_SPACE_SHAPING'] = repr(free_space)
    os.environ['SNEK_FREE_SPACE_GATE'] = str(free_space_gate)
    try:
        import snake_constants
        import state_helpers
        import Snake
        importlib.reload(snake_constants)
        importlib.reload(state_helpers)
        importlib.reload(Snake)
        assert Snake.CHASE_SAFE_SHAPING == c, Snake.CHASE_SAFE_SHAPING
        assert Snake.CHASE_SAFE_GATE == gate, Snake.CHASE_SAFE_GATE
        assert Snake.FREE_SPACE_SHAPING == free_space, Snake.FREE_SPACE_SHAPING
        assert Snake.FREE_SPACE_GATE == free_space_gate, Snake.FREE_SPACE_GATE
        return Snake
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def restored(module, body, food, facing, discount=GAMMA, score=0, step=0):
    """A `Game` holding a hand-built board, with the potential set from it by `restore_snapshot`."""
    game = module.Game(display=False, discount=discount)
    snapshot = module.GameSnapshot(
        body=tuple(body), head_move_dir=facing, tail_last_move_dir=facing, food=food,
        current_score=score, current_step=step, last_food_step=step,
        finished=False, starved=False, perfect_game=False)
    game.restore_snapshot(snapshot)
    return game


# -------------------------------------------------------------------------- the knobs

def test_chase_safe_shaping_defaults_to_off():
    """0.0, so every existing arm and every historical number is unaffected by this feature."""
    import snake_constants
    importlib.reload(snake_constants)
    assert snake_constants.CHASE_SAFE_SHAPING == 0.0
    assert snake_constants.DEFAULT_CHASE_SAFE_SHAPING == 0.0
    # The gate default is the variant Phase 0 selected, so turning the term on gets the validated
    # form rather than the ungated one.
    assert snake_constants.CHASE_SAFE_GATE == 85


def test_both_chase_safe_knobs_reach_the_module_that_consumes_them():
    # Snake.py holds its own copy from `import *`, which is the failure this asserts against.
    module = build_chase_game(0.1, gate=75)
    try:
        assert module.CHASE_SAFE_SHAPING == 0.1
        assert module.CHASE_SAFE_GATE == 75
    finally:
        restore()


# ------------------------------------------------------------------- the potential itself

def test_the_potential_is_zero_when_the_food_is_sealed_off_from_the_tail():
    """The intersection is what makes this 0: the head reaches the main region, the tail reaches
    both, and the food is in neither's shared one. Testing the food against the head's regions
    alone would call this 1 — the trap `group_obs`' comment describes."""
    module = build_chase_game(0.1, gate=0)
    try:
        game = restored(module, SEALED_POCKET, SEALED_FOOD, SEALED_FACING)
        assert game._chase_safe_potential() == 0.0
        # The same board with the food out in the open is 1, so the 0 above is about the pocket and
        # not about the fixture being broken.
        game = restored(module, SEALED_POCKET, (0, 0), SEALED_FACING)
        assert game._chase_safe_potential() == 1.0
    finally:
        restore()


def test_the_potential_tests_the_food_against_the_intersection_not_the_head_alone():
    """The head neighbours two regions at once here. Food in the one the tail cannot reach is a
    trap, not a meal — reaching it seals the snake in with it — so the correct answer is 0.

    This is the assertion the sealed-pocket fixtures cannot make, since there the head does not
    neighbour the pocket either and both rules agree. Verified by mutation: `head_groups` alone
    fails this and nothing else.
    """
    module = build_chase_game(0.1, gate=0)
    try:
        game = restored(module, HEAD_ONLY_TRAP, HEAD_ONLY_TRAP_FOOD, HEAD_ONLY_TRAP_FACING)
        assert game._chase_safe_potential() == 0.0, 'food the tail cannot reach is not chase-safe'
        # The same board with the food in the region both ends share is 1, so the 0 is about the
        # intersection and not about the fixture being unreachable in general.
        game = restored(module, HEAD_ONLY_TRAP, HEAD_ONLY_TRAP_SAFE_FOOD, HEAD_ONLY_TRAP_FACING)
        assert game._chase_safe_potential() == 1.0
    finally:
        restore()


def test_the_potential_is_zero_when_the_snake_is_coiled_around_the_food():
    """A snake wrapped all the way around the food: the head sits right on it and *can* eat, but the
    food is a sealed one-cell region and eating walls the head in — a death, not a meal — so Phi = 0.

    Distinct from the fixtures above, and worth its own. The head neighbours the food's region
    *directly* (in `HEAD_ONLY_TRAP` it neighbours a larger sealed region; in `SEALED_POCKET` it
    neighbours neither), and the intersection is empty because the corner-enclosed tail borders no
    open region at all — the `if not escape` path, which the other fixtures never take, since there
    both ends border something and the food simply sits outside the shared region. `head_groups`
    alone would call this 1, because the head is on the food, so the 0 pins the intersection.
    """
    module = build_chase_game(0.1, gate=0)
    try:
        game = restored(module, COILED_AROUND_FOOD, COILED_AROUND_FOOD_FOOD,
                        COILED_AROUND_FOOD_FACING)
        assert game._chase_safe_potential() == 0.0, 'eating a food the coil has sealed is not safe'
    finally:
        restore()


def test_the_chase_safe_observation_is_zero_on_the_eating_move_of_the_coil():
    """The policy-facing counterpart of the test above, on the same coil: obs[15-17] is the "safe to
    chase the food" per-action block, and the move that eats must read 0 even though it is legal.

    The eating action here — `left`, which turns the head down onto the food at (1,1) — is a legal
    move (obs[6] = 1, the head can take it) and it eats. But `group_obs`' eating branch sets the flag
    to whether the tail is still reachable *after* the meal, and it is not: the head lands walled in
    by its own body, so `can reach tail` (obs[9]) is 0 and the safe-to-chase flag follows it to 0.
    The policy is told "this meal traps you", not "go eat". The other two moves read 0 as well — one
    is a wall, the other follows the vacating tail and cannot reach the food — and index 29 confirms
    the food is sealed with no open neighbour.

    The obs[6] = 1 assertion is what makes this bite: the 0 is not the trivial 0 of an unreachable
    food, it is the flag correctly calling a reachable meal a trap.
    """
    module = build_chase_game(0.1, gate=0)
    try:
        game = restored(module, COILED_AROUND_FOOD, COILED_AROUND_FOOD_FOOD,
                        COILED_AROUND_FOOD_FACING)
        obs = game.get_observation()
        eat = module.ACTIONS.index('left')   # 'left' from a left-facing head turns down onto (1,1)
        assert float(obs[6 + eat]) == 1.0, 'the eating move must be legal, or the 0 below is trivial'
        assert float(obs[9 + 2 * eat]) == 0.0, 'after eating, the head cannot reach its own tail'
        assert float(obs[15 + eat]) == 0.0, 'so chasing this food is not safe'
        assert [float(obs[15 + i]) for i in range(3)] == [0.0, 0.0, 0.0], 'no move here is a safe chase'
        assert float(obs[29]) == 0.0, 'the food is sealed in, with no open neighbour'
    finally:
        restore()


def test_the_gate_zeroes_the_potential_below_the_length_threshold():
    """Below the gate the potential is 0 whatever the board says, which is what confines the term
    to the endgame. Above it, the same board reads its true chase-safety."""
    module = build_chase_game(0.1, gate=len(OPEN_BOARD) + 1)
    try:
        game = restored(module, OPEN_BOARD, OPEN_FOOD, OPEN_FACING)
        assert game._chase_safe_potential() == 0.0, 'a snake shorter than the gate must read 0'
    finally:
        restore()
    module = build_chase_game(0.1, gate=len(OPEN_BOARD))
    try:
        game = restored(module, OPEN_BOARD, OPEN_FOOD, OPEN_FACING)
        assert game._chase_safe_potential() == 1.0, 'at the gate the board decides'
    finally:
        restore()


def test_the_potential_is_zero_whenever_the_term_is_off():
    """So `c = 0` skips the flood fill entirely rather than computing a value nobody adds."""
    module = build_chase_game(0.0, gate=0)
    try:
        game = restored(module, OPEN_BOARD, OPEN_FOOD, OPEN_FACING)
        assert game._chase_safe_potential() == 0.0
    finally:
        restore()


# ------------------------------------------------------------------------- the arithmetic

def test_the_potential_going_up_pays_exactly_c_times_gamma():
    """0 -> 1 on a real move, engineered rather than hand-set: the tail vacates (9,8), which joins
    the sealed pocket to the main region. Two-sided, unlike the distance term it replaces."""
    module = build_chase_game(0.1, gate=0)
    try:
        game = restored(module, TAIL_UNSEALS, SEALED_FOOD, TAIL_UNSEALS_FACING)
        assert game.chase_safe_potential == 0.0, 'fixture must start chase-unsafe'
        finished, reward = game.step('forward')
        assert not finished, 'the fixture move must be legal, or this tests a death'
        assert game.chase_safe_potential == 1.0, 'the tail should have opened the pocket'
        assert abs(reward - 0.1 * GAMMA) < 1e-12, reward
    finally:
        restore()


def test_the_potential_going_down_costs_exactly_c():
    """1 -> 0 costs -c, with no gamma on it, because gamma multiplies only the *new* potential.

    The starting potential is set by hand here. Every legal move on this board leaves Phi at 0 (the
    pocket's seal is mid-body and the vacated cell is in the main region), so a 1 -> 0 transition
    cannot be produced from a restore alone — the snake opens space as it moves.
    """
    module = build_chase_game(0.1, gate=0)
    try:
        game = restored(module, SEALED_POCKET, SEALED_FOOD, SEALED_FACING)
        game.chase_safe_potential = 1.0
        finished, reward = game.step('forward')
        assert not finished
        assert game.chase_safe_potential == 0.0
        assert abs(reward - (-0.1)) < 1e-12, reward
    finally:
        restore()


def test_a_held_potential_costs_exactly_c_times_gamma_minus_one():
    """The discounting of a constant potential, and the test that gamma is wired at all: at gamma
    = 1.0 this would be exactly 0.0 and the term would be a per-step bonus for being chase-safe."""
    module = build_chase_game(0.1, gate=0)
    try:
        game = restored(module, OPEN_BOARD, OPEN_FOOD, OPEN_FACING)
        assert game.chase_safe_potential == 1.0
        finished, reward = game.step('forward')
        assert not finished
        assert game.chase_safe_potential == 1.0, 'the fixture must hold the potential'
        assert abs(reward - 0.1 * (GAMMA - 1.0)) < 1e-12, reward
    finally:
        restore()


def test_the_discount_is_threaded_and_not_defaulted():
    """`SnakeEnvironment` owns the true gamma; a Game built without it would silently use 1.0."""
    import snake_environment
    importlib.reload(snake_environment)
    env = snake_environment.SnakeEnvironment(discount=0.9975, display=False, policy_name='smoke')
    assert env._game.shaping_discount == 0.9975


# --------------------------------------------------------------- the ending, and the ablation

def test_the_potential_is_zero_at_a_death_so_the_last_step_pays_minus_c():
    """`Phi(terminal) = 0` is what the invariance requires, and the same branch keeps the potential
    off an unusable grid — on a death the head is off the board. A starvation and a perfect game
    take this identical branch, since both set `self.finished` before the term runs."""
    module = build_chase_game(0.1, gate=0)
    try:
        game = restored(module, OPEN_BOARD, OPEN_FOOD, OPEN_FACING)
        assert game.chase_safe_potential == 1.0
        # One left turn from facing right points the head up, then straight on into the wall at
        # y < 0. Six moves from (5,5); the loop is longer so the fixture is not one step from
        # silently asserting nothing if the opening tile ever moves.
        finished = False
        for index in range(9):
            finished, reward = game.step('left' if index == 0 else 'forward')
            if finished:
                break
        assert finished, 'the walk must reach the wall'
        assert game.chase_safe_potential == 0.0
        # The death pays DEATH_REWARD plus -c times whatever the potential was, which on this board
        # is 1.0 for every step up until the wall.
        assert abs(reward - (-5.0 - 0.1)) < 1e-12, reward
    finally:
        restore()


def test_a_starvation_also_ends_with_a_zero_potential():
    module = build_chase_game(0.1, gate=0)
    try:
        game = restored(module, OPEN_BOARD, OPEN_FOOD, OPEN_FACING)
        # One step from starving: the budget is spent, so the next step ends the episode.
        game.last_food_step = game.current_step - module.starve_budget(len(game.snake_group))
        finished, reward = game.step('forward')
        assert finished and game.starved, (finished, game.starved)
        assert game.chase_safe_potential == 0.0
        assert abs(reward - (-0.5 - 0.1)) < 1e-12, reward
    finally:
        restore()


def test_turning_the_chase_safe_knob_off_changes_no_reward_and_no_food_position():
    """The regression that protects every existing arm. `count_groups` draws no randomness, so
    skipping the term cannot shift the food stream — this asserts that rather than assuming it."""
    actions = ['forward', 'left', 'forward', 'right', 'forward'] * 4
    for seed in (1, 5, 9):
        off = play_chase(0.0, actions, seed=seed)
        on_but_zero = play_chase(0.0, actions, seed=seed, gate=0)
        assert off == on_but_zero, seed
        shaped = play_chase(0.1, actions, seed=seed, gate=0)
        assert [s['food'] for s in shaped] == [s['food'] for s in off], (
            'the shaped run must see the same food stream, or nothing below is comparable')


def play_chase(c, actions, seed=7, gate=0, discount=GAMMA):
    """Plays a fixed action sequence from a fresh episode under the chase-safe knobs."""
    module = build_chase_game(c, gate=gate)
    try:
        game = module.Game(display=False, discount=discount)
        random.seed(seed)
        game.reset()
        steps = []
        for action in actions:
            finished, reward = game.step(action)
            steps.append({'reward': round(reward, 12), 'finished': finished,
                          'head': game.head.tile_pos,
                          'food': (game.current_food.position
                                   if game.current_food != 'no food' else None)})
            if finished:
                break
        return steps
    finally:
        restore()


def test_the_shaped_rewards_telescope_to_minus_c_times_the_opening_potential():
    """The invariance property, end to end and on a real episode.

    Differencing a shaped run against an unshaped one on the same food stream isolates `F` exactly,
    and the discounted sum of `F` has to come out at `-c * Phi(s0)` — a constant that depends on the
    opening board and *not* on the policy, which is the whole reason this form is safe. A sign
    error, a missing gamma or a non-zero terminal potential each break it.
    """
    actions = ['forward', 'left', 'forward', 'right', 'forward'] * 6
    for seed in (1, 5, 9):
        plain = play_chase(0.0, actions, seed=seed)
        shaped = play_chase(0.1, actions, seed=seed, gate=0)
        assert len(plain) == len(shaped), 'the two runs must follow the same trajectory'
        telescope = sum((GAMMA ** index) * (shaped[index]['reward'] - plain[index]['reward'])
                        for index in range(len(shaped)))
        # Phi(s0) on the opening board: the snake starts at length 5 in open space, so 1 unless the
        # food happened to spawn sealed, which cannot happen on an opening board.
        assert abs(telescope - (-0.1 * 1.0)) < 1e-9, (seed, telescope)


# ------------------------------------------------------------------ the fork, and the snapshot

def test_restoring_a_snapshot_rebuilds_the_potential_from_the_restored_board():
    """`GameSnapshot` does not carry the potential, and `ForkingCollector` reuses a pool of envs, so
    without this a branch computes its first F against the value left by the previous branch."""
    module = build_chase_game(0.1, gate=0)
    try:
        game = restored(module, OPEN_BOARD, OPEN_FOOD, OPEN_FACING)
        assert game.chase_safe_potential == 1.0
        # Deliberately wrong before the restore, which is exactly the pooled-env situation.
        game.chase_safe_potential = 1.0
        snapshot = module.GameSnapshot(
            body=SEALED_POCKET, head_move_dir=SEALED_FACING, tail_last_move_dir=SEALED_FACING,
            food=SEALED_FOOD, current_score=0, current_step=0, last_food_step=0,
            finished=False, starved=False, perfect_game=False)
        game.restore_snapshot(snapshot)
        assert game.chase_safe_potential == 0.0, 'the stale 1.0 survived the restore'
    finally:
        restore()


def test_a_restored_branch_pays_the_same_shaped_reward_as_its_parent_would():
    """The same gap, end to end on the path the collector takes: fork, then step, and the branch's
    first shaped reward has to match the parent's."""
    module = build_chase_game(0.1, gate=0)
    try:
        parent = restored(module, TAIL_UNSEALS, SEALED_FOOD, TAIL_UNSEALS_FACING)
        snapshot = parent.snapshot()
        _, parent_reward = parent.step('forward')

        branch = module.Game(display=False, discount=GAMMA)
        # A pooled env that has already played something else, so its potential is whatever that
        # episode left behind.
        random.seed(3)
        branch.reset()
        branch.step('forward')
        branch.chase_safe_potential = 1.0
        branch.restore_snapshot(snapshot)
        _, branch_reward = branch.step('forward')
        assert abs(branch_reward - parent_reward) < 1e-12, (parent_reward, branch_reward)
    finally:
        restore()


def test_the_state_potential_agrees_with_the_observation_the_policy_reads():
    """`chase_safe_state` on the post-move board must mean the same thing as `obs[15 + a]`, which is
    what the policy actually sees. Asserted on a coiled fixture: on an open board the two agree
    whatever the rule is, so this only says anything because the pocket is here.

    Excludes nothing, because the fixture move neither eats nor follows the tail — the two
    documented cases where `group_obs` deliberately answers a different question.
    """
    module = build_chase_game(0.1, gate=0)
    try:
        game = restored(module, TAIL_UNSEALS, SEALED_FOOD, TAIL_UNSEALS_FACING)
        observation = game.get_observation()
        # 15-17 is the safe-to-chase block, ordered left, right, forward.
        forward_flag = float(observation[15 + module.ACTIONS.index('forward')])
        game.step('forward')
        assert game.chase_safe_potential == forward_flag, (forward_flag,
                                                           game.chase_safe_potential)
        assert forward_flag == 1.0, 'the fixture must exercise a 1, or the equality is trivial'
    finally:
        restore()


# ------------------------------------------------- the free-space potential (1 / open-region count)

def test_free_space_shaping_defaults_to_off():
    """0.0, so every existing arm and historical number is unaffected, and the gate default matches
    CHASE_SAFE_GATE so turning the term on with no gate override lands on the same endgame."""
    import snake_constants
    importlib.reload(snake_constants)
    assert snake_constants.FREE_SPACE_SHAPING == 0.0
    assert snake_constants.DEFAULT_FREE_SPACE_SHAPING == 0.0
    assert snake_constants.FREE_SPACE_GATE == 85


def test_the_free_space_potential_is_one_over_the_number_of_pieces():
    """1 on a single open piece, 1/3 on the coil (a sealed food pocket, the freed corner cell and
    the main board), so the value cliffs on fragmentation rather than tracking the stranded *size*."""
    module = build_chase_game(0.0, gate=0, free_space=0.1, free_space_gate=0)
    try:
        game = restored(module, OPEN_BOARD, OPEN_FOOD, OPEN_FACING)
        assert game._free_space_potential() == 1.0, 'one open region must read 1.0'
        game = restored(module, COILED_AROUND_FOOD, COILED_AROUND_FOOD_FOOD, COILED_AROUND_FOOD_FACING)
        assert abs(game._free_space_potential() - 1.0 / 3.0) < 1e-12, 'three regions must read 1/3'
    finally:
        restore()


def test_the_free_space_potential_frees_the_tail_before_counting():
    """The tail-aware decision, and the one a mutant is most likely to drop. On `TAIL_UNSEALS` the
    tail (9,8) is the only wall between the sealed food and the main region, so freeing it merges
    them and the board is one piece -> 1.0. Counting the tail as a wall would read 2 regions -> 0.5.

    Contrast with `SEALED_POCKET`, where the seal is mid-body and freeing the tail cannot help: it
    stays 2 regions -> 0.5. So the 1.0 above is specifically the freed tail, not the fixture.
    """
    module = build_chase_game(0.0, gate=0, free_space=0.1, free_space_gate=0)
    try:
        game = restored(module, TAIL_UNSEALS, SEALED_FOOD, TAIL_UNSEALS_FACING)
        assert game._free_space_potential() == 1.0, 'freeing the tail must merge the pocket'
        game = restored(module, SEALED_POCKET, SEALED_FOOD, SEALED_FACING)
        assert game._free_space_potential() == 0.5, 'a mid-body seal the tail cannot open stays split'
    finally:
        restore()


def test_the_free_space_gate_zeroes_the_potential_below_the_length_threshold():
    """Below the gate the potential is 0 whatever the board says; at the gate the board decides."""
    module = build_chase_game(0.0, gate=0, free_space=0.1, free_space_gate=len(OPEN_BOARD) + 1)
    try:
        game = restored(module, OPEN_BOARD, OPEN_FOOD, OPEN_FACING)
        assert game._free_space_potential() == 0.0, 'a snake shorter than the gate must read 0'
    finally:
        restore()
    module = build_chase_game(0.0, gate=0, free_space=0.1, free_space_gate=len(OPEN_BOARD))
    try:
        game = restored(module, OPEN_BOARD, OPEN_FOOD, OPEN_FACING)
        assert game._free_space_potential() == 1.0, 'at the gate the board decides'
    finally:
        restore()


def test_the_free_space_potential_is_zero_whenever_the_term_is_off():
    """So `c = 0` skips the flood fill entirely rather than computing a value nobody adds."""
    module = build_chase_game(0.0, gate=0, free_space=0.0, free_space_gate=0)
    try:
        game = restored(module, COILED_AROUND_FOOD, COILED_AROUND_FOOD_FOOD, COILED_AROUND_FOOD_FACING)
        assert game._free_space_potential() == 0.0
    finally:
        restore()


def test_a_held_free_space_potential_costs_exactly_c_times_gamma_minus_one():
    """Chase-safe off, free-space on: a forward move on the open board keeps the board one piece, so
    the only reward is the free-space term at a held potential, c*(gamma - 1). Pins that the term is
    wired into the reward at all, and — with chase-safe off — that it is wired *independently*."""
    module = build_chase_game(0.0, gate=0, free_space=0.1, free_space_gate=0)
    try:
        game = restored(module, OPEN_BOARD, OPEN_FOOD, OPEN_FACING)
        assert game.free_space_potential == 1.0
        finished, reward = game.step('forward')
        assert not finished
        assert game.free_space_potential == 1.0, 'the fixture must hold the potential'
        assert abs(reward - 0.1 * (GAMMA - 1.0)) < 1e-12, reward
    finally:
        restore()


def test_the_two_shaping_terms_add():
    """The whole basis of running both at once: PBRS terms sum, so with both on and each held at
    1.0 across a move the reward is the sum, 2 * c * (gamma - 1). A mutant that dropped either term,
    or shared one potential cache between them, fails this."""
    module = build_chase_game(0.1, gate=0, free_space=0.1, free_space_gate=0)
    try:
        game = restored(module, OPEN_BOARD, OPEN_FOOD, OPEN_FACING)
        assert game.chase_safe_potential == 1.0 and game.free_space_potential == 1.0
        finished, reward = game.step('forward')
        assert not finished
        assert abs(reward - 2 * 0.1 * (GAMMA - 1.0)) < 1e-12, reward
    finally:
        restore()
