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
