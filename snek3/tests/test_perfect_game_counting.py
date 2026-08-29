"""How a finished episode is identified as a perfect game — and why it may not be the reward.

The whole suite exists because of one defect. Every perfect-game counter used to ask
`final_reward == PERFECT_GAME_REWARD`, an exact float comparison. `CHASE_SAFE_SHAPING` then added
`-c * Phi(s)` to the winning step (the theory requires `Phi(terminal) = 0`), so a perfect game pays
99.9 instead of 100 whenever the pre-win board was chase-safe — which is the tail-chasing endgame a
competent policy plays. Three counters went silent at once: the training self-eval, the independent
eval worker and the batched close-out. Batch 27 on the desktop and batch 30 on the laptop both
reported **0% perfect games across 300k+ steps** while their own `max_score` field recorded filled
boards from step 9k, and because `training.epsilon_for` takes the trailing perfect rate as its skill
signal, epsilon sat pinned at its refinement ceiling of 0.0125 instead of annealing to the floor.
The runs were not merely mismeasured, they were handicapped.

So the counters read the score, `state_helpers.is_perfect_score` is the single definition, and the
tests below pin all three parts: the predicate, the reward that proves why the old test cannot work,
and the counter that consumes it. `test_no_perfect_game_counter_compares_a_reward` is a source-level
tripwire — the two eval paths do their counting inside a spawned worker and a batched step loop that
a unit test cannot reach, and a grep is a great deal better than nothing there.
"""
import ast
import glob
import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import numpy as np

from env import constants as snake_constants
from env.observations import is_perfect_score
from test_reward_shaping import build_chase_game, restore, restored


# ------------------------------------------------------------------ the predicate

def test_only_a_filled_board_is_perfect():
    perfect = int(snake_constants.MAX_POSSIBLE_SCORE)
    assert is_perfect_score(perfect)
    assert not is_perfect_score(perfect - 1)
    assert not is_perfect_score(0)


def test_a_float_score_from_a_numpy_array_still_reads_as_perfect():
    """The training self-eval's scores arrive as float32 out of a numpy array, so the predicate
    has to survive the round trip. `95.0 == 95` holds in Python, but a float32 95 that has been
    through `.tolist()` is the kind of value an `is` or a str comparison would silently drop."""
    scores = np.zeros(2, dtype=np.float32)
    scores[0] = snake_constants.MAX_POSSIBLE_SCORE
    scores[1] = snake_constants.MAX_POSSIBLE_SCORE - 1
    assert [is_perfect_score(value) for value in scores.tolist()] == [True, False]


def test_the_predicate_and_the_game_rule_are_the_same_test():
    """`Snake.check_perfect_game` is the rule that ends the episode. If the two ever disagree, an
    episode can end as a win that no counter counts, which is exactly the failure being fixed."""
    module = build_chase_game(0.0)
    try:
        game = module.Game(display=False)
        game.reset()
        for score in (0, 1, int(snake_constants.MAX_POSSIBLE_SCORE) - 1,
                      int(snake_constants.MAX_POSSIBLE_SCORE)):
            game.current_score = score
            assert game.check_perfect_game() == is_perfect_score(score), score
    finally:
        restore()


# ------------------------------------------------------- the reward at the winning step

def hamiltonian_cycle():
    """Cell order of a closed tour of the whole board, so cell N is adjacent to cell 0.

    Top row left to right, then a serpentine over the remaining rows restricted to columns 1..N,
    then back up column 0. That closes only when the serpentine's last row runs right to left,
    i.e. when the row count is even — asserted rather than assumed, because a board resize would
    otherwise leave the tour open and the fixture would test a board one cell short of full.
    """
    cols, rows = snake_constants.SCREENTILES[0] + 1, snake_constants.SCREENTILES[1] + 1
    assert rows % 2 == 0, 'the tour construction needs an even row count'
    cells = [(x, 0) for x in range(cols)]
    for y in range(1, rows):
        xs = range(cols - 1, 0, -1) if y % 2 else range(1, cols)
        cells += [(x, y) for x in xs]
    cells += [(0, y) for y in range(rows - 1, 0, -1)]
    assert len(cells) == cols * rows == int(snake_constants.PERFECT_SCORE), len(cells)
    return cells


def win_from_a_full_tour(c, gate=85):
    """Steps into the last free cell from a board that is one food short of perfect.

    The snake lies along the tour, so the single free cell is both the head's next cell and the
    tail's neighbour — the tail-chasing endgame, and the board on which Phi is 1. Returns the
    game and the winning step's reward.
    """
    module = build_chase_game(c, gate=gate)
    cells = hamiltonian_cycle()
    body, free = cells[:-1], cells[-1]
    game = restored(module, body, free, module.direction_between(body[1], body[0]),
                    score=int(snake_constants.MAX_POSSIBLE_SCORE) - 1, step=10)
    potential = game.chase_safe_potential
    finished, reward = game.step('left')
    assert finished and game.perfect_game, (finished, game.perfect_game)
    assert is_perfect_score(game.current_score), game.current_score
    return module, potential, reward


def test_a_perfect_game_pays_exactly_the_perfect_reward_when_nothing_shapes_it():
    """Why the old reward test ever worked, and the ground truth the shaped case is measured
    against: with no shaping term the winning step really does pay exactly PERFECT_GAME_REWARD."""
    try:
        module, potential, reward = win_from_a_full_tour(0.0)
        assert potential == 0.0, potential
        assert reward == snake_constants.PERFECT_GAME_REWARD, reward
    finally:
        restore()


def test_the_shaped_winning_step_pays_less_than_the_perfect_reward():
    """The defect itself, at the one step where it matters.

    The pre-win board is chase-safe, so `Phi(s)` is 1 and the terminal `Phi(s')` is 0 — the winning
    step pays `PERFECT_GAME_REWARD - c`. Anything comparing that with `==` counts zero perfect
    games. Asserted at the calibrated `c = 0.10` and the shipped gate of 85, so the numbers are the
    ones batches 27 and 30 actually ran.
    """
    try:
        module, potential, reward = win_from_a_full_tour(0.10)
        assert potential == 1.0, 'the tail-chasing endgame must be chase-safe'
        assert abs(reward - (snake_constants.PERFECT_GAME_REWARD - 0.10)) < 1e-9, reward
        assert reward != snake_constants.PERFECT_GAME_REWARD
    finally:
        restore()


def test_the_gate_does_not_save_the_reward_test_either():
    """A tempting reading of the fix is "the gate keeps the term away from the endgame". It does
    not: the gate is a *minimum* length, and a perfect game is the longest the snake ever gets, so
    every gate at or below the winning length lets the term fire on the winning step."""
    try:
        for gate in (0, 85, int(snake_constants.MAX_POSSIBLE_SCORE)):
            module, potential, reward = win_from_a_full_tour(0.10, gate=gate)
            assert potential == 1.0, gate
            assert reward != snake_constants.PERFECT_GAME_REWARD, gate
    finally:
        restore()


# ------------------------------------------------------------------ the tripwire

def test_no_perfect_game_counter_compares_a_reward():
    """Nothing outside `env/game.py` may mention PERFECT_GAME_REWARD in a comparison.

    `env/game.py` assigns it and is exempt. The counters are the point of the rule, and some of them
    live where a unit test cannot reach — inside a shard process, inside a step loop over live
    environments — so this reads the source instead.

    Parsed rather than grepped. A line-based version tripped on its own explanatory prose the first
    time it ran: these modules discuss the reward at length in comments and docstrings, and a rule
    that fires on documentation would be turned off within a week. `ast` sees only `==` and `!=`
    between real expressions.

    **The module list is a glob, not a list, and that has already earned itself.** In snek2 it was
    four hardcoded filenames, so `vec_engine.py` — a fourth counter, banking a perfect flag per
    episode — was outside the rule the day it was written, and a mutation confirmed the reward
    version of that line survived the whole suite. So this scans every module in every package that
    could hold a counter and exempts only `env/game.py`, which *assigns* the reward.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    packages = ('env', 'vectorized', 'dqn', 'ppo', 'tools')
    exempt = {os.path.join('env', 'game.py')}
    paths = sorted(glob.glob(os.path.join(here, '*.py')))
    for package in packages:
        paths += sorted(glob.glob(os.path.join(here, package, '*.py')))
    paths = [p for p in paths if os.path.relpath(p, here) not in exempt]

    # Two floors, because they fail on different mistakes. The named files catch a scan that ran
    # from the wrong directory or lost a package from the tuple — a count alone would pass on
    # whatever it happened to find. The count catches a package that silently stopped matching.
    found = {os.path.relpath(p, here) for p in paths}
    for required in (os.path.join('env', 'observations.py'),
                     os.path.join('vectorized', 'engine.py'),
                     os.path.join('vectorized', 'vec_env.py')):
        assert required in found, 'the scan missed {0} — run from snek3/'.format(required)
    assert len(paths) >= 8, 'the scan found only {0} modules'.format(len(paths))

    offenders = []
    for path in paths:
        name = os.path.relpath(path, here)
        with open(path) as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            if not any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
                continue
            names = {child.attr if isinstance(child, ast.Attribute) else child.id
                     for child in ast.walk(node)
                     if isinstance(child, (ast.Attribute, ast.Name))}
            # Substring, not equality. The rule used to look for the exact name, so
            # `final_reward == DEFAULT_PERFECT_GAME_REWARD` — the same bug spelled against the
            # default — was invisible to it. Matching any name containing it closes that, and the
            # constant-only exemption below is what keeps `snek2.py`'s override notice legal.
            if not any('PERFECT_GAME_REWARD' in candidate for candidate in names):
                continue
            # A comparison between two *configuration constants* is reporting, not classification:
            # `snek2.py` prints an override notice by testing the knob against its own default, and
            # widening the glob turned that into a false positive. The exemption is "every name in
            # the comparison is a constant", not "the other name looks like a default" — under the
            # looser rule `final_reward == DEFAULT_PERFECT_GAME_REWARD` would slip through, which is
            # precisely the bug wearing a different hat. Any runtime value in the expression is
            # lower-case by convention here, so it still trips.
            if all(candidate.isupper() or candidate.startswith('DEFAULT_') for candidate in names):
                continue
            offenders.append('{0}:{1}'.format(name, node.lineno))
    assert not offenders, ('a perfect game must be identified by its score, not its reward: '
                           + ', '.join(offenders))
