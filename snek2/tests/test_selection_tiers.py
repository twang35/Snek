"""The close-out selection tiers, and the granularity they depend on.

Raised on 2026-08-19: the graph eval went from 10 to 20 episodes, and the selection thresholds
from 90/60/top20 to 95/90/top50. The two changes are **coupled** and that is the whole point of
this module. A graph point reports `k/num_eval_episodes`, so the reportable values sit on a grid
of `100/num_eval_episodes` — at 10 episodes a 95% threshold collapses to exactly {100} and is
indistinguishable from ALWAYS_FULL_SINGLE, which would silently merge the mandatory tier into the
uncapped full-length tier and make a close-out *more* expensive, not less.

Raised again on 2026-08-27: 20 -> 100 episodes, when the self-eval moved to the vectorised engine.
**The thresholds survived that re-derivation unchanged** — 95 and 90 are both reportable on the
100-episode grid, the mandatory tier is {95..100} instead of {95, 100}, and the fill band widens from
{90} to {90..94}, so every property this module defends still holds. What did *not* survive was two
fixtures that had `n = 20` baked into their assertions while their docstrings claimed something
weaker, and one of them (`(n - 2) / n`) inverted at n=100: 98 is four points inside the mandatory
tier, so the fixture asserting it must *not* skip screening was asserting the opposite of the
invariant it documented. Both are now expressed off the thresholds and the grid.

Worth carrying: a fixture written against a literal that *coincides* with the invariant is
indistinguishable from one written against the invariant until the coincidence breaks.

Nothing else in the repo ties these two files together, so without these fixtures a future change
to `training.num_eval_episodes` looks local and is not.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_checkpoints
import training


def grid():
    """The perfect-percent values a single graph eval can actually report."""
    n = training.num_eval_episodes
    return [100.0 * k / n for k in range(n + 1)]


def fill_band():
    """The reportable values that `top<N>`'s count actually bounds.

    At or above MIN_EVAL_SINGLE and below ALWAYS_EVAL_SINGLE — one value (90) at n=20, five
    (90-94) at n=100.
    """
    return [x for x in grid()
            if eval_checkpoints.MIN_EVAL_SINGLE <= x < eval_checkpoints.ALWAYS_EVAL_SINGLE]


def test_the_graph_granularity_is_what_the_thresholds_assume():
    """Every threshold must land exactly on the reportable grid.

    A threshold between two grid points is not wrong so much as misleading — it silently behaves
    as the next grid point up, so the constant stops describing the behaviour.
    """
    g = grid()
    for name in ('ALWAYS_EVAL_SINGLE', 'MIN_EVAL_SINGLE', 'ALWAYS_FULL_SINGLE'):
        value = getattr(eval_checkpoints, name)
        assert any(abs(value - x) < 1e-9 for x in g), (
            '{0}={1} is not reportable at num_eval_episodes={2}; the grid is {3}'.format(
                name, value, training.num_eval_episodes, g[-4:]))


def test_the_mandatory_tier_is_the_full_length_tier():
    """Mandatory and full-length are the *same* tier, on purpose: 19/20 and 20/20 both skip the screen.

    An earlier version of this fixture asserted the opposite — that a value had to sit between the
    two thresholds — on the theory that collapsing them would let the uncapped full-length tier
    absorb the whole arm. That theory ignored the abandonment gate and was wrong: simulated on
    b43/b44's curves, moving ALWAYS_FULL_SINGLE from 100 to 95 changed total close-out episodes by
    -1%, because a 19/20 checkpoint whose true rate is under the gate is abandoned after 4 failures
    — often at the 20-episode floor, i.e. for what the screen would have cost anyway.

    What it buys is coverage: under the old split, several hundred 19/20 checkpoints per
    continuation arm were screened to 20 episodes and capped by EVAL_CONFIRM_COUNT, so the arm's
    best checkpoint could finish on a 20-episode row.
    """
    assert eval_checkpoints.ALWAYS_FULL_SINGLE == eval_checkpoints.ALWAYS_EVAL_SINGLE
    n = training.num_eval_episodes
    # The two lowest values of the mandatory tier, whatever the episode count makes them: at n=20
    # that is 19/20 and 20/20, at n=100 it is 95/100 and 96/100. Expressed off the threshold rather
    # than off `n` because the *tier* is what has to skip the screen — an `n - 1` literal only
    # happened to name the tier's floor while the grid was coarse enough for the tier to hold two
    # values, and at n=100 it names 99, four points inside it.
    mandatory = [x for x in grid() if x >= eval_checkpoints.ALWAYS_EVAL_SINGLE]
    assert len(mandatory) >= 2, 'mandatory tier {0} needs at least two reportable values'.format(
        mandatory)
    for single in (mandatory[0], mandatory[-1]):
        assert eval_checkpoints.skips_screening({'selected_by': 'x', 'single_eval': single}), (
            '{0}% is in the mandatory tier and must go straight to full length'.format(single))
    # **The whole fill band must NOT skip screening**, or `count` stops bounding selection. This is
    # the assertion an `n - 2` literal was standing in for: at n=20 the band was the single value 90
    # and `(n - 2) / n` happened to be exactly it, but at n=100 `(n - 2) / n` is 98 -- inside the
    # mandatory tier -- so the literal was testing the opposite of what it claimed.
    for single in fill_band():
        assert not eval_checkpoints.skips_screening(
            {'selected_by': 'x', 'single_eval': single}), (
            '{0}% is in the fill band and must be screened, not sent straight to full '
            'length'.format(single))


def test_the_uncapped_full_tier_is_only_affordable_because_of_the_gate():
    """The -1% result above is conditional on abandonment being on and tighter than the tier.

    With the gate off, or set at or below the full-length threshold, every 19/20 checkpoint runs the
    whole EVAL_EPISODES and an uncapped tier becomes the entire close-out bill. This fixture is the
    reason not to relax one of these without re-reading the other.
    """
    assert eval_checkpoints.DEFAULT_MIN_ACHIEVABLE > eval_checkpoints.ALWAYS_FULL_SINGLE, (
        'the gate must be stricter than the full-length tier, or that tier is unbounded')
    # And the gate must still leave the HOF re-measure something to select.
    assert eval_checkpoints.DEFAULT_MIN_ACHIEVABLE < eval_checkpoints.DEFAULT_ABOVE_THRESHOLD


def test_the_fill_band_is_not_empty():
    """>=MIN_EVAL_SINGLE and <ALWAYS_EVAL_SINGLE has to contain something, or `count` is dead.

    With an empty band the `top<N>` count can never do anything: selection becomes the mandatory
    tier alone and raising N has no effect at all.
    """
    band = fill_band()
    assert band, (
        'the fill band is empty at num_eval_episodes={0}: MIN_EVAL_SINGLE={1} and '
        'ALWAYS_EVAL_SINGLE={2} leave no reportable value between them, so `top<N>` is dead'.format(
            training.num_eval_episodes, eval_checkpoints.MIN_EVAL_SINGLE,
            eval_checkpoints.ALWAYS_EVAL_SINGLE))


def test_thresholds_are_ordered():
    assert (eval_checkpoints.MIN_EVAL_SINGLE
            < eval_checkpoints.ALWAYS_EVAL_SINGLE
            <= eval_checkpoints.ALWAYS_FULL_SINGLE)


def test_the_abandon_gate_stays_below_the_hof_selection_gate():
    """The close-out gate must be strictly below the HOF gate, and it now has one point of slack.

    HOF re-measures `above:98` out of the close-out's own file, and only rows that reach the
    close-out gate are measured full length — so a close-out gate at or above 98 would abandon
    exactly the rows the re-measure reads and starve it silently. At the old 96 this had two
    points of slack; at 97 it has one.
    """
    assert eval_checkpoints.DEFAULT_MIN_ACHIEVABLE < eval_checkpoints.DEFAULT_ABOVE_THRESHOLD
    assert eval_checkpoints.DEFAULT_MIN_ACHIEVABLE == 97.0
    assert eval_checkpoints.DEFAULT_ABOVE_THRESHOLD == 98.0


def test_the_abandon_gate_is_reachable_at_the_full_episode_count():
    """A gate has to be expressible at EVAL_EPISODES or it rounds into a stricter one.

    At 100 episodes a 97% gate means "stop once more than 3 have failed". A gate of, say, 97.5
    would behave as 98 and collide with the HOF gate without any constant saying so.
    """
    episodes = 100
    gate = eval_checkpoints.DEFAULT_MIN_ACHIEVABLE
    assert abs(gate * episodes / 100.0 - round(gate * episodes / 100.0)) < 1e-9


def test_the_selection_label_reports_the_real_threshold():
    """The `selected_by` label is derived, not hardcoded.

    It read `threshold90` for the whole life of the 90% tier; a literal is how a label ends up
    describing a threshold the code no longer uses.
    """
    label = 'threshold{0:g}'.format(eval_checkpoints.ALWAYS_EVAL_SINGLE)
    assert label == 'threshold95'
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'eval_checkpoints.py')).read()
    assert "'threshold90'" not in src, 'a hardcoded threshold90 label has come back'


def test_the_band_printout_does_not_hardcode_a_10_episode_step():
    """`ALWAYS_EVAL_SINGLE - 10` described the fill band only while a graph point was 10 episodes.

    At 20 episodes the band is [90, 95), and `95 - 10 = 85` names a value that is not in it.
    """
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'eval_checkpoints.py')).read()
    assert 'ALWAYS_EVAL_SINGLE - 10' not in src


def test_a_strong_arm_exceeds_the_count_on_purpose():
    """`count` is a target, not a quota: the mandatory tier is uncapped.

    This is why a continuation close-out is expensive, and it must stay true — capping the
    mandatory tier would silently drop checkpoints a close-out is supposed to cover.
    """
    n = 400
    candidates = [{'step': i * 1000, 'single': 100.0, 'smoothed': 100.0} for i in range(n)]
    mandatory = [c for c in candidates if c['single'] >= eval_checkpoints.ALWAYS_EVAL_SINGLE]
    fill = mandatory[:max(0, eval_checkpoints.DEFAULT_COUNT - len(mandatory))]
    assert len(mandatory) == n
    assert fill == []                      # no slots left once mandatory exceeds count
    assert len(mandatory) > eval_checkpoints.DEFAULT_COUNT
