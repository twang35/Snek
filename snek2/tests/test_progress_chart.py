"""Tests for the training progress chart's guides — `under_the_hood.display_progress`.

Only the guides are worth pinning. The traces themselves are one `plot` call each and a wrong one is
obvious on sight, but a guide that appears at the wrong value, on the wrong axis, or in front of the
data is the kind of thing that looks plausible in a 3.65x2.25in thumbnail and quietly misleads —
especially the perfect-score line, whose whole job is to be a reference the score trace is read
against.

`display_progress` renders through matplotlib and writes a PNG, so these use a stub screen and a
temporary path, and assert against the figure's artists via the axes matplotlib left behind.
"""
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import snake_constants
import under_the_hood

PERFECT = snake_constants.MAX_POSSIBLE_SCORE


class StubScreen:
    """Stands in for the pyformulas window, recording that it was handed a frame."""

    def __init__(self):
        self.frames = 0

    def update(self, image):
        self.frames += 1


def rows(scores):
    """Eval rows with the given average scores, one every 1000 steps."""
    return [{'step': (index + 1) * 1000, 'avg_score': score, 'perfect_percent': 0.0,
             'trailing_avg_score': score, 'epsilon': 0.002}
            for index, score in enumerate(scores)]


def draw(scores, resume_steps=()):
    """Runs display_progress and returns its figure's axes, newest figure last.

    display_progress builds its figure through the OO API (under_the_hood.Figure) rather than
    pyplot, to avoid the per-eval matplotlib leak — so the figure is captured by patching that
    Figure, less invasive than changing production code to hand one back for testing. The score
    axis is the figure's first (the percent axis is the twinx() second).
    """
    captured = {}
    real_figure = under_the_hood.Figure

    def capture(*args, **kwargs):
        figure = real_figure(*args, **kwargs)
        captured['figure'] = figure
        return figure

    handle, path = tempfile.mkstemp(suffix='.png')
    os.close(handle)
    under_the_hood.Figure = capture
    try:
        under_the_hood.display_progress(rows(scores), list(resume_steps), StubScreen(),
                                        graph_path=path)
    finally:
        under_the_hood.Figure = real_figure
        for leftover in (path, path + '.partial.png'):
            if os.path.exists(leftover):
                os.remove(leftover)
    return captured['figure'], captured['figure'].axes[0]


def horizontal_lines(axis):
    """(y, colour, linewidth, zorder) for every horizontal guide on `axis`."""
    guides = []
    for line in axis.get_lines():
        ydata = list(line.get_ydata())
        xdata = list(line.get_xdata())
        if len(ydata) == 2 and ydata[0] == ydata[1] and len(xdata) == 2:
            guides.append((ydata[0], line.get_color(), line.get_linewidth(), line.get_zorder()))
    return guides


# ----------------------------------------------------------- the perfect-score guide

def test_no_perfect_score_guide_while_the_arm_is_far_from_it():
    """Below the threshold the line would only stretch the y axis and squash the score trace."""
    figure, score_axis = draw([0.0, 5.0, 40.0, 70.0, 75.9])
    assert not horizontal_lines(score_axis), horizontal_lines(score_axis)
    plt.close(figure)


def test_the_perfect_score_guide_appears_once_the_arm_is_within_80_percent():
    figure, score_axis = draw([10.0, 50.0, 77.0])
    guides = horizontal_lines(score_axis)
    assert len(guides) == 1, guides
    y, _, _, _ = guides[0]
    assert y == PERFECT, 'the guide must sit at the perfect-game score, not a rounded value'
    plt.close(figure)


def test_the_guide_is_at_the_perfect_game_score_not_the_board_size():
    # 95, not PERFECT_SCORE's 100: a perfect game triggers at 95 food eaten because the snake
    # starts with START_SEGMENTS + 1 cells already placed. Drawing 100 would put the reference
    # somewhere the score can never reach.
    figure, score_axis = draw([90.0])
    y = horizontal_lines(score_axis)[0][0]
    assert y == snake_constants.MAX_POSSIBLE_SCORE == 95
    assert y != snake_constants.PERFECT_SCORE
    plt.close(figure)


def test_the_threshold_is_a_fraction_of_the_perfect_score_not_a_literal():
    # Exactly at 80% is not "above" it, one point past is.
    at = 0.8 * PERFECT
    figure, score_axis = draw([at])
    assert not horizontal_lines(score_axis), 'the guide fired at exactly 80%'
    plt.close(figure)

    figure, score_axis = draw([at + 0.1])
    assert len(horizontal_lines(score_axis)) == 1
    plt.close(figure)


def test_the_guide_stays_once_the_arm_has_ever_been_close():
    """Gated on the series maximum, so it does not flicker between renders.

    A drawdown is exactly when the reference is most useful, and a guide that vanished as the score
    fell would disappear at that moment.
    """
    figure, score_axis = draw([85.0, 40.0, 12.0, 3.0])
    assert len(horizontal_lines(score_axis)) == 1
    plt.close(figure)


def test_the_guide_is_behind_the_traces_and_thin():
    """It must not cover the score or perfect-rate lines it annotates."""
    # Several varied points, so there is a real multi-point trace to compare zorder against — with a
    # single point the trace is one marker and this test has nothing to check against.
    figure, score_axis = draw([85.0, 90.0, 88.0, 92.0])
    y, colour, width, zorder = horizontal_lines(score_axis)[0]
    score_trace = [line for line in score_axis.get_lines() if len(line.get_ydata()) > 2]
    assert score_trace, 'the score trace should be the multi-point line on this axis'
    assert zorder < score_trace[0].get_zorder(), 'the guide draws in front of the score trace'
    assert zorder > 0, 'the guide must still be above the axes patch or it is invisible'
    assert width <= 0.5, 'thicker than the traces it annotates'
    assert 'green' in str(colour)
    plt.close(figure)


def test_the_guide_is_on_the_score_axis_not_the_percent_axis():
    # On the percent axis a value of 95 would read as 95% perfect games, which is a different and
    # much stronger claim than "the score ceiling is 95".
    figure, score_axis = draw([90.0])
    assert len(horizontal_lines(score_axis)) == 1
    percent_axis = [axis for axis in figure.axes if axis is not score_axis][0]
    percent_guides = [y for y, _, _, _ in horizontal_lines(percent_axis)]
    assert PERFECT not in percent_guides, percent_guides
    plt.close(figure)


# ----------------------------------------------------------- the guides that were already there

def test_the_percent_guides_are_unchanged():
    figure, score_axis = draw([90.0])
    percent_axis = [axis for axis in figure.axes if axis is not score_axis][0]
    levels = sorted(y for y, _, _, _ in horizontal_lines(percent_axis))
    assert levels == [20, 40, 60, 80], levels
    for _, _, _, zorder in horizontal_lines(percent_axis):
        assert zorder == 1
    plt.close(figure)


def test_a_resume_draws_a_vertical_line_per_restart():
    figure, score_axis = draw([90.0] * 6, resume_steps=(2000, 5000))
    verticals = []
    for line in score_axis.get_lines():
        xdata = list(line.get_xdata())
        if len(xdata) == 2 and xdata[0] == xdata[1]:
            verticals.append(xdata[0])
    assert sorted(verticals) == [2000, 5000], verticals
    plt.close(figure)


def test_the_chart_is_written_and_the_window_updated():
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, 'nested', 'arm.png')
    screen = StubScreen()
    # No figure capture needed: display_progress builds an OO Figure that is not registered
    # with pyplot, so there is nothing to close -- it is collected when the call returns.
    under_the_hood.display_progress(rows([50.0, 90.0]), [], screen, graph_path=path)
    assert screen.frames == 1
    assert os.path.exists(path), 'the PNG was not written'
    assert not os.path.exists(path + '.partial.png'), 'the temporary file was left behind'
    os.remove(path)
    os.rmdir(os.path.dirname(path))
    os.rmdir(directory)
