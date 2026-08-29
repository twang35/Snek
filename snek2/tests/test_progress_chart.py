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


def perfect_rows(percents):
    """Eval rows carrying the given perfect-game percents, one every 1000 steps."""
    return [{'step': (index + 1) * 1000, 'avg_score': 90.0, 'perfect_percent': percent,
             'trailing_avg_score': 90.0, 'epsilon': 0.002}
            for index, percent in enumerate(percents)]


def render(eval_rows, resume_steps=(), policy_name=None, graph_name=None):
    """Runs display_progress on explicit rows and returns (figure, score_axis).

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
    # `graph_name` renames the temp file so the title's path fallback has something predictable to
    # derive from -- mkstemp's own stem is random, which is fine for every other test here but
    # untestable for that one branch.
    if graph_name is not None:
        renamed = os.path.join(os.path.dirname(path), graph_name + '.png')
        os.replace(path, renamed)
        path = renamed
    under_the_hood.Figure = capture
    try:
        under_the_hood.display_progress(eval_rows, list(resume_steps), StubScreen(),
                                        graph_path=path, policy_name=policy_name)
    finally:
        under_the_hood.Figure = real_figure
        for leftover in (path, path + '.partial.png'):
            if os.path.exists(leftover):
                os.remove(leftover)
    return captured['figure'], captured['figure'].axes[0]


def draw(scores, resume_steps=()):
    """Runs display_progress on rows built from `scores`; see `render`."""
    return render(rows(scores), resume_steps)


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


# ----------------------------------------------------------- the x-axis step label

def test_the_x_label_shows_the_latest_step():
    # rows() spaces evals every 1000 steps, so three of them end at step 3000 -> "3k steps".
    figure, score_axis = draw([10.0, 50.0, 90.0])
    assert score_axis.get_xlabel() == 'Iterations (3k steps)', score_axis.get_xlabel()
    plt.close(figure)


def test_the_x_label_groups_thousands_with_a_comma():
    # The latest step is the last row's, and thousands are comma-grouped so a mid-millions run
    # reads cleanly rather than as a wall of digits.
    rows_in = [{'step': 2685000, 'avg_score': 90.0, 'perfect_percent': 50.0,
                'trailing_avg_score': 90.0, 'epsilon': 0.002}]
    figure, score_axis = render(rows_in)
    assert score_axis.get_xlabel() == 'Iterations (2,685k steps)', score_axis.get_xlabel()
    plt.close(figure)


# ----------------------------------------------------------- the trailing-average trend line

def test_trailing_average_is_a_causal_moving_mean():
    # Element i is the mean of the last `window` values up to and including i; the window is
    # shorter at the start where fewer values exist. The last element never depends on values
    # after it — that is what "trailing, not centred" buys.
    assert under_the_hood.trailing_average([10, 20, 30], 5) == [10, 15, 20]
    assert under_the_hood.trailing_average([0, 0, 0, 0, 100], 5) == [0, 0, 0, 0, 20]


def test_trailing_average_window_only_looks_back():
    # A window of 2 over a step change lags the change, it does not anticipate it: the point at
    # the jump averages the jump with the value before, never with the one after.
    assert under_the_hood.trailing_average([0, 0, 100, 100], 2) == [0, 0, 50, 100]


def test_trailing_average_of_nothing_is_nothing():
    assert under_the_hood.trailing_average([], 5) == []


def test_the_trend_line_is_on_the_percent_axis_bold_and_above_the_raw_trace():
    """The whole point of the overlay: a bold smoothed line the eye can follow through the noisy
    thin raw trace, on the same axis and colour so it reads as the same quantity, drawn on top."""
    percents = [10.0, 90.0, 20.0, 80.0, 95.0, 30.0, 70.0]
    figure, score_axis = render(perfect_rows(percents))
    percent_axis = [axis for axis in figure.axes if axis is not score_axis][0]

    # Window 10 must match display_progress's overlay; change both together if it moves.
    expected_trend = [round(v, 6) for v in under_the_hood.trailing_average(percents, 10)]
    assert expected_trend != [round(v, 6) for v in percents], \
        'chosen percents must make the trend differ from the raw trace, or the test proves nothing'

    raw = trend = None
    for line in percent_axis.get_lines():
        ydata = [round(v, 6) for v in line.get_ydata()]
        if len(ydata) != len(percents):
            continue  # skip the 2-point dashed guides
        if ydata == [round(v, 6) for v in percents]:
            raw = line
        elif ydata == expected_trend:
            trend = line
    assert raw is not None, 'the raw perfect-% trace is missing'
    assert trend is not None, 'the trailing-average trend line is missing'
    assert trend.get_linewidth() > raw.get_linewidth(), 'the trend must be bolder than the raw trace'
    assert trend.get_zorder() > raw.get_zorder(), 'the trend must draw on top of the raw trace'
    assert 'red' in str(trend.get_color()), 'the trend belongs to the red perfect-% family'
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

# ----------------------------------------------------------- the chart title (the arm's name)
#
# The name has to be burned into the PNG: chart_viewer renders each arm as a bare `imshow` panel
# with `axis('off')` and reclaims the title space, and charts.md embeds the image with only a
# markdown caption. So an untitled figure is unidentifiable in the four-panel wave window, which
# is the one place it matters. The precedence below is the part worth pinning -- three branches
# that would be easy to collapse into "whichever is not None".


def test_the_chart_is_titled_with_the_policy_name():
    figure, score_axis = render(rows([90.0]), policy_name='b45c-lowlr8-b40b')
    assert score_axis.get_title() == 'b45c-lowlr8-b40b', score_axis.get_title()
    plt.close(figure)


def test_the_title_falls_back_to_the_graph_paths_stem():
    # A caller with the path but not the name still gets an identified chart, and the `.png`
    # comes off -- a title reading "b44a-lowlr7-b29b.png" would be the filename, not the arm.
    figure, score_axis = render(rows([90.0]), graph_name='b44a-lowlr7-b29b')
    assert score_axis.get_title() == 'b44a-lowlr7-b29b', score_axis.get_title()
    plt.close(figure)


def test_an_explicit_name_wins_over_the_path_stem():
    # The mutant this catches is checking graph_path first. On the real training path both are
    # present and agree, so a wrong precedence would never show up there -- only somewhere like
    # the k1000 re-measurements, where the directory is named per checkpoint and not per arm.
    figure, score_axis = render(rows([90.0]), policy_name='b29b-chase10g75seed2',
                                graph_name='k1000e-b29b-1447k')
    assert score_axis.get_title() == 'b29b-chase10g75seed2', score_axis.get_title()
    plt.close(figure)


def test_no_name_and_no_path_leaves_the_chart_untitled():
    # display_progress is called with neither by a headless caller that only wants the window,
    # and matplotlib's default title is the empty string -- so this asserts we add nothing rather
    # than a literal "None".
    captured = {}
    real_figure = under_the_hood.Figure

    def capture(*args, **kwargs):
        figure = real_figure(*args, **kwargs)
        captured['figure'] = figure
        return figure

    under_the_hood.Figure = capture
    try:
        under_the_hood.display_progress(rows([90.0]), [], StubScreen())
    finally:
        under_the_hood.Figure = real_figure
    assert captured['figure'].axes[0].get_title() == '', captured['figure'].axes[0].get_title()
    plt.close(captured['figure'])


def test_the_title_is_larger_than_the_axis_labels_so_it_reads_as_a_heading():
    # At 3.65x2.25in everything is small; the title being the same size as the axis labels made
    # it read as another annotation rather than as the chart's identity.
    figure, score_axis = render(rows([90.0]), policy_name='b45a-lowlr8-b29b')
    title_size = score_axis.title.get_fontsize()
    label_size = score_axis.xaxis.label.get_fontsize()
    assert title_size > label_size, (title_size, label_size)
    plt.close(figure)


def test_the_longest_real_arm_name_still_fits_the_figure_width():
    # 24 chars (`b40b-chasefree10g75seed2`) is the longest name this project has produced. Measured
    # against the figure's own width rather than eyeballed, so a later font bump cannot silently
    # start clipping it.
    figure, score_axis = render(rows([90.0]), policy_name='b40b-chasefree10g75seed2')
    figure.canvas.draw()
    extent = score_axis.title.get_window_extent(figure.canvas.get_renderer())
    assert extent.width < figure.get_window_extent().width, (
        extent.width, figure.get_window_extent().width)
    plt.close(figure)
