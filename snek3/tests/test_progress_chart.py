"""The per-arm chart. Three properties, each with an incident behind it.

**No figure may be registered with `pyplot`.** That registry is why snek2's trainer grew ~1 MB per
eval and eventually exhausted the desktop's memory: `plt.close()` does not fully release a figure's
artists in this matplotlib version, so a figure per eval leaked. A bare `Figure` is never registered.

**The trailing average must be trailing.** The chart is read live, so the newest point must reflect
only evals already seen; a centred window would pull it toward values that do not exist yet.

**The PNG must appear atomically.** The trainer rewrites it every eval while a viewer may be polling
it, so no reader may ever see a half-written file.
"""

import os

import matplotlib.pyplot as pyplot
import pytest

from env import constants
from tools import progress_chart


def evals(count, perfect=None, score=None):
    return [{'step': (index + 1) * 1000,
             'avg_score': score(index) if score else min(95.0, index * 0.1),
             'trailing_avg_score': min(95.0, index * 0.1),
             'perfect_percent': perfect(index) if perfect else min(100.0, index * 0.2)}
            for index in range(count)]


def axhline_levels(axis):
    """The y positions of the horizontal guide lines on an axis."""
    return sorted(line.get_ydata()[0] for line in axis.lines
                  if len(set(line.get_xdata())) <= 2 and len(set(line.get_ydata())) == 1)


# ------------------------------------------------------------------ the leak

def test_rendering_registers_no_pyplot_figure(tmp_path):
    """The assertion that stands in for "the trainer does not grow 1 MB an eval".

    Directly checkable, unlike the leak itself: a figure only leaks *because* it is registered, so
    an empty registry after a render is the property that matters.
    """
    before = set(pyplot.get_fignums())
    for _ in range(3):
        progress_chart.render(evals(50), str(tmp_path / 'a.png'), 'a')
    assert set(pyplot.get_fignums()) == before


def test_the_check_above_would_notice_a_registered_figure():
    # A fixture whose subject cannot violate it is not a fixture: this proves an empty registry is
    # not simply what `get_fignums` always returns.
    figure = pyplot.figure()
    try:
        assert pyplot.get_fignums()
    finally:
        pyplot.close(figure)


# ------------------------------------------------------------------ the trailing average

def test_the_average_is_trailing_not_centred():
    # Element i is the mean of the last `window` values up to and including i. A centred window
    # would make element 0 depend on element 1, which does not exist yet when the chart is drawn.
    assert progress_chart.trailing_average([0.0, 0.0, 3.0], 3) == [0.0, 0.0, 1.0]


def test_the_window_is_short_at_the_start_rather_than_padded():
    # Padding with zeroes would drag the opening of every curve down toward zero, which on this
    # project's charts is exactly where a reader looks for the arm's first signs of learning.
    assert progress_chart.trailing_average([10.0, 10.0], 5) == [10.0, 10.0]


def test_the_average_smooths_toward_the_mean():
    values = [0.0, 100.0] * 20
    smoothed = progress_chart.trailing_average(values, 10)
    assert all(30.0 <= value <= 70.0 for value in smoothed[10:])


# ------------------------------------------------------------------ the guides

def test_the_perfect_score_guide_appears_only_once_the_arm_is_close():
    """Below the gate the line sits far above every point and only squashes the score trace.

    Both directions are checked, because a guide that is always drawn and a guide that is never
    drawn both pass a one-sided test.
    """
    _, weak, _ = progress_chart.build_figure(evals(40, score=lambda i: 10.0), 'a')
    _, strong, _ = progress_chart.build_figure(evals(40, score=lambda i: 90.0), 'a')
    assert constants.MAX_POSSIBLE_SCORE not in axhline_levels(weak)
    assert constants.MAX_POSSIBLE_SCORE in axhline_levels(strong)


def test_the_guide_is_gated_on_the_maximum_not_the_latest_point():
    # Gated on the latest point it would flicker on and off between renders as the score oscillates,
    # which on a chart rewritten every eval is visible as a blinking line.
    rows = evals(40, score=lambda i: 90.0 if i == 20 else 10.0)
    _, score_axis, _ = progress_chart.build_figure(rows, 'a')
    assert rows[-1]['avg_score'] < progress_chart.GUIDE_FRACTION * constants.MAX_POSSIBLE_SCORE
    assert constants.MAX_POSSIBLE_SCORE in axhline_levels(score_axis)


def test_the_percent_axis_carries_its_own_guides_and_a_fixed_range():
    # The red trace cannot be read off the right-hand axis without them, since the left axis's ticks
    # are on a different scale. The range is fixed at 0-100 so two arms' charts are comparable by eye.
    _, _, percent_axis = progress_chart.build_figure(evals(40), 'a')
    assert axhline_levels(percent_axis) == [20.0, 40.0, 60.0, 80.0]
    assert percent_axis.get_ylim() == (0.0, 100.0)


def test_a_resume_is_marked_on_the_chart():
    # So a dip or a jump can be tied to a restart rather than to the policy itself.
    _, score_axis, _ = progress_chart.build_figure(evals(40), 'a', resume_steps=[20000])
    verticals = [line.get_xdata()[0] for line in score_axis.lines
                 if len(set(line.get_xdata())) == 1]
    assert 20000 in verticals


def test_the_name_is_burned_into_the_image():
    # Neither consumer can add it: a chart viewer renders each arm as a bare panel with the axes
    # off, and `charts.md` embeds the PNG with only a markdown caption. So an untitled figure is
    # unidentifiable in a four-panel wave window, which is exactly where identifying it matters.
    _, score_axis, _ = progress_chart.build_figure(evals(10), 'b1a-thing')
    assert score_axis.get_title() == 'b1a-thing'


def test_the_x_label_states_the_latest_step():
    # Late in training the traces fill the plot, and nothing else on the chart says where the run is.
    _, score_axis, _ = progress_chart.build_figure(evals(1200), 'a')
    assert '1,200k steps' in score_axis.get_xlabel()


# ------------------------------------------------------------------ the file

def test_the_png_appears_atomically(tmp_path):
    path = str(tmp_path / 'a.png')
    progress_chart.render(evals(20), path, 'a')
    assert sorted(os.listdir(str(tmp_path))) == ['a.png']
    assert os.path.getsize(path) > 0


def test_a_rerender_replaces_the_file(tmp_path):
    path = str(tmp_path / 'a.png')
    progress_chart.render(evals(20), path, 'a')
    progress_chart.render(evals(400), path, 'a')
    assert sorted(os.listdir(str(tmp_path))) == ['a.png']


def test_an_arm_with_one_eval_renders(tmp_path):
    # A chart is drawn from the very first eval, so the degenerate case has to work rather than
    # raising on an empty axis range.
    progress_chart.render(evals(1), str(tmp_path / 'a.png'), 'a')


def test_redraw_reports_nothing_rather_than_raising_for_an_untrained_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, 'RUNS_DIR', str(tmp_path))
    assert progress_chart.redraw('never-trained') is None
