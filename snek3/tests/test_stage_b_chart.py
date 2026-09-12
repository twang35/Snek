"""Fixtures for the stage-B chart: what it loads, what it counts, and what it draws."""

import json
import os

import numpy as np
import pytest

from env import constants
from tools import results, stage_b_chart


def row(step, perfect, episodes=100):
    """A stage-B row with only the fields the chart reads."""
    return {'step': int(step), 'episodes': episodes, 'perfect_games': int(perfect),
            'perfect_percent': round(100.0 * perfect / episodes, 1),
            'perfect_ci95': [90.0, 100.0]}


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, 'RUNS_DIR', str(tmp_path))
    return str(tmp_path)


def write_pass(policy, label, rows, shard=None, shards=None):
    path = results.stage_b_path(policy, label, shard=shard, shards=shards)
    results.write(path, {'policy': policy, 'rows': rows})
    return path


def test_load_prefers_the_merged_file_over_the_shards(runs_dir):
    write_pass('arm', 'p', [row(1000, 90)], shard=0, shards=2)
    write_pass('arm', 'p', [row(2000, 91)], shard=1, shards=2)
    write_pass('arm', 'p', [row(1000, 90), row(2000, 91), row(3000, 92)])
    assert [r['step'] for r in stage_b_chart.load('arm', 'p')] == [1000, 2000, 3000]


def test_load_pools_the_shards_while_a_wave_is_still_running(runs_dir):
    # No merged file yet: this is the case that makes the chart usable in flight.
    write_pass('arm', 'p', [row(3000, 92), row(1000, 90)], shard=0, shards=2)
    write_pass('arm', 'p', [row(2000, 91)], shard=1, shards=2)
    loaded = stage_b_chart.load('arm', 'p')
    assert [r['step'] for r in loaded] == [1000, 2000, 3000], 'pooled and sorted by step'


def test_load_keeps_the_longer_sample_of_a_duplicated_step(runs_dir):
    write_pass('arm', 'p', [row(1000, 90, episodes=100)], shard=0, shards=2)
    write_pass('arm', 'p', [row(1000, 480, episodes=500)], shard=1, shards=2)
    loaded = stage_b_chart.load('arm', 'p')
    assert len(loaded) == 1 and loaded[0]['episodes'] == 500, 'matches results.merge'


def test_load_is_empty_rather_than_an_error_when_nothing_has_been_measured(runs_dir):
    assert stage_b_chart.load('arm', 'p') == []


def test_a_labelled_pass_does_not_pick_up_the_unlabelled_one(runs_dir):
    write_pass('arm', None, [row(1000, 90)])
    assert stage_b_chart.load('arm', 'ab') == []


def test_widest_region_finds_the_longest_run_not_the_first(runs_dir):
    rows = [row(1000, 99), row(2000, 99),                      # a run of 2
            row(3000, 50),
            row(4000, 98), row(5000, 100), row(6000, 99)]      # a run of 3
    count, low, high = stage_b_chart.widest_region(rows, level=98.0)
    assert (count, low, high) == (3, 4000, 6000)


def test_widest_region_counts_adjacent_rows_not_consecutive_step_numbers():
    # A selector that skipped step 3000 must not split an otherwise solid plateau. Measuring
    # adjacency in step numbers would report 1 here instead of 3.
    rows = [row(1000, 99), row(2000, 99), row(9000, 99)]
    assert stage_b_chart.widest_region(rows, level=98.0)[0] == 3


def test_widest_region_includes_a_run_that_reaches_the_last_row():
    rows = [row(1000, 50), row(2000, 99), row(3000, 99)]
    assert stage_b_chart.widest_region(rows, level=98.0) == (2, 2000, 3000)


def test_widest_region_is_zero_when_nothing_clears_the_level():
    assert stage_b_chart.widest_region([row(1000, 50)], level=98.0) == (0, None, None)


def test_a_row_exactly_on_the_level_counts_as_inside_the_region():
    assert stage_b_chart.widest_region([row(1000, 98)], level=98.0)[0] == 1


def test_summarise_pools_episodes_rather_than_averaging_row_percentages():
    # 100/100 and 400/500 is 500/600 = 83.3%, not the mean of 100% and 80% = 90%.
    rows = [row(1000, 100, episodes=100), row(2000, 400, episodes=500)]
    facts = stage_b_chart.summarise(rows)
    assert facts['pooled_percent'] == pytest.approx(500 / 600 * 100, abs=0.01)
    assert facts['episodes'] == 600 and facts['perfect_games'] == 500


def test_summarise_reports_every_step_tied_for_best():
    rows = [row(1000, 100), row(2000, 90), row(3000, 100)]
    facts = stage_b_chart.summarise(rows)
    assert facts['best_percent'] == 100.0 and facts['best_steps'] == [1000, 3000]


def test_summarise_thresholds_are_at_or_above_not_above():
    facts = stage_b_chart.summarise([row(1000, 98), row(2000, 97)])
    assert facts['at_or_above'][98.0] == 1
    assert facts['at_or_above'][95.0] == 2


def test_summarise_is_empty_for_no_rows():
    assert stage_b_chart.summarise([]) == {}


def test_text_summary_says_so_rather_than_raising_on_no_rows():
    assert 'no stage-B rows' in stage_b_chart.text_summary([], 'arm')


def test_text_summary_prints_the_interval_in_percent_not_in_fractions():
    # perfect_ci95 is stored in percent, like perfect_percent beside it. Scaling it again printed
    # `[9630.0, 10000.0]`.
    text = stage_b_chart.text_summary([row(1000, 100)], 'arm')
    assert '[90.0, 100.0]' in text


def test_the_figure_carries_the_name_and_the_row_count():
    figure, axis = stage_b_chart.build_figure([row(1000, 99), row(2000, 98)], name='b48a')
    title = axis.get_title()
    assert 'b48a' in title and '2' in title


def test_an_empty_pass_still_builds_a_labelled_figure():
    figure, axis = stage_b_chart.build_figure([], name='b48a')
    assert 'b48a' in axis.get_title()


def test_the_y_floor_is_pinned_below_both_the_worst_row_and_the_level():
    figure, axis = stage_b_chart.build_figure([row(1000, 99), row(2000, 96)], name='x')
    low, high = axis.get_ylim()
    assert low <= 94.0 and high >= 100.0, (low, high)
    # A run of rows that all clear the level must not push the floor above the level's guide line.
    # Stated against `REGION_LEVEL` rather than a number, so moving the level cannot make this fail
    # while the property it is about still holds (it did, when the level moved to 99 on 2026-08-30).
    figure, axis = stage_b_chart.build_figure([row(1000, 100), row(2000, 100)], name='x')
    assert axis.get_ylim()[0] < stage_b_chart.REGION_LEVEL


def test_pooled_trend_pools_games_over_episodes_within_the_step_window_not_row_percentages():
    # Two rows a step apart, unequal sizes: the pooled rate is 150/200 = 75.0, not the mean of 50 and 100.
    rows = [row(1000, 50, episodes=100), row(2000, 100, episodes=100)]
    rows[0]['perfect_percent'] = 50.0
    steps, trend = stage_b_chart.pooled_trend(rows, half_window=1_000_000)
    assert list(steps) == [1000, 2000]
    assert np.allclose(trend, [75.0, 75.0])


def test_pooled_trend_only_pools_rows_inside_the_window():
    rows = [row(0, 100), row(500_000, 90), row(5_000_000, 80)]
    steps, trend = stage_b_chart.pooled_trend(rows, half_window=1_000_000)
    # The first two see each other (95.0); the far row sees only itself (80.0).
    assert np.isclose(trend[0], 95.0) and np.isclose(trend[1], 95.0)
    assert np.isclose(trend[-1], 80.0)


def test_pooled_trend_breaks_the_line_across_a_gap_wider_than_the_window():
    rows = [row(0, 100), row(500_000, 90), row(5_000_000, 80)]
    steps, trend = stage_b_chart.pooled_trend(rows, half_window=1_000_000)
    # A NaN sits between the two segments so a line through the result lifts the pen over the gap.
    assert len(steps) == 4 and np.isnan(trend[2]) and steps[2] == 500_000


def test_pooled_trend_is_sorted_by_step_and_empty_below_two_rows():
    steps, trend = stage_b_chart.pooled_trend([row(3000, 99), row(1000, 97), row(2000, 98)])
    assert list(steps) == [1000, 2000, 3000]
    assert stage_b_chart.pooled_trend([row(1000, 99)])[1].size == 0


def test_the_figure_draws_the_trend_in_its_colour_on_two_or_more_rows_and_not_on_one():
    figure, axis = stage_b_chart.build_figure([row(1000, 99), row(2000, 98), row(3000, 97)], name='x')
    trend_lines = [line for line in axis.get_lines() if line.get_color() == stage_b_chart.TREND_COLOR]
    assert len(trend_lines) == 1 and len(trend_lines[0].get_xdata()) == 3
    figure, axis = stage_b_chart.build_figure([row(1000, 99)], name='x')
    assert not [line for line in axis.get_lines() if line.get_color() == stage_b_chart.TREND_COLOR]


def test_render_writes_a_png_atomically_and_leaves_no_partial(runs_dir):
    path = os.path.join(runs_dir, 'chart.png')
    image = stage_b_chart.render([row(1000, 99)], path, name='arm')
    assert os.path.exists(path) and not os.path.exists(path + '.partial.png')
    assert image.ndim == 3 and image.shape[2] == 3


def test_redraw_names_the_png_after_the_result_file(runs_dir):
    write_pass('arm', 'ab3222', [row(1000, 99)])
    path, rows = stage_b_chart.redraw('arm', 'ab3222')
    assert os.path.basename(path) == 'arm_checkpoint_evals_ab3222.png'
    assert len(rows) == 1


def test_redraw_writes_nothing_when_there_are_no_rows(runs_dir):
    path, rows = stage_b_chart.redraw('arm', 'ab3222')
    assert path is None and rows == []


# --- no trend line --------------------------------------------------------------------------------

def connected_series(axis):
    """Lines that join the data. The two `axhline` guides are excluded by their 2 points.

    A trend line spans the rows, so it has one point per row; the level guide and the pooled line
    have exactly two and are meant to be there.
    """
    return [line for line in axis.get_lines()
            if line.get_linestyle() not in ('none', 'None') and len(line.get_xdata()) > 2]


def test_the_dots_are_never_connected_and_the_only_series_is_the_trend_at_every_row_count():
    """The dots stay unconnected (adjacent checkpoints are independent samples). The one connected
    series is the pooled-rate trend (2026-09-09, user's call), and it has no row-count gate: the
    2026-09-01 line was gated at 40 rows and so appeared on some arms of a batch and not others,
    which is why 39, 40 and 41 are the counts to test (`connected_series` cannot see a 2-row line)."""
    for count in (3, 5, 39, 40, 41, 200):
        rows = [row(1000 * i, 495 - (i % 3), episodes=500) for i in range(1, count + 1)]
        figure, axis = stage_b_chart.build_figure(rows, 'arm')
        drawn = connected_series(axis)
        assert len(drawn) == 1 and drawn[0].get_color() == stage_b_chart.TREND_COLOR, (
            '{0} rows drew {1} connected line(s); expected the trend alone'.format(count, len(drawn)))
        assert len(drawn[0].get_xdata()) == count


def test_the_two_horizontal_guides_survive():
    """The removal must not take the level guide or the pooled line with it."""
    rows = [row(1000 * i, 495, episodes=500) for i in range(1, 60)]
    figure, axis = stage_b_chart.build_figure(rows, 'arm')
    guides = [line for line in axis.get_lines()
              if line.get_linestyle() not in ('none', 'None') and len(line.get_xdata()) == 2]
    assert len(guides) == 2, 'expected the level guide and the pooled line'


def test_the_points_and_the_rug_are_still_drawn():
    # Stated against the pass's own level rather than a count out of 500, so moving the gate (99 -> 99.2
    # on 2026-09-09) cannot empty the rug while the property it is about still holds.
    level = stage_b_chart.region_level(None)
    rows = [row(1000 * i, 500, episodes=500) for i in range(1, 60)]
    assert rows[0]['perfect_percent'] >= level
    figure, axis = stage_b_chart.build_figure(rows, 'arm', level=level)
    markers = [line for line in axis.get_lines() if line.get_marker() not in ('', 'None', None)]
    assert len(markers) == 2, 'expected the point cloud and the >=level rug'


def test_summarise_pools_full_rows_only_and_counts_the_stopped_ones():
    """A stopped row (`abandoned`) is a short sample below its target: it is a row, and it is in no
    pooled rate, threshold count or top list (`plans/archive/early-stop.md`)."""
    stopped = dict(row(3000, 40, episodes=50), abandoned=True, episodes_planned=100)
    rows = [row(1000, 98), row(2000, 99), stopped]
    facts = stage_b_chart.summarise(rows)
    assert (facts['rows'], facts['stopped']) == (3, 1)
    assert facts['pooled_percent'] == pytest.approx(197 / 200 * 100, abs=0.01)
    assert facts['at_or_above'][95.0] == 2 and facts['episodes_per_row'] == [100]
    text = stage_b_chart.text_summary(rows, 'arm')
    assert '1 stopped early' in text and '3,000' not in text.split('top')[1]
    only_stopped = stage_b_chart.summarise([stopped])
    assert only_stopped['pooled_percent'] is None and only_stopped['best_percent'] is None
    assert 'nothing to pool' in stage_b_chart.text_summary([stopped], 'arm')
    figure, _ = stage_b_chart.build_figure(rows, "arm")    # draws, with the stopped row hollow
    assert figure is not None


def test_the_guide_sits_at_the_gate_of_the_pass_drawn():
    """Stage B and hof5000 draw the hof5000 cut, hof30k the hof30k cut (user, 2026-09-09), read from
    `eta` so the guide moves when a gate does; `--level` still overrides."""
    from tools import eta
    assert stage_b_chart.region_level(None) == eta.HOF_THRESHOLD == 99.2
    assert stage_b_chart.region_level('hof5000') == eta.HOF_THRESHOLD
    assert stage_b_chart.region_level('hof30k') == eta.HOF30K_THRESHOLD == 99.6
    assert stage_b_chart.region_level('something-else') == stage_b_chart.REGION_LEVEL
    facts = stage_b_chart.summarise([row(1000, 996, episodes=1000), row(2000, 992, episodes=1000)], level=99.6)
    assert facts['at_or_above'][99.6] == 1 and facts['at_or_above'][99.0] == 2
    assert 'at or above  99.6%' in stage_b_chart.text_summary([row(1000, 996, episodes=1000)], 'arm', level=99.6)


def test_redraw_draws_each_pass_at_its_own_gate(runs_dir, monkeypatch):
    seen = {}
    monkeypatch.setattr(stage_b_chart, 'render', lambda rows, path, name=None, level=None: seen.__setitem__(name or path, level))
    results.write(results.stage_b_path('arm'), {'rows': [row(1000, 99)]})
    results.write(results.stage_b_path('arm', 'hof30k'), {'rows': [row(1000, 99)]})
    stage_b_chart.redraw('arm')
    assert seen['arm'] == 99.2
    stage_b_chart.redraw('arm', 'hof30k')
    assert seen['arm'] == 99.6
    stage_b_chart.redraw('arm', 'hof30k', level=98.0)
    assert seen['arm'] == 98.0
