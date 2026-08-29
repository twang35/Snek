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
    figure, axis = stage_b_chart.build_figure([row(1000, 100), row(2000, 100)], name='x')
    assert axis.get_ylim()[0] <= 96.0


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
