"""Fixtures for the chart viewer's decision logic — which panels, and when to close.

Nothing here opens a window: the drawing is matplotlib's and the parts worth pinning are the two
that have historically been wrong, panel selection and the liveness answer.
"""

import os
import subprocess

import pytest

from tools import chart_viewer


def touch(directory, name, mtime):
    path = os.path.join(str(directory), name)
    with open(path, 'wb') as handle:
        handle.write(b'not really a png')
    os.utime(path, (mtime, mtime))
    return path


def test_panels_caps_hard_and_keeps_the_most_recently_written(tmp_path):
    for index in range(10):
        touch(tmp_path, 'arm{0}.png'.format(index), 1_000_000 + index)
    found = chart_viewer.panels([], os.path.join(str(tmp_path), '*.png'), max_panels=3)
    assert [os.path.basename(path) for path in found] == ['arm9.png', 'arm8.png', 'arm7.png']


def test_panels_re_expands_the_glob_so_a_new_chart_appears(tmp_path):
    pattern = os.path.join(str(tmp_path), '*.png')
    touch(tmp_path, 'a.png', 1_000_000)
    assert len(chart_viewer.panels([], pattern, max_panels=8)) == 1
    touch(tmp_path, 'b.png', 1_000_001)
    assert len(chart_viewer.panels([], pattern, max_panels=8)) == 2


def test_panels_drops_a_path_that_does_not_exist(tmp_path):
    real = touch(tmp_path, 'a.png', 1_000_000)
    found = chart_viewer.panels([real, os.path.join(str(tmp_path), 'gone.png')], None, max_panels=8)
    assert found == [real]


def test_panels_does_not_show_one_file_twice(tmp_path):
    real = touch(tmp_path, 'a.png', 1_000_000)
    found = chart_viewer.panels([real], os.path.join(str(tmp_path), '*.png'), max_panels=8)
    assert found == [real], 'an explicit path that the glob also matches is one panel'


def test_grid_shape_stays_wide_and_always_has_room(tmp_path):
    for count in range(1, 10):
        rows, columns = chart_viewer.grid_shape(count)
        assert rows * columns >= count, count
        assert columns >= rows, 'charts are wide, so the grid should be too'


def test_a_live_pid_reads_as_alive():
    assert chart_viewer.pids_alive([os.getpid()]) is True


def test_no_live_pid_reads_as_not_alive():
    assert chart_viewer.pids_alive([999_991, 999_992]) is False


def test_one_live_pid_among_dead_ones_still_reads_as_alive():
    # The bug this pins: macOS `ps -p 999991,<live>` rejects the whole list and prints nothing, so
    # asking that way closed the window on three running shards as soon as the first finished.
    assert chart_viewer.pids_alive([999_991, os.getpid()]) is True


def test_no_pids_to_watch_is_unanswerable_not_negative():
    assert chart_viewer.pids_alive([]) is None


def test_a_failed_scan_is_unanswerable_not_negative(monkeypatch):
    # `ps` printing nothing because it failed must never read the same as `ps` finding nothing.
    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=2, stdout=b'')
    monkeypatch.setattr(chart_viewer.subprocess, 'run', failed)
    assert chart_viewer.pids_alive([os.getpid()]) is None


def test_a_ps_that_cannot_be_launched_is_unanswerable(monkeypatch):
    def raises(*args, **kwargs):
        raise OSError('no ps')
    monkeypatch.setattr(chart_viewer.subprocess, 'run', raises)
    assert chart_viewer.pids_alive([1]) is None


def test_a_zombie_does_not_count_as_alive(monkeypatch):
    def scanned(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=0, stdout=b'  4242 Z+\n     1 Ss\n')
    monkeypatch.setattr(chart_viewer.subprocess, 'run', scanned)
    assert chart_viewer.pids_alive([4242]) is False
    assert chart_viewer.pids_alive([1]) is True


def test_a_negative_answer_has_to_repeat_before_it_is_believed():
    # One negative is a race with a launcher that has not spawned yet, or a lost `ps`.
    assert chart_viewer.NEGATIVE_CHECKS > 1
