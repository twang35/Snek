"""Fixtures for the chart viewer's decision logic — which panels, and what it follows.

Nothing here opens a window: the drawing is matplotlib's and the parts worth pinning are the two
that have historically been wrong, panel selection and the exit condition.
"""

import os

import pytest

from tools import chart_viewer


def touch(directory, name, mtime):
    path = os.path.join(str(directory), name)
    with open(path, 'wb') as handle:
        handle.write(b'not really a png')
    os.utime(path, (mtime, mtime))
    return path


def test_panels_caps_hard_and_keeps_the_most_recently_written(tmp_path):
    """mtime decides *which* panels survive the cap. It does not decide their order — see below."""
    for index in range(10):
        touch(tmp_path, 'arm{0}.png'.format(index), 1_000_000 + index)
    found = chart_viewer.panels([], os.path.join(str(tmp_path), '*.png'), max_panels=3)
    assert {os.path.basename(path) for path in found} == {'arm7.png', 'arm8.png', 'arm9.png'}


def test_panels_order_does_not_move_when_a_chart_is_rewritten(tmp_path):
    """The flash bug, pinned at its source.

    `panels` used to return its result in mtime order, so every eval an arm wrote permuted the list;
    `refresh` rebuilds on a changed panel set, and a rebuild closes the figure and opens a new
    window. Three arms on their own clocks flashed the desktop's window every few seconds and took
    any resize with it.
    """
    pattern = os.path.join(str(tmp_path), '*.png')
    for name in ('a.png', 'b.png', 'c.png'):
        touch(tmp_path, name, 1_000_000)
    before = chart_viewer.panels([], pattern, max_panels=8)
    touch(tmp_path, 'b.png', 2_000_000)          # the middle arm finishes an eval
    assert chart_viewer.panels([], pattern, max_panels=8) == before


def test_panels_order_is_stable_whatever_the_mtimes(tmp_path):
    pattern = os.path.join(str(tmp_path), '*.png')
    touch(tmp_path, 'c.png', 3_000_000)
    touch(tmp_path, 'a.png', 1_000_000)
    touch(tmp_path, 'b.png', 2_000_000)
    found = chart_viewer.panels([], pattern, max_panels=8)
    assert [os.path.basename(path) for path in found] == ['a.png', 'b.png', 'c.png']


# --- sizing ---------------------------------------------------------------------------------------
#
# The window opened at 1201x650 on a 3840x2160 display, 31% of its width, because the size was a
# fixed number of inches. These pin the screen-derived replacement; none of them needs a display.

def test_the_window_fills_the_screen_it_opens_on():
    screen = (38.4, 21.6)                        # 3840x2160 at 100 dpi
    width, height = chart_viewer.fit_dims(2, 2, 1.622, 1.0, screen)
    assert height == pytest.approx(21.6 * chart_viewer.SCREEN_HEIGHT_FRACTION, rel=1e-6)
    assert width > 30.0, 'a 4K panel should give a window far larger than the old fixed 12 inches'


def test_the_window_never_exceeds_the_screen():
    for aspect in (0.5, 1.0, 1.622, 2.083, 5.0):
        for rows, columns in ((1, 1), (2, 2), (2, 3), (3, 3)):
            width, height = chart_viewer.fit_dims(rows, columns, aspect, 1.0, (38.4, 21.6))
            assert width <= 38.4 * chart_viewer.SCREEN_WIDTH_FRACTION + 1e-9
            assert height <= 21.6 * chart_viewer.SCREEN_HEIGHT_FRACTION + 1e-9


def test_the_grid_aspect_is_preserved_so_panels_never_distort():
    aspect, rows, columns = 1.622, 2, 2
    width, height = chart_viewer.fit_dims(rows, columns, aspect, 1.0, (38.4, 21.6))
    assert width / height == pytest.approx((columns * aspect) / rows, rel=1e-9)


def test_scale_is_a_fraction_of_the_screen():
    full = chart_viewer.fit_dims(2, 2, 1.622, 1.0, (38.4, 21.6))
    half = chart_viewer.fit_dims(2, 2, 1.622, 0.5, (38.4, 21.6))
    assert half[0] == pytest.approx(full[0] / 2, rel=1e-9)
    assert half[1] == pytest.approx(full[1] / 2, rel=1e-9)


def test_no_screen_falls_back_to_fixed_inches():
    """A failed Tk probe must still give a usable window, sized for a laptop rather than a 4K panel."""
    width, height = chart_viewer.fit_dims(2, 2, 1.622, 1.0, None)
    assert width == pytest.approx(2 * chart_viewer.FALLBACK_PANEL_WIDTH_IN)
    assert 6.0 < height < 12.0


def test_a_panel_is_never_drawn_wider_than_the_png_behind_it():
    """The cap that makes the window shrink. One 1000px eval chart filled a 2858px panel here."""
    screen = (15.04, 8.46)                       # this laptop, 3008x1692 at dpi 200
    uncapped, _ = chart_viewer.fit_dims(1, 1, 2.083, 1.0, screen)
    capped, _ = chart_viewer.fit_dims(1, 1, 2.083, 1.0, screen, panel_width=5.0)
    assert uncapped > 14.0, 'without the cap the screen budget binds'
    assert capped == pytest.approx(5.0), 'with it, the chart\'s own width binds'


def test_the_natural_cap_scales_with_the_column_count():
    # It caps the *panel*, not the window: four columns of a 5-inch chart may be 20 inches wide, and
    # then the screen is what stops it.
    screen = (38.4, 21.6)
    width, _ = chart_viewer.fit_dims(2, 4, 2.083, 1.0, screen, panel_width=5.0)
    assert width == pytest.approx(20.0)


def test_the_screen_still_wins_when_the_charts_would_overflow_it():
    width, height = chart_viewer.fit_dims(2, 4, 2.083, 1.0, (15.04, 8.46), panel_width=5.0)
    assert width <= 15.04 * chart_viewer.SCREEN_WIDTH_FRACTION + 1e-9
    assert height <= 8.46 * chart_viewer.SCREEN_HEIGHT_FRACTION + 1e-9


def test_max_width_caps_growth_below_both_others():
    screen = (38.4, 21.6)
    width, _ = chart_viewer.fit_dims(2, 4, 2.083, 1.0, screen, panel_width=5.0, max_width=10.0)
    assert width == pytest.approx(10.0)


def test_max_width_never_grows_a_window():
    """It is a ceiling, not a request. A generous cap must leave a small window small."""
    width, _ = chart_viewer.fit_dims(1, 1, 2.083, 1.0, (15.04, 8.46),
                                     panel_width=5.0, max_width=100.0)
    assert width == pytest.approx(5.0)


def test_the_caps_preserve_the_grid_aspect_so_panels_never_distort():
    for panel_width, max_width in ((5.0, None), (None, 9.0), (5.0, 4.0), (None, None)):
        width, height = chart_viewer.fit_dims(2, 3, 2.083, 1.0, (38.4, 21.6),
                                              panel_width=panel_width, max_width=max_width)
        assert width / height == pytest.approx((3 * 2.083) / 2, rel=1e-9)


def test_a_failed_screen_probe_is_capped_too():
    """A probe that raised is not a reason to upscale a small chart."""
    width, _ = chart_viewer.fit_dims(1, 1, 2.083, 1.0, None, panel_width=2.0)
    assert width == pytest.approx(2.0)
    assert width < chart_viewer.FALLBACK_PANEL_WIDTH_IN


def test_panel_pixels_reads_width_and_height(tmp_path):
    import numpy as np
    import imageio.v2 as imageio
    path = os.path.join(str(tmp_path), 'chart.png')
    imageio.imwrite(path, np.zeros((480, 1000, 3), dtype=np.uint8))
    assert chart_viewer.panel_pixels(path) == (1000, 480)


def test_panel_pixels_is_none_on_a_torn_read(tmp_path):
    # The caller keeps its previous aspect and its previous cap rather than resizing to nonsense.
    assert chart_viewer.panel_pixels(touch(tmp_path, 'half-written.png', 1_000_000)) is None


def test_panel_aspect_is_read_from_the_image(tmp_path):
    import numpy as np
    import imageio.v2 as imageio
    path = os.path.join(str(tmp_path), 'chart.png')
    imageio.imwrite(path, np.zeros((450, 730, 3), dtype=np.uint8))
    assert chart_viewer.panel_aspect(path) == pytest.approx(730.0 / 450.0, rel=1e-6)


def test_panel_aspect_falls_back_on_a_torn_read(tmp_path):
    """A PNG caught mid-write must not resize the window to something absurd."""
    path = touch(tmp_path, 'half-written.png', 1_000_000)
    assert chart_viewer.panel_aspect(path, fallback=1.4) == 1.4


def test_the_render_dpi_doubles_on_retina_only():
    assert chart_viewer.viewer_dpi('darwin') == chart_viewer.HIDPI_DPI
    assert chart_viewer.viewer_dpi('linux') == 100
    assert chart_viewer.viewer_dpi('darwin', override=150) == 150


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


def test_follow_panels_reads_what_the_scheduler_wrote(tmp_path):
    import json
    status = str(tmp_path / '.status.json')
    with open(status, 'w') as handle:
        json.dump({'panels': ['runs/a.png', 'runs/b.png']}, handle)
    assert chart_viewer.follow_panels(status) == ['runs/a.png', 'runs/b.png']
    with open(status, 'w') as handle:
        json.dump({'panels': []}, handle)
    assert chart_viewer.follow_panels(status) == [], 'a file that says nothing is running says so'


def test_follow_panels_is_none_on_a_torn_or_missing_read_so_the_last_panels_stay(tmp_path):
    status = str(tmp_path / '.status.json')
    assert chart_viewer.follow_panels(status) is None
    with open(status, 'w') as handle:
        handle.write('{"panels": [')
    assert chart_viewer.follow_panels(status) is None


def test_parent_gone_is_the_exit_condition_for_a_followed_window():
    assert chart_viewer.parent_gone(os.getppid()) is False
    assert chart_viewer.parent_gone(-1) is True


def test_the_viewer_claims_no_slot_and_watches_no_pids():
    """What was deleted on 2026-09-05, pinned absent: the viewer draws what it is told and exits when
    its owner is gone. Every lifecycle rule it used to hold about other processes is the scheduler's."""
    import inspect
    source = inspect.getsource(chart_viewer)
    for gone in ('fcntl', 'take_window_slot', 'stand_by_for_slot', 'watch_pids', 'NEGATIVE_CHECKS',
                 'from tools import live_runs'):
        assert gone not in source, gone
