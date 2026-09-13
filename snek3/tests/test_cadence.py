from desktop.daemon import cadence

TEN = 600


def test_slot_is_the_last_instant_at_the_offset_minute():
    # 12:17:30 -> the :08 slot at 12:08:00; 12:07:59 -> 11:58:00
    assert cadence.slot(12 * 3600 + 17 * 60 + 30, TEN, 8 * 60) == 12 * 3600 + 8 * 60
    assert cadence.slot(12 * 3600 + 7 * 60 + 59, TEN, 8 * 60) == 11 * 3600 + 58 * 60
    assert cadence.slot(12 * 3600 + 8 * 60, TEN, 8 * 60) == 12 * 3600 + 8 * 60


def test_due_once_per_slot_boundary_and_immediately_when_never_published():
    assert cadence.due(None, 5.0, TEN, 480)
    t = 12 * 3600 + 8 * 60            # published at exactly :08:00
    assert not cadence.due(t, t + 20, TEN, 480)
    assert not cadence.due(t, t + 599, TEN, 480)
    assert cadence.due(t, t + 600, TEN, 480)
    # published a few seconds late, after the boundary: not due again until the next one
    assert not cadence.due(t + 15, t + 590, TEN, 480)
    assert cadence.due(t + 15, t + 601, TEN, 480)


def test_the_laptop_and_desktop_slots_are_a_minute_apart_in_the_right_order():
    now = 13 * 3600 + 9 * 60 + 10     # 13:09:10
    assert cadence.slot(now, TEN, 8 * 60) == 13 * 3600 + 8 * 60
    assert cadence.slot(now, TEN, 9 * 60) == 13 * 3600 + 9 * 60


def test_zero_period_is_always_due():
    assert cadence.due(100.0, 101.0, 0, 0)
