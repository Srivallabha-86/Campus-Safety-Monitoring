
from backend.zone import (
    is_inside_restricted_zone,
    get_zone_polygon,
    ZoneEntryTracker,
)


FRAME_SHAPE = (1080, 1920, 3)


def test_zone_polygon_has_four_points():
    polygon = get_zone_polygon(FRAME_SHAPE)

    assert polygon.shape == (4, 2)


def test_point_inside_right_zone():
    assert is_inside_restricted_zone(
        (1600, 500),
        FRAME_SHAPE,
    )


def test_point_outside_right_zone():
    assert not is_inside_restricted_zone(
        (500, 500),
        FRAME_SHAPE,
    )


def test_zone_entry_detected_once():
    tracker = ZoneEntryTracker()

    assert tracker.update(1, True, 0)
    assert not tracker.update(1, True, 1)


def test_zone_reentry_respects_cooldown():
    tracker = ZoneEntryTracker(cooldown_seconds=30)

    assert tracker.update(1, True, 0)

    tracker.update(1, False, 5)

    assert not tracker.update(1, True, 10)

    tracker.update(1, False, 30)

    assert tracker.update(1, True, 31)
