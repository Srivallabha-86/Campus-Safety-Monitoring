from backend.rules import (
    is_after_hours,
    restricted_entry_rule,
    temperature_rule,
    door_open_rule,
    motion_detected_rule,
    extreme_temperature_rule,
)


def test_after_hours():
    assert is_after_hours(23)
    assert not is_after_hours(14)


def test_restricted_entry():
    assert restricted_entry_rule(True, True, 23)
    assert not restricted_entry_rule(True, True, 14)
    assert not restricted_entry_rule(True, False, 23)


def test_temperature():
    assert temperature_rule(40)
    assert temperature_rule(45)
    assert not temperature_rule(35)


def test_door_open():
    assert door_open_rule(True, 23)
    assert not door_open_rule(True, 14)
    assert not door_open_rule(False, 23)


def test_motion_detected():
    assert motion_detected_rule(True, True, 23)
    assert not motion_detected_rule(True, True, 14)
    assert not motion_detected_rule(True, False, 23)
    assert not motion_detected_rule(False, True, 23)


def test_extreme_temperature():
    assert extreme_temperature_rule(50)
    assert extreme_temperature_rule(60)
    assert not extreme_temperature_rule(45)