from backend.rules import (
    is_after_hours,
    restricted_entry_rule,
    temperature_rule
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