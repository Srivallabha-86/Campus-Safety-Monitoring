from datetime import datetime


def is_after_hours(hour: int) -> bool:
    return hour < 6 or hour >= 18


def restricted_entry_rule(
    person_detected: bool,
    restricted_zone: bool,
    current_hour: int
) -> bool:

    return (
        person_detected
        and restricted_zone
        and is_after_hours(current_hour)
    )

def temperature_rule(temperature: float) -> bool:
    return temperature >= 38 and temperature < 50

def door_open_rule(door_open: bool, current_hour: int) -> bool:
    return door_open and is_after_hours(current_hour)

def motion_detected_rule(motion_detected: bool, current_hour: int) -> bool:
    return motion_detected and is_after_hours(current_hour)

def extreme_temperature_rule(temperature: float) -> bool:
    return temperature >= 50