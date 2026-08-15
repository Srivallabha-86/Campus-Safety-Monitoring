def calculate_severity(
    person_detected: bool,
    door_open: bool,
    motion_detected: bool,
    restricted_zone: bool,
    after_hours: bool,
    temperature: float
) -> str:

    if (
        person_detected
        and restricted_zone
        and after_hours
    ):
        return "CRITICAL"

    if (
        door_open
        and after_hours
    ):
        return "HIGH"

    if (
        motion_detected
        and after_hours
    ):
        return "CRITICAL"

    if temperature >= 50:
        return "CRITICAL"

    if temperature >= 38 and temperature < 50:
        return "HIGH"

    return "LOW"