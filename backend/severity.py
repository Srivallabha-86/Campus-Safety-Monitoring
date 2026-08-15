def calculate_event_severity(
    event_type: str,
    temperature: float,
    person_detected: bool,
    restricted_zone: bool,
    after_hours: bool,
) -> str:

    if event_type == "restricted_entry":
        if person_detected and restricted_zone and after_hours:
            return "CRITICAL"

    if event_type == "extreme_temperature":
        if temperature >= 50:
            return "CRITICAL"

    if event_type == "high_temperature":
        if 38 <= temperature < 50:
            return "HIGH"

    if event_type == "door_open_after_hours":
        if after_hours:
            return "HIGH"

    if event_type == "motion_detected":
        if restricted_zone and after_hours:
            return "HIGH"

    return "LOW"