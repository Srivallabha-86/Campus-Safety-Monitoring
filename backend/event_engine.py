from datetime import datetime

from severity import calculate_event_severity
from models import SafetyEvent
from rules import (
    is_after_hours,
    restricted_entry_rule,
    temperature_rule,
    extreme_temperature_rule,
    door_open_rule,
    motion_detected_rule,
)


def process_observation(
    person_detected: bool,
    restricted_zone: bool,
    door_open: bool,
    motion_detected: bool,
    location: str,
    temperature: float,
) -> list[SafetyEvent]:

    events = []

    # Get the actual current timestamp.
    timestamp = datetime.now()

    # Extract the actual current hour.
    current_hour = timestamp.hour

    # Determine whether it is after hours.
    after_hours = is_after_hours(current_hour)

    # 1. Restricted entry
    if restricted_entry_rule(
        person_detected,
        restricted_zone,
        current_hour,
    ):
        severity = calculate_event_severity(
            event_type="restricted_entry",
            temperature=temperature,
            person_detected=person_detected,
            restricted_zone=restricted_zone,
            after_hours=after_hours,
        )

        events.append(
            SafetyEvent(
                event_type="restricted_entry",
                location=location,
                severity=severity,
                timestamp=timestamp,
                description=(
                    "Person detected in restricted zone "
                    "after hours"
                ),
            )
        )

    # 2. High temperature
    if temperature_rule(temperature):
        severity = calculate_event_severity(
            event_type="high_temperature",
            temperature=temperature,
            person_detected=person_detected,
            restricted_zone=restricted_zone,
            after_hours=after_hours,
        )

        events.append(
            SafetyEvent(
                event_type="high_temperature",
                location=location,
                severity=severity,
                timestamp=timestamp,
                description=f"High temperature detected: {temperature}°C",
            )
        )

    # 3. Extreme temperature
    if extreme_temperature_rule(temperature):
        severity = calculate_event_severity(
            event_type="extreme_temperature",
            temperature=temperature,
            person_detected=person_detected,
            restricted_zone=restricted_zone,
            after_hours=after_hours,
        )

        events.append(
            SafetyEvent(
                event_type="extreme_temperature",
                location=location,
                severity=severity,
                timestamp=timestamp,
                description=f"Extreme temperature detected: {temperature}°C",
            )
        )

    # 4. Door opened after hours
    if door_open_rule(
        door_open,
        current_hour,
    ):
        severity = calculate_event_severity(
            event_type="door_open_after_hours",
            temperature=temperature,
            person_detected=person_detected,
            restricted_zone=restricted_zone,
            after_hours=after_hours,
        )

        events.append(
            SafetyEvent(
                event_type="door_open_after_hours",
                location=location,
                severity=severity,
                timestamp=timestamp,
                description=(
                    f"Door opened after hours "
                    f"at {current_hour}:00"
                ),
            )
        )

    # 5. Motion detected in restricted zone after hours
    if motion_detected_rule(
        motion_detected,
        restricted_zone,
        current_hour,
    ):
        severity = calculate_event_severity(
            event_type="motion_detected",
            temperature=temperature,
            person_detected=person_detected,
            restricted_zone=restricted_zone,
            after_hours=after_hours,
        )

        events.append(
            SafetyEvent(
                event_type="motion_detected",
                location=location,
                severity=severity,
                timestamp=timestamp,
                description=(
                    "Motion detected in restricted zone "
                    f"after hours at {current_hour}:00"
                ),
            )
        )

    return events