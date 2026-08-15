from datetime import datetime
from severity import calculate_severity
from models import SafetyEvent
from rules import (
    restricted_entry_rule,
    temperature_rule,
    extreme_temperature_rule,
    door_open_rule,
    motion_detected_rule
)


def process_observation(
    person_detected: bool,
    restricted_zone: bool,
    door_open: bool,
    motion_detected: bool,
    location: str,
    current_hour: int,
    temperature: float
) -> list[SafetyEvent]:

    events = []

    if restricted_entry_rule(
        person_detected,
        restricted_zone,
        current_hour
    ):
        events.append(
            SafetyEvent(
                event_type="restricted_entry",
                location=location,
                severity=calculate_severity(
                    person_detected=person_detected,
                    door_open=door_open,
                    motion_detected=motion_detected,
                    restricted_zone=restricted_zone,
                    after_hours=True,
                    temperature=temperature
                ),
                timestamp=datetime.now(),
                description="Person detected in restricted zone"
            )
        )

    if temperature_rule(temperature):
        events.append(
            SafetyEvent(
                event_type="high_temperature",
                location=location,
                severity=calculate_severity(
                    person_detected=person_detected,
                    door_open=door_open,
                    motion_detected=motion_detected,
                    restricted_zone=restricted_zone,
                    after_hours=True,
                    temperature=temperature
                ),
                timestamp=datetime.now(),
                description=f"High temperature detected: {temperature}°C"
            )
        )

    if extreme_temperature_rule(temperature):
        events.append(
            SafetyEvent(
                event_type="extreme_temperature",
                location=location,
                severity=calculate_severity(
                    person_detected=person_detected,
                    door_open=door_open,
                    motion_detected=motion_detected,
                    restricted_zone=restricted_zone,
                    after_hours=True,
                    temperature=temperature
                ),
                timestamp=datetime.now(),
                description=f"Extreme temperature detected: {temperature}°C"
            )
        )

    if door_open_rule(door_open, current_hour):
        events.append(
            SafetyEvent(
                event_type="door_open_after_hours",
                location=location,
                severity=calculate_severity(
                    person_detected=person_detected,
                    door_open=door_open,
                    motion_detected=motion_detected,
                    restricted_zone=restricted_zone,
                    after_hours=True,
                    temperature=temperature
                ),
                timestamp=datetime.now(),
                description=f"Door opened after hours {current_hour}"
            )
        )

    if motion_detected_rule(motion_detected, current_hour):
        events.append(
            SafetyEvent(
                event_type="motion_detected",
                location=location,
                severity=calculate_severity(
                    person_detected=person_detected,
                    door_open=door_open,
                    motion_detected=motion_detected,
                    restricted_zone=restricted_zone,
                    after_hours=True,
                    temperature=temperature
                ),
                timestamp=datetime.now(),
                description=f"Motion detected after hours {current_hour}"
            )
        )

    return events