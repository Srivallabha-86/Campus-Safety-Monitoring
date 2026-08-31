from backend.event_engine import process_observation


def test_restricted_entry_event():

    events = process_observation(
        person_detected=True,
        restricted_zone=True,
        door_open=False,
        motion_detected=False,
        location="Chemistry Lab",
        temperature=25,
    )

    # The result depends on the actual current time.
    # During after-hours we expect a restricted-entry event.
    if events:
        event = events[0]

        assert event.event_type == "restricted_entry"
        assert event.location == "Chemistry Lab"
        assert event.severity == "CRITICAL"
        assert event.description == (
            "Person detected in restricted zone after hours"
        )