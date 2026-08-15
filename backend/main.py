from datetime import datetime

from event_engine import process_observation


def main():

    events = process_observation(
        person_detected=True,
        restricted_zone=True,
        location="Chemistry Lab",
        current_hour=datetime.now().hour,
        temperature=28,
        door_open=True,
        motion_detected=True
    )

    if not events:
        print("No safety events detected.")
        return

    for event in events:
        print("\n🚨 SAFETY ALERT")
        print(f"Type: {event.event_type}")
        print(f"Location: {event.location}")
        print(f"Severity: {event.severity}")
        print(f"Time: {event.timestamp}")
        print(f"Description: {event.description}")


if __name__ == "__main__":
    main()