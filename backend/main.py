from backend.event_engine import process_observation
from backend.database import create_database


def main():

    # Initialize SQLite database.
    create_database()

    events = process_observation(
        person_detected=True,
        restricted_zone=True,
        door_open=True,
        motion_detected=True,
        location="Chemistry Lab",
        temperature=42,
    )

    if not events:
        print("No safety events detected.")
        return

    print(f"\n{len(events)} safety event(s) detected.")

    for event in events:
        print("\n🚨 SAFETY EVENT")
        print(f"Type: {event.event_type}")
        print(f"Location: {event.location}")
        print(f"Severity: {event.severity}")
        print(f"Time: {event.timestamp}")
        print(f"Description: {event.description}")


if __name__ == "__main__":
    main()