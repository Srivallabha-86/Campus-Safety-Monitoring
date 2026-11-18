
import sqlite3
import random
from datetime import datetime, timedelta

DB_PATH = "campus_safety.db"

demo_events = [
    ("motion_detected", "Library", "Routine movement detected"),
    ("door_activity", "Main Building", "Normal door activity observed"),
    ("temperature_normal", "Chemistry Lab", "Temperature within normal range"),
    ("motion_detected", "Parking Area", "Routine pedestrian movement"),
    ("campus_activity", "Academic Block", "Non-critical campus activity"),
    ("motion_detected", "Cafeteria", "Normal movement in common area"),
    ("door_activity", "Computer Lab", "Routine door opening activity"),
    ("temperature_normal", "Server Room", "Temperature observation within normal range"),
    ("motion_detected", "Sports Ground", "Routine movement detected"),
    ("campus_activity", "Administration Block", "Routine campus activity"),
    ("motion_detected", "Corridor A", "Normal corridor movement"),
    ("door_activity", "Library", "Routine entrance activity"),
    ("temperature_normal", "Physics Lab", "Normal temperature observation"),
    ("motion_detected", "Auditorium", "Routine pedestrian movement"),
    ("campus_activity", "Garden Area", "Non-critical campus activity"),
    ("motion_detected", "Hostel Entrance", "Routine movement near entrance"),
    ("door_activity", "Seminar Hall", "Normal door activity"),
    ("temperature_normal", "Electrical Lab", "Temperature within normal range"),
    ("motion_detected", "Reception", "Normal visitor movement"),
    ("campus_activity", "Parking Area", "Routine parking area activity"),
    ("motion_detected", "Block B Corridor", "Normal pedestrian movement"),
    ("door_activity", "Staff Room", "Routine door activity"),
    ("temperature_normal", "Workshop", "Normal temperature observation"),
    ("motion_detected", "Entrance Gate", "Routine pedestrian activity"),
    ("campus_activity", "Open Campus Area", "Non-critical movement observed"),
    ("motion_detected", "Reading Room", "Normal movement detected"),
    ("door_activity", "Laboratory Block", "Routine door activity"),
    ("temperature_normal", "Biology Lab", "Temperature within normal range"),
    ("motion_detected", "Central Courtyard", "Routine campus movement"),
    ("campus_activity", "Main Building", "Normal campus activity"),
]

def insert_demo_incidents():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    try:
        now = datetime.now()

        for index, event in enumerate(demo_events):
            event_type, location, description = event

            # Spread timestamps across the previous 30 days.
            timestamp = now - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            cursor.execute(
                """
                INSERT INTO events
                (event_type, location, severity, timestamp, description)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    location,
                    "LOW",
                    timestamp.isoformat(),
                    f"MANUAL DEMO EVENT: {description}"
                )
            )

        connection.commit()

        print(f"Successfully inserted {len(demo_events)} LOW-severity demo incidents.")

        print("\nUpdated severity distribution:")
        cursor.execute("""
            SELECT severity, COUNT(*)
            FROM events
            GROUP BY severity
            ORDER BY severity
        """)

        for severity, count in cursor.fetchall():
            print(f"{severity}: {count}")

        print("\nTotal database records:")
        cursor.execute("SELECT COUNT(*) FROM events")
        print(cursor.fetchone()[0])

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    insert_demo_incidents()
