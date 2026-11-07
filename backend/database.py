import sqlite3

from backend.models import SafetyEvent


DATABASE_NAME = "campus_safety.db"


def create_database():
    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            location TEXT NOT NULL,
            severity TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            description TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def insert_event(event: SafetyEvent):
    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO events (
            event_type,
            location,
            severity,
            timestamp,
            description
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        event.event_type,
        event.location,
        event.severity,
        event.timestamp.isoformat(),
        event.description,
    ))

    connection.commit()
    connection.close()


def get_all_events():
    connection = sqlite3.connect(DATABASE_NAME)

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM events
        ORDER BY id DESC
    """)

    events = cursor.fetchall()

    connection.close()

    return [dict(event) for event in events]
