
import sqlite3
from pathlib import Path

database_path = Path("campus_safety.db").resolve()

connection = sqlite3.connect(database_path)
cursor = connection.cursor()

print("DATABASE:", database_path)

total = cursor.execute(
    "SELECT COUNT(*) FROM events"
).fetchone()[0]

today = cursor.execute(
    "SELECT COUNT(*) FROM events "
    "WHERE date(timestamp) = date('now', 'localtime')"
).fetchone()[0]

print("\nTOTAL EVENTS:", total)
print("TODAY EVENTS:", today)

print("\nLATEST 10 EVENTS:")

events = cursor.execute("""
    SELECT id, event_type, location, severity,
           timestamp, description
    FROM events
    ORDER BY id DESC
    LIMIT 10
""").fetchall()

for event in events:
    print(event)

connection.close()
