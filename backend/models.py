from dataclasses import dataclass
from datetime import datetime


@dataclass
class SafetyEvent:
    event_type: str
    location: str
    severity: str
    timestamp: datetime
    description: str