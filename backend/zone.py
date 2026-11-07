
import cv2
import numpy as np


# Normalized coordinates: x and y range from 0.0 to 1.0.
# This defines the right-hand demonstration area.
RESTRICTED_ZONE = [
    (0.70, 0.15),
    (0.99, 0.15),
    (0.99, 0.92),
    (0.70, 0.92),
]


def get_zone_polygon(frame_shape):
    height, width = frame_shape[:2]

    return np.array(
        [
            (int(x * width), int(y * height))
            for x, y in RESTRICTED_ZONE
        ],
        dtype=np.int32,
    )


def is_inside_restricted_zone(point, frame_shape):
    polygon = get_zone_polygon(frame_shape)

    result = cv2.pointPolygonTest(
        polygon,
        (float(point[0]), float(point[1])),
        False,
    )

    return result >= 0


def draw_restricted_zone(frame):
    polygon = get_zone_polygon(frame.shape)

    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        [polygon],
        (0, 0, 255),
    )

    frame = cv2.addWeighted(
        overlay,
        0.12,
        frame,
        0.88,
        0,
    )

    cv2.polylines(
        frame,
        [polygon],
        True,
        (0, 0, 255),
        3,
    )

    cv2.putText(
        frame,
        "RESTRICTED ZONE",
        (polygon[0][0] + 10, polygon[0][1] + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    return frame


class ZoneEntryTracker:

    def __init__(self, cooldown_seconds=30):
        self.inside_state = {}
        self.last_alert_time = {}
        self.cooldown_seconds = cooldown_seconds

    def update(self, track_id, inside, video_seconds):
        previous_inside = self.inside_state.get(
            track_id,
            False,
        )

        self.inside_state[track_id] = inside

        entered = not previous_inside and inside

        if not entered:
            return False

        last_alert = self.last_alert_time.get(
            track_id,
            float("-inf"),
        )

        if video_seconds - last_alert < self.cooldown_seconds:
            return False

        self.last_alert_time[track_id] = video_seconds

        return True
