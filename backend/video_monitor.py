
import math
from datetime import datetime

from backend.zone import (
    is_inside_restricted_zone,
    draw_restricted_zone,
    ZoneEntryTracker,
)
from backend.event_engine import record_visual_event
from backend.rules import is_after_hours


# COCO class IDs supported by the standard YOLO11n weights.
BAG_CLASSES = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}


class VideoEventProcessor:

    def __init__(
        self,
        location="Right-side demonstration zone",
        stationary_seconds=10,
    ):
        self.location = location
        self.stationary_seconds = stationary_seconds

        self.zone_tracker = ZoneEntryTracker(
            cooldown_seconds=30,
        )

        self.bag_states = {}
        self.tracked_people = {}

        self.zone_entries = 0
        self.unattended_bags = 0
        self.detected_bags = 0
        self.bag_tracks_seen = set() 

    def process_result(self, result, video_seconds):

        frame = result.orig_img.copy()
        height, width = frame.shape[:2]

        diagonal = math.hypot(width, height)

        # Scale proximity and movement thresholds to resolution.
        proximity_threshold = diagonal * 0.08
        movement_threshold = max(20, diagonal * 0.012)

        people = []
        bags = []

        boxes = result.boxes

        if boxes is not None and boxes.id is not None:

            coordinates = boxes.xyxy.cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            track_ids = boxes.id.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for coordinates_box, class_id, track_id, confidence in zip(
                coordinates,
                classes,
                track_ids,
                confidences,
            ):

                x1, y1, x2, y2 = coordinates_box

                center = (
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                )

                if class_id == 0:

                    # Use the bottom-center point (feet).
                    foot_point = (
                        (x1 + x2) / 2,
                        y2,
                    )

                    inside = is_inside_restricted_zone(
                        foot_point,
                        frame.shape,
                    )

                    people.append(center)

                    self.tracked_people[track_id] = {
                        "Track ID": track_id,
                        "Class": "Person",
                        "Confidence": confidence,
                    }

                    entered = self.zone_tracker.update(
                        track_id,
                        inside,
                        video_seconds,
                    )

                    if entered:
                        self.zone_entries += 1

                        # Preserve the existing after-hours rule.
                        if is_after_hours(
                            datetime.now().hour
                        ):
                            record_visual_event(
                                event_type="restricted_entry",
                                location=self.location,
                                severity="CRITICAL",
                                description=(
                                    f"Person with Track ID {track_id} "
                                    "entered the restricted zone after hours."
                                ),
                            )

                elif class_id in BAG_CLASSES:
                    self.detected_bags += 1
                    self.bag_tracks_seen.add(track_id)

                    bags.append(
                        (
                            track_id,
                            center,
                            BAG_CLASSES[class_id],
                        )
                    )

        # Draw the restricted-zone overlay.
        annotated = result.plot()

        annotated = draw_restricted_zone(annotated)

        # Check each tracked bag for stationary behaviour.
        for track_id, center, bag_name in bags:

            state = self.bag_states.get(track_id)

            if state is None:
                state = {
                    "center": center,
                    "stationary_since": video_seconds,
                    "last_person_near": video_seconds,
                    "alerted": False,
                }

                self.bag_states[track_id] = state

            else:

                movement = math.dist(
                    center,
                    state["center"],
                )

                if movement > movement_threshold:
                    state["center"] = center
                    state["stationary_since"] = video_seconds
                    state["last_person_near"] = video_seconds
                    state["alerted"] = False

            # Find the nearest detected person.
            nearest_distance = min(
                (
                    math.dist(center, person)
                    for person in people
                ),
                default=float("inf"),
            )

            if nearest_distance < proximity_threshold:
                state["last_person_near"] = video_seconds

            stationary_for = (
                video_seconds - state["stationary_since"]
            )

            unattended_for = (
                video_seconds - state["last_person_near"]
            )

            if (
                stationary_for >= self.stationary_seconds
                and unattended_for >= self.stationary_seconds
                and not state["alerted"]
            ):

                record_visual_event(
                    event_type="unattended_baggage",
                    location=self.location,
                    severity="HIGH",
                    description=(
                        f"Potential unattended {bag_name} detected. "
                        f"Track ID: {track_id}."
                    ),
                )

                state["alerted"] = True
                self.unattended_bags += 1

        return annotated
