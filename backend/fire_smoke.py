
import cv2
from collections import deque


class FireSmokeProcessor:
    """
    Experimental fire/smoke detection with temporal validation.

    A candidate is registered when a class is detected in at
    least 2 of the most recent 5 frames.

    Candidates are displayed for evaluation only.
    They are NOT saved as safety incidents in SQLite.
    """

    def __init__(
        self,
        model,
        location="Campus Video Area",
        confidence=0.50,
        window_size=5,
        minimum_hits=2,
        clear_frames=10,
    ):
        self.model = model
        self.location = location
        self.confidence = confidence
        self.window_size = window_size
        self.minimum_hits = minimum_hits
        self.clear_frames = clear_frames

        self.classes = {
            name: {
                "history": deque(maxlen=window_size),
                "active": False,
                "missing_frames": 0,
            }
            for name in ("smoke", "fire")
        }

        # Counts experimental detection episodes, not confirmed incidents.
        self.alert_counts = {"fire": 0, "smoke": 0}

        model_names = self.model.names

        if isinstance(model_names, list):
            model_names = dict(enumerate(model_names))

        self.class_ids = {
            str(name).lower(): int(class_id)
            for class_id, name in model_names.items()
        }

        for required_class in ("smoke", "fire"):
            if required_class not in self.class_ids:
                raise ValueError(
                    f"Model is missing '{required_class}'. "
                    f"Available classes: {model_names}"
                )

    def process_frame(self, frame, display_frame=None):
        """
        Returns:
            annotated_frame, experimental_candidates

        Experimental candidates are not stored in SQLite.
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )

        output = (
            display_frame.copy()
            if display_frame is not None
            else frame.copy()
        )

        detected = {"smoke": False, "fire": False}
        max_confidence = {"smoke": 0.0, "fire": 0.0}

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                score = float(box.conf[0])

                class_name = next(
                    (
                        name
                        for name, value in self.class_ids.items()
                        if value == class_id
                    ),
                    None,
                )

                if class_name not in detected:
                    continue

                detected[class_name] = True
                max_confidence[class_name] = max(
                    max_confidence[class_name],
                    score,
                )

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = (
                    (0, 0, 255)
                    if class_name == "fire"
                    else (255, 165, 0)
                )

                cv2.rectangle(
                    output, (x1, y1), (x2, y2), color, 2
                )

                cv2.putText(
                    output,
                    f"EXPERIMENTAL {class_name.upper()} {score:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        candidates = []

        for class_name, state in self.classes.items():
            state["history"].append(detected[class_name])

            if detected[class_name]:
                state["missing_frames"] = 0
            else:
                state["missing_frames"] += 1

            hit_count = sum(state["history"])

            if hit_count >= self.minimum_hits and not state["active"]:
                candidates.append({
                    "event_type": f"{class_name}_candidate",
                    "location": self.location,
                    "confidence": max_confidence[class_name],
                    "frames_detected": hit_count,
                    "status": "EXPERIMENTAL - NOT A CONFIRMED INCIDENT",
                })

                state["active"] = True
                self.alert_counts[class_name] += 1

            if (
                state["active"]
                and state["missing_frames"] >= self.clear_frames
            ):
                state["active"] = False
                state["history"].clear()
                state["missing_frames"] = 0

        return output, candidates
