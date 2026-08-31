import cv2
from datetime import datetime
from backend.vision import (
    open_video,
    create_person_detector,
    detect_person,
)
from backend.severity import calculate_event_severity
from backend.rules import (
    restricted_entry_rule,
    is_after_hours,
)
from backend.models import SafetyEvent

WINDOW_NAME = "Campus Safety - Person Monitoring"


def close_camera(capture):
    """Release the camera and close OpenCV windows."""
    capture.release()
    cv2.destroyAllWindows()


capture = open_video()

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    960,
    540
)

detector = create_person_detector()

# For this test, treat the camera view as a restricted zone.
restricted_zone = True

try:
    while True:

        success, frame = capture.read()

        if not success:
            print("Could not read frame")
            break

        # YOLO person detection
        person_boxes = detect_person(
            frame,
            detector
        )

        person_detected = len(person_boxes) > 0

        current_hour = datetime.now().hour

        after_hours = is_after_hours(
          current_hour
        )

        restricted_entry = restricted_entry_rule(
         person_detected,
         restricted_zone,
         current_hour
        )
        
        severity = calculate_event_severity(
         event_type="restricted_entry",
         temperature=25,
         person_detected=person_detected,
         restricted_zone=restricted_zone,
         after_hours=after_hours,
        )

        if restricted_entry:

            event = SafetyEvent(
             event_type="restricted_entry",
             location="Chemistry Lab",
             severity=severity,
              timestamp=datetime.now(),
             description="Person detected in restricted zone after hours",
            )

            print("\nSAFETY EVENT DETECTED")
            print(f"Type: {event.event_type}")
            print(f"Location: {event.location}")
            print(f"Severity: {event.severity}")
            print(f"Time: {event.timestamp}")
            print(f"Description: {event.description}")

        # Draw person boxes
        for (
            x1,
            y1,
            x2,
            y2,
            confidence
        ) in person_boxes:

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Person {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Safety status
        if restricted_entry:

         status = f"CRITICAL: RESTRICTED ENTRY"
         status_color = (0, 0, 255)

        elif person_detected:

         status = f"PERSON DETECTED | {severity}"
         status_color = (0, 165, 255)

        else:

         status = "AREA CLEAR"
         status_color = (0, 255, 0)

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        # Q / q / ESC
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q") or key == 27:
            break

        # X button
        try:

            visible = cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_VISIBLE
            )

            if visible < 1:
                break

        except cv2.error:
            break


except KeyboardInterrupt:

    print("\nStopping camera...")


finally:

    close_camera(capture)

    print("Camera closed.")