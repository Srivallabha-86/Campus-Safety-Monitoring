import cv2

from backend.vision import (
    open_video,
    process_frame,
    detect_motion,
    create_person_detector,
    detect_person,
)

from backend.event_engine import process_observation


WINDOW_NAME = "Campus Safety Monitor"

LOCATION = "Chemistry Lab"

# Temporary values.
# These will later come from real sensors.
RESTRICTED_ZONE = True
DOOR_OPEN = False
TEMPERATURE = 25.0


def close_camera(capture):
    """Release the camera and close OpenCV windows."""
    capture.release()
    cv2.destroyAllWindows()


# Open camera
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

cv2.moveWindow(
    WINDOW_NAME,
    100,
    100
)

# Create YOLO detector
person_detector = create_person_detector()

previous_frame = None


try:
    while True:

        # ---------------------------------------------
        # Read camera frame
        # ---------------------------------------------

        success, frame = capture.read()

        if not success:
            print("Could not read frame")
            break

        # ---------------------------------------------
        # Motion detection
        # ---------------------------------------------

        processed_frame = process_frame(frame)

        motion_detected = detect_motion(
            previous_frame,
            processed_frame
        )

        previous_frame = processed_frame

        # ---------------------------------------------
        # Person detection
        # ---------------------------------------------

        person_boxes = detect_person(
            frame,
            person_detector
        )

        person_detected = len(person_boxes) > 0

        # ---------------------------------------------
        # Safety Event Engine
        # ---------------------------------------------

        events = process_observation(
            person_detected=person_detected,
            restricted_zone=RESTRICTED_ZONE,
            door_open=DOOR_OPEN,
            motion_detected=motion_detected,
            location=LOCATION,
            temperature=TEMPERATURE,
        )

        # ---------------------------------------------
        # Draw person bounding boxes
        # ---------------------------------------------

        for (
            x1,
            y1,
            x2,
            y2,
            confidence,
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

        # ---------------------------------------------
        # Determine display status
        # ---------------------------------------------

        if events:

            highest_severity = "LOW"

            for event in events:

                if event.severity == "CRITICAL":
                    highest_severity = "CRITICAL"
                    break

                if event.severity == "HIGH":
                    highest_severity = "HIGH"

            status = f"SAFETY EVENT: {highest_severity}"
            status_color = (0, 0, 255)

        elif person_detected:

            status = "PERSON DETECTED"
            status_color = (0, 165, 255)

        else:

            status = "AREA CLEAR"
            status_color = (0, 255, 0)

        # ---------------------------------------------
        # Status background
        # ---------------------------------------------

        cv2.rectangle(
            frame,
            (0, 0),
            (500, 65),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            frame,
            status,
            (20, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA
        )

        # ---------------------------------------------
        # Display camera
        # ---------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        # ---------------------------------------------
        # Keyboard controls
        # ---------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # Q / q / ESC
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