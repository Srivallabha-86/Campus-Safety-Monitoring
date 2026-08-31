import cv2

from backend.vision import (
    open_video,
    create_person_detector,
    detect_person,
)


WINDOW_NAME = "Person Detection"


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


try:
    while True:

        success, frame = capture.read()

        if not success:
            print("Could not read frame")
            break

        person_boxes = detect_person(
            frame,
            detector
        )

        # Draw detected people
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

            label = f"Person {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Overall status
        if person_boxes:
            status = "PERSON DETECTED"
            status_color = (0, 0, 255)
        else:
            status = "NO PERSON"
            status_color = (0, 255, 0)

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
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