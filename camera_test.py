import cv2

from backend.vision import (
    open_video,
    process_frame,
    detect_motion,
)


WINDOW_NAME = "Campus Safety Monitor"


def close_camera(capture):
    """Release the camera and close all OpenCV windows."""
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

previous_frame = None

try:
    while True:
        success, frame = capture.read()

        if not success:
            print("Could not read frame")
            break

        # Process the frame internally
        processed_frame = process_frame(frame)

        # Detect motion
        motion_detected = detect_motion(
            previous_frame,
            processed_frame
        )

        previous_frame = processed_frame

        # Create a fresh copy of the original color frame
        display_frame = frame.copy()

        # Determine current status
        if motion_detected:
            status = "MOTION DETECTED"
            text_color = (0, 0, 255)
        else:
            status = "NO MOTION"
            text_color = (0, 128, 0)


        # Draw ONLY the current status
        cv2.putText(
            display_frame,
            status,
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2,
            cv2.LINE_AA
        )

        # Show the updated frame
        cv2.imshow(
            WINDOW_NAME,
            display_frame
        )

        # Process keyboard events
        key = cv2.waitKey(10) & 0xFF

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
