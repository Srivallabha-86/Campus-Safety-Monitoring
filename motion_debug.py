import cv2

from backend.vision import open_video, process_frame


WINDOW_NAME = "Motion Debug"

capture = open_video()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

previous_frame = None

try:
    while True:
        success, frame = capture.read()

        if not success:
            print("Could not read frame")
            break

        current_frame = process_frame(frame)

        if previous_frame is None:
            motion_pixels = 0
        else:
            difference = cv2.absdiff(
                previous_frame,
                current_frame
            )

            _, thresholded = cv2.threshold(
                difference,
                30,
                255,
                cv2.THRESH_BINARY
            )

            motion_pixels = cv2.countNonZero(thresholded)

        previous_frame = current_frame

        # Show the actual measured difference
        text = f"Changed pixels: {motion_pixels}"

        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            WINDOW_NAME,
            display_frame
        )

        key = cv2.waitKey(10) & 0xFF

        if key == ord("q") or key == ord("Q") or key == 27:
            break

        try:
            visible = cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_VISIBLE
            )

            if visible < 1:
                break

        except cv2.error:
            break

finally:
    capture.release()
    cv2.destroyAllWindows()
    print("Camera closed.")