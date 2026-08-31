import cv2
from ultralytics import YOLO

def open_video(source=0):
    """
    Open a camera or video source.

    source=0 means the default webcam.
    """
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError("Could not open video source")

    return capture


def process_frame(frame):
    """
    Convert a camera frame to grayscale.

    The original frame is not modified.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def detect_motion(previous_frame, current_frame, threshold=25):
    """
    Detect significant motion between two grayscale frames.

    Returns:
        True  -> motion detected
        False -> no significant motion
    """

    if previous_frame is None:
        return False

    difference = cv2.absdiff(
        previous_frame,
        current_frame
    )

    _, thresholded = cv2.threshold(
        difference,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    motion_pixels = cv2.countNonZero(thresholded)

    return motion_pixels > 1000

def create_person_detector(model_path="yolo11n.pt"):
    """
    Create a YOLO person detector.
    """
    return YOLO(model_path)


def detect_person(frame, detector, confidence=0.5):
    """
    Detect people in a frame.

    Returns:
        List of bounding boxes for detected people.
    """

    results = detector(
        frame,
        verbose=False
    )

    person_boxes = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            score = float(box.conf[0])

            # COCO class 0 = person
            if class_id == 0 and score >= confidence:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                person_boxes.append(
                    (x1, y1, x2, y2, score)
                )

    return person_boxes