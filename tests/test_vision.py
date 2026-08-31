import numpy as np

from backend.vision import (
    process_frame,
    detect_motion,
    create_person_detector,
    detect_person,
)


def test_process_frame_converts_to_grayscale():
    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    result = process_frame(frame)

    assert len(result.shape) == 2
    assert result.shape == (100, 100)


def test_no_motion_when_frames_are_identical():
    frame = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    assert not detect_motion(
        frame,
        frame
    )


def test_motion_when_frames_are_different():
    previous_frame = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    current_frame = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    # 80 x 80 = 6400 changed pixels
    current_frame[10:90, 10:90] = 255

    assert detect_motion(
        previous_frame,
        current_frame
    )


def test_small_change_is_not_motion():
    previous_frame = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    current_frame = previous_frame.copy()

    current_frame[10, 10] = 255

    assert not detect_motion(
        previous_frame,
        current_frame
    )

def test_create_person_detector():
    detector = create_person_detector()

    assert detector is not None

def test_detect_person_returns_list():
    detector = create_person_detector()

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    boxes = detect_person(
        frame,
        detector
    )

    assert isinstance(boxes, list)