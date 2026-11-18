
from types import SimpleNamespace

import numpy as np

from backend.fire_smoke import FireSmokeProcessor


class FakeBox:
    def __init__(self, class_id, confidence=0.90):
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = [[20, 20, 100, 100]]


class FakeModel:
    names = {0: "smoke", 1: "fire"}

    def __init__(self, detections=None):
        self.detections = detections or []

    def predict(self, **kwargs):
        return [SimpleNamespace(boxes=self.detections)]


def test_smoke_requires_two_detection_frames():
    model = FakeModel([FakeBox(0)])
    processor = FireSmokeProcessor(model)
    frame = np.zeros((150, 150, 3), dtype=np.uint8)

    _, first = processor.process_frame(frame)
    _, second = processor.process_frame(frame)

    assert first == []
    assert len(second) == 1
    assert second[0]["event_type"] == "smoke_candidate"
    assert "EXPERIMENTAL" in second[0]["status"]


def test_fire_generates_experimental_candidate():
    model = FakeModel([FakeBox(1)])
    processor = FireSmokeProcessor(model)
    frame = np.zeros((150, 150, 3), dtype=np.uint8)

    processor.process_frame(frame)
    _, candidates = processor.process_frame(frame)

    assert len(candidates) == 1
    assert candidates[0]["event_type"] == "fire_candidate"


def test_continuous_smoke_does_not_duplicate_candidates():
    model = FakeModel([FakeBox(0)])
    processor = FireSmokeProcessor(model)
    frame = np.zeros((150, 150, 3), dtype=np.uint8)

    all_candidates = []

    for _ in range(20):
        _, candidates = processor.process_frame(frame)
        all_candidates.extend(candidates)

    assert len(all_candidates) == 1


def test_new_candidate_after_previous_detection_clears():
    model = FakeModel([FakeBox(0)])
    processor = FireSmokeProcessor(model)
    frame = np.zeros((150, 150, 3), dtype=np.uint8)

    processor.process_frame(frame)
    _, first = processor.process_frame(frame)

    assert len(first) == 1

    model.detections = []

    for _ in range(10):
        processor.process_frame(frame)

    model.detections = [FakeBox(0)]

    processor.process_frame(frame)
    _, second = processor.process_frame(frame)

    assert len(second) == 1


def test_fire_and_smoke_have_separate_states():
    model = FakeModel([FakeBox(0)])
    processor = FireSmokeProcessor(model)
    frame = np.zeros((150, 150, 3), dtype=np.uint8)

    processor.process_frame(frame)
    _, candidates = processor.process_frame(frame)

    assert len(candidates) == 1
    assert processor.alert_counts["smoke"] == 1
    assert processor.alert_counts["fire"] == 0
