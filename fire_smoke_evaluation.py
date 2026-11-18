
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "models/fire_smoke/best.pt"
VIDEO_DIR = Path("videos/dfire_samples")

THRESHOLDS = [0.40, 0.50, 0.60]
BASE_CONFIDENCE = min(THRESHOLDS)

model = YOLO(MODEL_PATH)

print("Loaded model classes:", model.names)

video_files = sorted(VIDEO_DIR.glob("*.mp4"))

if not video_files:
    raise FileNotFoundError("No test videos found!")

results_summary = {}

for threshold in THRESHOLDS:
    results_summary[threshold] = []

print("\nStarting Fire and Smoke Evaluation...")
print("=" * 65)

for video_path in video_files:

    category = video_path.stem[:2]

    if category not in ("FP", "VP"):
        print("Skipping unknown category:", video_path.name)
        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open:", video_path.name)
        continue

    frame_count = 0
    start_time = time.perf_counter()

    # Store detections for each confidence threshold
    video_results = {
        threshold: {
            "smoke_frames": 0,
            "fire_frames": 0,
            "smoke_boxes": 0,
            "fire_boxes": 0,
        }
        for threshold in THRESHOLDS
    }

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        predictions = model.predict(
            frame,
            conf=BASE_CONFIDENCE,
            verbose=False
        )

        frame_count += 1
        boxes = predictions[0].boxes

        if boxes is None or len(boxes) == 0:
            continue

        classes = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        for threshold in THRESHOLDS:

            detected_classes = []

            for class_id, confidence in zip(classes, confidences):

                if confidence < threshold:
                    continue

                class_name = model.names[int(class_id)].lower()

                if class_name in ("smoke", "fire"):
                    detected_classes.append(class_name)

                    if class_name == "smoke":
                        video_results[threshold]["smoke_boxes"] += 1

                    elif class_name == "fire":
                        video_results[threshold]["fire_boxes"] += 1

            if "smoke" in detected_classes:
                video_results[threshold]["smoke_frames"] += 1

            if "fire" in detected_classes:
                video_results[threshold]["fire_frames"] += 1

    cap.release()

    elapsed = time.perf_counter() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    print(f"\nVideo: {video_path.name}")
    print("Category:", category)
    print("Frames:", frame_count)
    print(f"Processing FPS: {fps:.2f}")

    for threshold in THRESHOLDS:

        data = video_results[threshold]

        has_detection = (
            data["smoke_frames"] > 0
            or data["fire_frames"] > 0
        )

        results_summary[threshold].append({
            "video": video_path.name,
            "category": category,
            "detected": has_detection,
            **data
        })

        print(
            f"  Confidence {threshold:.2f}: "
            f"Smoke frames={data['smoke_frames']}, "
            f"Fire frames={data['fire_frames']}"
        )

print("\n\nFINAL EVALUATION SUMMARY")
print("=" * 65)

for threshold in THRESHOLDS:

    results = results_summary[threshold]

    fp_videos = [r for r in results if r["category"] == "FP"]
    vp_videos = [r for r in results if r["category"] == "VP"]

    false_alarms = sum(r["detected"] for r in fp_videos)
    detected_positives = sum(r["detected"] for r in vp_videos)

    print(f"\nConfidence Threshold: {threshold:.2f}")

    print(
        f"Negative videos with detections: "
        f"{false_alarms}/{len(fp_videos)}"
    )

    print(
        f"Positive videos with detections: "
        f"{detected_positives}/{len(vp_videos)}"
    )

    print(
        f"Positive video detection rate: "
        f"{detected_positives / len(vp_videos) * 100:.1f}%"
        if vp_videos else "No positive videos"
    )

print("\nEvaluation completed!")
