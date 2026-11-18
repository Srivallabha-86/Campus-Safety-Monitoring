import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path("models/fire_smoke/best.pt")
VIDEO_PATH = Path("videos/fire_smoke_test.mp4")
CONFIDENCE = 0.40


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Fire/smoke test video not found: {VIDEO_PATH}\n"
            "Place a suitable fire/smoke video at this location."
        )

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    detection_counts = {
        int(class_id): 0 for class_id in model.names
    }

    frame_count = 0

    print("Starting Fire and Smoke Detection...")
    print(f"Video: {VIDEO_PATH}")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            results = model.predict(
                frame,
                conf=CONFIDENCE,
                verbose=False
            )

            result = results[0]
            frame_count += 1

            if result.boxes is not None:
                for class_id in result.boxes.cls.tolist():
                    detection_counts[int(class_id)] += 1

            annotated_frame = result.plot()

            cv2.imshow(
                "Campus Safety - Fire and Smoke Detection",
                annotated_frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Detection stopped by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\nDetection Summary")
    print("--------------------------")
    print("Frames Processed:", frame_count)

    for class_id, count in detection_counts.items():
        class_name = model.names[class_id]
        print(f"{class_name}: {count} detections")

    print("\nFire and Smoke Video Test Completed!")


if __name__ == "__main__":
    main()