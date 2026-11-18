
import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "models/fire_smoke/best.pt"
VIDEO_DIR = Path("videos/dfire_samples")
OUTPUT_DIR = Path("runs/fire_smoke_review/frames")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)

for video_name in ["VP2.mp4", "VP10.mp4"]:
    video_path = VIDEO_DIR / video_name
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Cannot open {video_name}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for percentage in [25, 50, 75]:
        frame_number = int(total_frames * percentage / 100)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(
            source=frame,
            conf=0.25,
            verbose=False
        )

        annotated = results[0].plot()

        output_path = OUTPUT_DIR / (
            f"{video_path.stem}_{percentage}.jpg"
        )

        cv2.imwrite(str(output_path), annotated)
        print("Saved:", output_path)

    cap.release()

print("\nAnnotated screenshots generated.")
