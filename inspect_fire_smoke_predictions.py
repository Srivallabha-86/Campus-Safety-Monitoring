
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "models/fire_smoke/best.pt"
VIDEO_DIR = Path("videos/dfire_samples")

model = YOLO(MODEL_PATH)

for video_name in ["VP2.mp4", "VP10.mp4"]:
    video_path = VIDEO_DIR / video_name

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        continue

    print(f"\nInspecting {video_name}...")

    model.predict(
        source=str(video_path),
        conf=0.25,
        save=True,
        project="runs/fire_smoke_review",
        name=video_path.stem,
        exist_ok=True,
        verbose=False
    )

    print(f"Completed: {video_name}")

print("\nPrediction review videos generated.")
