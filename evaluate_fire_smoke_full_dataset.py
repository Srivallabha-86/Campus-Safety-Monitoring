import csv
from pathlib import Path
import cv2
from ultralytics import YOLO

WEIGHTS = Path("runs/dfire_training/yolo11n_fire_smoke/weights/best.pt")
VIDEO_DIR = Path("videos/dfire_samples")
OUTPUT = Path("reports/fire_smoke_video_evaluation.csv")
CONF = 0.50
STRIDE = 2
MIN_POSITIVE_FRAMES = 2

def actual_label(path):
    name = path.stem.upper()
    if name.startswith("FP"): return 0
    if name.startswith("VP"): return 1
    return None

def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Missing trained weights: {WEIGHTS}")
    videos = sorted(p for p in VIDEO_DIR.glob("*.mp4") if actual_label(p) is not None)
    if not videos:
        raise FileNotFoundError(f"No FP/VP videos found in {VIDEO_DIR}")
    model = YOLO(str(WEIGHTS))
    rows = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"Skipping unreadable video: {video.name}")
            continue
        frame_no = sampled = positives = box_count = 0
        max_conf = 0.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok: break
                if frame_no % STRIDE == 0:
                    sampled += 1
                    boxes = model.predict(frame, conf=CONF, imgsz=640, verbose=False)[0].boxes
                    count = 0 if boxes is None else len(boxes)
                    if count:
                        positives += 1
                        box_count += count
                        max_conf = max(max_conf, max(boxes.conf.tolist(), default=0.0))
                frame_no += 1
        finally:
            cap.release()
        pred, actual = int(positives >= MIN_POSITIVE_FRAMES), actual_label(video)
        row = {"video": video.name, "actual_positive": actual, "predicted_positive": pred,
               "sampled_frames": sampled, "positive_frames": positives,
               "candidate_boxes": box_count, "max_confidence": round(max_conf, 4)}
        rows.append(row)
        print(f"{video.name}: actual={actual}, predicted={pred}, positive_frames={positives}, boxes={box_count}")

    if not rows:
        raise RuntimeError("No videos were evaluated.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    tp = sum(r["actual_positive"] == 1 and r["predicted_positive"] == 1 for r in rows)
    fp = sum(r["actual_positive"] == 0 and r["predicted_positive"] == 1 for r in rows)
    tn = sum(r["actual_positive"] == 0 and r["predicted_positive"] == 0 for r in rows)
    fn = sum(r["actual_positive"] == 1 and r["predicted_positive"] == 0 for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"\\nVIDEO-LEVEL RESULTS: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    print(f"CSV: {OUTPUT.resolve()}")
    print("Note: these are video-level classification metrics, not object-detection mAP.")

if __name__ == "__main__":
    main()
