from pathlib import Path
import cv2
from ultralytics import YOLO

WEIGHTS = Path(
    "runs/dfire_training/yolo11n_fire_smoke/weights/best.pt"
)

VIDEO = Path("videos/campus_negative.mp4")

CONF = 0.50
STRIDE = 2


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Trained weights not found: {WEIGHTS}")

    if not VIDEO.exists():
        raise FileNotFoundError(
            f"Negative test video not found: {VIDEO}\n"
            "Provide a suitable campus video without fire or smoke."
        )

    model = YOLO(str(WEIGHTS))
    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO}")

    frame_no = 0
    sampled = 0
    positive_frames = 0
    boxes_total = 0
    max_conf = 0.0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if frame_no % STRIDE == 0:
                sampled += 1

                result = model.predict(
                    frame,
                    conf=CONF,
                    imgsz=640,
                    verbose=False
                )[0]

                boxes = result.boxes
                count = 0 if boxes is None else len(boxes)

                if count:
                    positive_frames += 1
                    boxes_total += count

                    scores = boxes.conf.tolist()
                    max_conf = max(
                        max_conf,
                        max(scores, default=0.0)
                    )

                    print(
                        f"Frame {frame_no}: {count} candidate(s), "
                        f"confidence={scores}"
                    )

            frame_no += 1

    finally:
        cap.release()

    print("\nCAMPUS NEGATIVE VIDEO")
    print("--------------------------")
    print(f"Video: {VIDEO}")
    print(f"Sampled frames: {sampled}")
    print(f"Positive frames: {positive_frames}")
    print(f"Candidate boxes: {boxes_total}")
    print(f"Maximum confidence: {max_conf:.3f}")

    if boxes_total == 0:
        print("PASS: Zero candidate detections.")
    else:
        print(
            "REVIEW: False-positive candidates remain; "
            "keep dashboard alerts disabled."
        )


if __name__ == "__main__":
    main()