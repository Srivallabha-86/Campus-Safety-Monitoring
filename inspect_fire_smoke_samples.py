
import cv2
from pathlib import Path
import numpy as np

VIDEO_DIR = Path("videos/dfire_samples")
OUTPUT = VIDEO_DIR / "vp_contact_sheet.jpg"

video_files = sorted(
    VIDEO_DIR.glob("VP*.mp4"),
    key=lambda p: int(p.stem[2:])
)

columns = 5
cell_width = 260
cell_height = 165
frames_per_video = 3

rows = (len(video_files) * frames_per_video + columns - 1) // columns

sheet = np.zeros(
    (rows * cell_height, columns * cell_width, 3),
    dtype=np.uint8
)

index = 0

for video_path in video_files:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Cannot open:", video_path.name)
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for percentage in (25, 50, 75):
        frame_number = int(total_frames * percentage / 100)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (240, 135))

        row = index // columns
        col = index % columns

        x = col * cell_width
        y = row * cell_height

        sheet[y:y + 135, x:x + 240] = frame

        label = f"{video_path.stem} - {percentage}%"

        cv2.putText(
            sheet,
            label,
            (x + 5, y + 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        index += 1

    cap.release()

cv2.imwrite(str(OUTPUT), sheet)

print("Contact sheet created:", OUTPUT)
print("Frames extracted:", index)
