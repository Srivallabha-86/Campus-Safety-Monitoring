import cv2
import time
from ultralytics import YOLO


VIDEO_SOURCE = "videos/people-walking.mp4"

model = YOLO("yolo11n.pt")

capture = cv2.VideoCapture(VIDEO_SOURCE)

if not capture.isOpened():
    raise RuntimeError("Could not open video source")

frame_count = 0
start_time = time.time()

while True:

    success, frame = capture.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack_custom.yaml",
        classes=[0],
        conf=0.3,
        iou=0.5,
        verbose=False,
    )

    annotated_frame = results[0].plot()

    frame_count += 1

    elapsed_time = time.time() - start_time

    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    else:
        fps = 0

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow(
        "YOLO11n ByteTrack",
        annotated_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


capture.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time

if total_time > 0:
    average_fps = frame_count / total_time
else:
    average_fps = 0

print()
print("========== TRACKING RESULTS ==========")
print(f"Frames processed : {frame_count}")
print(f"Total time       : {total_time:.2f} seconds")
print(f"Average FPS      : {average_fps:.2f}")
print("Tracker          : ByteTrack")
print("Model            : YOLO11n")
print("=============================================")