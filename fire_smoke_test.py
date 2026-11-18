
from ultralytics import YOLO

MODEL_PATH = "models/fire_smoke/best.pt"

model = YOLO(MODEL_PATH)

print("\nFire and Smoke Model Loaded Successfully!")
print("Available Classes:", model.names)
