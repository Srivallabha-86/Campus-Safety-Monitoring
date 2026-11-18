from pathlib import Path
from ultralytics import YOLO

WEIGHTS = Path("runs/dfire_training/yolo11n_fire_smoke/weights/best.pt")
DATA = Path("datasets/dfire.yaml")

def report(label, metrics):
    print(f"\\n{label}")
    print(f"Precision: {float(metrics.box.mp):.4f}")
    print(f"Recall:    {float(metrics.box.mr):.4f}")
    print(f"mAP50:     {float(metrics.box.map50):.4f}")
    print(f"mAP50-95:  {float(metrics.box.map):.4f}")

def main():
    if not WEIGHTS.exists() or not DATA.exists():
        raise FileNotFoundError("Required weights or datasets/dfire.yaml are missing.")
    model = YOLO(str(WEIGHTS))
    report("VALIDATION SPLIT", model.val(data=str(DATA.resolve()), split="val", imgsz=640, batch=8, device="cpu"))
    report("HELD-OUT TEST SPLIT", model.val(data=str(DATA.resolve()), split="test", imgsz=640, batch=8, device="cpu"))

if __name__ == "__main__":
    main()
