
from pathlib import Path
from ultralytics import YOLO

DATA = Path("datasets/dfire.yaml")

MODELS = {
    "Dashboard Model": Path("models/fire_smoke/best.pt"),
    "Training Model": Path(
        "runs/dfire_training/yolo11n_fire_smoke/weights/best.pt"
    ),
}


def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    for name, weights in MODELS.items():
        if not weights.exists():
            print(f"\nSkipping missing model: {weights}")
            continue

        print("\n" + "=" * 65)
        print(name)
        print("=" * 65)

        model = YOLO(str(weights))

        for split in ("val", "test"):
            print(f"\n--- {split.upper()} ---")

            metrics = model.val(
                data=str(DATA.resolve()),
                split=split,
                imgsz=640,
                batch=8,
                device="cpu",
                plots=False,
                verbose=False,
            )

            print(f"Precision: {metrics.box.mp:.4f}")
            print(f"Recall:    {metrics.box.mr:.4f}")
            print(f"mAP50:     {metrics.box.map50:.4f}")
            print(f"mAP50-95:  {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
