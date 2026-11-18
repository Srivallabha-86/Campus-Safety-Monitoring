
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET_YAML = ROOT / "datasets" / "dfire.yaml"
OUTPUT_DIR = ROOT / "runs" / "dfire_training"


def main():
    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATASET_YAML}")

    print("=" * 65)
    print("STARTING YOLO11n FIRE AND SMOKE TRAINING")
    print("=" * 65)

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATASET_YAML),
        epochs=5,
        imgsz=416,
        batch=16,
        patience=3,
        workers=0,
        project=str(OUTPUT_DIR),
        name="yolo11n_fire_smoke",
        pretrained=True,
        seed=42,
        device="cpu",
        exist_ok=True
    )

    best_weights = (
        OUTPUT_DIR / "yolo11n_fire_smoke" / "weights" / "best.pt"
    )

    print("\nTraining finished.")
    print(f"Best weights: {best_weights}")
    print(f"Weights exist: {best_weights.exists()}")


if __name__ == "__main__":
    main()
