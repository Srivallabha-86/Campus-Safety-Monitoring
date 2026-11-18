from pathlib import Path
from zipfile import ZipFile
import random
import shutil

ARCHIVE = Path.home() / "Downloads" / "DFire-Downloads" / "D-Fire.zip"
OUTPUT = Path("datasets/dfire")
SEED = 42
VAL_RATIO = 0.10

def parts(name):
    return [p for p in name.replace("\\\\", "/").split("/") if p]

def index_split(zf, split):
    images, labels = {}, {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        p = parts(info.filename)
        for i in range(len(p) - 2):
            if p[i].lower() == split and p[i+1].lower() in ("images", "labels"):
                (images if p[i+1].lower() == "images" else labels)[Path(p[i+2]).stem] = info.filename
                break
    return images, labels

def copy_member(zf, member, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst)

def main():
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"D-Fire ZIP not found: {ARCHIVE}. Edit ARCHIVE in this script.")
    if OUTPUT.exists():
        if input(f"{OUTPUT} exists. Delete and rebuild? [y/N]: ").strip().lower() != "y":
            print("Stopped; existing dataset preserved.")
            return
        shutil.rmtree(OUTPUT)

    rng = random.Random(SEED)
    with ZipFile(ARCHIVE) as zf:
        train_i, train_l = index_split(zf, "train")
        test_i, test_l = index_split(zf, "test")
        train = sorted(set(train_i) & set(train_l))
        test = sorted(set(test_i) & set(test_l))
        if not train or not test:
            raise RuntimeError("Could not find paired train/test images and labels in archive.")
        rng.shuffle(train)
        n_val = max(1, round(len(train) * VAL_RATIO))
        val, train = sorted(train[:n_val]), sorted(train[n_val:])
        print(f"Train={len(train)} | Val={len(val)} | Test={len(test)}")

        for split, stems, image_map, label_map in (
            ("train", train, train_i, train_l),
            ("val", val, train_i, train_l),
            ("test", test, test_i, test_l),
        ):
            for stem in stems:
                img_member, lbl_member = image_map[stem], label_map[stem]
                copy_member(zf, img_member, OUTPUT / "images" / split / f"{stem}{Path(img_member).suffix}")
                copy_member(zf, lbl_member, OUTPUT / "labels" / split / f"{stem}.txt")

    yaml_path = Path("datasets/dfire.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        f"path: {OUTPUT.resolve().as_posix()}\\n"
        "train: images/train\\nval: images/val\\ntest: images/test\\n"
        "names:\\n  0: smoke\\n  1: fire\\n",
        encoding="utf-8",
    )
    print(f"Prepared dataset: {OUTPUT.resolve()}")
    print(f"Dataset YAML: {yaml_path.resolve()}")

if __name__ == "__main__":
    main()
