import csv
import random
from pathlib import Path


def main():
    random.seed(42)
    data_dir = Path("data/flower_photos")
    output_dir = Path("data/custom_split")
    output_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if len(class_dirs) != 5:
        raise ValueError("TensorFlow Flowers 数据集应包含 5 个类别")

    selected = {"train": [], "val": [], "test": []}
    for label, class_dir in enumerate(class_dirs):
        images = sorted(
            path for path in class_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        random.shuffle(images)
        test_count = int(len(images) * 0.15)
        val_count = int(len(images) * 0.15)
        selected["test"].extend((path, label) for path in images[:test_count])
        selected["val"].extend(
            (path, label)
            for path in images[test_count:test_count + val_count]
        )
        selected["train"].extend(
            (path, label)
            for path in images[test_count + val_count:]
        )

    for split, rows in selected.items():
        csv_path = output_dir / f"{split}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image_path", "label"])
            writer.writerows(rows)
        print(f"{split}: {len(rows)} images -> {csv_path}")


if __name__ == "__main__":
    main()
