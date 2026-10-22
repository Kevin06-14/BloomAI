import csv
import random
from collections import defaultdict
from pathlib import Path

from torchvision import datasets


def main():
    random.seed(42)
    root = Path("data")
    output_dir = root / "custom_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets.Flowers102(root=root, split="train", download=False)
    datasets.Flowers102(root=root, split="val", download=False)
    datasets.Flowers102(root=root, split="test", download=False)

    all_rows = {}
    for split in ("train", "val", "test"):
        dataset = datasets.Flowers102(root=root, split=split, download=False)
        for path, label in zip(dataset._image_files, dataset._labels):
            all_rows[str(Path(path))] = int(label)

    by_class = defaultdict(list)
    for image_path, label in all_rows.items():
        by_class[label].append(image_path)

    selected = {"train": [], "test": [], "val": []}
    for label, paths in sorted(by_class.items()):
        random.shuffle(paths)
        test_count = min(9, len(paths))
        train_count = min(60, len(paths) - test_count)
        selected["test"].extend((path, label) for path in paths[:test_count])
        selected["train"].extend(
            (path, label) for path in paths[test_count:test_count + train_count]
        )
        selected["val"].extend(
            (path, label) for path in paths[test_count + train_count:]
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
