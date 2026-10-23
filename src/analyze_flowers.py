import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    split_dir = Path("data/custom_split")
    rows = []
    for split in ("train", "val", "test"):
        with (split_dir / f"{split}.csv").open(
            "r", encoding="utf-8", newline=""
        ) as file:
            rows.extend(
                {"split": split, **row}
                for row in csv.DictReader(file)
            )

    df = pd.DataFrame(rows)
    counts = df.groupby(["split", "label"]).size().reset_index(
        name="image_count"
    )
    print("TensorFlow Flowers statistics:")
    print("Total images:", len(df))
    print("Total classes:", df["label"].nunique())
    print("Mean images per class:", np.mean(counts["image_count"]))
    print(counts)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(output_dir / "tf_flowers_class_statistics.csv", index=False)

    chart = counts.pivot(
        index="label", columns="split", values="image_count"
    ).fillna(0)
    chart.plot(kind="bar", figsize=(9, 5))
    plt.title("TensorFlow Flowers Split Distribution")
    plt.xlabel("Class index")
    plt.ylabel("Image count")
    plt.tight_layout()
    plt.savefig(output_dir / "tf_flowers_class_distribution.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
