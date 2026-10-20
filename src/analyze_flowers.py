
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import datasets


def main():
    # 读取真实 Flowers102 训练集，而不是手写演示数据。
    dataset = datasets.Flowers102(
        root="data",
        split="train",
        download=False,
    )

    labels = np.asarray(dataset._labels, dtype=np.int64)
    counts = pd.Series(labels).value_counts().sort_index()
    df = pd.DataFrame({
        "class_index": counts.index,
        "image_count": counts.values,
    })

    print("Flowers102 training category statistics:")
    print("Total images:", len(labels))
    print("Total classes:", df["class_index"].nunique())
    print("Mean images per class:", np.mean(df["image_count"]))
    print("Minimum images in a class:", np.min(df["image_count"]))
    print("Maximum images in a class:", np.max(df["image_count"]))
    print(counts)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(df["class_index"], df["image_count"])
    plt.title("Flowers102 Training Category Distribution")
    plt.xlabel("Class index")
    plt.ylabel("Image count")
    plt.tight_layout()

    output_path = output_dir / "flowers102_class_distribution.png"
    plt.savefig(output_path)
    plt.close()
    df.to_csv(output_dir / "flowers102_class_statistics.csv", index=False)

    print(f"Chart saved to: {output_path}")
    print(f"Statistics saved to: {output_dir / 'flowers102_class_statistics.csv'}")


if __name__ == "__main__":
    main()
