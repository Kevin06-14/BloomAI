
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    # 这是演示数据，不是真实花卉数据集
    data = {
        "flower": [
            "rose", "rose", "rose",
            "daisy", "daisy",
            "sunflower", "sunflower", "sunflower", "sunflower",
            "tulip",
        ]
    }

    # 将字典转换为表格
    df = pd.DataFrame(data)

    # 统计每个类别的数量
    counts = df["flower"].value_counts().sort_index()

    # 输出统计结果
    print("Flower category statistics (demo data):")
    print(counts)

    # 创建输出目录
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # 绘制柱状图
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Flower Category Distribution (Demo Data)")
    plt.xlabel("Flower Category")
    plt.ylabel("Count")
    plt.tight_layout()

    # 保存图片
    output_path = output_dir / "flower_distribution.png"
    plt.savefig(output_path)
    plt.close()

    print(f"\nChart saved to: {output_path}")


if __name__ == "__main__":
    main()
