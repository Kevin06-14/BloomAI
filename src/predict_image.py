import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


MODEL_PATH = Path("outputs/flowers102_resnet18_transfer.pth")
LABELS_PATH = Path("src/flowers102_labels.json")
LABELS_ZH_PATH = Path("src/flowers102_labels_zh.json")


def load_labels():
    with LABELS_PATH.open("r", encoding="utf-8") as file:
        labels = json.load(file)
    if len(labels) != 102:
        raise ValueError("类别名称文件必须包含 102 个名称")
    return labels


def load_chinese_labels():
    with LABELS_ZH_PATH.open("r", encoding="utf-8") as file:
        labels = json.load(file)
    if len(labels) != 102:
        raise ValueError("中文类别名称文件必须包含 102 个名称")
    return labels
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型文件: {MODEL_PATH}。"
            "请先运行 python src/train_flowers102_transfer.py"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 102)
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()
    return model


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="使用训练好的 Flowers102 模型预测一张图片"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="待预测图片的路径",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片文件: {image_path}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    model = load_model()
    labels = load_labels()
    labels_zh = load_chinese_labels()
    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)
        top_probabilities, top_indices = torch.topk(probabilities, k=5, dim=1)

    print("Image:", image_path)
    print("Top 5 predictions:")
    for rank, (probability, index) in enumerate(
        zip(top_probabilities[0], top_indices[0]),
        start=1,
    ):
        class_index = index.item()
        print(
            f"{rank}. {labels[class_index]} ({labels_zh[class_index]}) "
            f"(class index: {class_index}, probability: {probability.item():.4f})"
        )


if __name__ == "__main__":
    main()
