import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.small_resnet import ImprovedResNet


MODEL_PATH = Path("outputs/tf_flowers_resnet_improved.pth")
LABELS_PATH = Path("src/tf_flowers_labels.json")
LABELS_ZH_PATH = Path("src/tf_flowers_labels_zh.json")


def load_labels(path):
    with path.open("r", encoding="utf-8") as file:
        labels = json.load(file)
    if len(labels) != 5:
        raise ValueError(f"{path} 必须包含 5 个类别名称")
    return labels


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="使用手写 SmallResNet 预测一张图片"
    )
    parser.add_argument("--image", required=True, help="待预测图片的路径")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片文件: {image_path}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "找不到改进手写模型，请先运行 python src/train_flowers102.py"
        )

    image = Image.open(image_path).convert("RGB")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    base_transform = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize,
    ])
    crop_transform = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize,
    ])
    flip_transform = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ])
    input_tensors = torch.stack([
        base_transform(image),
        crop_transform(image),
        flip_transform(image),
    ])
    labels = load_labels(LABELS_PATH)
    labels_zh = load_labels(LABELS_ZH_PATH)

    model = ImprovedResNet(num_classes=5)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensors), dim=1).mean(dim=0)
        top_probabilities, top_indices = torch.topk(probabilities, k=5)

    print("Image:", image_path)
    print("Model: improved manually implemented ResNet")
    print("Inference: 3-view test-time augmentation")
    print("Top 5 predictions:")
    for rank, (probability, index) in enumerate(
        zip(top_probabilities, top_indices),
        start=1,
    ):
        print(
            f"{rank}. {labels[index.item()]} ({labels_zh[index.item()]}) "
            f"[class index {index.item()}], "
            f"probability {probability.item():.4f}"
        )


if __name__ == "__main__":
    main()
