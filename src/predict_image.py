import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


MODEL_PATH = Path("outputs/flowers102_resnet18_transfer.pth")
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
    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)
        confidence, prediction = probabilities.max(dim=1)

    print("Image:", image_path)
    class_index = prediction.item()
    print("Predicted class index:", class_index)
    print(f"Predicted class name: Flowers102 category {class_index:03d}")
    print(f"Confidence: {confidence.item():.4f}")


if __name__ == "__main__":
    main()
