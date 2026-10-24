from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.small_resnet import ImprovedResNet
from custom_dataset import FlowersCsvDataset


def main():
    model_path = Path("outputs/tf_flowers_resnet_improved.pth")
    if not model_path.exists():
        raise FileNotFoundError(
            "找不到手写 ResNet 模型，请先运行 "
            "python src/train_TensorFlowFlowers.py"
        )

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = FlowersCsvDataset(
        csv_path="data/custom_split/test.csv",
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = ImprovedResNet(num_classes=5)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print("Model: improved manually implemented ResNet")
    print("Custom test images:", total)
    print(f"Custom test accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    main()
