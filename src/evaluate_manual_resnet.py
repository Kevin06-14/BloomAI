from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.small_resnet import ImprovedResNet


def main():
    model_path = Path("outputs/flowers102_resnet_improved.pth")
    if not model_path.exists():
        raise FileNotFoundError(
            "找不到手写 ResNet 模型，请先运行 "
            "python src/train_flowers102.py"
        )

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.Flowers102(
        root="data",
        split="test",
        transform=transform,
        download=False,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = ImprovedResNet(num_classes=102)
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
    print("Test images:", total)
    print(f"Test accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    main()
