import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from custom_dataset import FlowersCsvDataset


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
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

    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 102)

    model.load_state_dict(
        torch.load(
            "outputs/flowers102_resnet18_transfer.pth",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print("Custom test images:", total)
    print(f"Custom test accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    main()
