import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.small_resnet import SmallResNet


def main():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = datasets.Flowers102(
        root="data",
        split="test",
        transform=transform,
        download=False,
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = SmallResNet(num_classes=102)
    model.load_state_dict(
        torch.load(
            "outputs/flowers102_resnet.pth",
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

    accuracy = correct / total
    print("Test images:", total)
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()