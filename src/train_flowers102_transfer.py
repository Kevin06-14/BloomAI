from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def main():
    torch.manual_seed(42)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_data = datasets.Flowers102(
        root="data",
        split="train",
        transform=train_transform,
        download=False,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
    )

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 102)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    model.train()

    for epoch in range(5):
        total_loss = 0.0

        for images, labels in train_loader:
            predictions = model(images)
            loss = loss_function(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/5, Loss: {average_loss:.4f}")

    output_path = Path("outputs") / "flowers102_resnet18_transfer.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()