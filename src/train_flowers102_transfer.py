from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import pandas as pd


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
    val_data = datasets.Flowers102(
        root="data",
        split="val",
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        download=False,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
    )
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 102)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    model.train()
    history = []

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
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        val_accuracy = correct / total
        history.append({
            "epoch": epoch + 1,
            "train_loss": average_loss,
            "val_accuracy": val_accuracy,
        })
        print(
            f"Epoch {epoch + 1}/5, Loss: {average_loss:.4f}, "
            f"Val accuracy: {val_accuracy:.4f}"
        )
        model.train()

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "flowers102_resnet18_transfer.pth"
    torch.save(model.state_dict(), output_path)
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        output_dir / "transfer_training_history.csv", index=False
    )
    figure, loss_axis = plt.subplots(figsize=(8, 5))
    loss_axis.plot(history_df["epoch"], history_df["train_loss"], marker="o")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Training loss")
    accuracy_axis = loss_axis.twinx()
    accuracy_axis.plot(
        history_df["epoch"],
        history_df["val_accuracy"],
        color="tab:orange",
        marker="s",
    )
    accuracy_axis.set_ylabel("Validation accuracy")
    figure.tight_layout()
    figure.savefig(output_dir / "transfer_training_curves.png", dpi=150)
    plt.close(figure)
    print(f"Model saved to: {output_path}")
    print("Training history saved to: outputs/transfer_training_history.csv")
    print("Training curves saved to: outputs/transfer_training_curves.png")


if __name__ == "__main__":
    main()
