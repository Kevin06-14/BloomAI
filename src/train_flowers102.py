import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd

from models.small_resnet import ImprovedResNet
from custom_dataset import FlowersCsvDataset


def main():
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = FlowersCsvDataset(
        csv_path="data/custom_split/train.csv",
        transform=transform,
    )
    val_dataset = FlowersCsvDataset(
        csv_path="data/custom_split/val.csv",
        transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ImprovedResNet(num_classes=102)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=40,
    )

    model.train()
    history = []

    best_val_accuracy = 0.0
    best_state = None

    for epoch in range(40):
        total_loss = 0.0

        for images, labels in loader:
            predictions = model(images)
            loss = loss_function(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)
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
            f"Epoch {epoch + 1}/40, Loss: {average_loss:.4f}, "
            f"Val accuracy: {val_accuracy:.4f}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        scheduler.step()
        model.train()

    output_dir = "outputs"
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/flowers102_resnet_improved.pth"
    torch.save(best_state or model.state_dict(), output_path)
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        f"{output_dir}/manual_improved_training_history.csv", index=False
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
    figure.savefig(
        f"{output_dir}/manual_improved_training_curves.png",
        dpi=150,
    )
    plt.close(figure)
    print(f"Model saved to: {output_path}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(
        "Training history saved to: "
        "outputs/manual_improved_training_history.csv"
    )
    print(
        "Training curves saved to: "
        "outputs/manual_improved_training_curves.png"
    )

if __name__ == "__main__":
    main()
