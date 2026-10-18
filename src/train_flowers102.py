import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.small_resnet import SmallResNet


def main():
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = datasets.Flowers102(
        root="data",
        split="train",
        transform=transform,
        download=False,
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = SmallResNet(num_classes=102)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(2):
        total_loss = 0.0

        for images, labels in loader:
            predictions = model(images)
            loss = loss_function(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/2, Loss: {average_loss:.4f}")

    output_path = "outputs/flowers102_resnet.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to: {output_path}")

if __name__ == "__main__":
    main()