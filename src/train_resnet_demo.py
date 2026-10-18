import torch
from torch import nn

from models.small_resnet import SmallResNet


def main():
    torch.manual_seed(42)

    model = SmallResNet(num_classes=4)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    images = torch.randn(20, 3, 32, 32)
    labels = torch.randint(0, 4, (20,))

    for epoch in range(5):
        predictions = model(images)
        loss = loss_function(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/5, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()