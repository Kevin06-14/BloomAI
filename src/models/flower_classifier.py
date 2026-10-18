import torch
from torch import nn


class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def main():
    model = FlowerClassifier()
    sample_images = torch.randn(2, 3, 32, 32)
    predictions = model(sample_images)

    print("Input shape:", sample_images.shape)
    print("Output shape:", predictions.shape)
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()