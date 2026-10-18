from pathlib import Path

import torch

from models.small_resnet import SmallResNet


def main():
    model = SmallResNet(num_classes=4)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "small_resnet.pth"
    torch.save(model.state_dict(), model_path)

    print(f"ResNet model saved to: {model_path}")


if __name__ == "__main__":
    main()