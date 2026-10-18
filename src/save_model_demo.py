from pathlib import Path

import torch

from models.flower_classifier import FlowerClassifier


def main():
    model = FlowerClassifier(num_classes=4)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "flower_classifier.pth"
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()