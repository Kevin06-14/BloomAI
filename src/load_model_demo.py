from pathlib import Path

import torch

from models.flower_classifier import FlowerClassifier


def main():
    model = FlowerClassifier(num_classes=4)
    model_path = Path("outputs") / "flower_classifier.pth"

    model.load_state_dict(
    torch.load(model_path, map_location="cpu", weights_only=True)
)
    model.eval()

    sample_image = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        prediction = model(sample_image)
        predicted_class = prediction.argmax(dim=1).item()

    print("Model loaded successfully")
    print("Predicted class index:", predicted_class)


if __name__ == "__main__":
    main()