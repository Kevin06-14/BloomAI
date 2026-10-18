from pathlib import Path

import torch

from models.flower_classifier import FlowerClassifier


def main():
    class_names = ["daisy", "rose", "sunflower", "tulip"]

    model = FlowerClassifier(num_classes=len(class_names))
    model_path = Path("outputs") / "flower_classifier.pth"

    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    sample_image = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        prediction = model(sample_image)
        probabilities = torch.softmax(prediction, dim=1)
        predicted_index = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_index].item()

    print("Model loaded successfully")
    print("Predicted flower:", class_names[predicted_index])
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()