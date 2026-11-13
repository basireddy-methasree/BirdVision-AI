
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# Add project src directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import create_model
from data_transforms import eval_transform


class BirdVisionModel:

    def __init__(self):

        # CPU configuration
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        self.device = torch.device("cpu")

        self.checkpoint_path = (
            PROJECT_ROOT / "models" / "best_resnet50.pth"
        )

        self.model = None
        self.classes = []

        self.load_model()

    def load_model(self):

        print("Loading BirdVision AI model...")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # Extract original species mapping
        self.classes = checkpoint["classes"]

        # Create the exact model architecture
        self.model = create_model(
            num_classes=len(self.classes),
            pretrained=False
        )

        # Load trained weights
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")
        print("Number of species:", len(self.classes))
        print("Device:", self.device)

    def predict(self, image: Image.Image):

        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        # Convert image to RGB
        image = image.convert("RGB")

        # Apply existing evaluation preprocessing
        image_tensor = eval_transform(image)

        image_tensor = (
            image_tensor.unsqueeze(0).to(self.device)
        )

        # Run inference
        with torch.no_grad():

            outputs = self.model(image_tensor)

            probabilities = F.softmax(
                outputs,
                dim=1
            )[0]

            top_values, top_indices = torch.topk(
                probabilities,
                k=min(5, len(self.classes))
            )

        # Build Top-5 results
        top_predictions = []

        for value, index in zip(top_values, top_indices):

            top_predictions.append({
                "species": self.classes[index.item()],
                "confidence": round(value.item() * 100, 2)
            })

        predicted_index = top_indices[0].item()

        return {
            "predicted_species": self.classes[predicted_index],
            "confidence": round(
                top_values[0].item() * 100,
                2
            ),
            "top_5_predictions": top_predictions
        }


if __name__ == "__main__":

    bird_model = BirdVisionModel()

    print("\nLoaded species:")

    for index, species in enumerate(bird_model.classes):
        print(f"{index}: {species}")
