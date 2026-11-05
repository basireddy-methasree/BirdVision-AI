
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from PIL import Image
from matplotlib import colormaps

from model import create_model
from data_transforms import eval_transform


# =====================================================
# 1. CONFIGURATION
# =====================================================

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEVICE = torch.device("cpu")

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = ROOT / "models" / "best_resnet50.pth"

INPUT_DIR = ROOT / "input_images"

OUTPUT_DIR = ROOT / "results" / "predictions" / "unseen"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224


# =====================================================
# 2. LOAD TRAINED MODEL
# =====================================================

print("Loading BirdVision AI model...", flush=True)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

classes = checkpoint["classes"]

model = create_model(
    num_classes=len(classes),
    pretrained=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Number of bird species:", len(classes))


# =====================================================
# 3. GRAD-CAM IMPLEMENTATION
# =====================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.activations = None
        self.gradients = None

        self.handle = target_layer.register_forward_hook(
            self.save_activation
        )

    def save_activation(self, module, inputs, output):

        self.activations = output.detach()

        if output.requires_grad:
            output.register_hook(self.save_gradient)

    def save_gradient(self, gradient):

        self.gradients = gradient.detach()

    def generate(self, image_tensor, class_index):

        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None

        image_tensor = image_tensor.detach()
        image_tensor.requires_grad_(True)

        with torch.enable_grad():

            output = self.model(image_tensor)

            output[:, class_index].sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Unable to capture Grad-CAM activations or gradients."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = torch.sum(
            weights * self.activations,
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()

    def close(self):
        self.handle.remove()


# =====================================================
# 4. GENERATE GRAD-CAM OVERLAY
# =====================================================

def create_overlay(original_image, heatmap):

    original = np.array(
        original_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
    )

    colored_heatmap = (
        colormaps["jet"](heatmap)[..., :3] * 255
    ).astype(np.uint8)

    overlay = (
        0.5 * original.astype(np.float32)
        + 0.5 * colored_heatmap.astype(np.float32)
    )

    return np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)


# =====================================================
# 5. PREDICT BIRD SPECIES
# =====================================================

def predict_bird(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print("\nProcessing:", image_path.name)

    original_image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = eval_transform(
        original_image
    )

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # Predict species
    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        top_values, top_indices = torch.topk(
            probabilities,
            k=min(5, len(classes))
        )

    predicted_index = top_indices[0].item()

    predicted_species = classes[predicted_index]

    confidence = top_values[0].item() * 100

    # Display prediction
    print("\n" + "=" * 55)
    print("BIRDVISION AI - PREDICTION RESULTS")
    print("=" * 55)

    print("Image:", image_path.name)
    print("Predicted species:", predicted_species)
    print(f"Confidence: {confidence:.2f}%")

    print("\nTOP-5 PREDICTIONS")
    print("-" * 55)

    for rank, (value, index) in enumerate(
        zip(top_values, top_indices),
        start=1
    ):

        print(
            f"{rank}. {classes[index.item()]} "
            f"- {value.item() * 100:.2f}%"
        )

    # =================================================
    # 6. GENERATE GRAD-CAM
    # =================================================

    gradcam = GradCAM(
        model,
        model.layer4[-1]
    )

    try:

        heatmap = gradcam.generate(
            image_tensor,
            predicted_index
        )

    finally:

        gradcam.close()

    overlay = create_overlay(
        original_image,
        heatmap
    )

    # =================================================
    # 7. VISUALIZE RESULTS
    # =================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 6)
    )

    axes[0].imshow(
        original_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
    )

    axes[0].set_title("Original Bird Image")

    axes[1].imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    axes[1].set_title("Grad-CAM Heatmap")

    axes[2].imshow(overlay)

    axes[2].set_title("Grad-CAM Overlay")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"BirdVision AI | {predicted_species} "
        f"| Confidence: {confidence:.2f}%",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    # Save image
    output_path = OUTPUT_DIR / (
        f"{image_path.stem}_prediction.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print("\nVisualization saved:", output_path)

    # Save prediction report
    report_path = OUTPUT_DIR / (
        f"{image_path.stem}_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:

        file.write("BIRDVISION AI - PREDICTION REPORT\n")
        file.write("=" * 45 + "\n")
        file.write(f"Image: {image_path.name}\n")
        file.write(f"Predicted species: {predicted_species}\n")
        file.write(f"Confidence: {confidence:.2f}%\n\n")
        file.write("TOP-5 PREDICTIONS\n")

        for rank, (value, index) in enumerate(
            zip(top_values, top_indices),
            start=1
        ):

            file.write(
                f"{rank}. {classes[index.item()]}: "
                f"{value.item() * 100:.2f}%\n"
            )

    print("Prediction report saved:", report_path)


# =====================================================
# 8. MAIN EXECUTION
# =====================================================

if __name__ == "__main__":

    if len(sys.argv) > 1:

        image_path = Path(sys.argv[1])

        if not image_path.is_absolute():
            image_path = ROOT / image_path

    else:

        image_files = sorted([
            path
            for path in INPUT_DIR.iterdir()
            if path.suffix.lower() in [
                ".jpg", ".jpeg", ".png"
            ]
        ])

        if not image_files:
            raise FileNotFoundError(
                "No images found in input_images. "
                "Add a bird JPG or PNG image first."
            )

        image_path = image_files[0]

    predict_bird(image_path)
