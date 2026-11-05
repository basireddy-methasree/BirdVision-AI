
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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

OUTPUT_DIR = ROOT / "results" / "gradcam_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224


# =====================================================
# 2. LOAD TRAINED RESNET-50 MODEL
# =====================================================

print("Loading ResNet-50 model...", flush=True)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
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

print("Model loaded successfully.", flush=True)
print("Number of classes:", len(classes))


# =====================================================
# 3. GRAD-CAM CLASS
# =====================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Register forward hook
        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

    def save_activation(self, module, inputs, output):

        # Store feature maps
        self.activations = output.detach()

        # Register gradient hook
        if output.requires_grad:
            output.register_hook(
                self.save_gradient
            )

    def save_gradient(self, gradient):

        self.gradients = gradient.detach()

    def generate(self, image_tensor, class_index=None):

        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None

        # Ensure input supports gradient calculation
        image_tensor = image_tensor.detach()
        image_tensor.requires_grad_(True)

        # Enable gradients even though backbone is frozen
        with torch.enable_grad():

            output = self.model(image_tensor)

            # Select predicted class
            if class_index is None:
                class_index = output.argmax(dim=1).item()

            score = output[:, class_index]

            # Backpropagate class score
            score.backward()

        # Verify activation capture
        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        # Verify gradient capture
        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # Global average pooling of gradients
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted combination of feature maps
        cam = torch.sum(
            weights * self.activations,
            dim=1,
            keepdim=True
        )

        # Remove negative values
        cam = F.relu(cam)

        # Resize heatmap
        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        # Normalize heatmap
        cam = cam - cam.min()

        cam = cam / (
            cam.max() + 1e-8
        )

        return cam.cpu().numpy(), class_index

    def close(self):

        self.forward_handle.remove()


# =====================================================
# 4. CREATE HEATMAP OVERLAY
# =====================================================

def create_overlay(original_image, heatmap):

    original = np.array(
        original_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
    )

    # Apply Jet colormap
    colored_heatmap = (
        colormaps["jet"](heatmap)[..., :3] * 255
    ).astype(np.uint8)

    # Blend original image and heatmap
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
# 5. GENERATE GRAD-CAM FOR AN IMAGE
# =====================================================

def generate_gradcam(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print(
        "\nProcessing image:",
        image_path.name,
        flush=True
    )

    # Load original image
    original_image = Image.open(
        image_path
    ).convert("RGB")

    # Apply evaluation transformation
    image_tensor = eval_transform(
        original_image
    )

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # Create Grad-CAM using final ResNet-50 block
    gradcam = GradCAM(
        model,
        model.layer4[-1]
    )

    try:

        heatmap, predicted_index = gradcam.generate(
            image_tensor
        )

    finally:

        gradcam.close()

    predicted_species = classes[predicted_index]

    # Calculate prediction confidence separately
    with torch.no_grad():

        probabilities = torch.softmax(
            model(image_tensor.detach()),
            dim=1
        )

        confidence = (
            probabilities[0, predicted_index].item()
            * 100
        )

    # Create overlay
    overlay = create_overlay(
        original_image,
        heatmap
    )

    # =================================================
    # 6. VISUALIZE RESULTS
    # =================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 6)
    )

    # Original image
    axes[0].imshow(
        original_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
    )

    axes[0].set_title(
        "Original Bird Image",
        fontsize=12
    )

    # Grad-CAM heatmap
    axes[1].imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        "Grad-CAM Heatmap",
        fontsize=12
    )

    # Overlay
    axes[2].imshow(overlay)

    axes[2].set_title(
        "Grad-CAM Overlay",
        fontsize=12
    )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"BirdVision AI | Predicted: {predicted_species} "
        f"| Confidence: {confidence:.2f}%",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    # =================================================
    # 7. SAVE OUTPUT
    # =================================================

    output_path = OUTPUT_DIR / (
        f"{image_path.stem}_gradcam.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print("\nGrad-CAM generated successfully!", flush=True)
    print("Predicted species:", predicted_species)
    print(f"Confidence: {confidence:.2f}%")
    print("Saved output:", output_path)



# =====================================================
# 8. GENERATE GRAD-CAM FOR MISCLASSIFIED IMAGES
# =====================================================

if __name__ == "__main__":

    import pandas as pd

    PREDICTIONS_DIR = ROOT / "results" / "predictions"

    REPORT_PATH = (
        PREDICTIONS_DIR / "misclassification_report.csv"
    )

    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Misclassification report not found: {REPORT_PATH}"
        )

    # Load misclassification report
    df = pd.read_csv(REPORT_PATH)

    if df.empty:
        print("No misclassified images found.")
        raise SystemExit

    print("\n" + "=" * 60)
    print("BIRDVISION AI - MISCLASSIFIED IMAGE GRAD-CAM")
    print("=" * 60)

    print("Total misclassified images:", len(df))

    successful = 0
    failed = 0

    # Generate Grad-CAM for each incorrect prediction
    for index, row in df.iterrows():

        image_path = PREDICTIONS_DIR / row["saved_image"]

        print("\n" + "-" * 50)
        print(f"Image {index + 1}/{len(df)}")
        print("Actual species:", row["actual_species"])
        print("Previously predicted:", row["predicted_species"])

        try:
            generate_gradcam(image_path)
            successful += 1

        except Exception as error:
            failed += 1
            print("Error processing image:", error)

    print("\n" + "=" * 60)
    print("GRAD-CAM BATCH SUMMARY")
    print("=" * 60)

    print("Total images:", len(df))
    print("Successful:", successful)
    print("Failed:", failed)

    print("\nOutput directory:", OUTPUT_DIR)
    print("Batch processing completed!")


