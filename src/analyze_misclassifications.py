
import torch
import pandas as pd
import shutil

from pathlib import Path
from dataset import create_dataloaders
from model import create_model

# ==========================================
# 1. CONFIGURATION
# ==========================================

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEVICE = torch.device("cpu")

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT / "results" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = ROOT / "models" / "best_resnet50.pth"


# ==========================================
# 2. LOAD DATASET AND MODEL
# ==========================================

print("Loading test dataset...")

_, _, test_loader, classes = create_dataloaders()

model = create_model(
    num_classes=len(classes),
    pretrained=False
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

if checkpoint["classes"] != classes:
    raise ValueError("Checkpoint and dataset classes do not match.")

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Test images:", len(test_loader.dataset))


# ==========================================
# 3. IDENTIFY MISCLASSIFIED IMAGES
# ==========================================

errors = []
sample_index = 0

print("\nAnalyzing predictions...")

with torch.inference_mode():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predictions = probabilities.max(dim=1)

        for i in range(len(labels)):

            actual_index = labels[i].item()
            predicted_index = predictions[i].item()

            image_path = test_loader.dataset.samples[
                sample_index
            ][0]

            sample_index += 1

            if actual_index != predicted_index:

                actual_species = classes[actual_index]
                predicted_species = classes[predicted_index]

                image_name = Path(image_path).name

                # Create a unique filename
                output_name = (
                    f"actual_{actual_species}"
                    f"__predicted_{predicted_species}"
                    f"__{image_name}"
                )

                destination = OUTPUT_DIR / output_name

                shutil.copy2(image_path, destination)

                errors.append({
                    "image_name": image_name,
                    "actual_species": actual_species,
                    "predicted_species": predicted_species,
                    "confidence_percent": round(
                        confidence[i].item() * 100, 2
                    ),
                    "saved_image": output_name
                })


# ==========================================
# 4. SAVE ERROR ANALYSIS
# ==========================================

report_path = OUTPUT_DIR / "misclassification_report.csv"

df = pd.DataFrame(errors)

df.to_csv(report_path, index=False)

print("\n" + "=" * 60)
print("MISCLASSIFICATION ANALYSIS")
print("=" * 60)

print("Total test images:", len(test_loader.dataset))
print("Incorrect predictions:", len(errors))
print("Correct predictions:", len(test_loader.dataset) - len(errors))

if not df.empty:
    print("\nMisclassified images:")
    print(df.to_string(index=False))
else:
    print("\nNo misclassified images found.")

print("\nSaved report:", report_path)
print("Saved images:", OUTPUT_DIR)
print("\nAnalysis completed successfully!")
