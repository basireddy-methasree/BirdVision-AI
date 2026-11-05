
import torch
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from dataset import create_dataloaders
from model import create_model


# ==========================================
# 1. CONFIGURATION
# ==========================================

# Force CPU inference and limit threads
DEVICE = torch.device("cpu")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_resnet50.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. MODEL EVALUATION
# ==========================================

def evaluate_model():

    print("=" * 60, flush=True)
    print("BIRDVISION AI - MODEL EVALUATION", flush=True)
    print("=" * 60, flush=True)

    # Load test dataset
    _, _, test_loader, classes = create_dataloaders()

    print("\nDataset loaded successfully.", flush=True)
    print("Device:", DEVICE, flush=True)
    print("CPU threads:", torch.get_num_threads(), flush=True)
    print("Number of species:", len(classes), flush=True)
    print("Test images:", len(test_loader.dataset), flush=True)

    # Check checkpoint
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Create model architecture
    model = create_model(
        num_classes=len(classes),
        pretrained=False
    )

    # Load trained checkpoint
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # Verify checkpoint class mapping
    if checkpoint["classes"] != classes:
        raise ValueError(
            "Checkpoint classes do not match test dataset classes."
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    print("\nTrained ResNet-50 checkpoint loaded.", flush=True)
    print("Best epoch:", checkpoint["epoch"], flush=True)
    print(
        "Best validation accuracy:",
        checkpoint["best_val_accuracy"],
        flush=True
    )

    # Store predictions and actual labels
    all_labels = []
    all_predictions = []

    top5_correct = 0
    total_images = 0

    # ==========================================
    # 3. TEST DATA EVALUATION
    # ==========================================

    print("\nEvaluating test dataset...", flush=True)

    with torch.inference_mode():

        for batch_index, (images, labels) in enumerate(test_loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Top-1 predictions
            predictions = torch.argmax(outputs, dim=1)

            # Top-5 predictions
            top5_predictions = torch.topk(
                outputs,
                k=5,
                dim=1
            ).indices

            # Check Top-5 correctness
            correct_top5 = (
                top5_predictions == labels.unsqueeze(1)
            ).any(dim=1)

            top5_correct += correct_top5.sum().item()

            total_images += labels.size(0)

            # Store predictions
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

            print(
                f"Processed batch {batch_index + 1}/"
                f"{len(test_loader)}",
                flush=True
            )

    # ==========================================
    # 4. CALCULATE ACCURACY
    # ==========================================

    top1_accuracy = (
        accuracy_score(all_labels, all_predictions) * 100
    )

    top5_accuracy = (
        top5_correct / total_images
    ) * 100

    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)

    print(f"Total test images : {total_images}")
    print(f"Top-1 Accuracy    : {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy    : {top5_accuracy:.2f}%")

    # ==========================================
    # 5. CLASSIFICATION REPORT
    # ==========================================

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # ==========================================
    # 6. CONFUSION MATRIX
    # ==========================================

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(classes)))
    )

    # ==========================================
    # 7. SAVE EVALUATION RESULTS
    # ==========================================

    report_path = RESULTS_DIR / "evaluation_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:

        file.write("BIRDVISION AI - FINAL EVALUATION\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Device: {DEVICE}\n")
        file.write(f"Test images: {total_images}\n")
        file.write(f"Number of species: {len(classes)}\n")
        file.write(
            f"Top-1 Accuracy: {top1_accuracy:.2f}%\n"
        )
        file.write(
            f"Top-5 Accuracy: {top5_accuracy:.2f}%\n\n"
        )

        file.write("CLASSIFICATION REPORT\n")
        file.write(report)

    # Save confusion matrix
    matrix_path = RESULTS_DIR / "confusion_matrix.csv"

    np.savetxt(
        matrix_path,
        matrix,
        delimiter=",",
        fmt="%d"
    )

    # Save class names for matrix interpretation
    class_names_path = RESULTS_DIR / "class_names.txt"

    with open(class_names_path, "w", encoding="utf-8") as file:
        file.write("\n".join(classes))

    # ==========================================
    # 8. FINAL SUMMARY
    # ==========================================

    print("\n" + "=" * 60)
    print("EVALUATION FILES SAVED")
    print("=" * 60)

    print("1. results/evaluation_report.txt")
    print("2. results/confusion_matrix.csv")
    print("3. results/class_names.txt")

    print("\nModel evaluation completed successfully!")


# ==========================================
# 9. MAIN
# ==========================================

if __name__ == "__main__":
    evaluate_model()
