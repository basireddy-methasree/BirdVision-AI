
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from torch.utils.data import SequentialSampler

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

DEVICE = torch.device("cpu")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_resnet50.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
MISCLASSIFIED_DIR = RESULTS_DIR / "misclassified"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MISCLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CONFUSION MATRIX VISUALIZATION
# ==========================================

def save_confusion_matrix(matrix, classes, normalized=False):

    plt.figure(figsize=(18, 15))

    display_matrix = matrix.astype(float)

    if normalized:
        row_sums = display_matrix.sum(axis=1, keepdims=True)

        display_matrix = np.divide(
            display_matrix,
            row_sums,
            out=np.zeros_like(display_matrix),
            where=row_sums != 0
        )

    plt.imshow(
        display_matrix,
        interpolation="nearest",
        cmap="Blues"
    )

    plt.title(
        "BirdVision AI - Normalized Confusion Matrix"
        if normalized
        else "BirdVision AI - Confusion Matrix",
        fontsize=16,
        pad=20
    )

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(
        tick_marks,
        classes,
        rotation=90,
        fontsize=8
    )

    plt.yticks(
        tick_marks,
        classes,
        fontsize=8
    )

    plt.xlabel("Predicted Species", fontsize=12)
    plt.ylabel("Actual Species", fontsize=12)

    plt.tight_layout()

    filename = (
        "normalized_confusion_matrix.png"
        if normalized
        else "confusion_matrix.png"
    )

    plt.savefig(
        RESULTS_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================
# 3. SAVE MISCLASSIFIED IMAGES
# ==========================================

def save_misclassified_images(records):

    print("\nSaving misclassified images...", flush=True)

    report_path = MISCLASSIFIED_DIR / "misclassification_report.csv"

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "image_name",
            "actual_species",
            "predicted_species",
            "confidence"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for index, record in enumerate(records, start=1):

            source_path = Path(record["image_path"])

            actual_species = record["actual_species"]
            predicted_species = record["predicted_species"]

            filename = (
                f"image_{index:02d}_"
                f"actual_{actual_species.replace(' ', '_')}_"
                f"predicted_{predicted_species.replace(' ', '_')}.jpg"
            )

            destination = MISCLASSIFIED_DIR / filename

            # Save original test image
            with Image.open(source_path) as image:
                image.convert("RGB").save(
                    destination,
                    format="JPEG",
                    quality=95
                )

            writer.writerow({
                "image_name": filename,
                "actual_species": actual_species,
                "predicted_species": predicted_species,
                "confidence": f"{record['confidence']:.2f}%"
            })

            print(
                f"{index}. {actual_species} -> "
                f"{predicted_species} "
                f"({record['confidence']:.2f}%)",
                flush=True
            )

    print(
        f"\nMisclassified images saved: {len(records)}",
        flush=True
    )

    print(f"Folder: {MISCLASSIFIED_DIR}", flush=True)
    print(f"Report: {report_path}", flush=True)


# ==========================================
# 4. MODEL EVALUATION
# ==========================================

def evaluate_model():

    print("=" * 65, flush=True)
    print("BIRDVISION AI - MODEL EVALUATION", flush=True)
    print("=" * 65, flush=True)

    # Load datasets
    _, _, test_loader, classes = create_dataloaders()

    print("\nDataset loaded successfully.", flush=True)
    print("Device:", DEVICE, flush=True)
    print("CPU threads:", torch.get_num_threads(), flush=True)
    print("Number of species:", len(classes), flush=True)
    print("Test images:", len(test_loader.dataset), flush=True)

    if len(test_loader.dataset) == 0:
        raise ValueError("Test dataset is empty.")

    if len(classes) < 5:
        raise ValueError("At least five species are required for Top-5 evaluation.")

    # We use dataset.samples to associate each prediction
    # with its original image file.
    if not hasattr(test_loader.dataset, "samples"):
        raise ValueError(
            "Test dataset does not expose image paths through .samples."
        )

    if not isinstance(test_loader.sampler, SequentialSampler):
        raise ValueError(
            "Test DataLoader must use sequential sampling "
            "to match predictions with original image paths."
        )

    test_image_paths = [
        sample[0] for sample in test_loader.dataset.samples
    ]

    # Check checkpoint
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Create model
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

    # Store evaluation information
    all_labels = []
    all_predictions = []

    top5_correct = 0
    total_images = 0

    misclassified_records = []

    sample_cursor = 0

    # ==========================================
    # 5. TEST DATA EVALUATION
    # ==========================================

    print("\nEvaluating test dataset...", flush=True)

    with torch.inference_mode():

        for batch_index, (images, labels) in enumerate(test_loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)

            # Top-1 predictions
            predictions = torch.argmax(outputs, dim=1)

            # Top-5 predictions
            top5_predictions = torch.topk(
                outputs,
                k=5,
                dim=1
            ).indices

            correct_top5 = (
                top5_predictions == labels.unsqueeze(1)
            ).any(dim=1)

            top5_correct += correct_top5.sum().item()

            # Store predictions
            batch_labels = labels.cpu().tolist()
            batch_predictions = predictions.cpu().tolist()

            all_labels.extend(batch_labels)
            all_predictions.extend(batch_predictions)

            batch_confidences = (
                probabilities.max(dim=1).values.cpu().tolist()
            )

            # Identify incorrect predictions
            for i in range(len(batch_labels)):

                actual_index = batch_labels[i]
                predicted_index = batch_predictions[i]

                image_path = test_image_paths[sample_cursor]

                if actual_index != predicted_index:

                    misclassified_records.append({
                        "image_path": image_path,
                        "actual_species": classes[actual_index],
                        "predicted_species": classes[predicted_index],
                        "confidence": batch_confidences[i] * 100
                    })

                sample_cursor += 1

            total_images += labels.size(0)

            print(
                f"Processed batch {batch_index + 1}/"
                f"{len(test_loader)}",
                flush=True
            )

    # ==========================================
    # 6. OVERALL ACCURACY
    # ==========================================

    top1_accuracy = (
        accuracy_score(all_labels, all_predictions) * 100
    )

    top5_accuracy = (
        top5_correct / total_images
    ) * 100

    print("\n" + "=" * 65)
    print("FINAL EVALUATION RESULTS")
    print("=" * 65)

    print(f"Total test images : {total_images}")
    print(f"Top-1 Accuracy    : {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy    : {top5_accuracy:.2f}%")
    print(f"Misclassified     : {len(misclassified_records)}")

    # ==========================================
    # 7. CLASSIFICATION REPORT
    # ==========================================

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )

    report_dict = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT")
    print("=" * 65)
    print(report)

    # ==========================================
    # 8. CONFUSION MATRIX
    # ==========================================

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(classes)))
    )

    # ==========================================
    # 9. PER-SPECIES PERFORMANCE
    # ==========================================

    species_results = []

    print("\nPER-SPECIES PERFORMANCE")
    print("=" * 65)

    for species in classes:

        metrics = report_dict[species]

        result = {
            "species": species,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1-score"],
            "support": int(metrics["support"])
        }

        species_results.append(result)

        print(
            f"{species:30s} "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1-score']:.4f}"
        )

    ranked_species = sorted(
        species_results,
        key=lambda item: item["f1_score"]
    )

    print("\nLOWEST PERFORMING SPECIES:")

    for item in ranked_species[:3]:

        print(
            f"{item['species']}: "
            f"F1 = {item['f1_score']:.4f}"
        )

    print("\nHIGHEST PERFORMING SPECIES:")

    for item in ranked_species[-3:][::-1]:

        print(
            f"{item['species']}: "
            f"F1 = {item['f1_score']:.4f}"
        )

    # ==========================================
    # 10. SAVE EVALUATION REPORT
    # ==========================================

    report_path = RESULTS_DIR / "evaluation_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:

        file.write("BIRDVISION AI - MODEL EVALUATION\n")
        file.write("=" * 65 + "\n\n")

        file.write(f"Device: {DEVICE}\n")
        file.write(f"Test images: {total_images}\n")
        file.write(f"Number of species: {len(classes)}\n")
        file.write(f"Top-1 Accuracy: {top1_accuracy:.2f}%\n")
        file.write(f"Top-5 Accuracy: {top5_accuracy:.2f}%\n")
        file.write(
            f"Misclassified images: {len(misclassified_records)}\n\n"
        )

        file.write("CLASSIFICATION REPORT\n")
        file.write("=" * 65 + "\n")
        file.write(report)

    # ==========================================
    # 11. SAVE CSV FILES
    # ==========================================

    # Confusion matrix CSV
    matrix_path = RESULTS_DIR / "confusion_matrix.csv"

    with open(matrix_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow(["Actual / Predicted"] + classes)

        for index, row in enumerate(matrix):
            writer.writerow([classes[index]] + row.tolist())

    # Per-species metrics CSV
    metrics_path = RESULTS_DIR / "species_metrics.csv"

    with open(metrics_path, "w", newline="", encoding="utf-8") as file:

        fieldnames = [
            "species",
            "precision",
            "recall",
            "f1_score",
            "support"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(species_results)

    # Class names
    class_names_path = RESULTS_DIR / "class_names.txt"

    with open(class_names_path, "w", encoding="utf-8") as file:
        file.write("\n".join(classes))

    # ==========================================
    # 12. SAVE MISCLASSIFIED IMAGES
    # ==========================================

    save_misclassified_images(misclassified_records)

    # ==========================================
    # 13. SAVE CONFUSION MATRIX VISUALIZATIONS
    # ==========================================

    print("\nGenerating confusion matrix visualizations...")

    save_confusion_matrix(
        matrix,
        classes,
        normalized=False
    )

    save_confusion_matrix(
        matrix,
        classes,
        normalized=True
    )

    # ==========================================
    # 14. FINAL SUMMARY
    # ==========================================

    print("\n" + "=" * 65)
    print("EVALUATION FILES SAVED")
    print("=" * 65)

    print("1. results/evaluation_report.txt")
    print("2. results/confusion_matrix.csv")
    print("3. results/species_metrics.csv")
    print("4. results/class_names.txt")
    print("5. results/confusion_matrix.png")
    print("6. results/normalized_confusion_matrix.png")
    print("7. results/misclassified/")

    print("\nModel evaluation completed successfully!")


# ==========================================
# 15. MAIN
# ==========================================

if __name__ == "__main__":
    evaluate_model()
