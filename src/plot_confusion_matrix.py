
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MATRIX_PATH = PROJECT_ROOT / "results" / "confusion_matrix.csv"
CLASS_NAMES_PATH = PROJECT_ROOT / "results" / "class_names.txt"

OUTPUT_DIR = PROJECT_ROOT / "results" / "confusion_matrix"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "confusion_matrix_heatmap.png"


# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading confusion matrix...")

matrix = np.loadtxt(
    MATRIX_PATH,
    delimiter=",",
    dtype=int
)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = [line.strip() for line in file if line.strip()]

if matrix.shape != (len(class_names), len(class_names)):
    raise ValueError("Confusion matrix dimensions do not match class names.")

print("Matrix shape:", matrix.shape)
print("Number of species:", len(class_names))


# ==========================================
# 3. CALCULATE PERFORMANCE
# ==========================================

total_images = matrix.sum()
correct_predictions = np.trace(matrix)
incorrect_predictions = total_images - correct_predictions

accuracy = (correct_predictions / total_images) * 100

print("\nEvaluation Summary")
print("-" * 40)
print("Total images:", total_images)
print("Correct predictions:", correct_predictions)
print("Incorrect predictions:", incorrect_predictions)
print(f"Accuracy: {accuracy:.2f}%")


# ==========================================
# 4. GENERATE HEATMAP
# ==========================================

plt.figure(figsize=(18, 15))

sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor="white",
    cbar=True
)

plt.title(
    "BirdVision AI - ResNet-50 Confusion Matrix",
    fontsize=18,
    fontweight="bold",
    pad=20
)

plt.xlabel(
    "Predicted Bird Species",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "Actual Bird Species",
    fontsize=13,
    fontweight="bold"
)

plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()

# Save high-resolution heatmap
plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix heatmap saved successfully!")
print("Output:", OUTPUT_PATH)
