
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Load saved training history
history_path = Path("models/training_history.pth")
history = torch.load(
    history_path,
    map_location="cpu",
    weights_only=False
)

epochs = range(1, len(history["train_loss"]) + 1)

# Create results directory
Path("results").mkdir(exist_ok=True)

# 1. Loss graph
plt.figure(figsize=(9, 5))
plt.plot(epochs, history["train_loss"], marker="o", label="Training Loss")
plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet-50 Training and Validation Loss")
plt.xticks(list(epochs))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/loss_curve.png", dpi=300)
plt.show()

# 2. Accuracy graph
plt.figure(figsize=(9, 5))
plt.plot(epochs, history["train_accuracy"], marker="o", label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("ResNet-50 Training and Validation Accuracy")
plt.xticks(list(epochs))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/accuracy_curve.png", dpi=300)
plt.show()

print("Performance graphs generated successfully!")
print("Saved: results/loss_curve.png")
print("Saved: results/accuracy_curve.png")
