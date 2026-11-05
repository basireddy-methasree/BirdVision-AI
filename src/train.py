
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from model import create_model
from dataset import create_dataloaders
from training_config import configure_training


# ---------------- CONFIGURATION ----------------

EPOCHS = 10
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_resnet50.pth"


# ---------------- TRAINING FUNCTION ----------------

def train_one_epoch(model, loader, criterion, optimizer):

    model.eval()
    model.fc.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training"):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# ---------------- VALIDATION FUNCTION ----------------

def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in tqdm(loader, desc="Validation"):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# ---------------- MAIN TRAINING ----------------

def main():

    print("=" * 55)
    print("BIRDVISION AI - RESNET50 TRAINING")
    print("=" * 55)

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)

    train_loader, val_loader, test_loader, classes = (
        create_dataloaders()
    )

    model = create_model(
        num_classes=len(classes),
        pretrained=True
    ).to(DEVICE)

    criterion, optimizer = configure_training(model)

    best_val_accuracy = 0.0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(EPOCHS):

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                    "classes": classes
                },
                BEST_MODEL_PATH
            )

            print("Best model checkpoint saved!")

    print("\n" + "=" * 55)
    print("TRAINING FINISHED")
    print("Best Validation Accuracy:", best_val_accuracy)
    print("Model saved at:", BEST_MODEL_PATH)
    print("=" * 55)

    torch.save(
        history,
        MODEL_DIR / "training_history.pth"
    )


if __name__ == "__main__":
    main()
