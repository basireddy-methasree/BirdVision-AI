
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

from data_transforms import train_transform, eval_transform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "processed" / "birds_20"
SPECIES_FILE = PROJECT_ROOT / "config" / "selected_species.txt"

BATCH_SIZE = 16
NUM_WORKERS = 0  # Safe starting point for Windows


def create_dataloaders():
    selected_species = [
        line.strip()
        for line in SPECIES_FILE.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if len(selected_species) != 20:
        raise ValueError("Expected exactly 20 selected species.")

    train_dataset = datasets.ImageFolder(
        DATASET_PATH / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        DATASET_PATH / "validation",
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        DATASET_PATH / "test",
        transform=eval_transform
    )

    # Confirm identical class mapping across all splits
    expected_mapping = {
        name: index
        for index, name in enumerate(sorted(selected_species))
    }

    for name, dataset in [
        ("train", train_dataset),
        ("validation", val_dataset),
        ("test", test_dataset)
    ]:
        if dataset.class_to_idx != expected_mapping:
            raise ValueError(
                f"{name} class mapping does not match selected species."
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = (
        create_dataloaders()
    )

    print("Number of classes:", len(classes))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Testing batches:", len(test_loader))
    print("Class names:", classes)
