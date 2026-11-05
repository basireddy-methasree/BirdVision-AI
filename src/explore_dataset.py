
from pathlib import Path

DATASET_PATH = Path("dataset/raw/birds_525")

splits = ["train", "validation", "test"]

for split in splits:
    split_path = DATASET_PATH / split

    if not split_path.exists():
        print(f"\n{split}: Folder not found")
        continue

    species = sorted(
        folder.name for folder in split_path.iterdir()
        if folder.is_dir()
    )

    total_images = sum(
        1 for folder in split_path.iterdir()
        if folder.is_dir()
        for file in folder.iterdir()
        if file.is_file()
    )

    print(f"\n{'=' * 45}")
    print(f"Split: {split}")
    print(f"Number of species: {len(species)}")
    print(f"Total images: {total_images}")
    print("First 10 species:", species[:10])
