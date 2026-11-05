
from pathlib import Path
import shutil

SOURCE = Path("dataset/raw/birds_525")
DESTINATION = Path("dataset/processed/birds_20")
SPECIES_FILE = Path("config/selected_species.txt")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ["train", "validation", "test"]

# Read the selected 20 species
selected_species = [
    name.strip()
    for name in SPECIES_FILE.read_text(encoding="utf-8-sig").splitlines()
    if name.strip()
]

if len(selected_species) != 20:
    raise ValueError(
        f"Expected 20 species, found {len(selected_species)}"
    )

print("=" * 55)
print("BIRDVISION AI - 20 SPECIES DATASET PREPARATION")
print("=" * 55)

total_images = 0

for split in SPLITS:
    print(f"\nProcessing {split}...")

    split_count = 0

    for species in selected_species:
        source_folder = SOURCE / split / species
        destination_folder = DESTINATION / split / species

        if not source_folder.exists():
            raise FileNotFoundError(
                f"Missing source folder: {source_folder}"
            )

        destination_folder.mkdir(parents=True, exist_ok=True)

        for image_file in source_folder.iterdir():
            if (
                image_file.is_file()
                and image_file.suffix.lower() in IMAGE_EXTENSIONS
            ):
                destination_file = destination_folder / image_file.name

                if not destination_file.exists():
                    shutil.copy2(image_file, destination_file)

                split_count += 1

    total_images += split_count
    print(f"{split}: {split_count} images")

print("\n" + "=" * 55)
print("DATASET PREPARATION COMPLETED")
print("Selected species:", len(selected_species))
print("Total images:", total_images)
print("Destination:", DESTINATION.resolve())
print("=" * 55)
