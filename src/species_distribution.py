
from pathlib import Path

TRAIN_PATH = Path("dataset/raw/birds_525/train")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

species_counts = []

for folder in TRAIN_PATH.iterdir():
    if folder.is_dir():
        count = sum(
            1 for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        species_counts.append((folder.name, count))

# Sort by image count, highest first
species_counts.sort(key=lambda item: item[1], reverse=True)

print("\nTOTAL SPECIES:", len(species_counts))
print("TOTAL TRAINING IMAGES:", sum(c for _, c in species_counts))

print("\nTOP 30 SPECIES:")
for name, count in species_counts[:30]:
    print(f"{name:40} {count:5} images")

print("\nBOTTOM 10 SPECIES:")
for name, count in species_counts[-10:]:
    print(f"{name:40} {count:5} images")
