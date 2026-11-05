
from pathlib import Path

DATASET_PATH = Path("dataset/raw/birds_525")

SELECTED_SPECIES = [
    "HOUSE FINCH",
    "OVENBIRD",
    "ASIAN GREEN BEE EATER",
    "WOOD DUCK",
    "CASPIAN TERN",
    "WOOD THRUSH",
    "MERLIN",
    "OYSTER CATCHER",
    "LAUGHING GULL",
    "EASTERN YELLOW ROBIN",
    "JACOBIN PIGEON",
    "DARK EYED JUNCO",
    "RED TAILED HAWK",
    "YELLOW BREASTED CHAT",
    "VIOLET BACKED STARLING",
    "VIOLET GREEN SWALLOW",
    "LIMPKIN",
    "RED KNOT",
    "SORA",
    "GREY HEADED CHACHALACA",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

print("SELECTED SPECIES:", len(SELECTED_SPECIES))

all_valid = True

for species in SELECTED_SPECIES:
    print(f"\n{species}")

    for split in ["train", "validation", "test"]:
        folder = DATASET_PATH / split / species

        if not folder.exists():
            print(f"  {split}: MISSING")
            all_valid = False
            continue

        count = sum(
            1 for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"  {split}: {count} images")

        if count == 0:
            all_valid = False

print("\n" + "=" * 45)

if all_valid:
    print("SUCCESS: All 20 species exist in all splits.")
else:
    print("WARNING: Some species or images are missing.")
