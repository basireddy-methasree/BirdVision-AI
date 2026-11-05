
import os
from pathlib import Path

from datasets import load_dataset
from datasets.features import ClassLabel
from tqdm import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

REPO_ID = "yashikota/birds-525-species-image-classification"

# Store Hugging Face cache on D drive
HF_CACHE = Path(r"D:\HuggingFaceCache")

# Store extracted bird images on D drive
OUTPUT_DIR = Path(r"D:\BirdVisionData\birds_525")

# Configure Hugging Face cache before loading the dataset
os.environ["HF_HOME"] = str(HF_CACHE)

# Create required directories
HF_CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DOWNLOAD AND LOAD DATASET
# ============================================================

def load_bird_dataset():

    print("\n" + "=" * 60)
    print("       BIRDVISION AI - DATASET DOWNLOADER")
    print("=" * 60)

    print("\nDataset repository:", REPO_ID)
    print("Cache location:", HF_CACHE)
    print("Output location:", OUTPUT_DIR)

    print("\nLoading dataset. Please wait...")

    dataset = load_dataset(REPO_ID)

    print("\nDataset loaded successfully!")

    return dataset


# ============================================================
# 3. CONVERT AND SAVE IMAGES
# ============================================================

def process_dataset(dataset):

    total_images = 0
    skipped_images = 0
    failed_images = 0

    split_summary = {}

    for split_name, split_data in dataset.items():

        print("\n" + "-" * 60)
        print(f"PROCESSING SPLIT: {split_name.upper()}")
        print("-" * 60)

        split_total = len(split_data)

        split_summary[split_name] = {
            "total": split_total,
            "saved": 0,
            "skipped": 0,
            "failed": 0
        }

        label_feature = split_data.features["label"]

        progress_bar = tqdm(
            enumerate(split_data),
            total=split_total,
            desc=f"Processing {split_name}",
            unit="image"
        )

        for index, item in progress_bar:

            try:

                image = item["image"]
                label = item["label"]

                # Convert label ID into species name
                if isinstance(label_feature, ClassLabel):
                    species = label_feature.int2str(label)
                else:
                    species = str(label)

                # Remove unsafe path characters
                species = "".join(
                    char if char.isalnum() or char in " _-" else "_"
                    for char in species
                ).strip()

                # Create species folder
                species_folder = (
                    OUTPUT_DIR / split_name / species
                )

                species_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )

                # Generate image filename
                image_path = species_folder / f"{index:06d}.jpg"

                # Skip existing image
                if image_path.exists():
                    skipped_images += 1
                    split_summary[split_name]["skipped"] += 1
                    continue

                # Convert image into RGB JPEG
                image = image.convert("RGB")

                image.save(
                    image_path,
                    format="JPEG",
                    quality=95
                )

                total_images += 1
                split_summary[split_name]["saved"] += 1

            except Exception as error:

                failed_images += 1
                split_summary[split_name]["failed"] += 1

                print(
                    f"\nError processing image {index}: {error}"
                )

    return (
        total_images,
        skipped_images,
        failed_images,
        split_summary
    )


# ============================================================
# 4. DISPLAY FINAL SUMMARY
# ============================================================

def display_summary(
    total_images,
    skipped_images,
    failed_images,
    split_summary
):

    print("\n" + "=" * 60)
    print("              DATASET PROCESSING SUMMARY")
    print("=" * 60)

    for split_name, details in split_summary.items():

        print(f"\n{split_name.upper()} SPLIT")

        print("Total images :", details["total"])
        print("Saved images :", details["saved"])
        print("Skipped      :", details["skipped"])
        print("Failed       :", details["failed"])

    print("\n" + "-" * 60)

    print("Total newly saved images :", total_images)
    print("Total skipped images     :", skipped_images)
    print("Total failed images      :", failed_images)

    print("\nDataset location:")
    print(OUTPUT_DIR.resolve())

    print("\n" + "=" * 60)

    if failed_images == 0:
        print("DATASET PROCESSING FINISHED!")
    else:
        print("PROCESSING FINISHED WITH ERRORS.")
        print("Check the errors above before proceeding.")

    print("=" * 60)


# ============================================================
# 5. MAIN FUNCTION
# ============================================================

def main():

    try:

        dataset = load_bird_dataset()

        (
            total_images,
            skipped_images,
            failed_images,
            split_summary
        ) = process_dataset(dataset)

        display_summary(
            total_images,
            skipped_images,
            failed_images,
            split_summary
        )

    except Exception as error:

        print("\nDATASET DOWNLOAD FAILED!")
        print("Error:", error)

        raise


if __name__ == "__main__":
    main()
