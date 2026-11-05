
from pathlib import Path
from PIL import Image

from data_transforms import train_transform, eval_transform

DATASET_PATH = Path("dataset/processed/birds_20")

# Find one training image
image_files = list(
    (DATASET_PATH / "train").glob("*/*")
)

if not image_files:
    raise RuntimeError("No training images found!")

image_path = image_files[0]

# Load image
image = Image.open(image_path).convert("RGB")

# Apply transformations
train_tensor = train_transform(image)
eval_tensor = eval_transform(image)

print("Image:", image_path.name)
print("Original image size:", image.size)
print("Training tensor shape:", train_tensor.shape)
print("Evaluation tensor shape:", eval_tensor.shape)
print("Tensor data type:", train_tensor.dtype)
print("Preprocessing successful!")
