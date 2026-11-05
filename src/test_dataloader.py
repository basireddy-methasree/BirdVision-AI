
import torch

from dataset import create_dataloaders

train_loader, val_loader, test_loader, classes = (
    create_dataloaders()
)

# Get one batch
images, labels = next(iter(train_loader))

print("\n========== DATALOADER TEST ==========")

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Image data type:", images.dtype)
print("Label data type:", labels.dtype)

print("\nFirst 5 labels:")

for label in labels[:5]:
    index = label.item()
    print(index, "->", classes[index])

print("\nTraining batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Testing batches:", len(test_loader))

print("\nAll class labels:", len(classes))

assert images.shape[1:] == (3, 224, 224)
assert len(classes) == 20
assert labels.dtype == torch.int64

print("\nDATALOADER WORKING SUCCESSFULLY!")
