
import torch

from model import create_model
from dataset import create_dataloaders
from training_config import configure_training

# CPU execution
device = torch.device("cpu")

# Load data
train_loader, val_loader, test_loader, classes = (
    create_dataloaders()
)

# Create model
model = create_model(num_classes=len(classes))
model = model.to(device)

# Configure loss and optimizer
criterion, optimizer = configure_training(model)

# Get one real batch
images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)

# Forward pass
model.eval()

with torch.no_grad():
    outputs = model(images)
    loss = criterion(outputs, labels)

print("\n========== BIRDVISION MODEL TEST ==========")

print("Device:", device)
print("Input shape:", images.shape)
print("Output shape:", outputs.shape)
print("Number of classes:", len(classes))
print("Loss:", loss.item())
print("Trainable parameters:",
      sum(p.numel() for p in model.parameters()
          if p.requires_grad))

assert outputs.shape == (16, 20)

print("\nRESNET-50 MODEL WORKING SUCCESSFULLY!")
