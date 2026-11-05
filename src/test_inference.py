
import torch
from dataset import create_dataloaders
from model import create_model

print("Starting inference test...", flush=True)

# Limit CPU threads
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

print("CPU threads:", torch.get_num_threads(), flush=True)

# Load test dataset
_, _, test_loader, classes = create_dataloaders()

print("Classes:", len(classes), flush=True)

# Load model
model = create_model(
    num_classes=len(classes),
    pretrained=False
)

checkpoint = torch.load(
    "models/best_resnet50.pth",
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully.", flush=True)

# Load only one batch
images, labels = next(iter(test_loader))

print("Image batch shape:", images.shape, flush=True)

# Run inference
with torch.inference_mode():
    outputs = model(images)

print("Inference successful!", flush=True)
print("Output shape:", outputs.shape, flush=True)
print("Predictions:", outputs.argmax(dim=1).tolist(), flush=True)
print("Actual labels:", labels.tolist(), flush=True)
