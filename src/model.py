
import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=20, pretrained=True):
    """
    Create a ResNet-50 model for BirdVision AI.
    """

    # Load pretrained ResNet-50
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet50(weights=weights)

    # Freeze pretrained backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer
    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        num_classes
    )

    # New classification layer is trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


if __name__ == "__main__":
    model = create_model(num_classes=20)

    print(model)
    print("\nModel created successfully!")
    print("Output classes:", model.fc.out_features)

    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel() for p in model.parameters()
    )

    print("Total parameters:", total)
    print("Trainable parameters:", trainable)
