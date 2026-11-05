
import torch.nn as nn
import torch.optim as optim


def configure_training(model):
    # Multi-class classification loss
    criterion = nn.CrossEntropyLoss()

    # Optimize only the trainable classification layer
    optimizer = optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=0.001,
        weight_decay=0.01
    )

    return criterion, optimizer

