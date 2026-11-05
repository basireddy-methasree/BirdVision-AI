# BirdVision AI - Deep Learning Identification of Bird Species

## Project Overview

BirdVision AI is a deep learning-based bird species identification system designed to classify bird images into 20 different species.

The project uses transfer learning with a pretrained ResNet-50 architecture and PyTorch. It also integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the image regions that contribute to the model's predictions.

The objective is to support biodiversity monitoring by making bird species identification more accessible and interpretable.

## Key Features

- Classification of 20 bird species.
- Transfer learning using pretrained ResNet-50.
- Data augmentation to improve model generalization.
- Top-1 and Top-5 prediction results.
- Grad-CAM-based visual explainability.
- Prediction on individual input images.
- Visualization of model attention regions.
- CPU-based inference without paid APIs or services.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.10 |
| Deep Learning Framework | PyTorch |
| Model Architecture | ResNet-50 |
| Computer Vision | Torchvision, OpenCV |
| Image Processing | Pillow |
| Data Analysis | NumPy, Pandas |
| Visualization | Matplotlib |
| Explainable AI | Grad-CAM |
| Development Environment | VS Code |

## Dataset

**Dataset:** Birds 525 Species Image Classification

**Source:** Hugging Face â€“ `yashikota/birds-525-species-image-classification`

Original dataset statistics:

| Split | Images | Species |
|---|---:|---:|
| Training | 84,635 | 525 |
| Validation | 2,625 | 525 |
| Testing | 2,625 | 525 |
| Total | 89,885 | 525 |

For this project, 20 species were selected and extracted into a separate processed dataset.

### Selected Dataset Statistics

| Split | Images |
|---|---:|
| Training | 4,178 |
| Validation | 100 |
| Testing | 100 |
| **Total** | **4,378** |

The raw dataset remains unchanged.

### Selected Bird Species

1. House Finch
2. Ovenbird
3. Asian Green Bee Eater
4. Wood Duck
5. Caspian Tern
6. Wood Thrush
7. Merlin
8. Oyster Catcher
9. Laughing Gull
10. Eastern Yellow Robin
11. Jacobin Pigeon
12. Dark Eyed Junco
13. Red Tailed Hawk
14. Yellow Breasted Chat
15. Violet Backed Starling
16. Violet Green Swallow
17. Limpkin
18. Red Knot
19. Sora
20. Grey Headed Chachalaca

## Model Architecture

BirdVision AI uses ResNet-50 with transfer learning.

- Loads pretrained ImageNet weights.
- Freezes the pretrained backbone.
- Replaces the final fully connected layer with a 20-class classification layer.
- Trains the classification head using the selected bird dataset.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Architecture | ResNet-50 |
| Input Image Size | 224 Ã— 224 |
| Batch Size | 16 |
| Epochs | 10 |
| Optimizer | AdamW |
| Learning Rate | 0

