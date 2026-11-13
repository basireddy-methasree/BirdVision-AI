
import base64
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from matplotlib import colormaps
from torchvision.transforms import functional as TF


class GradCAMService:
    def __init__(self, model):
        self.model = model

        # Final convolutional layer of ResNet-50
        self.target_layer = model.layer4[-1]

        self.activations = None
        self.gradients = None

        # Register forward hook
        self.forward_handle = self.target_layer.register_forward_hook(
            self.save_activation
        )

    def save_activation(self, module, inputs, output):
        self.activations = output.detach()

        if output.requires_grad:
            output.register_hook(self.save_gradient)

    def save_gradient(self, gradient):
        self.gradients = gradient.detach()

    def generate(self, image_tensor):
        """
        Generate Grad-CAM heatmap for the predicted class.
        """

        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None

        image_tensor = image_tensor.detach()
        image_tensor.requires_grad_(True)

        with torch.enable_grad():

            # Forward pass
            output = self.model(image_tensor)

            probabilities = torch.softmax(output, dim=1)

            predicted_index = output.argmax(dim=1).item()

            confidence = (
                probabilities[0, predicted_index].item() * 100
            )

            # Backpropagate predicted class score
            score = output[:, predicted_index]
            score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # Calculate channel-wise importance weights
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted combination of feature maps
        cam = torch.sum(
            weights * self.activations,
            dim=1,
            keepdim=True
        )

        # Remove negative values
        cam = F.relu(cam)

        # Resize heatmap to model input dimensions
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        # Normalize heatmap
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        heatmap = cam.detach().cpu().numpy()

        return heatmap, predicted_index, confidence

    @staticmethod
    def image_to_base64(image):
        """
        Convert PIL image into Base64 PNG.
        """

        buffer = BytesIO()

        image.save(buffer, format="PNG")

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return f"data:image/png;base64,{encoded}"

    def create_visualizations(self, original_image, heatmap):
        """
        Create original image, heatmap and overlay.

        The original image uses the same resize and center-crop
        transformation as the model's evaluation preprocessing.
        """

        # Convert to RGB
        original_image = original_image.convert("RGB")

        # Match eval_transform:
        # Resize(256) -> CenterCrop(224)
        original = TF.center_crop(
            TF.resize(original_image, 256),
            [224, 224]
        )

        original_array = np.array(original)

        # Convert heatmap into a colored image
        colored_heatmap = (
            colormaps["jet"](heatmap)[..., :3] * 255
        ).astype(np.uint8)

        # Generate overlay
        overlay_array = (
            0.5 * original_array.astype(np.float32)
            + 0.5 * colored_heatmap.astype(np.float32)
        )

        overlay_array = np.clip(
            overlay_array,
            0,
            255
        ).astype(np.uint8)

        heatmap_image = Image.fromarray(
            colored_heatmap
        )

        overlay_image = Image.fromarray(
            overlay_array
        )

        return {
            "original_image": self.image_to_base64(original),
            "heatmap_image": self.image_to_base64(heatmap_image),
            "overlay_image": self.image_to_base64(overlay_image)
        }

    def close(self):
        """
        Remove the registered forward hook.
        """

        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
