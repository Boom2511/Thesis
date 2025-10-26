# -*- coding: utf-8 -*-
"""
Enhanced Grad-CAM Service for Model Interpretability
Goal 1.2.2: Explain model decisions with visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Tuple, Optional
import base64
from io import BytesIO


class EnhancedGradCAM:
    """Enhanced Grad-CAM with multi-layer support"""

    def __init__(self, model: nn.Module, target_layers: List[nn.Module]):
        """
        Initialize Grad-CAM

        Args:
            model: PyTorch model
            target_layers: List of layers to visualize
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks"""

        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook

        # Register hooks for each target layer
        for idx, layer in enumerate(self.target_layers):
            name = f"layer_{idx}"
            self.hooks.append(
                layer.register_forward_hook(forward_hook(name))
            )
            self.hooks.append(
                layer.register_full_backward_hook(backward_hook(name))
            )

    def generate_cam(self, input_tensor: torch.Tensor,
                     target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmaps

        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class for CAM (None = predicted class)

        Returns:
            Dict of layer_name -> heatmap
        """
        self.model.eval()
        self.gradients.clear()
        self.activations.clear()

        # Forward pass
        output = self.model(input_tensor)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        target = output[0, target_class]
        target.backward()

        # Generate CAMs for each layer
        cams = {}
        for idx, layer in enumerate(self.target_layers):
            name = f"layer_{idx}"

            if name not in self.gradients or name not in self.activations:
                continue

            # Get gradients and activations
            gradients = self.gradients[name]  # (B, C, H, W)
            activations = self.activations[name]  # (B, C, H, W)

            # Global average pooling of gradients
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

            # Weighted combination of activation maps
            cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)

            # ReLU to keep positive values only
            cam = F.relu(cam)

            # Normalize
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            cams[name] = cam

        return cams

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()


class GradCAMVisualizer:
    """Visualize Grad-CAM with various overlays"""

    @staticmethod
    def apply_colormap(heatmap: np.ndarray,
                       colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Apply colormap to heatmap

        Args:
            heatmap: Normalized heatmap (0-1)
            colormap: OpenCV colormap

        Returns:
            RGB heatmap image
        """
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        return colored_heatmap

    @staticmethod
    def overlay_heatmap(image: np.ndarray,
                        heatmap: np.ndarray,
                        alpha: float = 0.5) -> np.ndarray:
        """
        Overlay heatmap on image

        Args:
            image: Original image (H, W, 3), values 0-255
            heatmap: Heatmap (H, W)
            alpha: Overlay transparency (0-1)

        Returns:
            Overlayed image
        """
        # Resize heatmap to match image
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap
        colored_heatmap = GradCAMVisualizer.apply_colormap(heatmap_resized)

        # Ensure image is in correct format
        if image.max() <= 1.0:
            image = np.uint8(255 * image)

        # Overlay
        overlayed = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)

        return overlayed

    @staticmethod
    def create_multi_view(image: np.ndarray,
                          heatmap: np.ndarray,
                          title: str = "Grad-CAM") -> np.ndarray:
        """
        Create multi-view visualization

        Args:
            image: Original image
            heatmap: Heatmap
            title: Title text

        Returns:
            Multi-view image grid
        """
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Original image
        img_orig = image.copy()

        # Heatmap only
        img_heatmap = GradCAMVisualizer.apply_colormap(heatmap_resized)

        # Overlay
        img_overlay = GradCAMVisualizer.overlay_heatmap(image, heatmap, alpha=0.5)

        # Combine into grid
        top_row = np.hstack([img_orig, img_heatmap])
        bottom_row = np.hstack([img_overlay, img_overlay])  # Can add more views
        grid = np.vstack([top_row, bottom_row])

        # Add title
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(grid, title, (10, 30), font, 1, (255, 255, 255), 2)

        return grid

    @staticmethod
    def encode_image_base64(image: np.ndarray) -> str:
        """
        Encode image as base64

        Args:
            image: NumPy array image (RGB)

        Returns:
            Base64 encoded string with data URI
        """
        # Convert to PIL Image
        if image.max() > 1:
            image = np.uint8(image)
        else:
            image = np.uint8(255 * image)

        pil_image = Image.fromarray(image)

        # Encode
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return f"data:image/jpeg;base64,{img_base64}"


class ModelExplainer:
    """Explain model predictions using Grad-CAM"""

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize explainer

        Args:
            model: PyTorch model
            device: torch device
        """
        self.model = model
        self.device = device
        self.grad_cam = None

        self._setup_gradcam()

    def _setup_gradcam(self):
        """Setup Grad-CAM for the model"""
        # Find target layers
        target_layers = self._find_target_layers()

        if target_layers:
            self.grad_cam = EnhancedGradCAM(self.model, target_layers)
            print(f"[+] Grad-CAM initialized with {len(target_layers)} layers")
        else:
            print("[!] Warning: No suitable layers found for Grad-CAM")

    def _find_target_layers(self) -> List[nn.Module]:
        """Find suitable layers for Grad-CAM"""
        target_layers = []

        # For EfficientNet
        if hasattr(self.model, 'conv_head'):
            target_layers.append(self.model.conv_head)

        # For Xception
        if hasattr(self.model, 'block12'):
            target_layers.append(self.model.block12.rep[-2])

        # For ResNet-like models
        if hasattr(self.model, 'layer4'):
            target_layers.append(self.model.layer4[-1])

        # Generic: find last conv layer
        if not target_layers:
            for layer in reversed(list(self.model.modules())):
                if isinstance(layer, nn.Conv2d):
                    target_layers.append(layer)
                    break

        return target_layers

    def explain(self, image_tensor: torch.Tensor,
                original_image: np.ndarray,
                prediction: str,
                confidence: float) -> Dict:
        """
        Generate explanation for prediction

        Args:
            image_tensor: Preprocessed input tensor
            original_image: Original image (RGB, 0-255)
            prediction: Model prediction (FAKE/REAL)
            confidence: Prediction confidence

        Returns:
            Dict with explanations and visualizations
        """
        if self.grad_cam is None:
            return {
                'explanation': "Grad-CAM not available for this model",
                'heatmap': None,
                'overlay': None
            }

        # Generate Grad-CAM
        cams = self.grad_cam.generate_cam(image_tensor, target_class=None)

        if not cams:
            return {
                'explanation': "Failed to generate Grad-CAM",
                'heatmap': None,
                'overlay': None
            }

        # Get first (usually most informative) CAM
        cam = list(cams.values())[0]

        # Generate visualizations
        overlay = GradCAMVisualizer.overlay_heatmap(
            original_image, cam, alpha=0.5
        )

        # Generate explanation text
        explanation = self._generate_explanation_text(
            cam, prediction, confidence
        )

        # Encode visualizations
        heatmap_base64 = GradCAMVisualizer.encode_image_base64(
            GradCAMVisualizer.apply_colormap(cam)
        )
        overlay_base64 = GradCAMVisualizer.encode_image_base64(overlay)

        return {
            'explanation': explanation,
            'heatmap': heatmap_base64,
            'overlay': overlay_base64,
            'attention_score': float(cam.max())
        }

    def _generate_explanation_text(self, cam: np.ndarray,
                                   prediction: str,
                                   confidence: float) -> str:
        """Generate human-readable explanation"""

        # Analyze heatmap
        high_attention = (cam > 0.7).sum() / cam.size
        focused = high_attention < 0.3  # < 30% of image has high attention

        explanation_parts = []

        # Confidence explanation
        if confidence > 0.9:
            explanation_parts.append(f"High confidence ({confidence*100:.1f}%) in {prediction} classification.")
        elif confidence > 0.7:
            explanation_parts.append(f"Moderate confidence ({confidence*100:.1f}%) in {prediction} classification.")
        else:
            explanation_parts.append(f"Low confidence ({confidence*100:.1f}%) - results may be uncertain.")

        # Attention pattern explanation
        if focused:
            explanation_parts.append(
                "The model focused on specific regions (shown in red/yellow), "
                "suggesting localized artifacts or features."
            )
        else:
            explanation_parts.append(
                "The model examined distributed patterns across the image, "
                "indicating global inconsistencies."
            )

        # Prediction-specific explanation
        if prediction == "FAKE":
            explanation_parts.append(
                "Red/yellow regions show areas where the model detected potential manipulation, "
                "such as: facial inconsistencies, blending artifacts, or unnatural patterns."
            )
        else:
            explanation_parts.append(
                "The highlighted regions show areas the model used to verify authenticity, "
                "such as: natural skin texture, consistent lighting, and realistic features."
            )

        return " ".join(explanation_parts)

    def cleanup(self):
        """Clean up resources"""
        if self.grad_cam:
            self.grad_cam.remove_hooks()


# Example usage
def demo_gradcam():
    """Demonstrate Grad-CAM usage"""
    print("\n" + "="*60)
    print("Enhanced Grad-CAM Demo".center(60))
    print("="*60)

    print("\n[*] Grad-CAM provides visual explanations by highlighting")
    print("    the regions of the image that most influenced the model's decision.")

    print("\n[*] Key features:")
    print("    - Multi-layer CAM generation")
    print("    - Multiple visualization modes")
    print("    - Automatic explanation generation")
    print("    - Base64 encoding for web display")

    print("\n[*] Integration with detection service:")
    print("    - Automatic CAM generation after prediction")
    print("    - Overlay on original image")
    print("    - Human-readable explanations")

    print("="*60 + "\n")


if __name__ == "__main__":
    demo_gradcam()
