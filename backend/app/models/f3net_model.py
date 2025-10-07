import torch
import torch.nn as nn
import timm
from typing import Tuple
import torch.nn.functional as F

class F3NetModel:
    """F3Net model wrapper - handles 12-channel input (RGB + noise features)"""

    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.model = self._load_model(weights_path)
        self.model.eval()
        print("✅ F3Net model loaded")

    def _load_model(self, weights_path: str) -> nn.Module:
        """Load F3Net model with 12-channel input"""
        # Create base Xception model
        model = timm.create_model('xception', pretrained=False, num_classes=2)

        # Modify first conv to accept 12 channels
        original_conv1 = model.conv1
        model.conv1 = nn.Conv2d(
            12,  # 12 input channels instead of 3
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )

        # Load weights
        checkpoint = torch.load(weights_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        else:
            state_dict = checkpoint

        # Clean up state dict
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove prefixes
            k = k.replace('module.', '').replace('encoder.', '').replace('backbone.', '')

            # Map last_linear to fc
            if 'last_linear.' in k:
                k = k.replace('last_linear.', 'fc.')

            # Skip adjust_channel layers
            if 'adjust_channel' not in k:
                new_state_dict[k] = v

        result = model.load_state_dict(new_state_dict, strict=False)

        print(f"🔍 F3Net loading: {len(new_state_dict)} keys loaded")
        if result.missing_keys:
            print(f"⚠️  Missing keys: {len(result.missing_keys)}")

        model.to(self.device)
        return model

    def _extract_noise_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract noise features for F3Net
        F3Net uses RGB (3ch) + various noise features (9ch) = 12 channels total
        """
        # Simple noise extraction using high-pass filters
        # Sobel filters for edges
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=image_tensor.dtype, device=image_tensor.device)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=image_tensor.dtype, device=image_tensor.device)

        noise_features = []

        # Process each RGB channel
        for i in range(3):
            channel = image_tensor[:, i:i+1, :, :]

            # Edge detection (2 features per channel = 6 total)
            edge_x = F.conv2d(channel, sobel_x.unsqueeze(1), padding=1)
            edge_y = F.conv2d(channel, sobel_y.unsqueeze(1), padding=1)

            noise_features.append(edge_x)
            noise_features.append(edge_y)

        # Add 3 more noise features (simple high-frequency components)
        for i in range(3):
            channel = image_tensor[:, i:i+1, :, :]
            # High-pass filter
            blurred = F.avg_pool2d(channel, kernel_size=3, stride=1, padding=1)
            high_freq = channel - blurred
            noise_features.append(high_freq)

        # Concatenate: 3 RGB + 9 noise = 12 channels
        return torch.cat([image_tensor] + noise_features, dim=1)

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Predict fake/real probabilities
        Returns: (fake_prob, real_prob)

        Note: Class 0 = REAL, Class 1 = FAKE (swapped)
        """
        image_tensor = image_tensor.to(self.device)

        # Extract noise features and concatenate with RGB
        input_12ch = self._extract_noise_features(image_tensor)

        # Forward pass
        logits = self.model(input_12ch)
        probs = torch.softmax(logits, dim=1)

        # SWAPPED: class 0 = REAL, class 1 = FAKE
        real_prob = probs[0][0].item()
        fake_prob = probs[0][1].item()

        return fake_prob, real_prob
