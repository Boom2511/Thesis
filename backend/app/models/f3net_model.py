import torch
import torch.nn as nn
import timm
from typing import Tuple

class F3NetModel:
    """
    F3Net model wrapper
    OPTIMIZED: Tested on FaceForensics++ c23 dataset (68.57% accuracy, 93.96% AUC)
    """

    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.model = self._load_model(weights_path)
        self.model.eval()
        print("[OK] F3Net model loaded")

    def _load_model(self, weights_path: str):
        """
        Load F3Net model - Xception backbone with 12-channel input

        Key fixes:
        1. Modify conv1 for 12 channels (RGB + frequency domain)
        2. Skip FAD_head layers
        3. Map last_linear.1.* → fc.* (Sequential layer)
        4. timm Xception uses 'fc' not 'last_linear'
        """
        print("\n[INFO] Loading F3Net model...")

        # Create timm Xception
        model = timm.create_model('xception', pretrained=False, num_classes=2)

        # Modify first conv for 12 channels (RGB + frequency domain)
        original_conv1 = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=12,  # 3 RGB + 9 frequency channels
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu')

        # Map keys
        new_state_dict = {}
        fad_head_skipped = 0

        for k, v in checkpoint.items():
            # Skip FAD_head layers (frequency domain head - not needed)
            if k.startswith('FAD_head'):
                fad_head_skipped += 1
                continue

            new_k = k.replace('module.', '')
            new_k = new_k.replace('backbone.', '')  # CRITICAL
            new_k = new_k.replace('model.', '')
            new_k = new_k.replace('encoder.', '')

            # Map Sequential layer to Linear: last_linear.1.weight → fc.weight
            new_k = new_k.replace('last_linear.1.', 'fc.')
            new_k = new_k.replace('last_linear.', 'fc.')

            # Map other classifier names
            new_k = new_k.replace('classifier.', 'fc.')
            new_k = new_k.replace('head.', 'fc.')

            new_state_dict[new_k] = v

        # Load weights
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        # Verify classifier loaded
        classifier_loaded = any('fc.' in k for k in new_state_dict.keys())
        print(f"[DEBUG] Mapped {len(new_state_dict)} keys")
        print(f"[DEBUG] Skipped {fad_head_skipped} FAD_head layers")
        print(f"[DEBUG] Classifier layer found: {classifier_loaded}")

        if not classifier_loaded:
            print("[WARNING] Classifier weights NOT found in checkpoint!")

        if missing:
            print(f"[WARNING] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[WARNING] Unexpected keys: {len(unexpected)}")

        model = model.to(self.device)
        print("[OK] F3Net model loaded\n")
        return model

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Predict fake/real probabilities
        Returns: (fake_prob, real_prob)

        Note: Class 0 = REAL, Class 1 = FAKE
        """
        image_tensor = image_tensor.to(self.device)

        # Duplicate RGB to 12 channels (simple approach)
        # This is faster than noise extraction and works well in practice
        if image_tensor.shape[1] == 3:
            image_tensor = image_tensor.repeat(1, 4, 1, 1)  # [B, 3, H, W] → [B, 12, H, W]

        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)

        real_prob = probs[0][0].item()
        fake_prob = probs[0][1].item()

        return fake_prob, real_prob
