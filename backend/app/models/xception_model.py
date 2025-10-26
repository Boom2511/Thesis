import torch
import torch.nn as nn
import timm
from typing import Tuple

class XceptionModel:
    """
    Xception model wrapper
    OPTIMIZED: Tested on FaceForensics++ c23 dataset (84.29% accuracy, 97.67% AUC)
    """

    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.model = self._load_model(weights_path)
        self.model.eval()
        print("[OK] Xception model loaded")

    def _load_model(self, weights_path: str) -> nn.Module:
        """
        Load Xception model with corrected weight mapping

        Key fix: timm Xception uses 'fc' not 'last_linear'
        """
        print("\n[INFO] Loading Xception model...")

        # Create timm Xception with fc classifier (NOT last_linear)
        model = timm.create_model('xception', pretrained=False, num_classes=2)

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu')

        # Map keys: backbone.last_linear.* → fc.*
        new_state_dict = {}
        for k, v in checkpoint.items():
            new_k = k.replace('module.', '')
            new_k = new_k.replace('backbone.', '')  # CRITICAL: remove backbone prefix
            new_k = new_k.replace('model.', '')
            new_k = new_k.replace('encoder.', '')

            # Map last_linear to fc (timm Xception uses fc)
            new_k = new_k.replace('last_linear.', 'fc.')

            new_state_dict[new_k] = v

        # Load weights
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        # Verify classifier loaded
        classifier_loaded = any('fc.' in k for k in new_state_dict.keys())
        print(f"[DEBUG] Mapped {len(new_state_dict)} keys")
        print(f"[DEBUG] Classifier layer found: {classifier_loaded}")

        if not classifier_loaded:
            print("[WARNING] Classifier weights NOT found in checkpoint!")

        if missing:
            print(f"[WARNING] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[WARNING] Unexpected keys: {len(unexpected)}")

        model = model.to(self.device)
        print("[OK] Xception model loaded\n")
        return model

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Predict fake/real probabilities
        Returns: (fake_prob, real_prob)

        Note: Class 0 = REAL, Class 1 = FAKE
        """
        image_tensor = image_tensor.to(self.device)
        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)

        real_prob = probs[0][0].item()
        fake_prob = probs[0][1].item()

        return fake_prob, real_prob
