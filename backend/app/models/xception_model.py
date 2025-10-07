import torch
import torch.nn as nn
import timm
from typing import Tuple

class XceptionModel:
    """Xception model wrapper"""
    
    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.model = self._load_model(weights_path)
        self.model.eval()
        print("✅ Xception model loaded")
    
    def _load_model(self, weights_path: str) -> nn.Module:
        """Load Xception model with weights"""
        # Create model
        model = timm.create_model('xception', pretrained=False, num_classes=2)
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove prefixes and map layer names
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove common prefixes
            k = k.replace('module.', '').replace('encoder.', '').replace('backbone.', '')

            # Map last_linear to fc
            if 'last_linear.' in k:
                k = k.replace('last_linear.', 'fc.')

            # Skip adjust_channel layers
            if 'adjust_channel' not in k:
                new_state_dict[k] = v

        result = model.load_state_dict(new_state_dict, strict=False)

        # Debug: Print loading results
        print(f"🔍 Xception loading: {len(new_state_dict)} keys loaded")
        if result.missing_keys:
            print(f"⚠️  Missing keys: {len(result.missing_keys)} (first 3: {result.missing_keys[:3]})")
        if result.unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(result.unexpected_keys)} (first 3: {result.unexpected_keys[:3]})")
        model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Predict fake/real probabilities
        Returns: (fake_prob, real_prob)

        Note: The model was trained with class 0 = REAL, class 1 = FAKE
        So we swap the indices when reading from the output
        """
        image_tensor = image_tensor.to(self.device)
        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)

        # SWAPPED: class 0 = REAL, class 1 = FAKE
        real_prob = probs[0][0].item()
        fake_prob = probs[0][1].item()

        return fake_prob, real_prob