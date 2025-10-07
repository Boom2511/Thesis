import torch
import torch.nn as nn
import timm
from typing import Tuple

class EfficientNetModel:
    """EfficientNet-B4 model wrapper"""
    
    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.model = self._load_model(weights_path)
        self.model.eval()
        print("✅ EfficientNet-B4 model loaded")
    
    def _load_model(self, weights_path: str) -> nn.Module:
        """Load EfficientNet model with weights"""
        model = timm.create_model('tf_efficientnet_b4', pretrained=False, num_classes=2)
        
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove prefixes
            k = k.replace('module.', '').replace('encoder.', '').replace('backbone.', '').replace('efficientnet.', '')

            # Map layer names
            if 'last_linear.' in k:
                k = k.replace('last_linear.', 'classifier.')
            if '_conv_stem.' in k:
                k = k.replace('_conv_stem.', 'conv_stem.')
            if '_bn0.' in k:
                k = k.replace('_bn0.', 'bn1.')

            # Skip adjust_channel layers
            if 'adjust_channel' not in k:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Predict fake/real probabilities
        Returns: (fake_prob, real_prob)

        Note: Class 0 = REAL, Class 1 = FAKE (swapped)
        """
        image_tensor = image_tensor.to(self.device)
        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)

        # SWAPPED: class 0 = REAL, class 1 = FAKE
        real_prob = probs[0][0].item()
        fake_prob = probs[0][1].item()

        return fake_prob, real_prob