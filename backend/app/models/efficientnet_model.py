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
        print("[OK] EfficientNet-B4 model loaded")
    
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
            orig_k = k

            # Remove prefixes FIRST
            k = k.replace('module.', '').replace('encoder.', '').replace('backbone.', '').replace('efficientnet.', '')

            # Map classifier
            if 'last_layer.' in k:
                k = k.replace('last_layer.', 'classifier.')
            if 'last_linear.' in k:
                k = k.replace('last_linear.', 'classifier.')

            # Map stem layers
            if '_conv_stem.' in k:
                k = k.replace('_conv_stem.', 'conv_stem.')
            if '_bn0.' in k:
                k = k.replace('_bn0.', 'bn1.')

            # Map block structure: _blocks.X -> blocks.X.0
            # Example: _blocks.0._depthwise_conv -> blocks.0.0.conv_dw
            if '_blocks.' in k:
                import re
                # Extract block number
                match = re.match(r'_blocks\.(\d+)\.(.*)', k)
                if match:
                    block_num = match.group(1)
                    rest = match.group(2)

                    # Map layer names within blocks
                    rest = rest.replace('_depthwise_conv.', 'conv_dw.')
                    rest = rest.replace('_project_conv.', 'conv_pw.')
                    rest = rest.replace('_expand_conv.', 'conv_pwl.')
                    rest = rest.replace('_se_reduce.', 'se.conv_reduce.')
                    rest = rest.replace('_se_expand.', 'se.conv_expand.')

                    # Reconstruct: blocks.X.0.layer
                    k = f'blocks.{block_num}.0.{rest}'

            # Map head layers
            if '_conv_head.' in k:
                k = k.replace('_conv_head.', 'conv_head.')
            if '_bn1.' in k and 'blocks' not in k:
                k = k.replace('_bn1.', 'bn2.')

            # Skip adjust_channel layers
            if 'adjust_channel' not in k:
                new_state_dict[k] = v

        print(f"[DEBUG] Mapped {len(new_state_dict)} keys")

        # Check if classifier layer is present
        has_classifier = any('classifier.' in k for k in new_state_dict.keys())
        print(f"[DEBUG] Classifier layer found: {has_classifier}")

        result = model.load_state_dict(new_state_dict, strict=False)
        print(f"[DEBUG] EfficientNet loading: {len(new_state_dict)} keys loaded")
        if result.missing_keys:
            print(f"[WARNING] Missing keys: {len(result.missing_keys)} (first 3: {result.missing_keys[:3]})")
        if result.unexpected_keys:
            print(f"[WARNING] Unexpected keys: {len(result.unexpected_keys)}")
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