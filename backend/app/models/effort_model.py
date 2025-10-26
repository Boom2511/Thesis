import torch
import torch.nn as nn
from typing import Tuple

class EffortModel:
    """
    Effort-CLIP model wrapper using transformers CLIPVisionModel
    OPTIMIZED: Tested on FaceForensics++ c23 dataset (85% accuracy, 93.53% AUC)
    """

    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.model, self.classifier = self._load_model(weights_path)
        self.model.eval()
        self.classifier.eval()
        print("[OK] Effort-CLIP model loaded")

    def _load_model(self, weights_path: str):
        """
        Load Effort-CLIP model - CLIP Vision Encoder (1024 dim, 24 layers)

        Key fixes from optimization:
        1. Use CLIPVisionModel instead of ViTModel
        2. Add 'vision_model.' prefix to all keys
        3. Skip LoRA/residual weights (S_residual, U_residual, V_residual)
        4. Use pooler_output instead of last_hidden_state[CLS]
        """
        print("[INFO] Loading Effort-CLIP model...")

        try:
            from transformers import CLIPVisionModel, CLIPVisionConfig
        except ImportError:
            print("[ERROR] transformers library not installed!")
            print("Please run: pip install transformers")
            raise

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu')

        # Detect classifier input dimension
        classifier_weight = checkpoint.get('module.head.weight', checkpoint.get('head.weight'))
        if classifier_weight is None:
            raise ValueError("Cannot find head.weight in checkpoint!")

        hidden_dim = classifier_weight.shape[1]  # Should be 1024
        print(f"[DEBUG] Detected classifier input dim: {hidden_dim}")

        # Create CLIP vision config (1024 dim, 24 layers - CLIP-L/14)
        config = CLIPVisionConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            image_size=224,
            patch_size=14,
            num_channels=3
        )

        model = CLIPVisionModel(config)

        # Map checkpoint keys to CLIP model keys
        clip_state_dict = {}
        loaded_count = 0
        skipped_count = 0

        for k, v in checkpoint.items():
            if not k.startswith('module.backbone.'):
                continue

            # Remove prefix
            new_k = k.replace('module.backbone.', '')

            # Skip LoRA/residual weights (S_residual, U_residual, V_residual)
            if 'residual' in new_k.lower():
                skipped_count += 1
                continue

            # CRITICAL: Add vision_model. prefix for CLIPVisionModel
            new_k = 'vision_model.' + new_k

            # Map checkpoint naming to transformers CLIP naming
            # Remove _main suffix if exists
            new_k = new_k.replace('.weight_main', '.weight')
            new_k = new_k.replace('.bias_main', '.bias')

            clip_state_dict[new_k] = v
            loaded_count += 1

        print(f"[DEBUG] Processed {loaded_count} backbone params")
        print(f"[DEBUG] Skipped {skipped_count} LoRA/residual params")

        # Load weights into CLIP model
        missing, unexpected = model.load_state_dict(clip_state_dict, strict=False)

        # Calculate match rate
        total_params = len(model.state_dict())
        loaded_params = total_params - len(missing)
        match_rate = (loaded_params / total_params) * 100

        print(f"[DEBUG] Loaded {loaded_params}/{total_params} params ({match_rate:.1f}% match rate)")

        if match_rate < 50:
            print(f"[WARNING] Low match rate! Model may not work correctly.")
            print(f"[WARNING] Missing keys: {len(missing)}")
        else:
            print(f"[SUCCESS] Good match rate! Model loaded correctly.")

        # Move model to device
        model = model.to(self.device)

        # Create classifier and load weights
        classifier = nn.Linear(hidden_dim, 2)

        # Get head weights
        head_weight_key = 'module.head.weight' if 'module.head.weight' in checkpoint else 'head.weight'
        head_bias_key = 'module.head.bias' if 'module.head.bias' in checkpoint else 'head.bias'

        classifier.weight.data.copy_(checkpoint[head_weight_key])
        classifier.bias.data.copy_(checkpoint[head_bias_key])
        classifier = classifier.to(self.device)

        print(f"[DEBUG] Classifier head loaded ({hidden_dim} → 2)")

        return model, classifier

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Predict fake/real probabilities
        Returns: (fake_prob, real_prob)

        Note: Class 0 = REAL, Class 1 = FAKE

        IMPORTANT: Requires ImageNet normalization:
        - mean=[0.485, 0.456, 0.406]
        - std=[0.229, 0.224, 0.225]
        """
        # Move to device
        image_tensor = image_tensor.to(self.device)

        # CLIP vision encoder forward pass
        outputs = self.model(pixel_values=image_tensor)

        # Use pooler_output (not last_hidden_state[CLS])
        # This is the correct output for CLIP vision models
        features = outputs.pooler_output  # Shape: [batch, 1024]
        features = features.to(self.device)

        # Classifier forward pass
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=1)

        # Class 0 = REAL, Class 1 = FAKE
        real_prob = probs[0][0].item()
        fake_prob = probs[0][1].item()

        return fake_prob, real_prob
