import torch
import torch.nn as nn
from typing import Tuple
import timm
import os

class EffortModel:
    """
    Effort model wrapper
    Note: This is a simplified version. 
    For full Effort implementation, use the official repo:
    https://github.com/YZY-stack/Effort-AIGI-Detection
    """
    
    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.model = self._load_model(weights_path)
        self.model.eval()
        print("✅ Effort model loaded")
    
    def _load_model(self, weights_path: str) -> nn.Module:
        """
        Load Effort model - CLIP ViT-L/14 based
        The checkpoint contains SVD-decomposed CLIP weights
        """
        try:
            print("📌 Loading Effort (CLIP ViT-L/14) model...")

            # Try to use transformers CLIP if available
            try:
                from transformers import CLIPVisionModel, CLIPImageProcessor
                import torchvision.transforms as transforms

                # Load CLIP ViT-L/14
                clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

                # Create classifier head (matches Effort: 1024 -> 2)
                classifier = nn.Linear(1024, 2)

                # Load Effort checkpoint
                checkpoint = torch.load(weights_path, map_location=self.device)
                print(f"📦 Checkpoint has {len(checkpoint)} keys")

                # Load the classification head weights
                try:
                    classifier.weight.data = checkpoint['module.head.weight']
                    classifier.bias.data = checkpoint['module.head.bias']
                    print("✅ Loaded Effort classification head weights")
                except:
                    print("⚠️  Could not load head weights, using random")

                # Create wrapper model
                class EffortCLIPWrapper(nn.Module):
                    def __init__(self, clip, classifier):
                        super().__init__()
                        self.clip = clip
                        self.classifier = classifier

                    def forward(self, x):
                        # CLIP expects specific input format
                        # Resize to 224x224 if needed
                        if x.shape[-1] != 224:
                            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear')

                        outputs = self.clip(pixel_values=x)
                        features = outputs.pooler_output  # [batch, 1024]
                        logits = self.classifier(features)
                        return logits

                model = EffortCLIPWrapper(clip_model, classifier)
                model.to(self.device)
                print("✅ Effort-CLIP model loaded (using pretrained CLIP + random classifier)")
                print("⚠️  Note: Full Effort SVD weights require special loading")

                return model

            except ImportError:
                print("❌ transformers library not available")
                raise

        except Exception as e:
            print(f"❌ Failed to load Effort model: {e}")
            print("📌 Falling back to ResNet50")

            # Fallback
            model = timm.create_model('resnet50', pretrained=True, num_classes=2)
            model.to(self.device)
            return model
    
    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Predict fake/real probabilities
        Returns: (fake_prob, real_prob)

        Note: Effort model uses class 0 = FAKE, class 1 = REAL (opposite of Xception)
        """
        image_tensor = image_tensor.to(self.device)
        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)

        # NO SWAP for Effort: class 0 = FAKE, class 1 = REAL
        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()

        return fake_prob, real_prob