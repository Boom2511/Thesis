import json
import torch
import os
from typing import Dict, List, Tuple
from models.xception_model import XceptionModel
from models.efficientnet_model import EfficientNetModel
from models.f3net_model import F3NetModel
from models.effort_model import EffortModel

class EnsembleModelManager:
    """Manages multiple models and performs ensemble prediction"""
    
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # Setup device
        device_str = self.config.get("device", "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            print("[WARNING]  CUDA not available, falling back to CPU")
            device_str = "cpu"
        
        self.device = torch.device(device_str)
        print(f"[INIT] Using device: {self.device}")
        
        # Load models
        self.models = {}
        self._load_all_models()
        
        # Warm up
        self._warmup()
    
    def _load_all_models(self):
        """Load all enabled models"""
        print("\n[LOAD] Loading models...")
        
        model_configs = self.config["models"]
        
        # Load Xception
        if model_configs["xception"]["enabled"]:
            try:
                xception_path = model_configs["xception"]["path"]
                if os.path.exists(xception_path):
                    self.models["xception"] = XceptionModel(xception_path, self.device)
                else:
                    print(f"[WARNING]  Xception weights not found at {xception_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load Xception: {e}")
        
        # Load EfficientNet-B4
        if model_configs["efficientnet_b4"]["enabled"]:
            try:
                effnet_path = model_configs["efficientnet_b4"]["path"]
                if os.path.exists(effnet_path):
                    self.models["efficientnet_b4"] = EfficientNetModel(effnet_path, self.device)
                else:
                    print(f"[WARNING]  EfficientNet weights not found at {effnet_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load EfficientNet: {e}")

        # Load F3Net
        if model_configs.get("f3net", {}).get("enabled", False):
            try:
                f3net_path = model_configs["f3net"]["path"]
                if os.path.exists(f3net_path):
                    self.models["f3net"] = F3NetModel(f3net_path, self.device)
                else:
                    print(f"[WARNING]  F3Net weights not found at {f3net_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load F3Net: {e}")

        # Load Effort
        if model_configs["effort"]["enabled"]:
            try:
                effort_path = model_configs["effort"]["path"]
                self.models["effort"] = EffortModel(effort_path, self.device)
            except Exception as e:
                print(f"[ERROR] Failed to load Effort: {e}")
        
        if not self.models:
            raise RuntimeError("[ERROR] No models loaded! Check your config and weights files.")
        
        print(f"\n[OK] Loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def _warmup(self):
        """Warm up all models"""
        print("\n[INIT] Warming up models...")
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        for name, model in self.models.items():
            try:
                model.predict(dummy_input)
                print(f"   [OK] {name} ready")
            except Exception as e:
                print(f"   [ERROR] {name} failed: {e}")

        print("[OK] All models ready!\n")
    
    @torch.no_grad()
    def predict_ensemble(self, image_tensor: torch.Tensor) -> Dict:
        """
        Ensemble prediction from all models
        
        Returns:
            {
                'ensemble': {'fake_prob': float, 'real_prob': float, 'prediction': str},
                'individual': {
                    'xception': {'fake_prob': float, 'real_prob': float},
                    'efficientnet_b4': {'fake_prob': float, 'real_prob': float},
                    'effort': {'fake_prob': float, 'real_prob': float}
                }
            }
        """
        ensemble_config = self.config["ensemble"]
        model_configs = self.config["models"]
        
        individual_results = {}
        weighted_fake_prob = 0.0
        weighted_real_prob = 0.0
        total_weight = 0.0
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                fake_prob, real_prob = model.predict(image_tensor)
                
                individual_results[name] = {
                    'fake_prob': fake_prob,
                    'real_prob': real_prob,
                    'prediction': 'FAKE' if fake_prob > real_prob else 'REAL'
                }
                
                # Weighted ensemble
                weight = model_configs[name]["weight"]
                weighted_fake_prob += fake_prob * weight
                weighted_real_prob += real_prob * weight
                total_weight += weight
                
            except Exception as e:
                print(f"[WARNING]  Model {name} prediction failed: {e}")
        
        # Normalize weights
        if total_weight > 0:
            weighted_fake_prob /= total_weight
            weighted_real_prob /= total_weight
        
        # Ensemble result
        ensemble_prediction = 'FAKE' if weighted_fake_prob > weighted_real_prob else 'REAL'
        confidence = max(weighted_fake_prob, weighted_real_prob)
        
        return {
            'ensemble': {
                'fake_prob': weighted_fake_prob,
                'real_prob': weighted_real_prob,
                'prediction': ensemble_prediction,
                'confidence': confidence
            },
            'individual': individual_results,
            'models_used': list(self.models.keys()),
            'total_models': len(self.models)
        }