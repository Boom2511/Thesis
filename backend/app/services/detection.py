from PIL import Image
import torch
import numpy as np
import cv2
import base64
from io import BytesIO
from torchvision import transforms
from facenet_pytorch import MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.manager import EnsembleModelManager

class EnsembleDetectionService:
    """Detection service with ensemble of 3 models"""
    
    def __init__(self, config_path: str = "config.json"):
        print("🚀 Initializing Ensemble Detection Service...")
        
        # Load ensemble model manager
        self.model_manager = EnsembleModelManager(config_path=config_path)
        
        # Face detector
        print("📸 Loading face detector...")
        self.face_detector = MTCNN(
            keep_all=False,
            device=self.model_manager.device,
            post_process=False,
            min_face_size=40
        )
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Grad-CAM (use first available model)
        first_model_name = list(self.model_manager.models.keys())[0]
        first_model = self.model_manager.models[first_model_name].model

        if hasattr(first_model, 'conv_head'):
            target_layers = [first_model.conv_head]
        elif hasattr(first_model, 'block12'):
            # Xception model
            target_layers = [first_model.block12.rep[-2]]
        elif hasattr(first_model, 'layer4'):
            target_layers = [first_model.layer4[-1]]
        else:
            # Find last conv layer
            for layer in reversed(list(first_model.modules())):
                if isinstance(layer, torch.nn.Conv2d):
                    target_layers = [layer]
                    break

        self.grad_cam = GradCAM(
            model=first_model,
            target_layers=target_layers
        )
        
        print("✅ Ensemble Detection Service ready!")
    
    def _detect_face(self, image: Image.Image) -> tuple:
        """Detect and crop face"""
        img_np = np.array(image)
        boxes, probs = self.face_detector.detect(image)
        
        if boxes is None or len(boxes) == 0:
            raise ValueError("No face detected. Please upload a clear face photo.")
        
        best_idx = np.argmax(probs)
        box = boxes[best_idx]
        confidence = probs[best_idx]
        
        if confidence < 0.9:
            raise ValueError(f"Face detection confidence too low ({confidence:.2f})")
        
        # Crop with padding
        x1, y1, x2, y2 = map(int, box)
        padding = 30
        h, w = img_np.shape[:2]
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        face = image.crop((x1, y1, x2, y2))
        return face, box, float(confidence)
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image"""
        return self.transform(image)
    
    def _generate_gradcam(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Generate Grad-CAM"""
        grayscale_cam = self.grad_cam(input_tensor=image_tensor, targets=None)
        return grayscale_cam[0, :]
    
    def _create_heatmap_overlay(self, original_image: Image.Image, cam: np.ndarray) -> str:
        """Create heatmap overlay as base64"""
        img_resized = original_image.resize((224, 224))
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        
        visualization = show_cam_on_image(img_np, cam, use_rgb=True)
        vis_img = Image.fromarray(visualization)
        
        buffered = BytesIO()
        vis_img.save(buffered, format="JPEG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_base64}"
    
    def process(self, image: Image.Image, generate_heatmap: bool = True) -> dict:
        """
        Full ensemble detection pipeline
        """
        # 1. Detect face
        face_image, bbox, face_confidence = self._detect_face(image)

        # Ensure face is RGB
        face_image = face_image.convert('RGB')

        # 2. Preprocess
        image_tensor = self._preprocess_image(face_image).unsqueeze(0)
        image_tensor = image_tensor.to(self.model_manager.device)

        # 3. Ensemble prediction
        ensemble_results = self.model_manager.predict_ensemble(image_tensor)

        # 4. Build result
        ensemble = ensemble_results['ensemble']
        
        result = {
            "prediction": ensemble['prediction'],
            "confidence": ensemble['confidence'],
            "fake_probability": ensemble['fake_prob'],
            "real_probability": ensemble['real_prob'],
            "face_detection_confidence": face_confidence,
            "total_faces_detected": 1,
            
            # Individual model results
            "model_predictions": ensemble_results['individual'],
            "models_used": ensemble_results['models_used'],
            "total_models": ensemble_results['total_models'],
            
            "device": str(self.model_manager.device)
        }
        
        # 5. Generate Grad-CAM
        if generate_heatmap:
            try:
                cam = self._generate_gradcam(image_tensor)
                result["gradcam"] = self._create_heatmap_overlay(face_image, cam)
            except Exception as e:
                print(f"⚠️  Grad-CAM generation failed: {e}")
                result["gradcam"] = None
        
        return result