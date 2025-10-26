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
from typing import Optional

# MLflow integration
try:
    from services.mlflow_service import MLflowService
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    MLflowService = None

# Heatmap analyzer
try:
    from services.heatmap_analyzer import HeatmapAnalyzer
    HEATMAP_ANALYZER_AVAILABLE = True
    print("DEBUG: HeatmapAnalyzer import SUCCESS")
except ImportError as e:
    HEATMAP_ANALYZER_AVAILABLE = False
    HeatmapAnalyzer = None
    print(f"DEBUG: HeatmapAnalyzer import FAILED: {e}")

class EnsembleDetectionService:
    """Detection service with ensemble of 3 models"""

    def __init__(self, config_path: str = "config.json", enable_mlflow: bool = True):
        print("INFO: Initializing Ensemble Detection Service...")

        # Load ensemble model manager
        self.model_manager = EnsembleModelManager(config_path=config_path)

        # Initialize MLflow
        self.mlflow = None
        if enable_mlflow and MLFLOW_AVAILABLE:
            try:
                self.mlflow = MLflowService(experiment_name="deepfake_detection")
                self.mlflow.log_model_info(self.model_manager.config)
            except Exception as e:
                print(f"WARNING: MLflow initialization failed: {e}")
                self.mlflow = None

        # Initialize Heatmap Analyzer
        self.heatmap_analyzer = None
        if HEATMAP_ANALYZER_AVAILABLE:
            try:
                self.heatmap_analyzer = HeatmapAnalyzer()
                print("SUCCESS: Heatmap Analyzer initialized")
            except Exception as e:
                print(f"ERROR: Heatmap Analyzer init failed: {e}")
                self.heatmap_analyzer = None
        else:
            print("WARNING: HeatmapAnalyzer not available (import failed)")

        # Face detector (Force CPU due to torchvision::nms CUDA compatibility issue)
        print("INFO: Loading face detector...")
        self.face_detector = MTCNN(
            keep_all=False,
            device='cpu',  # Force CPU to avoid torchvision::nms CUDA error
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

        print("SUCCESS: Ensemble Detection Service ready!")
    
    def _detect_face(self, image: Image.Image) -> tuple:
        """Detect and crop face"""
        img_np = np.array(image)
        boxes, probs = self.face_detector.detect(image)
        
        if boxes is None or len(boxes) == 0:
            raise ValueError("No face detected. Please upload a clear face photo.")
        
        best_idx = np.argmax(probs)
        box = boxes[best_idx]
        confidence = probs[best_idx]
        
        if confidence < 0.85:
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
    
    def process(self, image: Image.Image, generate_heatmap: bool = True,
                metadata: Optional[dict] = None) -> dict:
        """
        Full ensemble detection pipeline

        Args:
            image: Input image
            generate_heatmap: Generate Grad-CAM visualization
            metadata: Additional metadata for logging (filename, source, etc.)
        """
        import time
        start_time = time.time()

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

        processing_time = time.time() - start_time

        result = {
            "prediction": ensemble['prediction'],
            "confidence": ensemble['confidence'],
            "fake_probability": ensemble['fake_prob'],
            "real_probability": ensemble['real_prob'],
            "face_detection_confidence": face_confidence,
            "total_faces_detected": 1,
            "processing_time": processing_time,

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

                # 5.1 Analyze heatmap to detect suspicious regions
                if self.heatmap_analyzer:
                    try:
                        is_fake = (ensemble['prediction'] == 'FAKE')
                        heatmap_analysis = self.heatmap_analyzer.analyze_heatmap(cam, is_fake)
                        result["heatmap_analysis"] = heatmap_analysis
                        print(f"DEBUG: Heatmap analysis generated - Top region: {heatmap_analysis['top_3_regions'][0]['region_id']}")
                    except Exception as e:
                        print(f"ERROR: Heatmap analysis failed: {e}")
                        result["heatmap_analysis"] = None
                else:
                    print("DEBUG: heatmap_analyzer is None - skipping analysis")
                    result["heatmap_analysis"] = None

            except Exception as e:
                print(f"WARNING: Grad-CAM generation failed: {e}")
                result["gradcam"] = None
                result["heatmap_analysis"] = None

        # 6. Log to MLflow
        if self.mlflow:
            try:
                self.mlflow.log_prediction("image", result, metadata)
            except Exception as e:
                print(f"WARNING: MLflow logging failed: {e}")

        return result