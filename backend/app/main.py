import time
import io
from contextlib import asynccontextmanager
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from services.detection import EnsembleDetectionService
from pydantic import BaseModel
from typing import Optional, Dict, List

service: EnsembleDetectionService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    print("🚀 Starting Ensemble Deepfake Detection Service...")
    try:
        service = EnsembleDetectionService(config_path="config.json")
        print("✅ Service ready!")
    except Exception as e:
        print(f"❌ Failed to start service: {e}")
        raise
    yield
    print("👋 Shutting down service.")

app = FastAPI(
    title="Ensemble Deepfake Detection API",
    version="3.0.0",
    description="Advanced deepfake detection using ensemble of 3 models: Xception + EfficientNet-B4 + Effort",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class DetectionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    fake_probability: float
    real_probability: float
    processing_time: float
    face_detection_confidence: Optional[float]
    gradcam: Optional[str]
    model_predictions: Dict
    models_used: List[str]
    total_models: int
    device: str
    error: Optional[str] = None

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Ensemble Deepfake Detection API",
        "version": "3.0.0",
        "status": "running",
        "models": service.model_manager.models.keys() if service else []
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(service.model_manager.models.keys()) if service else [],
        "total_models": len(service.model_manager.models) if service else 0,
        "device": str(service.model_manager.device) if service else "unknown"
    }

@app.post("/api/detect/image", response_model=DetectionResponse, tags=["Detection"])
async def detect_image(
    file: UploadFile = File(...),
    generate_heatmap: bool = Query(True, description="Generate Grad-CAM heatmap")
):
    """
    🎯 Ensemble deepfake detection using 3 models
    
    - **file**: Image file (JPG, PNG, etc.)
    - **generate_heatmap**: Generate Grad-CAM visualization
    
    Returns ensemble prediction with individual model results
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        start_time = time.time()
        
        contents = await file.read()
        
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        if image.size[0] < 100 or image.size[1] < 100:
            raise HTTPException(status_code=400, detail="Image too small (min 100x100)")
        
        # Process with ensemble
        result = service.process(image, generate_heatmap=generate_heatmap)
        
        result["filename"] = file.filename
        result["processing_time"] = time.time() - start_time
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/models", tags=["Info"])
async def get_models_info():
    """Get information about loaded models"""
    if not service:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "models": list(service.model_manager.models.keys()),
        "config": service.model_manager.config["models"],
        "ensemble_method": service.model_manager.config["ensemble"]["method"]
    }